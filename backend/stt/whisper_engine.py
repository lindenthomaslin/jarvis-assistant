"""
本地语音识别引擎
基于 faster-whisper，支持中英文混合识别。
"""
import logging
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from backend.config import CFG, ROOT_DIR

logger = logging.getLogger(__name__)

# Languages we actually care about; other detections are usually
# mis-detections on short noisy audio.
PRIMARY_LANGUAGES = {"zh", "en"}


class WhisperSTT:
    """
    faster-whisper 封装。
    支持 PCM bytes 输入，输出识别文本。
    """

    def __init__(
        self,
        model_size: str = CFG.STT_MODEL_SIZE,
        device: str = CFG.STT_DEVICE,
        compute_type: str = CFG.STT_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """延迟加载模型。"""
        try:
            from faster_whisper import WhisperModel

            logger.info("正在加载 faster-whisper 模型: %s", self.model_size)
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(Path(ROOT_DIR) / "models" / "whisper"),
            )
            logger.info("faster-whisper 模型加载完成")
        except Exception as e:
            logger.error("faster-whisper 加载失败: %s", e)
            raise

    def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = CFG.AUDIO_SAMPLE_RATE,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """
        识别 PCM16 音频数据，返回文本。
        language: 'zh' 或 'en'，None 则自动检测。
        """
        if not self.model:
            raise RuntimeError("STT 模型未加载")

        try:
            # 转换为 numpy float32 数组
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 去除前后静音
            audio_np = self._trim_silence(audio_np)
            if len(audio_np) < sample_rate * 0.3:
                logger.debug("音频过短，可能未检测到有效语音")
                return ""

            # 临时文件方式传给 faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, sample_rate)
                temp_path = f.name

            segments, info = self.model.transcribe(
                temp_path,
                language=language,
                initial_prompt=initial_prompt or "以下是普通话和 English 混合的句子。",
                vad_filter=True,
                condition_on_previous_text=False,
            )
            detected = info.language or "auto"
            text = " ".join([seg.text.strip() for seg in segments]).strip()

            # 短音频上 Whisper 偶尔会把 initial_prompt 原样吐出来，
            # 这既不是用户指令也不是唤醒词，直接剔除。
            prompt = initial_prompt or "以下是普通话和 English 混合的句子。"
            if text.startswith(prompt):
                text = text[len(prompt) :].strip()
            if text == prompt:
                text = ""

            # If the model hallucinates a non-target language on short/noisy
            # audio, re-run with the most likely target language.
            if language is None and detected not in PRIMARY_LANGUAGES and text:
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
                fallback_lang = "zh" if has_chinese else "en"
                logger.warning(
                    "STT 误识别为 %s，使用 %s 重新识别: %s", detected, fallback_lang, text
                )
                segments, _ = self.model.transcribe(
                    temp_path,
                    language=fallback_lang,
                    initial_prompt=initial_prompt or "以下是普通话和 English 混合的句子。",
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join([seg.text.strip() for seg in segments]).strip()
                if text.startswith(prompt):
                    text = text[len(prompt) :].strip()
                detected = fallback_lang

            os.unlink(temp_path)

            logger.info("STT 结果 [%s]: %s", detected, text)
            return text
        except Exception as e:
            logger.error("语音识别失败: %s", e)
            return ""

    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = 40.0,
        keep_silence: int = 1000,
    ) -> np.ndarray:
        """使用能量阈值简单去除前后静音。"""
        # 计算 RMS 能量
        frame_length = 512
        hop_length = 256
        rms = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            rms.append(np.sqrt(np.mean(frame**2)))
        rms = np.array(rms)
        if len(rms) == 0:
            return audio

        threshold = 10 ** (-threshold_db / 20)
        active = rms > threshold
        if not np.any(active):
            return audio

        # 找到第一个和最后一个活跃帧
        first = np.argmax(active)
        last = len(active) - np.argmax(active[::-1]) - 1

        start_sample = max(0, first * hop_length - keep_silence)
        end_sample = min(len(audio), last * hop_length + frame_length + keep_silence)
        return audio[start_sample:end_sample]
