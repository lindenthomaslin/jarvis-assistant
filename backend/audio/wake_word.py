"""
唤醒词检测模块
优先使用 openWakeWord；若模型不存在或导入失败，则降级为 STT 关键词检测。
"""
import difflib
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from backend.config import CFG, ROOT_DIR

logger = logging.getLogger(__name__)

# STT 识别结果与目标词/短语匹配的最低相似度。
# openWakeWord 模型只认识英文 "Hey Jarvis"；中文“贾维斯”以及打断命令
# 依赖 Whisper 转写后的模糊匹配（含拼音匹配），所以阈值不宜过高。
MATCH_MIN_RATIO = 0.62

try:
    from pypinyin import lazy_pinyin
except ImportError:  # pragma: no cover - pypinyin 为可选依赖
    lazy_pinyin = None


def _pinyin_syllables(text: str) -> List[str]:
    """Convert Chinese text to pinyin syllables, or return [] when unavailable."""
    if lazy_pinyin is None:
        return []
    try:
        return lazy_pinyin(text)
    except Exception as exc:
        logger.debug("拼音转换失败: %s", exc)
        return []


def _fuzzy_contains(phrase: str, text: str, min_ratio: float = MATCH_MIN_RATIO) -> bool:
    """
    判断 phrase 是否“近似”出现在 text 中。

    Whisper 对短词（贾维斯/停/打断）经常转写出同音字，例如：
    - 贾维斯 -> 假为似 / 假威斯 / 夏威斯
    - 停一下 -> 评以下
    因此同时使用三种策略：
    1. 精确子串匹配；
    2. 字符 n-gram 的 difflib 相似度；
    3. 拼音音节序列的 difflib 相似度（处理同音不同字）。
    """
    phrase = phrase.lower().strip()
    text = text.lower().strip()
    if not phrase or not text:
        return False

    # 1. 精确匹配
    if phrase in text:
        return True

    # 2. 字符级模糊匹配（仅对多字短语有意义）
    n = len(phrase)
    if n >= 2 and len(text) >= n:
        best = 0.0
        for i in range(len(text) - n + 1):
            best = max(
                best,
                difflib.SequenceMatcher(None, phrase, text[i : i + n]).ratio(),
            )
        if best >= min_ratio:
            return True

    # 3. 拼音级模糊匹配
    target_py = _pinyin_syllables(phrase)
    text_py = _pinyin_syllables(text)
    if target_py and len(text_py) >= len(target_py):
        # 单音节词（如“停”）拼音必须精确出现，否则“平/评”等尾音相同的
        # 无关字会频繁误触发；多音节短语仍允许 1-2 个音节轻微偏差。
        if len(target_py) == 1:
            return target_py[0] in text_py

        # 注意用音节列表而不是字符串做 SequenceMatcher：
        # "ting xia" 与 "ping xia" 在字符串层面共享 "ing xia"（相似度 0.875），
        # 但音节层面只有 1/2 相同（0.5），后者才符合“命令是否真的接近”的直觉。
        for i in range(len(text_py) - len(target_py) + 1):
            window = text_py[i : i + len(target_py)]
            if difflib.SequenceMatcher(None, target_py, window).ratio() >= min_ratio:
                return True
        # 也检查文本尾部（打断指令通常出现在缓冲区末尾）
        tail = text_py[-len(target_py) :]
        if difflib.SequenceMatcher(None, target_py, tail).ratio() >= min_ratio:
            return True

    return False


class WakeWordDetector:
    """
    唤醒词检测器。
    支持 openWakeWord 模型或基于 STT 的滚动检测。
    """

    def __init__(
        self,
        wake_words: List[str] = None,
        interrupt_commands: List[str] = None,
        model_path: Optional[str] = None,
        sample_rate: int = CFG.AUDIO_SAMPLE_RATE,
    ):
        self.wake_words = [w.lower() for w in (wake_words or CFG.WAKE_WORDS)]
        self.interrupt_commands = [w.lower() for w in (interrupt_commands or CFG.INTERRUPTION_COMMANDS)]
        self.sample_rate = sample_rate
        self.model_path = model_path or CFG.OPENWAKEWORD_MODEL_PATH
        self.oww = None
        self.use_oww = False
        self.use_stt_fallback = False

        self._stt_buffer: List[bytes] = []
        self._stt_buffer_max_seconds = 2.0
        self._callbacks: List[Callable[[str], None]] = []

        self._init_openwakeword()

    def _init_openwakeword(self):
        """尝试初始化 openWakeWord。"""
        try:
            from openwakeword.model import Model

            full_path = Path(ROOT_DIR) / self.model_path
            if full_path.exists():
                self.oww = Model(wakeword_models=[str(full_path)])
                self.use_oww = True
                logger.info("openWakeWord 已加载: %s", full_path)
            else:
                logger.warning(
                    "未找到 openWakeWord 模型 %s，将使用 STT 关键词检测降级方案",
                    full_path,
                )
                self.use_stt_fallback = True
        except ImportError:
            logger.warning("未安装 openWakeWord，将使用 STT 关键词检测降级方案")
            self.use_stt_fallback = True
        except Exception as e:
            logger.error("openWakeWord 初始化失败: %s", e)
            self.use_stt_fallback = True

    def set_stt_fallback(self, stt_engine):
        """设置 STT 引擎用于降级检测。"""
        self.stt_engine = stt_engine

    def on_wake_word(self, callback: Callable[[str], None]):
        """注册唤醒词回调。"""
        self._callbacks.append(callback)

    def process_chunk(self, audio_bytes: bytes):
        """
        处理一帧音频数据。
        若检测到唤醒词，触发所有回调并返回检测到的词。
        """
        if self.use_oww and self.oww:
            return self._process_oww(audio_bytes)
        return self._process_stt_fallback(audio_bytes)

    def _process_oww(self, audio_bytes: bytes) -> Optional[str]:
        """使用 openWakeWord 检测。"""
        try:
            # openWakeWord 的流式预处理器要求 16-bit PCM（int16），
            # 直接传 float32 会被静音化，导致永远无法触发唤醒。
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            prediction = self.oww.predict(audio_np)

            # prediction 是字典，键为模型名，值为得分
            for model_name, score in prediction.items():
                if score > 0.5:
                    detected = self.wake_words[0] if self.wake_words else "jarvis"
                    self._trigger(detected)
                    return detected
        except Exception as e:
            logger.error("openWakeWord 预测失败: %s", e)
        return None

    def _process_stt_fallback(self, audio_bytes: bytes) -> Optional[str]:
        """
        STT 降级方案：累积约 2 秒音频，定时识别是否包含唤醒词。
        注意：这里只是一个轻量缓冲，真正的 STT 识别由调用方周期性执行。
        """
        self._stt_buffer.append(audio_bytes)
        return None

    def check_buffer_with_stt(self, transcript: str) -> Optional[str]:
        """
        外部 STT 识别结果传入，检查是否包含唤醒词。
        返回匹配到的唤醒词或 None。
        """
        text = transcript.lower().strip()
        for word in self.wake_words:
            if _fuzzy_contains(word, text, MATCH_MIN_RATIO):
                self._trigger(word)
                return word
        return None

    def check_buffer_for_interrupt(self, transcript: str) -> Optional[str]:
        """
        在 JARVIS 说话时检查用户是否说了打断命令。
        返回匹配到的命令词或 None。
        """
        text = transcript.lower().strip()
        for cmd in self.interrupt_commands:
            if _fuzzy_contains(cmd, text, MATCH_MIN_RATIO):
                logger.info("检测到打断命令: %s (文本: %s)", cmd, transcript)
                return cmd
        return None

    def _trigger(self, word: str):
        """触发唤醒回调。"""
        logger.info("唤醒词 detected: %s", word)
        for cb in self._callbacks:
            try:
                cb(word)
            except Exception as e:
                logger.error("唤醒回调执行失败: %s", e)

    def reset(self):
        """重置检测状态。"""
        self._stt_buffer.clear()
        if self.oww:
            try:
                self.oww.reset()
            except Exception:
                pass
