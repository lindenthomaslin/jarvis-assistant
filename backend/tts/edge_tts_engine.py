"""
Edge-TTS 语音合成引擎
免费、低延迟，并可通过 pydub 添加电子音效，营造未来感。
"""
import asyncio
import logging
import re
from io import BytesIO
from typing import Optional

from backend.config import CFG
from backend.utils.audio_effects import apply_jarvis_effects

logger = logging.getLogger(__name__)


class EdgeTTSEngine:
    """
    基于 edge-tts 的语音合成器。
    自动根据文本语言选择中英文音色，并添加 JARVIS 风格的电子音效。
    """

    def __init__(self):
        self.voice_zh = CFG.TTS_EDGE_VOICE
        self.voice_en = CFG.TTS_EDGE_VOICE_EN
        self.speed = CFG.TTS_SPEED
        self.pitch = CFG.TTS_PITCH
        self.volume = CFG.TTS_VOLUME
        self.effects_enabled = CFG.TTS_EFFECTS_ENABLED
        self.effects = CFG.TTS_EFFECTS

    def _detect_language(self, text: str) -> str:
        """简单判断文本主要语言，返回 'zh' 或 'en'。"""
        # 统计中文字符数量
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_chars = len(re.sub(r"\s", "", text))
        if total_chars == 0:
            return "zh"
        ratio = chinese_chars / total_chars
        return "zh" if ratio > 0.3 else "en"

    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        add_effects: bool = True,
    ) -> bytes:
        """
        合成语音，返回 MP3 字节数据。
        若指定 output_path，则同时保存到文件。
        """
        try:
            import edge_tts
        except ImportError as e:
            raise RuntimeError("未安装 edge-tts，无法合成语音") from e

        lang = self._detect_language(text)
        voice = self.voice_zh if lang == "zh" else self.voice_en

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=self.speed,
            pitch=self.pitch,
            volume=self.volume,
        )

        mp3_buffer = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        mp3_data = mp3_buffer.getvalue()
        if not mp3_data:
            raise RuntimeError("TTS 合成失败，无音频数据")

        if add_effects and self.effects_enabled:
            mp3_data = apply_jarvis_effects(mp3_data, self.effects)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(mp3_data)
            logger.info("TTS 音频已保存: %s", output_path)

        return mp3_data

    async def synthesize_to_file(self, text: str, file_path: str) -> str:
        """合成语音并保存到指定路径。"""
        await self.synthesize(text, output_path=file_path)
        return file_path
