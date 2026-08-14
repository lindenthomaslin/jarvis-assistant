"""
J.A.R.V.I.S 配置中心
统一加载 .env 与 config.yaml
"""
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 加载环境变量。桌面版把密钥留在用户目录，避免将其嵌入 .app 中；
# 开发模式仍兼容项目根目录下的 .env。
if os.sys.platform == "darwin":
    load_dotenv(Path.home() / "Library" / "Application Support" / "JARVIS" / ".env")
load_dotenv(ROOT_DIR / ".env")


def load_yaml() -> dict:
    """加载 YAML 配置，若不存在则返回空字典。"""
    config_path = ROOT_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


YAML_CFG = load_yaml()


def _get(key: str, default=None, section: Optional[str] = None):
    """优先从环境变量读取，其次 YAML，最后默认值。"""
    env_key = key.upper().replace(".", "_")
    env_val = os.getenv(env_key)
    if env_val is not None:
        # 简单类型转换
        if isinstance(default, bool):
            return env_val.lower() in ("true", "1", "yes")
        if isinstance(default, int):
            try:
                return int(env_val)
            except ValueError:
                return default
        if isinstance(default, float):
            try:
                return float(env_val)
            except ValueError:
                return default
        return env_val

    # 从 YAML 读取
    data = YAML_CFG
    if section:
        data = data.get(section, {})
    val = data.get(key, default)
    return val


class Config:
    """集中式配置对象。"""

    # DeepSeek
    DEEPSEEK_API_KEY: str = _get("deepseek_api_key", "")
    DEEPSEEK_BASE_URL: str = _get("deepseek_base_url", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = _get("deepseek_model", "deepseek-v4-flash")
    DEEPSEEK_REASONING_EFFORT: str = _get("deepseek_reasoning_effort", "medium")

    # TTS
    TTS_ENGINE: str = _get("tts_engine", "edge_tts")
    TTS_VOICE: str = _get("tts_voice", "zh-CN-YunxiNeural")
    ELEVENLABS_API_KEY: Optional[str] = _get("elevenlabs_api_key", None)
    ELEVENLABS_VOICE_ID: Optional[str] = _get("elevenlabs_voice_id", None)

    # Wake word
    WAKE_WORD: str = _get("wake_word", "jarvis")
    WAKE_WORD_MODEL: str = _get("wake_word_model", "openwakeword")
    OPENWAKEWORD_MODEL_PATH: str = _get(
        "openwakeword_model_path", "models/hey_jarvis_v0.1.tflite"
    )

    # STT
    STT_MODEL_SIZE: str = _get("stt_model_size", "base")
    STT_DEVICE: str = _get("stt_device", "cpu")
    STT_COMPUTE_TYPE: str = _get("stt_compute_type", "int8")

    # Audio
    AUDIO_SAMPLE_RATE: int = _get("audio_sample_rate", 16000)
    AUDIO_CHANNELS: int = _get("audio_channels", 1)
    AUDIO_CHUNK_SIZE: int = _get("audio_chunk_size", 1024)
    VAD_AGGRESSIVENESS: int = _get("vad_aggressiveness", 2, section="speech")
    PHRASE_TIMEOUT: float = _get("phrase_timeout", 1.2, section="speech")
    MAX_RECORD_SECONDS: float = _get("max_record_seconds", 12, section="speech")
    INTERRUPTION_THRESHOLD: float = _get(
        "interruption_threshold", 0.015, section="speech"
    )
    INTERRUPTION_ENABLED: bool = _get(
        "interruption_enabled", False, section="speech"
    )
    INTERRUPTION_COMMANDS: List[str] = _get(
        "interruption_commands",
        ["打断", "停", "停一下", "停下", "别说了", "不要再说了", "闭嘴", "安静", "stop"],
        section="speech",
    )
    # 播放开始后忽略麦克风输入的时间（秒），避免一开口就被自己的回声打断
    INTERRUPTION_GRACE_SECONDS: float = _get(
        "interruption_grace_seconds", 0.8, section="speech"
    )
    # 能量打断时，麦克风能量需达到“正在播放回声能量”的多少倍才判定为用户发声
    INTERRUPTION_ECHO_RATIO: float = _get(
        "interruption_echo_ratio", 1.6, section="speech"
    )
    STT_WAKE_FALLBACK_ENABLED: bool = _get(
        "stt_wake_fallback_enabled", True, section="speech"
    )
    STT_FALLBACK_INTERVAL: float = _get(
        "stt_fallback_interval", 1.0, section="speech"
    )
    STT_FALLBACK_ENERGY_THRESHOLD: float = _get(
        "stt_fallback_energy_threshold", 0.008, section="speech"
    )

    # Server
    WS_HOST: str = _get("ws_host", "0.0.0.0")
    WS_PORT: int = _get("ws_port", 18790)

    # Personality
    SYSTEM_PROMPT: str = _get("system_prompt", "", section="personality")
    WAKE_WORDS: List[str] = _get("wake_words", ["jarvis", "贾维斯"], section="speech")

    # TTS 音效
    TTS_EFFECTS: dict = _get("effects", {}, section="tts")
    TTS_EDGE_VOICE: str = _get("edge_voice", "zh-CN-YunxiNeural", section="tts")
    TTS_EDGE_VOICE_EN: str = _get(
        "edge_voice_en", "en-US-SteffanNeural", section="tts"
    )
    TTS_SPEED: str = _get("speed", "+10%", section="tts")
    TTS_PITCH: str = _get("pitch", "+5Hz", section="tts")
    TTS_VOLUME: str = _get("volume", "+0%", section="tts")
    TTS_EFFECTS_ENABLED: bool = _get("effects_enabled", True, section="tts")


# 实例化供全局使用
CFG = Config()
