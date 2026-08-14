"""
音频后期处理：为 TTS 添加 JARVIS 风格的电子、混响、机器人质感。
"""
import logging
from io import BytesIO
from typing import Optional

import numpy as np
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize

logger = logging.getLogger(__name__)


def apply_jarvis_effects(audio_data: bytes, effects_cfg: Optional[dict] = None) -> bytes:
    """
    对 MP3/WAV 字节数据应用 JARVIS 风格音效。
    包括：高通滤波、轻微机器人化、短混响、EQ 提亮。
    """
    if effects_cfg is None:
        effects_cfg = {}

    try:
        audio = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
    except Exception:
        try:
            audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")
        except Exception as e:
            logger.warning("无法识别音频格式，跳过音效处理: %s", e)
            return audio_data

    # 使用 22.05kHz/16-bit 单声道处理：在音效质量与处理速度之间取平衡。
    # 48kHz 对短句 TTS 的延迟影响明显，22kHz 可将音效耗时降低约 50%。
    audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

    high_pass = int(effects_cfg.get("high_pass", 95))
    presence_frequency = int(effects_cfg.get("presence_frequency", 3200))
    presence_boost_db = float(effects_cfg.get("presence_boost_db", 3.0))
    robot_mix = float(effects_cfg.get("robot_mix", 0.10))
    ring_frequency = float(effects_cfg.get("ring_frequency", 86))
    reverb_delay = int(effects_cfg.get("reverb_delay", 46))
    reverb_decay = float(effects_cfg.get("reverb_decay", 0.20))
    second_echo_decay = float(effects_cfg.get("second_echo_decay", 0.09))

    # 1. 清理次声与低频浑浊，同时保留沉稳男声的主体。
    audio = audio.high_pass_filter(max(40, min(high_pass, 220)))

    # 2. 叠加一层高频“存在感”，避免依赖未注册的 pydub scipy EQ 扩展。
    if presence_boost_db > 0:
        presence = audio.high_pass_filter(
            max(1000, min(presence_frequency, 7000))
        ) + min(presence_boost_db, 6.0)
        audio = audio.overlay(presence)

    # 3. 低比例 ring modulation 提供电子边缘，不破坏中文辅音清晰度。
    robot_mix = max(0.0, min(robot_mix, 0.25))
    if robot_mix > 0:
        audio = _apply_robot_effect(
            audio,
            mix=robot_mix,
            carrier_hz=max(45.0, min(ring_frequency, 180.0)),
        )

    # 4. 两级短回声营造全息空间，尾音很短，避免与下一句重叠。
    reverb_decay = max(0.0, min(reverb_decay, 0.35))
    second_echo_decay = max(0.0, min(second_echo_decay, 0.18))
    if reverb_decay > 0:
        audio = _apply_holographic_reverb(
            audio,
            delay_ms=max(20, min(reverb_delay, 100)),
            first_decay=reverb_decay,
            second_decay=second_echo_decay,
        )

    # 5. 轻量压缩 + 留出 1dB 余量，降低扬声器削波风险。
    audio = compress_dynamic_range(
        audio,
        threshold=-10.0,
        ratio=3.0,
        attack=4.0,
        release=48.0,
    )
    audio = normalize(audio, headroom=1.0)

    out = BytesIO()
    audio.export(out, format="mp3", bitrate="96k")
    return out.getvalue()


def _apply_robot_effect(
    audio: AudioSegment,
    mix: float = 0.10,
    carrier_hz: float = 86.0,
) -> AudioSegment:
    """
    简单的机器人效果：用 ring modulation 生成谐波，与原声混合。
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate

    # 低频载波只形成轻微金属边缘。
    t = np.arange(len(samples)) / sample_rate
    carrier = np.sin(2 * np.pi * carrier_hz * t)

    # Ring modulation
    robot = samples * (1 + carrier) * 0.5

    # 混合
    mixed = samples * (1 - mix) + robot * mix
    mixed = np.clip(mixed, -32768, 32767)

    # 转回 AudioSegment
    robot_seg = audio._spawn(mixed.astype(audio.array_type).tobytes())
    return robot_seg


def _apply_holographic_reverb(
    audio: AudioSegment,
    delay_ms: int = 46,
    first_decay: float = 0.20,
    second_decay: float = 0.09,
) -> AudioSegment:
    """
    两级短回声。扩展尾部后再叠加，避免旧实现截断或提前回声。
    """
    tail_ms = delay_ms * 2 + 30
    silence = (
        AudioSegment.silent(duration=tail_ms, frame_rate=audio.frame_rate)
        .set_channels(audio.channels)
        .set_sample_width(audio.sample_width)
    )
    combined = audio + silence

    first_gain = -20 * np.log10(1 / max(first_decay, 0.01))
    combined = combined.overlay(audio + first_gain, position=delay_ms)

    if second_decay > 0:
        second_gain = -20 * np.log10(1 / max(second_decay, 0.01))
        combined = combined.overlay(audio + second_gain, position=delay_ms * 2)

    return combined
