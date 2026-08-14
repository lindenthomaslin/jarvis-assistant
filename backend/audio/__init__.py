"""
音频模块：录音、播放、唤醒词检测。
"""
from .recorder import AudioRecorder
from .player import AudioPlayer
from .wake_word import WakeWordDetector

__all__ = ["AudioRecorder", "AudioPlayer", "WakeWordDetector"]
