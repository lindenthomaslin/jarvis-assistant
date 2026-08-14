"""
音频录制模块
支持 VAD 检测说话起止，并输出 PCM 字节流。
"""
import collections
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from backend.config import CFG

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    基于 PyAudio + WebRTC VAD 的录音器。
    支持两种模式：
    1. 持续流式监听（用于唤醒词检测）
    2. VAD 触发式录音（用于指令识别）
    """

    def __init__(
        self,
        sample_rate: int = CFG.AUDIO_SAMPLE_RATE,
        chunk_size: int = CFG.AUDIO_CHUNK_SIZE,
        channels: int = CFG.AUDIO_CHANNELS,
        vad_aggressiveness: int = CFG.VAD_AGGRESSIVENESS,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = None  # pyaudio.paInt16，延迟初始化
        self.vad_aggressiveness = vad_aggressiveness
        self.vad = None
        self.pa = None
        self.stream: Optional[object] = None
        self.is_running = False
        self.is_recording = False

        # 回调
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_ended: Optional[Callable[[bytes], None]] = None

        # 录音缓冲区
        self._record_buffer = bytearray()
        # 0.3 秒前置缓冲，既能保留语音开头，又能降低等待时间
        self._ring_buffer = collections.deque(maxlen=int(sample_rate / chunk_size * 0.3))
        self._speech_started = False
        self._last_speech_time = 0.0
        self._silence_frames = 0
        self._lock = threading.Lock()

        # 延迟导入音频库，提升启动兼容性
        self._init_audio()

    def _init_audio(self):
        """初始化 PyAudio 与 WebRTC VAD。"""
        try:
            import pyaudio
            import webrtcvad

            self.format = pyaudio.paInt16
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            self.pa = pyaudio.PyAudio()
            logger.info("音频子系统初始化成功")
        except ImportError as e:
            logger.error("缺少音频依赖，请安装 pyaudio 与 webrtcvad: %s", e)
            raise

    def list_devices(self) -> list:
        """列出可用音频输入设备。"""
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info.get("name"),
                    "channels": info.get("maxInputChannels"),
                    "rate": info.get("defaultSampleRate"),
                })
        return devices

    def start_stream(
        self,
        on_chunk: Optional[Callable[[bytes], None]] = None,
    ):
        """启动持续音频流，用于唤醒词监听。"""
        self.on_audio_chunk = on_chunk
        self.is_running = True

        self.stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._stream_callback,
        )
        self.stream.start_stream()
        logger.info("音频流已启动，采样率 %d Hz", self.sample_rate)

    def stop_stream(self):
        """停止音频流。"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logger.info("音频流已停止")

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 回调：持续接收音频块。"""
        import pyaudio

        if self.on_audio_chunk:
            self.on_audio_chunk(in_data)

        # 如果在录音状态，同时把数据写入录音缓冲
        with self._lock:
            if self.is_recording:
                self._process_vad(in_data)

        return (in_data, pyaudio.paContinue)

    def start_recording(
        self,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[bytes], None]] = None,
        max_seconds: float = CFG.MAX_RECORD_SECONDS,
    ) -> bytes:
        """
        阻塞式 VAD 录音，直到说话结束或超时。
        返回 PCM 音频数据。
        """
        self.on_speech_started = on_speech_started
        self.on_speech_ended = on_speech_ended

        with self._lock:
            self.is_recording = True
            self._record_buffer = bytearray()
            self._ring_buffer.clear()
            self._speech_started = False
            self._silence_frames = 0

        start_time = time.time()
        logger.info("开始 VAD 录音，等待用户说话...")

        try:
            while self.is_recording and (time.time() - start_time) < max_seconds:
                # 实际数据处理在回调中完成，这里只等待
                time.sleep(0.05)

            with self._lock:
                self.is_recording = False
                pcm = bytes(self._record_buffer)
        except Exception as e:
            logger.error("录音过程出错: %s", e)
            with self._lock:
                self.is_recording = False
            pcm = bytes(self._record_buffer)

        logger.info("录音结束，长度 %.2f 秒", len(pcm) / (self.sample_rate * 2))
        return pcm

    def stop_recording(self):
        """手动停止当前录音。"""
        with self._lock:
            self.is_recording = False

    def _process_vad(self, in_data: bytes):
        """WebRTC VAD 处理音频块，检测说话起止。"""
        # VAD 要求 10/20/30ms 的帧
        frame_duration = 30  # ms
        frame_bytes = int(self.sample_rate * 2 * frame_duration / 1000)

        # 简单做法：直接判断当前 chunk 的能量 + 局部 VAD
        # 由于 chunk_size 可能不是严格的 VAD 帧长，这里使用能量作为辅助
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        energy = rms / 32768.0

        # 缓存到环形缓冲区
        self._ring_buffer.append(in_data)

        # 判断是否包含语音
        is_speech = False
        if len(in_data) >= frame_bytes:
            try:
                is_speech = self.vad.is_speech(in_data[:frame_bytes], self.sample_rate)
            except Exception:
                is_speech = energy > 0.01
        else:
            is_speech = energy > 0.015

        if not self._speech_started:
            if is_speech:
                self._speech_started = True
                # 把环形缓冲区的前置音频也加入
                self._record_buffer.extend(b"".join(self._ring_buffer))
                self._record_buffer.extend(in_data)
                self._last_speech_time = time.time()
                if self.on_speech_started:
                    self.on_speech_started()
                logger.info("检测到语音开始")
        else:
            self._record_buffer.extend(in_data)
            if is_speech:
                self._last_speech_time = time.time()
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                silence_duration = time.time() - self._last_speech_time
                if silence_duration > CFG.PHRASE_TIMEOUT:
                    self.is_recording = False
                    if self.on_speech_ended:
                        self.on_speech_ended(bytes(self._record_buffer))
                    logger.info("检测到语音结束，静默 %.2f 秒", silence_duration)

    def close(self):
        """释放资源。"""
        self.stop_stream()
        self.stop_recording()
        if self.pa:
            self.pa.terminate()
