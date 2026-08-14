"""
音频播放模块
支持连续播放 PCM/WAV/MP3 音频，并可在播放过程中被中断。
"""
import logging
import queue
import threading
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydub import AudioSegment

from backend.config import CFG

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    非阻塞连续音频播放器。

    - 多次调用 play() 会按顺序排队播放，中间不会关闭/重开音频流，避免句间卡顿。
    - 调用 stop() 可立即清空队列并中断当前播放。
    """

    def __init__(self, sample_rate: int = CFG.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.pa = None
        self.stream: Optional[object] = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._stop_event = threading.Event()
        self._close_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self.is_playing = False
        self._lock = threading.Lock()
        # 最近播放样本的 RMS 历史（用于打断检测的回声参考）
        self._playback_rms = deque(maxlen=32)
        # 播放代际：stop() 后旧线程不得再写入 RMS 参考，避免污染新播放
        self._playback_generation = 0
        self._init_audio()

    def _init_audio(self):
        """延迟初始化 PyAudio。"""
        try:
            import pyaudio

            self.pa = pyaudio.PyAudio()
        except ImportError as e:
            logger.error("缺少 pyaudio，无法播放音频: %s", e)
            raise

    def play(
        self,
        audio_input: Union[bytes, Path, str, BytesIO],
        format_hint: Optional[str] = None,
    ):
        """
        排队播放音频。
        audio_input: bytes 视为 WAV/MP3 数据；Path/str 视为文件路径。
        """
        pcm = self._load_audio(audio_input, format_hint)
        if not pcm:
            logger.warning("音频为空，跳过播放")
            return

        self._audio_queue.put(pcm)

        # 启动播放线程（如果尚未运行）
        with self._lock:
            if self._play_thread is None or not self._play_thread.is_alive():
                self._stop_event.clear()
                self._close_event.clear()
                self._play_thread = threading.Thread(
                    target=self._play_worker,
                    daemon=True,
                )
                self._play_thread.start()

    def _play_worker(self):
        """后台播放线程：保持一个打开的音频流，顺序写入队列中的 PCM 数据。"""
        import pyaudio

        try:
            generation = self._playback_generation
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=CFG.AUDIO_CHUNK_SIZE,
            )
            self.stream.start_stream()

            while not self._close_event.is_set():
                try:
                    pcm = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # stop() 被调用时退出循环
                    if self._stop_event.is_set():
                        break
                    continue

                self.is_playing = True
                chunk_size = CFG.AUDIO_CHUNK_SIZE * 2
                for i in range(0, len(pcm), chunk_size):
                    if self._stop_event.is_set():
                        break
                    chunk = pcm[i : i + chunk_size]
                    self.stream.write(chunk)
                    # 记录每个播放块的 RMS，供能量打断判断“用户是否盖过回声”
                    try:
                        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(audio**2)))
                        with self._lock:
                            if generation == self._playback_generation:
                                self._playback_rms.append((time.monotonic(), rms))
                    except Exception:
                        pass

                # 一段播完后，如果队列里还有数据就继续播放
                if self._audio_queue.empty():
                    self.is_playing = False

            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        except Exception as e:
            logger.error("播放线程异常: %s", e)
        finally:
            self.is_playing = False

    def _load_audio(
        self,
        audio_input: Union[bytes, Path, str, BytesIO],
        format_hint: Optional[str],
    ) -> Optional[bytes]:
        """统一加载音频并转换为 16kHz 单声道 16bit PCM。"""
        try:
            if isinstance(audio_input, (str, Path)):
                audio = AudioSegment.from_file(audio_input, format=format_hint)
            elif isinstance(audio_input, bytes):
                audio = AudioSegment.from_file(BytesIO(audio_input), format=format_hint)
            elif isinstance(audio_input, BytesIO):
                audio = AudioSegment.from_file(audio_input, format=format_hint)
            else:
                raise ValueError(f"不支持的音频输入类型: {type(audio_input)}")

            audio = audio.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
            return audio.raw_data
        except Exception as e:
            logger.error("加载音频失败: %s", e)
            return None

    def stop(self):
        """清空队列并中断当前播放。"""
        self._stop_event.set()
        with self._lock:
            self._playback_generation += 1
            self._playback_rms.clear()
        # 清空待播放数据
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self.is_playing = False
        self._stop_event.clear()
        self._play_thread = None
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self._lock:
            # 清除旧线程在退出前可能写入的最后一条参考数据
            self._playback_rms.clear()
        logger.info("音频播放已中断")

    def get_recent_playback_rms(self, window: float = 0.6) -> float:
        """返回最近 window 秒内播放音频的最大 RMS（0.0 表示当前没有播放）。"""
        now = time.monotonic()
        with self._lock:
            values = [rms for ts, rms in self._playback_rms if now - ts <= window]
        return max(values) if values else 0.0

    def close(self):
        """释放资源。"""
        self._close_event.set()
        self.stop()
        if self.pa:
            self.pa.terminate()
