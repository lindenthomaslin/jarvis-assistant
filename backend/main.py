"""
J.A.R.V.I.S 主程序
FastAPI + WebSocket  orchestrator：
  音频流 -> 唤醒词 -> VAD 录音 -> STT -> DeepSeek -> TTS -> 播放
同时向前端实时推送状态、语音波形、对话内容。
"""
import asyncio
from collections import deque
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.audio.player import AudioPlayer
from backend.audio.recorder import AudioRecorder
from backend.audio.wake_word import WakeWordDetector
from backend.config import CFG, ROOT_DIR
from backend.llm.deepseek_client import DeepSeekClient
from backend.stt.whisper_engine import WhisperSTT
from backend.tools.skills import SKILLS
from backend.tts.edge_tts_engine import EdgeTTSEngine

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class JarvisState(str, Enum):
    """系统状态枚举。"""

    BOOTING = "booting"
    IDLE = "idle"
    WAKE = "wake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class ClientSession:
    """One browser connection and its non-blocking outbound message buffer."""

    MAX_MESSAGES = 256

    def __init__(self):
        self.messages: deque[dict] = deque()
        self.latest_spectrum: Optional[dict] = None
        self.ready = asyncio.Event()
        self.closed = False
        self.sender_task: Optional[asyncio.Task] = None

    def enqueue(self, message: dict):
        """Queue important messages and coalesce disposable spectrum updates.

        A slow or backgrounded WebView must never be able to block the AI
        pipeline.  Spectrum is visual decoration, so retaining only its latest
        value is enough.  Reply deltas are merged when a client falls behind.
        """
        if message.get("type") == "spectrum":
            self.latest_spectrum = message
            self.ready.set()
            return

        if len(self.messages) >= self.MAX_MESSAGES:
            if message.get("type") == "response" and not message.get("done"):
                delta = message.get("delta", "")
                for queued in reversed(self.messages):
                    if queued.get("type") == "response" and not queued.get("done"):
                        queued["delta"] = queued.get("delta", "") + delta
                        queued["text"] = message.get("text", queued.get("text", ""))
                        self.ready.set()
                        return
            # State telemetry is recoverable.  Keeping the latest messages is
            # more useful than stalling the producer indefinitely.
            self.messages.popleft()

        self.messages.append(message)
        self.ready.set()


class JarvisCore:
    """
    J.A.R.V.I.S 核心控制器。
    负责状态机、音频管道、WebSocket 广播。
    """

    def __init__(self):
        self.state = JarvisState.BOOTING
        self.clients: dict[WebSocket, ClientSession] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # 音频与 AI 组件
        self.recorder: Optional[AudioRecorder] = None
        self.player: Optional[AudioPlayer] = None
        self.wake_detector: Optional[WakeWordDetector] = None
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[EdgeTTSEngine] = None
        self.llm: Optional[DeepSeekClient] = None

        # 音频数据队列（线程安全）
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.spectrum_queue: asyncio.Queue = asyncio.Queue()

        # 控制标志
        self.is_running = False
        self.is_interrupted = False
        self._state_lock = threading.Lock()
        self._started_at = time.monotonic()
        self._process = psutil.Process()

        # STT 唤醒降级方案缓冲
        self._stt_fallback_buffer = bytearray()
        self._stt_fallback_lock = threading.Lock()
        self._stt_fallback_last_check = 0.0

        # Text generation and TTS deliberately run on separate paths.  TTS is
        # a network request, so awaiting it inside the token loop used to pause
        # the visible answer after every sentence.
        self._speech_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._turn_id = 0
        self._background_tasks: list[asyncio.Task] = []

    # ---------- 生命周期 ----------

    def initialize(self):
        """初始化所有组件。"""
        logger.info("正在初始化 J.A.R.V.I.S 核心...")
        self.loop = asyncio.get_event_loop()

        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.wake_detector = WakeWordDetector()
        self.stt = WhisperSTT()
        self.tts = EdgeTTSEngine()
        self.llm = DeepSeekClient()

        # 注册唤醒回调
        self.wake_detector.on_wake_word(self._on_wake_word)

        # 注册音频流回调
        self.recorder.start_stream(on_chunk=self._on_audio_chunk)

        self.is_running = True
        self.set_state(JarvisState.IDLE)

        # 启动后台任务
        self._background_tasks.extend([
            asyncio.create_task(self._spectrum_broadcaster()),
            asyncio.create_task(self._telemetry_broadcaster()),
            asyncio.create_task(self._pipeline_worker()),
            asyncio.create_task(self._speech_worker()),
        ])
        if CFG.STT_WAKE_FALLBACK_ENABLED:
            self._background_tasks.append(asyncio.create_task(self._stt_fallback_worker()))
            logger.info("已启用 STT 关键词唤醒/打断降级方案")

        logger.info("J.A.R.V.I.S 初始化完成，等待唤醒...")

    def shutdown(self):
        """关闭所有资源。"""
        logger.info("正在关闭 J.A.R.V.I.S...")
        self.is_running = False
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
        for session in list(self.clients.values()):
            session.closed = True
            session.ready.set()
            if session.sender_task:
                session.sender_task.cancel()
        self.clients.clear()
        if self.player:
            self.player.close()
        if self.recorder:
            self.recorder.close()
        self.set_state(JarvisState.BOOTING)

    # ---------- WebSocket 管理 ----------

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        session = ClientSession()
        self.clients[websocket] = session
        session.sender_task = asyncio.create_task(self._client_sender(websocket, session))
        logger.info("前端已连接，当前连接数: %d", len(self.clients))

        # 发送当前状态
        session.enqueue({
            "type": "state",
            "state": self.state.value,
            "message": "J.A.R.V.I.S 在线",
        })

    async def disconnect(self, websocket: WebSocket):
        session = self.clients.pop(websocket, None)
        if session:
            session.closed = True
            session.ready.set()
            if session.sender_task:
                session.sender_task.cancel()
        logger.info("前端断开连接，当前连接数: %d", len(self.clients))

    async def broadcast(self, message: dict):
        """Queue a broadcast without allowing a slow client to block AI work."""
        for session in list(self.clients.values()):
            session.enqueue(message.copy())

    async def send_to(self, websocket: WebSocket, message: dict):
        """Queue a direct message without awaiting network I/O."""
        session = self.clients.get(websocket)
        if session:
            session.enqueue(message)

    async def _client_sender(self, websocket: WebSocket, session: ClientSession):
        """The only coroutine allowed to write to an individual WebSocket."""
        try:
            while not session.closed:
                await session.ready.wait()
                while not session.closed:
                    if session.messages:
                        message = session.messages.popleft()
                    elif session.latest_spectrum is not None:
                        message = session.latest_spectrum
                        session.latest_spectrum = None
                    else:
                        session.ready.clear()
                        break
                    await websocket.send_json(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("前端发送连接已关闭: %s", exc)
        finally:
            session.closed = True
            self.clients.pop(websocket, None)

    # ---------- 状态机 ----------

    def set_state(self, state: JarvisState, message: str = ""):
        def _update():
            self.state = state
            logger.info("状态切换 -> %s | %s", state.value, message)
            asyncio.create_task(self.broadcast({
                "type": "state",
                "state": state.value,
                "message": message,
            }))

        with self._state_lock:
            self.state = state

        # 确保在主事件循环线程中执行，避免 PyAudio 回调线程直接调用 asyncio.create_task
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(_update)
        else:
            _update()

    # ---------- 音频回调 ----------

    def _on_audio_chunk(self, chunk: bytes):
        """持续音频流回调（运行于 PyAudio 线程）。"""
        if not self.is_running:
            return

        # 将音频块放入队列供前端可视化
        try:
            self.loop.call_soon_threadsafe(self.spectrum_queue.put_nowait, chunk)
        except Exception:
            pass

        # IDLE: 监听唤醒词；SPEAKING: 监听打断
        if self.state in (JarvisState.IDLE, JarvisState.SPEAKING):
            if self.wake_detector:
                # openWakeWord 同时承担：
                # - IDLE 下的 "Hey Jarvis" 唤醒
                # - SPEAKING 下的 "Hey Jarvis" 语音打断
                self.wake_detector.process_chunk(chunk)

            # STT 降级方案：无论 openWakeWord 是否可用都缓存音频，
            # 用于识别中文“贾维斯”唤醒以及“停/打断”等打断命令。
            if CFG.STT_WAKE_FALLBACK_ENABLED:
                with self._stt_fallback_lock:
                    self._stt_fallback_buffer.extend(chunk)
                    # 保留最近 2.5 秒的音频
                    max_bytes = int(CFG.AUDIO_SAMPLE_RATE * 2 * 2.5)
                    if len(self._stt_fallback_buffer) > max_bytes:
                        self._stt_fallback_buffer = self._stt_fallback_buffer[-max_bytes:]

        # 能量打断检测：在 SPEAKING 状态检测用户说话
        if self.state == JarvisState.SPEAKING:
            self._check_interruption(chunk)

    def _on_wake_word(self, word: str):
        """唤醒词回调。"""
        if self.state == JarvisState.IDLE:
            self.loop.call_soon_threadsafe(self._handle_wake, word)
        elif (
            self.state == JarvisState.SPEAKING
            and CFG.INTERRUPTION_ENABLED
            and word.lower() in ("jarvis", "hey jarvis", "贾维斯")
        ):
            # 说话时再次喊出唤醒词 = 语音打断
            logger.info("唤醒词打断: %s", word)
            self.loop.call_soon_threadsafe(self._handle_interruption, "wakeword")
        else:
            return

    def _handle_wake(self, word: str):
        """处理唤醒事件。"""
        with self._stt_fallback_lock:
            self._stt_fallback_buffer.clear()
        self.set_state(JarvisState.WAKE, f"唤醒词识别: {word}")
        self._play_sound("wake")
        # 短暂延迟后进入录音
        asyncio.create_task(self._enter_listening())

    async def _enter_listening(self, delay: float = 0.12):
        # 播放完确认音后几乎立刻进入聆听，避免明显等待感
        await asyncio.sleep(delay)
        self.set_state(JarvisState.LISTENING, "正在聆听您的指令")
        # 启动录音（阻塞在线程池）
        asyncio.create_task(self._record_speech())

    async def _record_speech(self):
        """执行 VAD 录音并转文字。"""
        def record():
            return self.recorder.start_recording(
                on_speech_started=lambda: self.set_state(
                    JarvisState.LISTENING, "检测到语音"
                ),
                on_speech_ended=lambda pcm: None,
            )

        try:
            pcm = await asyncio.get_event_loop().run_in_executor(None, record)
            if not pcm:
                self.set_state(JarvisState.IDLE, "未检测到有效语音")
                return

            # 播放确认音
            self._play_sound("confirm")
            self.set_state(JarvisState.THINKING, "正在识别与思考")

            # STT
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.stt.transcribe, pcm
            )

            if not text:
                self.set_state(JarvisState.IDLE, "未能识别您的指令")
                return

            await self.broadcast({
                "type": "transcript",
                "text": text,
                "final": True,
            })

            # 优先执行本地技能
            skill_result = SKILLS.execute(text)
            if skill_result:
                text = f"用户指令：{text}\n本地技能执行结果：{skill_result}\n请结合以上信息，用 J.A.R.V.I.S 的风格回应用户。"

            # 进入 LLM + TTS 流程
            await self._process_llm_and_tts(text)
        except Exception as e:
            logger.error("录音流程异常: %s", e)
            self.set_state(JarvisState.IDLE, "处理过程中出现错误")

    async def _process_llm_and_tts(self, user_text: str):
        """Stream text immediately while TTS is generated in the background."""
        self._turn_id += 1
        turn_id = self._turn_id
        self.is_interrupted = False
        self.set_state(JarvisState.THINKING, "J.A.R.V.I.S 正在思考")

        # Sentence splitting is only for speech.  It must not pause the LLM
        # stream or the browser's first visible token.
        sentence_buffer = ""
        current_response = ""

        try:
            async for chunk in self.llm.chat(user_text, stream=True):
                if turn_id != self._turn_id:
                    return
                # 处理 thinking 标签
                if chunk.startswith("[thinking]") and chunk.endswith("[/thinking]"):
                    reasoning = chunk[10:-12]
                    await self.broadcast({
                        "type": "reasoning",
                        "text": reasoning,
                    })
                    continue

                current_response += chunk
                sentence_buffer += chunk

                # 向前端发送打字机效果文本
                await self.broadcast({
                    "type": "response",
                    "text": current_response,
                    "delta": chunk,
                    "done": False,
                })

                # 判断句子结束
                if self._is_sentence_end(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    if sentence:
                        self._queue_speech(turn_id, sentence)

            # 处理剩余文本
            if sentence_buffer.strip():
                self._queue_speech(turn_id, sentence_buffer.strip())

            # 最终响应完成
            if turn_id == self._turn_id:
                await self.broadcast({
                    "type": "response",
                    "text": current_response,
                    "done": True,
                })

            # The user can already read the complete answer.  This wait only
            # keeps the conversation state accurate until queued speech ends.
            await self._speech_queue.join()
            if turn_id != self._turn_id:
                return
            while turn_id == self._turn_id and self.player and self.player.is_playing:
                await asyncio.sleep(0.05)

            if turn_id == self._turn_id:
                self._clear_stt_fallback_buffer()
                self.set_state(JarvisState.IDLE, "等待下一次唤醒")
        except Exception as e:
            logger.error("LLM/TTS 流程异常: %s", e)
            if turn_id == self._turn_id:
                self._clear_stt_fallback_buffer()
                self.set_state(JarvisState.IDLE, "回复生成失败")

    def _is_sentence_end(self, text: str) -> bool:
        """判断是否构成完整句子（以句号、问号、感叹号、分号结尾）。"""
        text = text.strip()
        if not text:
            return False
        # 至少 8 个字符且以标点结尾
        if len(text) < 8:
            return False
        return text[-1] in "。.?!;？！；"

    def _queue_speech(self, turn_id: int, text: str):
        """Add a sentence to the serialized TTS queue without blocking text."""
        if text and turn_id == self._turn_id:
            self._speech_queue.put_nowait((turn_id, text))

    async def _speech_worker(self):
        """Generate and play speech independently from model token streaming."""
        while self.is_running:
            turn_id, text = await self._speech_queue.get()
            try:
                if turn_id != self._turn_id:
                    continue
                mp3_data = await self.tts.synthesize(text, add_effects=True)
                if turn_id != self._turn_id:
                    continue
                self.set_state(JarvisState.SPEAKING, "J.A.R.V.I.S 正在回复")
                # Briefly ignore microphone input at the start of playback to
                # avoid the speaker's own output triggering a false interruption.
                self._speaking_start_time = time.time()
                await asyncio.get_running_loop().run_in_executor(
                    None, self.player.play, mp3_data, "mp3"
                )
                # 竞态防护：打断可能恰好在 play() 排队期间发生（turn_id 已更新），
                # 此时立即停掉刚被错误排入队列的旧回复。
                if turn_id != self._turn_id:
                    self.player.stop()
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("语音合成/播放失败: %s", exc)
            finally:
                self._speech_queue.task_done()

    def cancel_current_response(self):
        """Invalidate pending LLM/TTS work and remove speech not yet started."""
        self._turn_id += 1
        self.is_interrupted = True
        while True:
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # ---------- 打断机制 ----------

    def _check_interruption(self, chunk: bytes):
        """检测用户是否在 AI 说话时发声。"""
        if not CFG.INTERRUPTION_ENABLED:
            return

        # Ignore echoes at the very start of playback.
        if time.time() - getattr(self, "_speaking_start_time", 0) < CFG.INTERRUPTION_GRACE_SECONDS:
            return

        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(audio**2)))
        # 单设备扬声器+麦克风下，AI 自己的声音会进入麦克风。
        # 用正在播放的音频能量作参考：只有当用户声音明显盖过回声时才判定为打断。
        playback_rms = self.player.get_recent_playback_rms() if self.player else 0.0
        if playback_rms > 0.003:
            threshold = max(
                CFG.INTERRUPTION_THRESHOLD,
                playback_rms * CFG.INTERRUPTION_ECHO_RATIO,
            )
        else:
            threshold = CFG.INTERRUPTION_THRESHOLD
        now = time.time()
        last_check = getattr(self, "_last_interruption_check", 0)

        if energy > threshold:
            # Require several consecutive loud frames to avoid brief noise.
            if now - last_check > 0.3:
                self._interruption_frame_count = 1
            else:
                self._interruption_frame_count = getattr(self, "_interruption_frame_count", 0) + 1
            self._last_interruption_check = now

            if self._interruption_frame_count >= 4:
                logger.info(
                    "检测到用户打断 (mic=%.3f threshold=%.3f playback=%.3f)",
                    energy,
                    threshold,
                    playback_rms,
                )
                self.loop.call_soon_threadsafe(self._handle_interruption, "energy")
        else:
            # Quiet frame: if the gap is long enough, reset the counter.
            if now - last_check > 0.3:
                self._interruption_frame_count = 0

    def _handle_interruption(self, source: str = "unknown"):
        """执行打断：停止播放，回到录音状态。"""
        if self.state != JarvisState.SPEAKING:
            return
        logger.info("执行打断 (source=%s)", source)
        self.cancel_current_response()
        if self.player:
            self.player.stop()
        self._clear_stt_fallback_buffer()
        self.set_state(JarvisState.LISTENING, "已打断，请继续")
        asyncio.create_task(self._record_speech_after_interrupt())

    async def _record_speech_after_interrupt(self, delay: float = 0.35):
        """打断后稍等扬声器余音消散，再开始录音，避免把回声当指令。"""
        await asyncio.sleep(delay)
        await self._record_speech()

    def _clear_stt_fallback_buffer(self):
        """清空 STT 滚动缓冲，避免 JARVIS 自己的尾音被误识别为唤醒/打断。"""
        with self._stt_fallback_lock:
            self._stt_fallback_buffer.clear()

    # ---------- 频谱广播 ----------

    async def _spectrum_broadcaster(self):
        """定期向前端发送音频频谱数据。"""
        while self.is_running:
            try:
                # 聚合最近几帧
                chunks = []
                for _ in range(5):
                    try:
                        chunk = self.spectrum_queue.get_nowait()
                        chunks.append(chunk)
                    except asyncio.QueueEmpty:
                        break

                if chunks:
                    combined = b"".join(chunks)
                    spectrum = self._compute_spectrum(combined)
                    await self.broadcast({
                        "type": "spectrum",
                        "data": spectrum,
                    })
                await asyncio.sleep(0.04)
            except Exception as e:
                logger.error("频谱广播异常: %s", e)
                await asyncio.sleep(0.1)

    async def _telemetry_broadcaster(self):
        """以低频率推送真实系统负载和模块就绪状态。"""
        psutil.cpu_percent(interval=None)
        while self.is_running:
            try:
                memory = psutil.virtual_memory()
                process_memory = self._process.memory_info().rss / (1024 * 1024)
                await self.broadcast({
                    "type": "telemetry",
                    "cpu": round(psutil.cpu_percent(interval=None), 1),
                    "memory": round(memory.percent, 1),
                    "process_memory_mb": round(process_memory, 1),
                    "uptime": int(time.monotonic() - self._started_at),
                    "clients": len(self.clients),
                    "state": self.state.value,
                    "audio_ready": bool(self.recorder and self.player),
                    "stt_ready": self.stt is not None,
                    "llm_ready": self.llm is not None,
                    "tts_ready": self.tts is not None,
                })
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.debug("遥测广播异常: %s", e)
                await asyncio.sleep(2.0)

    def _compute_spectrum(self, audio_bytes: bytes, bins: int = 64) -> list:
        """计算音频频谱，返回归一化的分贝值列表。"""
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio) < bins * 2:
                return [0.0] * bins

            # 加窗 FFT
            window = np.hanning(len(audio))
            fft = np.fft.rfft(audio * window)
            magnitude = np.abs(fft)

            # 分桶并取对数
            bin_size = max(1, len(magnitude) // bins)
            result = []
            for i in range(bins):
                start = i * bin_size
                end = (i + 1) * bin_size
                val = np.mean(magnitude[start:end])
                db = 20 * math.log10(val + 1e-6)
                # 归一化到 0-1
                normalized = (db + 60) / 60.0
                normalized = max(0.0, min(1.0, normalized))
                result.append(round(normalized, 3))
            return result
        except Exception:
            return [0.0] * bins

    # ---------- 音效播放 ----------

    def _play_sound(self, name: str):
        """播放内置音效（唤醒、确认等），优先 mp3，回退 wav。"""
        for ext in ("mp3", "wav"):
            sound_path = Path(ROOT_DIR) / "static" / "sounds" / f"{name}.{ext}"
            if sound_path.exists():
                threading.Thread(
                    target=self.player.play,
                    args=(str(sound_path), ext),
                    daemon=True,
                ).start()
                return
        logger.debug("音效文件不存在: %s", name)

    # ---------- 后台工作器 ----------

    async def _stt_fallback_worker(self):
        """
        STT 唤醒/打断降级工作器。
        每隔约 1 秒对缓存音频执行一次识别：
        - 在 idle 状态检测中文“贾维斯”等唤醒词
        - 在 speaking 状态检测“停/别说了/打断”等命令词
        识别前先做能量门控，环境安静时跳过，避免 Whisper 空转占 CPU。
        """
        while self.is_running:
            await asyncio.sleep(CFG.STT_FALLBACK_INTERVAL)

            if not self.wake_detector or not self.stt:
                continue
            if self.state not in (JarvisState.IDLE, JarvisState.SPEAKING):
                continue

            with self._stt_fallback_lock:
                audio = bytes(self._stt_fallback_buffer)

            # 至少 0.6 秒音频才检测
            min_bytes = int(CFG.AUDIO_SAMPLE_RATE * 2 * 0.6)
            if len(audio) < min_bytes:
                continue

            # 能量门控：缓冲里基本是安静环境时不做 STT，省 CPU 也减少误报
            energy = float(
                np.sqrt(np.mean(np.frombuffer(audio, dtype=np.int16).astype(np.float32) ** 2))
            ) / 32768.0
            if energy < CFG.STT_FALLBACK_ENERGY_THRESHOLD:
                continue

            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.stt.transcribe(audio, language=None)
                )
                if not text:
                    continue

                if self.state == JarvisState.IDLE:
                    detected = self.wake_detector.check_buffer_with_stt(text)
                    if detected:
                        with self._stt_fallback_lock:
                            self._stt_fallback_buffer.clear()
                elif self.state == JarvisState.SPEAKING:
                    cmd = self.wake_detector.check_buffer_for_interrupt(text)
                    if cmd:
                        with self._stt_fallback_lock:
                            self._stt_fallback_buffer.clear()
                        self._handle_interruption("stt")
            except Exception as e:
                logger.error("STT 唤醒/打断检测失败: %s", e)

    async def _pipeline_worker(self):
        """保留的异步工作器，用于未来扩展。"""
        while self.is_running:
            await asyncio.sleep(1)


# ---------- FastAPI 应用 ----------

core = JarvisCore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.initialize()
    yield
    core.shutdown()


app = FastAPI(title="J.A.R.V.I.S", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源服务
static_dir = Path(ROOT_DIR) / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """返回前端主页面。"""
    index_path = Path(ROOT_DIR) / "frontend" / "index.html"
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health():
    return {"status": "ok", "state": core.state.value}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await core.connect(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            await _handle_client_message(websocket, msg)
    except WebSocketDisconnect:
        await core.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket 异常: %s", e)
        await core.disconnect(websocket)


async def _handle_client_message(websocket: WebSocket, msg: dict):
    """处理前端发来的控制命令。"""
    cmd = msg.get("command")
    if cmd == "wakeup":
        # 手动触发唤醒
        core.loop.call_soon_threadsafe(core._handle_wake, "manual")
    elif cmd == "stop":
        core.cancel_current_response()
        if core.player:
            core.player.stop()
        core._clear_stt_fallback_buffer()
        core.set_state(JarvisState.IDLE, "已停止")
    elif cmd == "clear_history":
        if core.llm:
            core.llm.clear_history()
        await core.send_to(websocket, {"type": "info", "message": "对话历史已清空"})
    elif cmd == "ping":
        await core.send_to(websocket, {
            "type": "pong",
            "timestamp": msg.get("timestamp"),
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=CFG.WS_HOST,
        port=CFG.WS_PORT,
        reload=False,
        log_level="info",
    )
