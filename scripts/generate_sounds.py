"""
生成 J.A.R.V.I.S 内置音效
运行方式：python scripts/generate_sounds.py
"""
import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "static" / "sounds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000


def save_wav(data, path):
    """保存浮点音频数据为 WAV 文件。"""
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(data)


def float_to_bytes(samples):
    """将 [-1, 1] 浮点样本转为 16bit PCM 字节。"""
    pcm = bytearray()
    for s in samples:
        val = max(-1, min(1, s))
        pcm.extend(struct.pack("<h", int(val * 32767)))
    return bytes(pcm)


def generate_chirp(start_freq, end_freq, duration, amplitude=1.0, attack=0.08):
    """生成带平滑包络的线性扫频。"""
    samples = []
    phase = 0.0
    total = int(SAMPLE_RATE * duration)
    attack_samples = max(1, int(total * attack))
    for i in range(total):
        progress = i / max(1, total - 1)
        freq = start_freq + (end_freq - start_freq) * progress
        phase += 2 * math.pi * freq / SAMPLE_RATE
        fade_in = min(1.0, i / attack_samples)
        fade_out = math.sin(math.pi * progress) ** 0.7
        samples.append(math.sin(phase) * amplitude * fade_in * fade_out)
    return samples


def mix_layers(duration, layers):
    """按起始时间叠加音轨，并以保留动态的方式归一化。"""
    output = [0.0] * int(SAMPLE_RATE * duration)
    for start_seconds, samples in layers:
        offset = int(start_seconds * SAMPLE_RATE)
        for index, sample in enumerate(samples):
            target = offset + index
            if target >= len(output):
                break
            output[target] += sample

    peak = max(1.0, max(abs(value) for value in output) / 0.88)
    return [value / peak for value in output]


def generate_tone(freq, duration, envelope="sine", vibrato=0):
    """生成单一频率音调。"""
    samples = []
    for i in range(int(SAMPLE_RATE * duration)):
        t = i / SAMPLE_RATE
        env_t = t / duration
        freq_now = freq + math.sin(2 * math.pi * 12 * t) * vibrato
        sample = math.sin(2 * math.pi * freq_now * t)
        if envelope == "sine":
            sample *= math.sin(math.pi * env_t)
        elif envelope == "attack":
            sample *= min(1, env_t * 4) * (1 - env_t)
        samples.append(sample)
    return samples


def generate_boot():
    """开机音效：低频核心启动、宽频扫描与最终锁定和弦。"""
    duration = 2.3
    layers = [
        (0.00, generate_chirp(72, 118, 2.15, 0.38, attack=0.18)),
        (0.18, generate_chirp(180, 1480, 1.55, 0.42)),
        (0.25, generate_chirp(360, 2960, 1.30, 0.12)),
        (1.52, generate_chirp(520, 780, 0.52, 0.40)),
        (1.62, generate_chirp(780, 1170, 0.46, 0.28)),
        (1.76, generate_chirp(1170, 1560, 0.38, 0.20)),
    ]
    samples = mix_layers(duration, layers)
    save_wav(float_to_bytes(samples), OUT_DIR / "boot.wav")
    print("已生成 boot.wav")


def generate_wake():
    """唤醒音效：三层上扬脉冲，匹配核心向外扩散的波纹。"""
    layers = [
        (0.00, generate_chirp(540, 920, 0.18, 0.50)),
        (0.08, generate_chirp(920, 1840, 0.24, 0.48)),
        (0.15, generate_chirp(1380, 2460, 0.22, 0.18)),
    ]
    samples = mix_layers(0.42, layers)
    save_wav(float_to_bytes(samples), OUT_DIR / "wake.wav")
    print("已生成 wake.wav")


def generate_confirm():
    """确认音效：短促下行锁定音，不与语音首句争抢听感。"""
    layers = [
        (0.00, generate_chirp(1560, 840, 0.22, 0.50, attack=0.04)),
        (0.03, generate_chirp(780, 520, 0.20, 0.24, attack=0.04)),
    ]
    samples = mix_layers(0.28, layers)
    save_wav(float_to_bytes(samples), OUT_DIR / "confirm.wav")
    print("已生成 confirm.wav")


if __name__ == "__main__":
    generate_boot()
    generate_wake()
    generate_confirm()
    print(f"音效已保存到: {OUT_DIR}")
