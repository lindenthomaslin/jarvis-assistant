# J.A.R.V.I.S 语音助手

> Just A Rather Very Intelligent System.

一个受《钢铁侠》启发的**本地化智能语音助手**：内置本地唤醒词检测、 faster-whisper 语音识别、DeepSeek 大模型对话、edge-tts 语音合成与全息 HUD 界面，支持 macOS 桌面应用打包。所有语音处理优先在本地运行，大模型通过 DeepSeek API 接入，兼顾隐私与智能。

## 核心特性

- 🎙 **多方式唤醒**：支持 "Hey Jarvis"（openWakeWord 本地模型）以及 "Hi Jarvis" / "Jarvis" / "贾维斯"（STT 关键词降级）
- 🗣 **中英混合识别**：基于 faster-whisper，自动识别中文与英文输入
- 🧠 **大模型对话**：接入 DeepSeek API，可选 `deepseek-v4-flash` / `deepseek-v4-pro`
- 🔊 **科幻音色 TTS**：edge-tts + pydub 后期电子音效，低延迟、未来感
- 🖥 **全息 HUD 界面**：单文件原生 HTML/CSS/Canvas，无需构建工具
- ⏹ **随时打断**：说话时说出命令词即可强制打断，用户优先级最高
- 📦 **macOS 桌面版**：一键打包为 `.app`，自包含、无需浏览器

## 项目结构

```
jarvis-assistant/
├── backend/                  # Python 后端
│   ├── main.py              # FastAPI + WebSocket 主控
│   ├── config.py            # 配置中心
│   ├── audio/               # 录音、播放、唤醒词
│   ├── stt/                 # faster-whisper 语音识别
│   ├── tts/                 # edge-tts 语音合成
│   ├── llm/                 # DeepSeek 大模型
│   └── utils/               # 音频音效处理
├── frontend/
│   └── index.html           # 单文件全息 HUD 前端
├── static/sounds/           # 内置音效
├── scripts/
│   └── generate_sounds.py   # 音效生成脚本
├── models/                  # 本地模型缓存目录
├── .env.example             # 环境变量示例
├── config.yaml              # YAML 配置
├── requirements.txt         # Python 依赖
└── run.sh                   # 一键启动脚本
```

## 技术栈说明

- **后端**：Python + FastAPI + WebSocket
- **语音唤醒**：openWakeWord（首选）/ STT 关键词降级
- **语音识别**：faster-whisper（本地运行，中英混合）
- **大模型**：DeepSeek API（OpenAI 兼容接口）
- **语音合成**：edge-tts + pydub 电子音效（免费、低延迟）
- **前端**：原生 HTML5 Canvas + CSS3，单文件无构建
- **通信**：WebSocket 实时推送状态、频谱、对话

## 环境准备

### 1. 安装系统依赖

**macOS**（使用 Homebrew）：

```bash
brew install python portaudio ffmpeg
```

**Windows**：
- 安装 Python 3.10+
- 安装 [PortAudio](http://www.portaudio.com/download.html)
- 安装 FFmpeg 并加入 PATH

**Linux (Ubuntu/Debian)**：

```bash
sudo apt update
sudo apt install python3-pip python3-venv portaudio19-dev ffmpeg
```

### 2. 创建虚拟环境并安装依赖

```bash
cd jarvis-assistant
python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

> 注意：若 `openwakeword` 安装失败，系统会自动降级为 STT 关键词唤醒。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

其他配置可按需调整：
- `TTS_VOICE`：edge-tts 音色
- `STT_MODEL_SIZE`：whisper 模型大小（tiny/base/small/medium/large）
- `WAKE_WORD`：自定义唤醒词

## 启动

### 方式一：一键脚本

```bash
bash run.sh
```

### 方式二：手动启动

```bash
source venv/bin/activate
python -m backend.main
```

启动后打开浏览器访问：

```
http://localhost:18790
```

## macOS 桌面版

桌面版会在原生窗口中加载同一套界面，不再依赖浏览器标签页。需要在 Apple Silicon Mac 上构建：

```bash
cd jarvis-assistant
bash scripts/build_macos.sh
open dist/JARVIS.app
```

为避免把 API Key 打进应用，请在首次运行前保存配置到：

```bash
mkdir -p "$HOME/Library/Application Support/JARVIS"
cp .env "$HOME/Library/Application Support/JARVIS/.env"
```

生成的应用位于 `dist/JARVIS.app`。首次启动时 macOS 会请求麦克风权限；请在“系统设置 → 隐私与安全性 → 麦克风”允许 JARVIS。

## 使用说明

1. **语音唤醒**：
   - 说 **“Hey Jarvis”**（openWakeWord 本地模型，低延迟）
   - 说 **“Hi Jarvis” / “Hello Jarvis” / “Jarvis” / “贾维斯”**（STT 关键词降级通道，约 1 秒内响应）
2. **打断**：JARVIS 说话时，你的优先级最高，说出以下任一命令词即可强制打断：
   - `打断`、`停`、`stop`、`jarvis`、`贾维斯`
   - 打断后 JARVIS 会立即停止回复并重新聆听
3. **手动触发**：按空格键或点击屏幕中央
4. **停止**：按 ESC 键
5. **清空历史**：向后端发送 `clear_history` 命令（可在前端扩展按钮）

> 单设备扬声器+麦克风场景下，语音打断容易受回声影响。当前默认使用**命令词打断**而非纯音量打断，稳定性更高；如需“直接开口就打断”，建议戴耳机或在 `config.yaml` 中将 `interruption_enabled` 设为 `true`。

## 唤醒词模型（可选）

下载 openWakeWord 的 `hey_jarvis` 模型到 `models/` 目录：

```bash
mkdir -p models
cd models
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite
```

若未下载模型，系统会自动使用 STT 关键词检测作为降级方案。
内置模型文件 `hey_jarvis_v0.1.onnx` / `.tflite` 只能识别英文 “Hey Jarvis”；
中文“贾维斯”始终走 STT 关键词通道（`stt_wake_fallback_enabled: true`）。

## 后续扩展

- 接入 ElevenLabs 获得更高品质科幻男声
- 添加自定义技能（天气、打开软件、控制智能家居）
- 接入向量数据库实现长期记忆
- 用 Electron 打包为桌面应用，支持开机自启与系统托盘
- 接入本地 LLM（Ollama/LM Studio）实现完全离线

## 许可证

MIT
