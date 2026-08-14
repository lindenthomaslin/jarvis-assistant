#!/bin/bash
# Build a self-contained arm64 macOS .app from a clean virtual environment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DIST_DIR="$PROJECT_DIR/dist"
BUILD_DIR="$PROJECT_DIR/build"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "未找到 venv。请先执行：python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

cd "$PROJECT_DIR"
"$VENV_PYTHON" -m pip install --upgrade pyinstaller
"$VENV_PYTHON" -m pip install -r requirements.txt

rm -rf "$BUILD_DIR" "$DIST_DIR"
"$VENV_PYTHON" -m PyInstaller \
  --noconfirm \
  --windowed \
  --name "JARVIS" \
  --osx-bundle-identifier "com.jarvis.assistant" \
  --add-data "frontend:frontend" \
  --add-data "static:static" \
  --add-data "config.yaml:." \
  --add-data "models:models" \
  --hidden-import pyaudio \
  --hidden-import webrtcvad \
  --additional-hooks-dir "scripts/pyinstaller-hooks" \
  --hidden-import faster_whisper \
  --hidden-import edge_tts \
  --hidden-import pypinyin \
  --collect-all webview \
  --collect-all openwakeword \
  --collect-all faster_whisper \
  desktop.py

echo "已生成：$DIST_DIR/JARVIS.app"
echo "首次运行前，请将 .env 放到：$HOME/Library/Application Support/JARVIS/.env"
