#!/bin/bash
# J.A.R.V.I.S 一键启动脚本

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "未找到 venv，请先创建虚拟环境并安装依赖："
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 检查 .env
if [ ! -f ".env" ]; then
    echo "警告：未找到 .env 文件，将使用 .env.example"
    cp .env.example .env
fi

# 生成音效（如果不存在）
if [ ! -f "static/sounds/boot.wav" ]; then
    echo "正在生成默认音效..."
    python3 scripts/generate_sounds.py
fi

# 启动服务
echo "正在启动 J.A.R.V.I.S..."
echo "启动后请在浏览器打开 http://localhost:18790"
python3 -m backend.main
