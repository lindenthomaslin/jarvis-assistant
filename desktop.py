"""Native macOS launcher for J.A.R.V.I.S.

Runs the existing FastAPI server on loopback and presents it in a native WebKit
window, so users no longer need to keep a browser tab open.
"""
import multiprocessing
import os
import socket
import threading
import time
import urllib.request

import uvicorn
import webview

from backend.config import CFG
from backend.main import app


def _available_port(preferred: int) -> int:
    """Use the configured port, falling back cleanly when it is already busy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


def _wait_until_ready(url: str, timeout_seconds: float = 12.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.4) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("J.A.R.V.I.S 本地服务未能启动")


def main():
    # Prevent packaged builds from exposing the service to the local network.
    host = "127.0.0.1"
    port = _available_port(CFG.WS_PORT)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    thread = threading.Thread(target=server.run, name="jarvis-server", daemon=True)
    thread.start()

    url = f"http://{host}:{port}"
    _wait_until_ready(f"{url}/api/health")
    window = webview.create_window(
        "J.A.R.V.I.S",
        url,
        width=1280,
        height=800,
        min_size=(980, 640),
        background_color="#01040a",
    )
    try:
        webview.start(gui="cocoa", debug=bool(os.getenv("JARVIS_DEBUG")))
    finally:
        server.should_exit = True
        thread.join(timeout=3)


if __name__ == "__main__":
    # Required for PyInstaller onefile builds on macOS so that multiprocessing
    # spawn children do not re-run the GUI launcher and create extra windows.
    multiprocessing.freeze_support()
    main()
