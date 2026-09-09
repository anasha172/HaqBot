"""Start HaqBot as a background Streamlit server.

    python start.py            # start on http://localhost:8501
    HAQBOT_PORT=9000 python start.py

Writes .haqbot_run.json (pid + port) and streams output to .haqbot.log.
Use `python stop.py` to shut it down.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_FILE = ROOT / ".haqbot_run.json"
LOG_FILE = ROOT / ".haqbot.log"
PORT = int(os.environ.get("HAQBOT_PORT", "8501"))
HOST = os.environ.get("HAQBOT_HOST", "localhost")

# Air-gap: block huggingface_hub / transformers network access.
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    if RUN_FILE.exists():
        try:
            existing = json.loads(RUN_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        pid = existing.get("pid")
        if pid and _pid_alive(int(pid)):
            print(
                f"HaqBot already running (pid {pid}) on "
                f"http://{existing.get('host', HOST)}:{existing.get('port', PORT)}"
            )
            print("Run `python stop.py` first if you want to restart it.")
            return 0
        RUN_FILE.unlink(missing_ok=True)

    if _port_listening(HOST, PORT):
        print(f"Port {PORT} is already in use. Set HAQBOT_PORT to a free port.")
        return 1

    env = os.environ.copy()
    for key, value in config.OFFLINE_ENV.items():
        env.setdefault(key, value)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", str(PORT),
        "--server.address", HOST,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )

    log = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )

    RUN_FILE.write_text(
        json.dumps({"pid": proc.pid, "port": PORT, "host": HOST}, indent=2),
        encoding="utf-8",
    )

    url = f"http://{HOST}:{PORT}"
    print(f"Starting HaqBot (pid {proc.pid}) …")
    for _ in range(60):
        if proc.poll() is not None:
            print(f"HaqBot exited early (code {proc.returncode}). See {LOG_FILE.name}:")
            print(LOG_FILE.read_text(encoding="utf-8")[-2000:])
            RUN_FILE.unlink(missing_ok=True)
            return 1
        if _port_listening(HOST, PORT):
            print(f"HaqBot is up  ->  {url}")
            print(f"logs: {LOG_FILE.name}   ·   stop: python stop.py")
            return 0
        time.sleep(0.5)

    print(f"HaqBot did not report ready in time; check {LOG_FILE.name}. URL: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
