"""Stop the HaqBot background Streamlit server started by start.py.

    python stop.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_FILE = ROOT / ".haqbot_run.json"


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


def _kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main() -> int:
    if not RUN_FILE.exists():
        print("HaqBot is not running (no .haqbot_run.json).")
        return 0

    try:
        data = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    pid = data.get("pid")
    port = data.get("port")

    if not pid:
        RUN_FILE.unlink(missing_ok=True)
        print("No pid recorded; cleared stale run file.")
        return 0

    pid = int(pid)
    if not _pid_alive(pid):
        RUN_FILE.unlink(missing_ok=True)
        print(f"HaqBot process {pid} was not running; cleared run file.")
        return 0

    print(f"Stopping HaqBot (pid {pid}) …")
    _kill(pid)
    for _ in range(20):
        if not _pid_alive(pid):
            break
        time.sleep(0.25)

    if _pid_alive(pid):
        print(f"Could not stop process {pid}. Kill it manually.")
        return 1

    RUN_FILE.unlink(missing_ok=True)
    if port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            still_up = sock.connect_ex(("localhost", int(port))) == 0
        if still_up:
            print(f"Process stopped but port {port} still answers; give it a moment.")
    print("HaqBot stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
