"""
IPC module for daemon <-> message communication.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger("IPC")

IPC_DIR = "data/ipc"
INPUT_PATH = "data/ipc/input.json"
OUTPUT_PATH = "data/ipc/output.json"

_pending_message = None
_pending_event = threading.Event()


def ensure_dir():
    os.makedirs(IPC_DIR, exist_ok=True)


def read_input():
    """Read user message from IPC input file. Returns (message, msg_id) or None."""
    if not os.path.exists(INPUT_PATH):
        return None
    try:
        with open(INPUT_PATH) as f:
            data = json.load(f)
        os.remove(INPUT_PATH)
        msg = data.get("message")
        msg_id = data.get("id")
        if msg:
            return msg, msg_id
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"IPC read error: {e}")
        try:
            os.remove(INPUT_PATH)
        except OSError:
            pass
    return None


def write_output(response: str, msg_id=None):
    """Write agent response to IPC output file."""
    data = {"response": response, "id": msg_id, "timestamp": time.time()}
    try:
        ensure_dir()
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning(f"IPC write error: {e}")


def send_message(message: str) -> str | None:
    """Send a message to the daemon via IPC and wait for response."""
    if not os.path.exists(IPC_DIR):
        return None
    ensure_dir()
    msg_id = str(time.time())
    try:
        with open(INPUT_PATH, "w") as f:
            json.dump({"message": message, "id": msg_id}, f)
    except OSError as e:
        logger.warning(f"IPC send error: {e}")
        return None
    for _ in range(120):
        if os.path.exists(OUTPUT_PATH):
            try:
                with open(OUTPUT_PATH) as f:
                    data = json.load(f)
                if data.get("id") == msg_id:
                    os.remove(OUTPUT_PATH)
                    return data.get("response", "")
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.5)
    return None


def start_poller():
    """Start background thread that polls for IPC messages."""
    thread = threading.Thread(target=_poll_loop, daemon=True)
    thread.start()
    return thread


def _poll_loop():
    """Background loop: periodically check for IPC messages and set event."""
    global _pending_message
    while True:
        msg = read_input()
        if msg:
            _pending_message = msg
            _pending_event.set()
        time.sleep(1)


def has_pending():
    """Check if a pending IPC message exists (non-destructive)."""
    return _pending_event.is_set()


def get_pending():
    """Get pending IPC message (non-blocking). Returns (message, msg_id) or None."""
    global _pending_message
    if _pending_event.is_set():
        _pending_event.clear()
        msg = _pending_message
        _pending_message = None
        return msg
    return None


def wait_pending(timeout=0.5):
    """Wait for a pending IPC message. Returns (message, msg_id) or None."""
    if _pending_event.wait(timeout=timeout):
        return get_pending()
    return None


STATUS_PATH = "data/ipc/status.json"


def update_status(snapshot: dict):
    """Write daemon status snapshot (drives, env, current goal, etc.)."""
    snapshot["_updated"] = time.time()
    try:
        ensure_dir()
        with open(STATUS_PATH, "w") as f:
            json.dump(snapshot, f)
    except OSError as e:
        logger.warning(f"Status write error: {e}")
