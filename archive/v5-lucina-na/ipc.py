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
    """Read user message from IPC input file.

    Returns (message, msg_id, history) or None.
    v3.5: history = 直前の会話ターン（[{"role": ..., "text": ...}]）リスト。
    """
    if not os.path.exists(INPUT_PATH):
        return None
    try:
        with open(INPUT_PATH) as f:
            data = json.load(f)
        os.remove(INPUT_PATH)
        msg = data.get("message")
        msg_id = data.get("id")
        history = data.get("history") or []
        if msg:
            return msg, msg_id, history
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
    # v4.0.3: ローカルLLM（gemma4）の応答生成は数分かかるため、待機時間を
    # 60秒 → 15分に延長（WebUI側 _ipc_send_and_wait と同じ方針）。
    for _ in range(1800):
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
    """Get pending IPC message (non-blocking). Returns (message, msg_id, history) or None."""
    global _pending_message
    if _pending_event.is_set():
        _pending_event.clear()
        msg = _pending_message
        _pending_message = None
        return msg
    return None


def wait_pending(timeout=0.5):
    """Wait for a pending IPC message. Returns (message, msg_id, history) or None."""
    if _pending_event.wait(timeout=timeout):
        return get_pending()
    return None


PROACTIVE_PATH = "data/ipc/proactive.json"


def write_proactive(message: str) -> str | None:
    """
    v4.0: 能動的発話を書き込む（エージェントから自発的にユーザーへ）。

    WebUI がポーリングしてチャットに表示する。id で重複を防ぐ。
    戻り値は発話 ID（書き込み失敗時は None）。
    """
    msg_id = f"pro_{int(time.time() * 1000)}"
    data = {"id": msg_id, "message": message, "timestamp": time.time()}
    try:
        ensure_dir()
        with open(PROACTIVE_PATH, "w") as f:
            json.dump(data, f)
        return msg_id
    except OSError as e:
        logger.warning(f"Proactive write error: {e}")
        return None


def read_proactive() -> dict | None:
    """最新の能動発話を読み出す（無ければ None）。"""
    if not os.path.exists(PROACTIVE_PATH):
        return None
    try:
        with open(PROACTIVE_PATH) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Proactive read error: {e}")
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


# ── Process management (daemon <-> WebUI <-> launcher) ──
#
# The launcher script (lucina.sh) supervises both the daemon and the WebUI.
# Each process writes its PID to data/run/<name>.pid. A "wanted" flag file
# (data/run/<name>.wanted) tells the supervisor whether the process should be
# kept alive: removing it stops the process permanently, creating it starts it.
# A process that exits with code 42 requests an immediate restart by its
# supervisor. The WebUI requests daemon stop/restart by writing a control file.

RUN_DIR = "data/run"
DAEMON_PID_FILE = os.path.join(RUN_DIR, "daemon.pid")
WEBUI_PID_FILE = os.path.join(RUN_DIR, "webui.pid")
DAEMON_WANTED_FILE = os.path.join(RUN_DIR, "daemon.wanted")
WEBUI_WANTED_FILE = os.path.join(RUN_DIR, "webui.wanted")
LAUNCHER_LOCK_FILE = os.path.join(RUN_DIR, "launcher.lock")
DAEMON_LOCK_FILE = os.path.join(RUN_DIR, "daemon.lock")
CONTROL_PATH = "data/ipc/control.json"

RESTART_EXIT_CODE = 42  # exit code that tells the supervisor to restart the process

# Module-level handle so the lock is held for the whole process lifetime.
_daemon_lock_fd = None


def ensure_run_dir():
    os.makedirs(RUN_DIR, exist_ok=True)


def acquire_daemon_lock() -> bool:
    """
    Acquire the single-instance daemon lock (advisory file lock).

    Prevents double-daemon races: the WebUI's self-heal spawn and a manual
    (supervisor / terminal) spawn could otherwise run two daemons at once,
    which made them fight over long_term_plan.json (stale empty-state
    overwrites) and the spawn log (interleaved writes / NUL bytes).

    Returns True if the lock was acquired (this process is the only daemon).
    """
    global _daemon_lock_fd
    if _daemon_lock_fd is not None:
        return True  # already held by this process
    ensure_run_dir()
    try:
        import fcntl
        fd = open(DAEMON_LOCK_FILE, "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ImportError):
        try:
            fd.close()
        except Exception:
            pass
        return False
    _daemon_lock_fd = fd
    fd.write(str(os.getpid()))
    fd.flush()
    return True


def release_daemon_lock():
    """Release the single-instance daemon lock (if held by this process)."""
    global _daemon_lock_fd
    if _daemon_lock_fd is None:
        return
    try:
        import fcntl
        fcntl.flock(_daemon_lock_fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        _daemon_lock_fd.close()
    except Exception:
        pass
    _daemon_lock_fd = None


def write_pid(pid_file: str):
    """Write the current process PID to the given pid file."""
    ensure_run_dir()
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        logger.warning(f"PID write error: {e}")


def remove_file(path: str):
    """Remove a file if it exists (no error if missing)."""
    try:
        os.remove(path)
    except OSError:
        pass


def read_pid(pid_file: str):
    """Read a PID from a pid file. Returns int or None."""
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid):
    """Check whether a process with the given PID is alive."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False


def process_running(pid_file: str):
    """Return (running: bool, pid: int|None) for a pid file."""
    pid = read_pid(pid_file)
    return pid_alive(pid), pid


def launcher_running() -> bool:
    """Check whether the lucina.sh supervisor launcher is alive."""
    return pid_alive(read_pid(LAUNCHER_LOCK_FILE))


def send_control(command: str) -> bool:
    """Request a daemon control action ("stop" / "restart") via control file."""
    try:
        ensure_dir()
        with open(CONTROL_PATH, "w") as f:
            json.dump({"command": command, "timestamp": time.time()}, f)
        return True
    except OSError as e:
        logger.warning(f"Control write error: {e}")
        return False


def control_pending() -> bool:
    """Check whether a control command is waiting to be processed."""
    return os.path.exists(CONTROL_PATH)


def get_control(min_timestamp=None):
    """Read and clear the daemon control command. Returns command or None.

    If min_timestamp is given, only commands written at/after that time are
    returned (older ones are discarded) so stale control files written while
    the daemon was stopped cannot trigger actions on a fresh start.
    """
    if not os.path.exists(CONTROL_PATH):
        return None
    try:
        with open(CONTROL_PATH) as f:
            data = json.load(f)
        remove_file(CONTROL_PATH)
        if min_timestamp is not None and data.get("timestamp", 0) < min_timestamp:
            return None
        return data.get("command")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Control read error: {e}")
        remove_file(CONTROL_PATH)
        return None
