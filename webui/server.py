"""
WebUI: FastAPI backend for lucina-NA monitoring.

IPC-only mode — requires daemon to be running.
- Chat: sends message via IPC, polls for response
- Status: reads daemon status snapshot (data/ipc/status.json)
- Memory: reads episode files from data/episodes/
- Plan: reads LTP state from data/long_term_plan.json
- Logs: tails data/logs/system.log + SSE
- Control: start/stop/restart daemon & UI via control file + pid files
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# No layer imports — IPC only

logger = logging.getLogger("WebUI")
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Allow direct launch (`python webui/server.py`) from any cwd inside the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ipc

STATIC_DIR = Path(__file__).parent / "static"
IPC_DIR = Path("data/ipc")
EPISODE_DIR = Path("data/episodes")
LOG_PATH = Path("data/logs/system.log")
LTP_PATH = Path("data/long_term_plan.json")
BENCHMARK_DIR = Path("data/benchmarks")

_log_ring = []


class LogInterceptor(logging.Handler):
    def emit(self, record):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": record.name,
            "level": record.levelname,
            "msg": self.format(record),
        }
        _log_ring.append(entry)
        if len(_log_ring) > 500:
            _log_ring[:] = _log_ring[-500:]

logging.getLogger().addHandler(LogInterceptor())

app = FastAPI(title="lucina-NA WebUI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return {"error": "index.html not found"}


# ── Daemon auto-start (self-healing when the daemon is down) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spawned_daemon: subprocess.Popen | None = None


def _spawn_daemon_process() -> subprocess.Popen | None:
    """Spawn the daemon directly (no lucina.sh supervisor present).

    Writes the daemon's stdout/stderr to data/logs/daemon_spawn.log so startup
    errors are captured even if the daemon dies before logging is configured.

    v4.0.1: 二重起動ガード — 既に別のデーモンが動いている（PIDファイルが
    生きている）場合はスパウンしない。手動起動との競合で二重デーモンになる
    のを防ぐ（main.py 側のファイルロックと合わせて両側で防御）。
    """
    global _spawned_daemon
    if _spawned_daemon is not None and _spawned_daemon.poll() is None:
        return _spawned_daemon  # already running
    running, pid = ipc.process_running(ipc.DAEMON_PID_FILE)
    if running:
        logger.info(f"Daemon already running (PID {pid}); skip self-heal spawn")
        return None
    log_dir = PROJECT_ROOT / "data" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        spawn_log = (log_dir / "daemon_spawn.log").open("a")
    except OSError as e:
        logger.warning(f"Daemon spawn log open failed: {e}")
        spawn_log = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "main.py", "--daemon"],
            cwd=str(PROJECT_ROOT),
            stdout=spawn_log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if spawn_log else subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        logger.warning(f"Daemon spawn failed: {e}")
        return None
    finally:
        # Child inherits its own fd copy; close the parent's handle.
        if spawn_log is not None:
            spawn_log.close()
    _spawned_daemon = proc
    asyncio.get_running_loop().create_task(_watch_daemon(proc))
    logger.info(f"Daemon spawned directly (PID {proc.pid})")
    return proc


async def _watch_daemon(proc: subprocess.Popen):
    """Mini-supervisor for a WebUI-spawned daemon: respawn on restart request."""
    while proc.poll() is None:
        await asyncio.sleep(1)
    code = proc.poll()
    if code == ipc.RESTART_EXIT_CODE:
        logger.info("Daemon requested restart -> respawning (WebUI supervisor)")
        _spawn_daemon_process()
    else:
        logger.info(f"WebUI-spawned daemon exited (code={code})")


async def _ensure_daemon(timeout: float = 25.0) -> bool:
    """Make sure the daemon is running, starting it if needed.

    - Daemon already running -> True
    - lucina.sh supervisor alive -> arm its wanted flag and wait
    - No supervisor -> spawn main.py --daemon directly and wait
    Returns True once the daemon PID file is alive.
    """
    running, _ = ipc.process_running(ipc.DAEMON_PID_FILE)
    if running:
        return True
    if ipc.launcher_running():
        _touch(ipc.DAEMON_WANTED_FILE)
    else:
        _spawn_daemon_process()
    deadline = time.time() + timeout
    while time.time() < deadline:
        running, _ = ipc.process_running(ipc.DAEMON_PID_FILE)
        if running:
            return True
        await asyncio.sleep(0.5)
    return False


# ── Chat via IPC ──
# v3.5 (A): 会話履歴バッファ。WebUI プロセスが生きている間、直近の
# 会話ターン（ユーザー/エージェント発話）を保持し、IPC 経由でデーモンへ送る。
# これにより「続けて」「さっきの話」のような文脈依存発話でも直前の
# 会話を参照できるようになる。
_chat_history: deque = deque(maxlen=20)


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()

    async def send_msg(kind, data):
        await ws.send_text(json.dumps({"kind": kind, **data}, ensure_ascii=False))

    await send_msg("system", {"text": "IPC mode (daemon required)"})

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            user_msg = msg.get("message", "").strip()
            if not user_msg:
                continue

            await send_msg("user", {"text": user_msg})

            # Self-healing: if the daemon is down, auto-start it first.
            daemon_down = not ipc.process_running(ipc.DAEMON_PID_FILE)[0]
            if daemon_down:
                await send_msg("thinking", {"phase": "starting",
                               "label": "デーモンが停止中です。自動で起動しています..."})
                if not await _ensure_daemon():
                    await send_msg("error", {"text": "デーモンを起動できませんでした。"
                                              "コントロールタブの「デーモン起動」を試すか、"
                                              "ターミナルで ./lucina.sh を実行してください。"})
                    continue
                await send_msg("thinking", {"phase": "processing",
                               "label": "デーモン起動完了。メッセージを送信しています..."})
            else:
                await send_msg("thinking", {"phase": "processing",
                               "label": "Sending to daemon via IPC..."})

            # v3.5 (A): 今回の発話を履歴に追加し、直近のターン（今回を除く）を送る
            _chat_history.append({"role": "user", "text": user_msg})
            history = list(_chat_history)[:-1]

            # Send via IPC and poll
            # v4.0.2: ローカルLLMの応答生成は数分かかるため、20秒ごとに
            # 進捗ハートビートを送って「処理中」であることを表示する。
            async def progress_cb(elapsed: int):
                await send_msg("thinking", {"phase": "processing",
                               "label": f"ローカルLLMが応答を生成中です...（{elapsed}秒経過）"})

            response = await _ipc_send_and_wait(user_msg, history=history,
                                                on_progress=progress_cb)
            if response:
                _chat_history.append({"role": "assistant", "text": response})
                await send_msg("agent", {"text": response})
            else:
                # 応答なし: ぶら下がりターンにならないよう今回の発話だけ履歴から除く
                _chat_history.pop()
                await send_msg("error", {"text": "デーモンが応答しませんでした（タイムアウト）。"
                                          "デーモンの起動状況をコントロールタブで確認してください。"})

    except WebSocketDisconnect:
        pass


async def _ipc_send_and_wait(message: str, timeout: float = 900,
                             history: list | None = None,
                             on_progress=None) -> str | None:
    """Send message to daemon via IPC and wait for response.

    v4.0.2: タイムアウトを 120s → 900s（15分）に延長。ローカルLLM
    （gemma4）はチャット応答の生成に数分かかるため、デーモンが正常に
    処理していても 120s では必ずタイムアウトしてしまう。
    on_progress が与えられた場合は 20秒ごとに経過秒数を通知する。
    """
    msg_id = str(time.time())
    input_path = IPC_DIR / "input.json"
    output_path = IPC_DIR / "output.json"
    start = time.time()

    try:
        IPC_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"message": message, "id": msg_id}
        # v3.5 (A): 直前の会話ターンをペイロードに含める
        if history:
            payload["history"] = history
        with open(input_path, "w") as f:
            json.dump(payload, f)
    except OSError as e:
        logger.warning(f"IPC write error: {e}")
        return None

    deadline = time.time() + timeout
    last_progress = time.time()
    while time.time() < deadline:
        if output_path.exists():
            try:
                with open(output_path) as f:
                    data = json.load(f)
                if data.get("id") == msg_id:
                    output_path.unlink(missing_ok=True)
                    return data.get("response", "")
            except (json.JSONDecodeError, OSError):
                pass
        if on_progress and time.time() - last_progress >= 20:
            last_progress = time.time()
            await on_progress(int(time.time() - start))
        await asyncio.sleep(0.5)
    return None


# ── Read status from IPC snapshot ──
@app.get("/api/status")
async def api_status():
    daemon_running, daemon_pid = ipc.process_running(ipc.DAEMON_PID_FILE)
    status_path = IPC_DIR / "status.json"
    if not status_path.exists():
        return {
            "drives": {},
            "primary_drive": None,
            "personality": {},
            "memory": {"episodes": 0, "avg_importance": 0},
            "env": {},
            "plan": {},
            "daemon": {"running": daemon_running, "pid": daemon_pid, "mode": "unknown"},
            "error": "Daemon not running",
        }
    try:
        with open(status_path) as f:
            data = json.load(f)
        # Reformat for frontend
        drives = data.get("drives", {})
        pers = data.get("personality", {})
        plan = data.get("plan", {})
        env_p = data.get("env_cpu")
        # v5.0: サプライズは cycle_latest.json の drive.surprise から供給する
        # （実測予測誤差。tier2/3 サイクルでのみ値を持つ）
        surprise = None
        cycle_path = IPC_DIR / "cycle_latest.json"
        if cycle_path.exists():
            try:
                with open(cycle_path) as f:
                    surprise = json.load(f).get("drive", {}).get("surprise")
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "drives": drives,
            "primary_drive": data.get("primary_drive"),
            "surprise": surprise,
            "personality": pers,
            "memory": {
                "episodes": data.get("memory_episodes", 0),
                "avg_importance": 0,
            },
            "env": {
                "cpu": data.get("env_cpu"),
                "memory": data.get("env_memory"),
                "network": data.get("env_network"),
            } if env_p is not None else {},
            "plan": plan,
            "daemon": {"running": daemon_running, "pid": daemon_pid,
                        "mode": data.get("mode", "auto")},
            "rate_limit": data.get("rate_limit", {}),
        }
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


# ── Process control (daemon / UI start-stop-restart) ──
# The daemon honours data/ipc/control.json (stop/restart) and exits with
# code 42 to request a restart from the launcher supervisor. The WebUI
# restarts itself the same way, and removes its "wanted" flag before a
# permanent stop so the supervisor does not bring it back.


def _touch(path: str):
    ipc.ensure_run_dir()
    with open(path, "w") as f:
        f.write("")


async def _delayed_exit(code: int, delay: float = 0.6):
    """Exit the WebUI process shortly after the HTTP response is flushed."""
    await asyncio.sleep(delay)
    ipc.remove_file(ipc.WEBUI_PID_FILE)
    os._exit(code)


@app.get("/api/control/status")
async def api_control_status():
    daemon_running, daemon_pid = ipc.process_running(ipc.DAEMON_PID_FILE)
    webui_running, webui_pid = ipc.process_running(ipc.WEBUI_PID_FILE)
    return {
        "daemon": {
            "running": daemon_running,
            "pid": daemon_pid,
            "wanted": os.path.exists(ipc.DAEMON_WANTED_FILE),
        },
        "webui": {
            "running": webui_running,
            "pid": webui_pid,
            "wanted": os.path.exists(ipc.WEBUI_WANTED_FILE),
        },
        # A supervisor (lucina.sh) is present when its lock file is alive.
        "supervisor": ipc.launcher_running(),
    }


@app.post("/api/control/daemon/stop")
async def control_daemon_stop():
    ok = ipc.send_control("stop")
    return {"ok": ok,
            "message": "デーモン停止リクエスト送信" if ok else "制御ファイルの書き込みに失敗"}


@app.post("/api/control/daemon/start")
async def control_daemon_start():
    running, pid = ipc.process_running(ipc.DAEMON_PID_FILE)
    if running:
        return {"ok": False, "message": f"デーモンは既に起動中 (PID {pid})"}
    ok = await _ensure_daemon()
    if ok:
        _, pid = ipc.process_running(ipc.DAEMON_PID_FILE)
        return {"ok": True, "message": f"デーモンを起動しました (PID {pid})"}
    return {"ok": False, "message": "デーモンの起動に失敗しました。"
                                 "ログ (data/logs/daemon_spawn.log / system.log) を確認してください。"}


@app.post("/api/control/daemon/restart")
async def control_daemon_restart():
    running, _ = ipc.process_running(ipc.DAEMON_PID_FILE)
    if not running:
        ok = await _ensure_daemon()
        return {"ok": ok,
                "message": "デーモンを起動しました" if ok else "デーモンの起動に失敗しました"}
    # Restart works when someone can bring the process back: the lucina.sh
    # supervisor or the WebUI's own watcher (for a WebUI-spawned daemon).
    if ipc.launcher_running() or _spawned_daemon is not None:
        ok = ipc.send_control("restart")
        return {"ok": ok,
                "message": "デーモン再起動リクエスト送信" if ok else "制御ファイルの書き込みに失敗"}
    return {"ok": False, "message": "スーパーバイザが動作していないため再起動できません。"
                                 "「デーモン起動」で停止中を再開できます。"}


@app.post("/api/control/ui/restart")
async def control_ui_restart():
    if not ipc.launcher_running():
        return {"ok": False, "message": "スーパーバイザ (lucina.sh) が動作していないため再起動できません。起動には ./lucina.sh を実行してください。"}
    asyncio.create_task(_delayed_exit(ipc.RESTART_EXIT_CODE))
    return {"ok": True, "message": "UIを再起動します... (数秒間ページが繋がりません)"}


@app.post("/api/control/ui/stop")
async def control_ui_stop():
    ipc.remove_file(ipc.WEBUI_WANTED_FILE)
    asyncio.create_task(_delayed_exit(0))
    return {"ok": True, "message": "UIを停止します... 再起動は ./lucina.sh から"}


@app.post("/api/control/all/restart")
async def control_all_restart():
    if not ipc.launcher_running():
        return {"ok": False, "message": "スーパーバイザ (lucina.sh) が動作していないため再起動できません。起動には ./lucina.sh を実行してください。"}
    # Re-arm the daemon so a previously stopped daemon also comes back up.
    _touch(ipc.DAEMON_WANTED_FILE)
    ipc.send_control("restart")
    asyncio.create_task(_delayed_exit(ipc.RESTART_EXIT_CODE))
    return {"ok": True, "message": "デーモンとUIを再起動します..."}


@app.post("/api/control/all/stop")
async def control_all_stop():
    ipc.send_control("stop")
    ipc.remove_file(ipc.WEBUI_WANTED_FILE)
    asyncio.create_task(_delayed_exit(0))
    return {"ok": True, "message": "デーモンとUIを停止します..."}


# ── Read episodes directly ──
@app.get("/api/memory")
async def api_memory(top_k: int = 50, min_importance: float = 0.0):
    if top_k > 200:
        top_k = 200
    episodes = []
    try:
        files = sorted(EPISODE_DIR.glob("*.json"), reverse=True)[:top_k]
        for fpath in files:
            try:
                with open(fpath) as f:
                    ep = json.load(f)
                imp = ep.get("importance", 0)
                if imp >= min_importance:
                    episodes.append({
                        "id": ep.get("id", fpath.stem),
                        "time": ep.get("timestamp", ""),
                        "event": str(ep.get("event", ""))[:60],
                        "importance": round(imp, 2),
                        "tags": ep.get("tags", []),
                        "result": str(ep.get("result", ""))[:40],
                    })
            except (json.JSONDecodeError, OSError):
                pass
    except OSError:
        pass
    return {"episodes": episodes, "total": len(episodes)}


# ── Read logs directly ──
@app.get("/api/logs")
async def api_logs(lines: int = 100):
    if not LOG_PATH.exists():
        return {"logs": []}
    try:
        with open(LOG_PATH) as f:
            all_lines = f.readlines()
        return {"logs": all_lines[-lines:]}
    except OSError:
        return {"logs": []}


# ── Read plan from LTP file directly ──
@app.get("/api/plan")
async def api_plan():
    if not LTP_PATH.exists():
        return {"goals": [], "routines": [], "focus_area": "", "identity_policy": ""}
    try:
        with open(LTP_PATH) as f:
            data = json.load(f)
        goals = data.get("goals", [])
        routines_raw = data.get("routines", [])
        return {
            "goals": [{"goal": g.get("goal", "")[:80], "progress": g.get("progress", 0),
                        "priority": g.get("priority", "medium")} for g in goals],
            "routines": [{"name": r.get("name", ""), "description": r.get("action", "")[:60],
                           "interval_hours": r.get("interval_hours")}
                          for r in routines_raw],
            "focus_area": data.get("focus_area", ""),
            "identity_policy": data.get("identity_policy", "")[:100],
            # v4.0: 願望
            "aspirations": data.get("aspirations", []) or [],
        }
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


# ── v4.0: 能動発話（エージェントから自発的に話しかける）──
@app.get("/api/proactive")
async def api_proactive():
    data = ipc.read_proactive()
    if not data:
        return {"id": None, "message": None, "timestamp": None}
    return {"id": data.get("id"), "message": data.get("message"),
            "timestamp": data.get("timestamp")}


# ── v4.0: 日記 ──
DIARY_DIR = Path("data/diary")


@app.get("/api/diary")
async def api_diary():
    if not DIARY_DIR.exists():
        return {"diaries": []}
    try:
        files = sorted(DIARY_DIR.glob("*.md"), reverse=True)[:10]
        diaries = []
        for f in files:
            diaries.append({
                "date": f.stem,
                "content": f.read_text(encoding="utf-8")[:2000],
            })
        return {"diaries": diaries}
    except OSError:
        return {"diaries": []}


# ── v4.0: 自分の部屋（ワークスペース）──
WORKSPACE_DIR = Path("data/workspace")


@app.get("/api/workspace")
async def api_workspace():
    if not WORKSPACE_DIR.exists():
        return {"files": []}
    try:
        files = []
        for f in sorted(WORKSPACE_DIR.rglob("*")):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(WORKSPACE_DIR)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        return {"files": files}
    except OSError:
        return {"files": []}


# ── v5.0: Phase 3 ベンチマーク（data/benchmarks/*.json）──
@app.get("/api/benchmarks")
async def api_benchmarks():
    if not BENCHMARK_DIR.exists():
        return {"reports": [], "generated_at": None}
    reports = []
    try:
        for f in sorted(BENCHMARK_DIR.glob("*.json")):
            try:
                reports.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass
    return {"reports": reports}


@app.post("/api/benchmarks/run")
async def api_benchmarks_run():
    """ベンチマークを再実行する（決定論的・実LLM不使用・数秒で完了）。"""
    try:
        from benchmarks.run_all import run_all
        ok = run_all()
        return {"ok": bool(ok), "message": "ベンチマーク再実行が完了しました"}
    except Exception as e:
        logger.warning(f"Benchmark run failed: {e}")
        return {"ok": False, "message": f"実行失敗: {e}"}


# ── Read latest cycle details from IPC ──
@app.get("/api/cycle")
async def api_cycle():
    cycle_path = IPC_DIR / "cycle_latest.json"
    if not cycle_path.exists():
        return {"error": "No cycle data yet", "cycle": None}
    try:
        with open(cycle_path) as f:
            data = json.load(f)
        return {"cycle": data}
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e), "cycle": None}


# ── SSE: Real-time log stream ──
@app.get("/stream/logs")
async def stream_logs(request: Request):
    async def event_generator():
        last_index = len(_log_ring)
        while True:
            if await request.is_disconnected():
                break
            while len(_log_ring) > last_index:
                entry = _log_ring[last_index]
                last_index += 1
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main():
    port = int(os.environ.get("WEBUI_PORT", "8765"))
    host = os.environ.get("WEBUI_HOST", "127.0.0.1")
    ipc.write_pid(ipc.WEBUI_PID_FILE)
    print(f"lucina-NA WebUI: http://{host}:{port}  (IPC mode — requires daemon)")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        ipc.remove_file(ipc.WEBUI_PID_FILE)


if __name__ == "__main__":
    main()
