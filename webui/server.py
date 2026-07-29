"""
WebUI: FastAPI backend for lucina-NA monitoring.

IPC-only mode — requires daemon to be running.
- Chat: sends message via IPC, polls for response
- Status: reads daemon status snapshot (data/ipc/status.json)
- Memory: reads episode files from data/episodes/
- Plan: reads LTP state from data/long_term_plan.json
- Logs: tails data/logs/system.log + SSE
"""

import asyncio
import json
import logging
import os
import sys
import time
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

STATIC_DIR = Path(__file__).parent / "static"
IPC_DIR = Path("data/ipc")
EPISODE_DIR = Path("data/episodes")
LOG_PATH = Path("data/logs/system.log")
LTP_PATH = Path("data/long_term_plan.json")

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


# ── Chat via IPC ──
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
            await send_msg("thinking", {"phase": "processing",
                           "label": "Sending to daemon via IPC..."})

            # Send via IPC and poll
            response = await _ipc_send_and_wait(user_msg)
            if response:
                await send_msg("agent", {"text": response})
            else:
                await send_msg("error", {"text": "Daemon not responding. Start with: python main.py --daemon"})

    except WebSocketDisconnect:
        pass


async def _ipc_send_and_wait(message: str, timeout: float = 120) -> str | None:
    """Send message to daemon via IPC and wait for response."""
    msg_id = str(time.time())
    input_path = IPC_DIR / "input.json"
    output_path = IPC_DIR / "output.json"

    try:
        IPC_DIR.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w") as f:
            json.dump({"message": message, "id": msg_id}, f)
    except OSError as e:
        logger.warning(f"IPC write error: {e}")
        return None

    deadline = time.time() + timeout
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
        await asyncio.sleep(0.5)
    return None


# ── Read status from IPC snapshot ──
@app.get("/api/status")
async def api_status():
    status_path = IPC_DIR / "status.json"
    if not status_path.exists():
        return {
            "drives": {},
            "primary_drive": None,
            "personality": {},
            "memory": {"episodes": 0, "avg_importance": 0},
            "env": {},
            "plan": {},
            "daemon": None,
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
        return {
            "drives": drives,
            "primary_drive": data.get("primary_drive"),
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
            "daemon": {"mode": data.get("mode", "auto")},
            "rate_limit": data.get("rate_limit", {}),
        }
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


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
        }
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


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
    print(f"lucina-NA WebUI: http://{host}:{port}  (IPC mode — requires daemon)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
