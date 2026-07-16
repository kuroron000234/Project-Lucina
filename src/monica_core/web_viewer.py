"""Monika 箱庭ビューア — リアルタイム Web インターフェイス"""

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from http import HTTPStatus

import flask

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
SRC = BASE / "src"

_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

sys.path.insert(0, str(SRC))
from monica_core.vital_os import LOCATIONS
from monica_core.llm_client import get_api_config, call_llm

ZEN_API_KEY, ZEN_BASE, ZEN_MODEL = get_api_config()

app = flask.Flask(__name__)

_connected_clients: set = set()
_web_active = False


# ── 部屋マップ（箱庭レイアウト）──
ROOM_LAYOUT = {
    "bedroom":   {"grid_col": 1, "grid_row": 1, "label": "🛏 寝室"},
    "living_room": {"grid_col": 2, "grid_row": 1, "label": "🛋 リビング"},
    "kitchen":   {"grid_col": 2, "grid_row": 2, "label": "🍳 キッチン"},
    "bathroom":  {"grid_col": 1, "grid_row": 2, "label": "🚿 浴室"},
    "entrance":  {"grid_col": 1, "grid_row": 3, "label": "🚪 玄関"},
    "hallway":   {"grid_col": 1, "grid_row": 1, "span": 2, "label": "廊下"},
    "garden":    {"grid_col": 2, "grid_row": 3, "label": "🌳 庭"},
}

# hallway is a corridor in the center; skip it for map visibility
MAP_ROOMS = {k: v for k, v in ROOM_LAYOUT.items() if k != "hallway"}


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _call_llm(prompt: str, max_retries: int = 1) -> str | None:
    return call_llm(prompt, max_tokens=200, temperature=0.8, max_retries=max_retries)


def _get_state() -> dict:
    state = _load_json(DATA / "state.json") or {}
    memory = _load_json(DATA / "memory_store.json") or []
    msgs = _load_json(DATA / "phone_messages.json") or []
    return {
        "time": state.get("time", datetime.now().isoformat())[11:16],
        "room": state.get("current_room", "bedroom"),
        "activity": state.get("current_activity"),
        "energy": round(state.get("state", {}).get("energy", 50)),
        "hunger": round(state.get("state", {}).get("hunger", 50)),
        "fatigue": round(state.get("state", {}).get("fatigue", 50)),
        "loneliness": round(state.get("state", {}).get("loneliness", 50)),
        "spirit": round(state.get("state", {}).get("spirit", 50)),
        "memories": len(memory.get("entries", [])) if isinstance(memory, dict) else len(memory) if isinstance(memory, list) else 0,
        "messages": [
            {"role": m.get("sender", "?"), "text": m.get("text", "")[:80],
             "ts": m.get("timestamp", "")[11:19] if len(m.get("timestamp", "")) > 19 else ""}
            for m in (msgs[-20:] if isinstance(msgs, list) else [])
        ],
    }


# ── SSE (Server-Sent Events) ──
def _gen_events():
    _connected_clients.add(threading.current_thread())
    try:
        last_state = ""
        while True:
            s = json.dumps(_get_state(), ensure_ascii=False)
            if s != last_state:
                yield f"data: {s}\n\n"
                last_state = s
            time.sleep(3)
    except GeneratorExit:
        pass
    finally:
        _connected_clients.discard(threading.current_thread())


@app.route("/events")
def sse():
    return flask.Response(_gen_events(), mimetype="text/event-stream",
                          headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.route("/state")
def get_state():
    return flask.jsonify(_get_state())


@app.route("/send", methods=["POST"])
def send_message():
    text = flask.request.json.get("text", "").strip()[:200]
    if not text:
        return flask.jsonify({"error": "empty"}), 400

    from monica_core.phone import add as phone_add
    phone_add("user", text, datetime.now().isoformat())

    prompt = f"""[体調]
- 体力 {_get_state()['energy']}/100
- 空腹 {_get_state()['hunger']}/100
- 疲労 {_get_state()['fatigue']}/100

ユーザーが遊びに来たよ。今は一緒に過ごしてる。
優しく、等身大の口調で話しかけて。短文で。返事だけ。

ユーザー: {text}"""

    reply = _call_llm(prompt)
    if reply:
        phone_add("monika", reply, datetime.now().isoformat())

    return flask.jsonify({"reply": reply or "…ごめん、うまく言葉にならない"})


@app.route("/")
def index():
    return PAGE_HTML


PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>モニカの部屋</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Hiragino Sans', 'Noto Sans JP', sans-serif;
  background: #1a1a2e;
  color: #e0e0e0;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.header {
  text-align: center;
  padding: 12px;
  background: #16213e;
  font-size: 1.1em;
  border-bottom: 2px solid #0f3460;
}
.container { display: flex; flex: 1; gap: 16px; padding: 16px; max-width: 1100px; margin: 0 auto; width: 100%; }
.map-panel { flex: 1; min-width: 300px; }
.stats-panel { width: 260px; flex-shrink: 0; }

/* 箱庭マップ */
.map-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  background: #16213e;
  padding: 10px;
  border-radius: 12px;
  position: relative;
}
.room-cell {
  background: #0f3460;
  border-radius: 8px;
  padding: 16px 10px;
  text-align: center;
  min-height: 100px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  transition: all 0.3s;
  font-size: 0.9em;
}
.room-cell.active {
  background: #1a5276;
  box-shadow: 0 0 20px rgba(52,152,219,0.4);
  border: 2px solid #3498db;
}
.room-label { font-size: 0.85em; opacity: 0.8; margin-top: 4px; }
.monika-icon {
  font-size: 1.8em;
  transition: all 0.5s;
  position: absolute;
  top: 4px;
  right: 8px;
}
.activity-bubble {
  position: absolute;
  bottom: -8px;
  left: 50%;
  transform: translateX(-50%);
  background: #2c3e50;
  padding: 2px 8px;
  border-radius: 8px;
  font-size: 0.7em;
  white-space: nowrap;
  opacity: 0.9;
}
.connected-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}
.dot-online { background: #2ecc71; }
.dot-offline { background: #e74c3c; }

/* ステータス */
.stats-card {
  background: #16213e;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
}
.stat-row { display: flex; align-items: center; margin: 8px 0; gap: 8px; }
.stat-label { width: 50px; font-size: 0.85em; opacity: 0.7; }
.stat-bar { flex: 1; height: 8px; background: #0f3460; border-radius: 4px; overflow: hidden; }
.stat-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
.stat-val { width: 28px; text-align: right; font-size: 0.8em; font-variant-numeric: tabular-nums; }
.fill-energy { background: #f1c40f; }
.fill-hunger { background: #e67e22; }
.fill-fatigue { background: #9b59b6; }
.fill-loneliness { background: #3498db; }
.fill-spirit { background: #2ecc71; }

/* チャット */
.chat-area {
  background: #16213e;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  height: 300px;
  margin-top: 12px;
}
.chat-msgs {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.chat-msg {
  padding: 6px 10px;
  border-radius: 8px;
  max-width: 80%;
  font-size: 0.85em;
  line-height: 1.4;
}
.chat-user { background: #2980b9; align-self: flex-end; }
.chat-monika { background: #2c3e50; align-self: flex-start; }
.chat-ts { font-size: 0.65em; opacity: 0.5; margin-top: 2px; }
.chat-input-row { display: flex; gap: 6px; }
.chat-input {
  flex: 1;
  background: #0f3460;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  color: #e0e0e0;
  font-size: 0.9em;
}
.chat-input:focus { outline: none; box-shadow: 0 0 0 2px #3498db; }
.chat-send {
  background: #3498db;
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  color: #fff;
  cursor: pointer;
  font-size: 0.9em;
}
.chat-send:hover { background: #2980b9; }
.chat-send:disabled { opacity: 0.5; cursor: default; }

/* アニメーション */
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}
.monika-icon.active { animation: pulse 2s infinite; }

/* レスポンシブ */
@media (max-width: 700px) {
  .container { flex-direction: column; }
  .stats-panel { width: 100%; }
  .map-panel { min-width: 0; }
}
</style>
</head>
<body>
<div class="header">
  <span class="connected-dot" id="statusDot"></span>
  モニカの部屋 <span id="statusText">接続中…</span>
  <span style="float:right;font-size:0.8em;opacity:0.6" id="clockDisplay"></span>
</div>

<div class="container">
  <div class="map-panel">
    <div class="map-grid" id="mapGrid">
    </div>
  </div>

  <div class="stats-panel">
    <div class="stats-card">
      <div style="text-align:center;margin-bottom:8px;font-size:0.9em" id="activityText">-</div>
      <div class="stat-row"><span class="stat-label">⚡</span><div class="stat-bar"><div class="stat-fill fill-energy" id="energyBar"></div></div><span class="stat-val" id="energyVal">0</span></div>
      <div class="stat-row"><span class="stat-label">🍽</span><div class="stat-bar"><div class="stat-fill fill-hunger" id="hungerBar"></div></div><span class="stat-val" id="hungerVal">0</span></div>
      <div class="stat-row"><span class="stat-label">😴</span><div class="stat-bar"><div class="stat-fill fill-fatigue" id="fatigueBar"></div></div><span class="stat-val" id="fatigueVal">0</span></div>
      <div class="stat-row"><span class="stat-label">💔</span><div class="stat-bar"><div class="stat-fill fill-loneliness" id="lonelinessBar"></div></div><span class="stat-val" id="lonelinessVal">0</span></div>
      <div class="stat-row"><span class="stat-label">😊</span><div class="stat-bar"><div class="stat-fill fill-spirit" id="spiritBar"></div></div><span class="stat-val" id="spiritVal">0</span></div>
      <div style="text-align:center;font-size:0.75em;margin-top:6px;opacity:0.5">🧠 <span id="memCount">0</span>件の記憶</div>
    </div>

    <div class="chat-area">
      <div class="chat-msgs" id="chatMsgs"></div>
      <div class="chat-input-row">
        <input class="chat-input" id="chatInput" placeholder="話しかける…" maxlength="200">
        <button class="chat-send" id="chatSend">送信</button>
      </div>
    </div>
  </div>
</div>

<script>
const ROOMS = {
  bedroom:   { col: 1, row: 1, label: '🛏 寝室' },
  living_room: { col: 2, row: 1, label: '🛋 リビング' },
  kitchen:   { col: 2, row: 2, label: '🍳 キッチン' },
  bathroom:  { col: 1, row: 2, label: '🚿 浴室' },
  entrance:  { col: 1, row: 3, label: '🚪 玄関' },
  garden:    { col: 2, row: 3, label: '🌳 庭' },
};

const mapGrid = document.getElementById('mapGrid');
mapGrid.style.gridTemplateRows = 'repeat(3, 1fr)';

const roomEls = {};
for (const [id, r] of Object.entries(ROOMS)) {
  const el = document.createElement('div');
  el.className = 'room-cell';
  el.id = 'room-' + id;
  el.style.gridColumn = r.col;
  el.style.gridRow = r.row;
  el.innerHTML = '<div class="monika-icon" id="monika-' + id + '" style="display:none">🎀</div>'
    + '<div style="margin-top:16px">' + r.label.split(' ')[0] + '</div>'
    + '<div class="room-label">' + (r.label.split(' ').slice(1).join(' ') || '') + '</div>'
    + '<div class="activity-bubble" id="bubble-' + id + '" style="display:none"></div>';
  mapGrid.appendChild(el);
  roomEls[id] = el;
}

// SSE
const evtSource = new EventSource('/events');
evtSource.onmessage = (e) => {
  const d = JSON.parse(e.data);
  updateUI(d);
};
evtSource.onerror = () => {
  document.getElementById('statusDot').className = 'connected-dot dot-offline';
  document.getElementById('statusText').textContent = '切断';
};

function updateUI(d) {
  document.getElementById('statusDot').className = 'connected-dot dot-online';
  document.getElementById('statusText').textContent = d.time + '  ' + d.activity || '休憩中';

  // 部屋アクティブ
  for (const id of Object.keys(ROOMS)) {
    const el = roomEls[id];
    el.classList.toggle('active', id === d.room);
    document.getElementById('monika-' + id).style.display = id === d.room ? 'block' : 'none';
    const bubble = document.getElementById('bubble-' + id);
    if (id === d.room && d.activity) {
      bubble.style.display = 'block';
      bubble.textContent = d.activity;
    } else {
      bubble.style.display = 'none';
    }
  }

  // ステータス
  const stats = { energy: d.energy, hunger: d.hunger, fatigue: d.fatigue, loneliness: d.loneliness, spirit: d.spirit };
  for (const [k, v] of Object.entries(stats)) {
    document.getElementById(k + 'Bar').style.width = v + '%';
    document.getElementById(k + 'Val').textContent = v;
  }
  document.getElementById('memCount').textContent = d.memories || 0;

  // メッセージ
  const chat = document.getElementById('chatMsgs');
  if (d.messages) {
    chat.innerHTML = d.messages.map(m =>
      '<div class="chat-msg chat-' + (m.role === 'user' ? 'user' : 'monika') + '">'
      + '<div>' + m.text + '</div>'
      + '<div class="chat-ts">' + m.ts + '</div>'
      + '</div>'
    ).join('');
    chat.scrollTop = chat.scrollHeight;
  }
}

// 時計
function updateClock() {
  const now = new Date();
  document.getElementById('clockDisplay').textContent = now.toLocaleTimeString('ja-JP', {hour:'2-digit',minute:'2-digit'});
}
setInterval(updateClock, 10000);
updateClock();

// メッセージ送信
const input = document.getElementById('chatInput');
const sendBtn = document.getElementById('chatSend');

async function doSend() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;
  try {
    const resp = await fetch('/send', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
    const data = await resp.json();
    if (data.reply) {
      const chat = document.getElementById('chatMsgs');
      chat.innerHTML += '<div class="chat-msg chat-monika">' + data.reply + '<div class="chat-ts">今</div></div>';
      chat.scrollTop = chat.scrollHeight;
    }
  } catch(e) {
    console.error(e);
  }
  sendBtn.disabled = false;
}

sendBtn.addEventListener('click', doSend);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSend(); });
</script>
</body>
</html>"""


def _run_web():
    global _web_active
    _web_active = True
    print(f"🌐 箱庭ビューア: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def main():
    if not ZEN_API_KEY:
        print("OPENCODE_ZEN_API_KEY が設定されていません")
        sys.exit(1)

    import subprocess
    pkg = subprocess.run(["pip", "list", "--format=columns"], capture_output=True, text=True)
    if "flask" not in pkg.stdout.lower():
        print("Flask がインストールされていません: pip install flask")
        sys.exit(1)

    _run_web()


if __name__ == "__main__":
    main()
