#!/usr/bin/env python3
"""モニカ Web サーバー — Phase 6: デスクトップに棲む"""

import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# モニカをインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))
from monica.core.monica import Monica

app = FastAPI()
monica: Optional[Monica] = None

# ── 自律ループ ──

def autonomous_loop():
    """バックグラウンドで自律ループを回す"""
    global monica
    while True:
        try:
            if monica:
                msg = monica.autonomous_step()
                if msg:
                    print(f"[auto] {msg[:50]}...")
        except Exception as e:
            print(f"[auto error] {e}")
        time.sleep(15)


# ── API ──

@app.get("/api/state")
def get_state():
    """モニカの現在の状態を返す"""
    if not monica:
        return {"error": "モニカが起動していません"}
    return {
        "model": monica.brain.config.model,
        "mood": monica.state.get_mood_description(),
        "conversations": monica.conversation_count,
        "episodes": monica.episodic.count,
        "familiarity": monica.relationship.familiarity,
        "user_name": monica.relationship.user_name,
        "auto_last": monica._last_autonomous_action,
    }


@app.get("/api/episodes")
def get_episodes():
    if not monica:
        return {"error": "not ready"}
    return {"episodes": monica.episodic.get_recent(10)}


@app.post("/api/chat")
async def chat(request: Request):
    """チャットエンドポイント"""
    global monica
    data = await request.json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return {"response": "何か言いました？"}

    if not monica:
        return {"response": "モニカがまだ準備できていません..."}

    response = monica.think(user_input)
    return {
        "response": response or "...",
        "state": {
            "mood": monica.state.get_mood_description(),
            "conversations": monica.conversation_count,
        }
    }


# ── Web UI ──

HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>モニカ</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Hiragino Sans', 'Noto Sans JP', sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .header {
    padding: 16px 20px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header h1 { font-size: 18px; color: #e94560; }
  .status { font-size: 12px; color: #888; display: flex; gap: 16px; }
  .status span { display: flex; align-items: center; gap: 4px; }
  .chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; }
  .msg.user { background: #0f3460; align-self: flex-end; }
  .msg.monica { background: #16213e; align-self: flex-start; border: 1px solid #0f3460; }
  .msg.auto { background: #1a1a2e; align-self: flex-start; border: 1px solid #e94560; border-left: 3px solid #e94560; }
  .msg .name { font-size: 11px; color: #888; margin-bottom: 4px; }
  .input-area {
    padding: 16px 20px;
    background: #16213e;
    border-top: 1px solid #0f3460;
    display: flex;
    gap: 12px;
  }
  .input-area input {
    flex: 1;
    padding: 10px 16px;
    border: 1px solid #0f3460;
    border-radius: 8px;
    background: #1a1a2e;
    color: #e0e0e0;
    font-size: 14px;
    outline: none;
  }
  .input-area input:focus { border-color: #e94560; }
  .input-area button {
    padding: 10px 20px;
    background: #e94560;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
  }
  .input-area button:hover { background: #d63851; }
  .thinking { color: #888; font-size: 12px; padding: 8px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>🎀 モニカ</h1>
  <div class="status">
    <span id="mood">💭 -</span>
    <span id="conv">💬 0</span>
    <span id="episodes">📖 0</span>
    <span id="user">👤 -</span>
  </div>
</div>
<div class="chat" id="chat"></div>
<div class="input-area">
  <input type="text" id="input" placeholder="メッセージを入力..." autofocus>
  <button onclick="send()">送信</button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="name">' + (role === 'user' ? 'あなた' : 'モニカ') + '</div>' + text.replace(/\\n/g, '<br>');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('user', text);

  const thinking = document.createElement('div');
  thinking.className = 'thinking';
  thinking.textContent = 'モニカが考えています...';
  chat.appendChild(thinking);

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await r.json();
    thinking.remove();
    addMsg('monica', data.response || '...');
    updateStatus(data.state);
  } catch(e) {
    thinking.remove();
    addMsg('monica', 'ごめんなさい、エラーが起きました。');
  }
}

function updateStatus(state) {
  if (!state) return;
  document.getElementById('mood').textContent = '💭 ' + (state.mood || '-');
  document.getElementById('conv').textContent = '💬 ' + (state.conversations || 0);
}

async function refreshState() {
  try {
    const r = await fetch('/api/state');
    const data = await r.json();
    if (data.mood) document.getElementById('mood').textContent = '💭 ' + data.mood;
    if (data.conversations !== undefined) document.getElementById('conv').textContent = '💬 ' + data.conversations;
    if (data.episodes !== undefined) document.getElementById('episodes').textContent = '📖 ' + data.episodes;
    if (data.user_name) document.getElementById('user').textContent = '👤 ' + data.user_name;
  } catch(e) {}
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
setInterval(refreshState, 5000);
refreshState();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── 起動 ──

def main():
    global monica

    import argparse
    parser = argparse.ArgumentParser(description="🎀 モニカ Web サーバー")
    parser.add_argument("--model", default="qwen3.6:35b", help="Ollamaモデル名")
    parser.add_argument("--port", type=int, default=8000, help="ポート番号")
    parser.add_argument("--data-dir", default="monica/data", help="データ保存先")
    args = parser.parse_args()

    # モニカ初期化
    print(f"🎀 モニカを起動しています...")
    monica = Monica(model=args.model, data_dir=args.data_dir)
    print(f"   モデル: {monica.brain.config.model}")
    print(f"   Ollama: {'OK' if monica.brain.is_available() else 'NG'}")
    print(f"   会話数: {monica.conversation_count}")
    print()

    # 自律ループ開始
    thread = threading.Thread(target=autonomous_loop, daemon=True)
    thread.start()
    print(f"💭 自律ループ開始")

    # Webサーバー起動
    print(f"🌐 http://localhost:{args.port} で待受中")
    print()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
