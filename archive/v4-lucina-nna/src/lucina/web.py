"""Lucina Web UI（v1.14）— ブラウザでチャット＋Drive・記憶・自律イベントのダッシュボード。

アーキテクチャ:
- `run_agent.py --web` で起動。core のイベントループとは**別スレッド**で uvicorn サーバーを動かす。
- イベントループ側（run_agent の `_web_bridge_loop`）が OutputChannel のイベント（発話・質問）と
  Drive スナップショットを `WebBridge`（スレッドセーフな queue.Queue）に転送する。
- サーバー側のブロードキャスタが bridge のイベント＋ `reports/{memory,autonomy}.jsonl` の追記
  （記憶コミット・自律イベント）を WebSocket で全クライアントへ配信する。
- 人間の応答は `POST /send` で受けて `InterruptChannel.inject()` に渡す（スレッドセーフ・C1）。

実行例:
    PYTHONPATH=src:scripts python3 scripts/run_agent.py --config config/demo_interact.yaml \\
        --web --executor --scheduled
    # ブラウザで http://127.0.0.1:8787 を開く
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --------------------------------------------------------------------------- #
# bridge（core のループ → サーバースレッド のスレッドセーフ転送）
# --------------------------------------------------------------------------- #
class WebBridge:
    """core のイベントループが put し、サーバーのブロードキャスタが get するスレッドセーフなキュー。

    v1.15: チャット履歴（chat イベント）をリングバッファに保持し、再接続時に復元できるようにする。
    """

    def __init__(self, history_size: int = 200) -> None:
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._history: list[dict[str, Any]] = []
        self._history_size = int(history_size)
        self._lock = threading.Lock()
        self._ready = False  # v1.16: 起動完了（status done）済みフラグ。done は一時イベントのため、
                             #   後から接続したクライアントにもスプラッシュを閉じさせるのに使う

    def put(self, event: dict[str, Any]) -> None:
        self._q.put(event)
        if event.get("type") == "status" and event.get("stage") == "done":
            self._ready = True
        if event.get("type") == "chat":
            with self._lock:
                self._history.append(event)
                if len(self._history) > self._history_size:
                    del self._history[: len(self._history) - self._history_size]

    @property
    def ready(self) -> bool:
        return self._ready

    def get(self, timeout: float = 0.2) -> dict[str, Any] | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def history(self) -> list[dict[str, Any]]:
        """再接続クライアントへ渡すチャット履歴（最新順ではない・時系列のまま）。"""
        with self._lock:
            return list(self._history)


def tail_jsonl(path: Path, last_pos: int) -> tuple[int, list[dict[str, Any]]]:
    """JSON Lines ファイルの前回位置からの新規行を返す（ファイル追記の監視用）。

    ファイルが切り詰められた（再起動・rm）場合は位置を0に戻す。
    戻り値: (新しい位置, 新規イベントのリスト)
    """
    try:
        size = path.stat().st_size
    except OSError:
        return last_pos, []
    if size < last_pos:
        last_pos = 0
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(last_pos)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        last_pos = fh.tell()
    return last_pos, events


# --------------------------------------------------------------------------- #
# WebSocket ブロードキャスタ
# --------------------------------------------------------------------------- #
class Broadcaster:
    """接続中の全 WebSocket クライアントへイベントを配信する。"""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001 - クライアント切断等
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def __len__(self) -> int:
        return len(self._clients)


async def _broadcast_loop(
    bridge: WebBridge,
    broadcaster: Broadcaster,
    log_dir: Path | str,
    stop: asyncio.Event,
) -> None:
    """bridge のイベント＋ログファイルの追記を全クライアントへ配信する。"""
    positions: dict[str, int] = {}
    # 起動時点のファイル末尾を初期位置にする（過去の履歴を流さない）
    for name in ("memory.jsonl", "autonomy.jsonl"):
        try:
            positions[name] = (Path(log_dir) / name).stat().st_size
        except OSError:
            positions[name] = 0
    while not stop.is_set():
        ev = bridge.get(timeout=0.2)
        if ev is not None:
            await broadcaster.broadcast(ev)
        for name, event_type in (("memory.jsonl", "memory"), ("autonomy.jsonl", "autonomy")):
            path = Path(log_dir) / name
            pos, events = tail_jsonl(path, positions.get(name, 0))
            if pos != positions.get(name):
                positions[name] = pos
                for e in events:
                    await broadcaster.broadcast({"type": event_type, **e})
        await asyncio.sleep(0.05)


# --------------------------------------------------------------------------- #
# FastAPI アプリ
# --------------------------------------------------------------------------- #
def create_app(
    bridge: WebBridge,
    log_dir: Path | str,
    *,
    on_send: Callable[[str], None] | None = None,
) -> FastAPI:
    """Web UI の FastAPI アプリを組み立てる。

    - GET /      : チャット＋ダッシュボードの HTML
    - GET /ws    : イベント配信（WebSocket）
    - POST /send : 人間の応答（{message: str}）→ on_send へ（InterruptChannel.inject を渡す想定）
    """
    broadcaster = Broadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = asyncio.create_task(_broadcast_loop(bridge, broadcaster, log_dir, stop))
        yield
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="Lucina", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _PAGE_HTML

    @app.get("/history")
    async def history() -> dict[str, Any]:
        """v1.15: 再接続クライアント向けにチャット履歴を返す。

        v1.16: `ready`（起動完了済みか）も返す。done は一時イベントのため、起動完了後に
        接続したクライアントがスプラッシュを閉じるために使う。
        """
        return {"events": bridge.history(), "ready": bridge.ready}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await broadcaster.connect(ws)
        try:
            while True:
                await ws.receive_text()  # クライアントからの受信待ち（切断検出用）
        except WebSocketDisconnect:
            broadcaster.disconnect(ws)
        except Exception:  # noqa: BLE001
            broadcaster.disconnect(ws)

    @app.post("/send")
    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        msg = (payload.get("message") or "").strip()
        if not msg:
            return {"ok": False, "error": "メッセージが空です"}
        if on_send is not None:
            on_send(msg)
        return {"ok": True}

    return app


def open_browser(url: str) -> bool:
    """既定ブラウザで URL を開く（v1.16）。

    webbrowser.open は環境によってはブロックし得るため、バックグラウンドスレッドで実行する。
    ヘッドレス環境などでブラウザを開けない場合は False を返す（起動自体は継続する）。
    """
    import webbrowser

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        ok = webbrowser.get()  # 利用可能なブラウザがあるか事前確認
        if ok is None:
            return False
    except Exception:  # noqa: BLE001
        return False
    threading.Thread(target=_open, daemon=True).start()
    return True


def find_pid_on_port(port: int) -> int | None:
    """指定ポートで LISTEN しているプロセスの PID を返す（なければ None）。

    psutil 優先・`ss -tlnp` をフォールバックに使う。
    """
    port = int(port)
    try:
        import psutil

        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == "LISTEN" and conn.laddr is not None and conn.laddr.port == port:
                return conn.pid
    except Exception:  # noqa: BLE001
        pass
    try:
        import re
        import subprocess

        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if f":{port} " in line or f":{port}\t" in line:
                m = re.search(r"pid=(\d+)", line)
                if m:
                    return int(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return None


def is_lucina_web_process(pid: int) -> bool:
    """PID が lucina の run_agent.py プロセスかを判定する（無関係なサービスのキル防止）。"""
    try:
        import psutil

        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline() or [])
        return "run_agent.py" in cmdline
    except Exception:  # noqa: BLE001
        return False


def kill_stale_web_process(port: int) -> int | None:
    """ポートを掴んでいる同一の lucina Web プロセス（run_agent.py）を終了し、解放を待つ。

    戻り値: 終了した PID。掴んでいない・無関係なプロセスの場合は None（殺さない）。
    解放を最大5秒待つが、それでも解放されない場合は None を返す（start_server 側の
    バインド失敗検出に委ねる）。
    """
    import os
    import signal
    import time

    pid = find_pid_on_port(port)
    if pid is None or not is_lucina_web_process(pid):
        return None
    try:
        import psutil

        psutil.Process(pid).kill()
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    # ポート解放を最大5秒待つ
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if find_pid_on_port(port) is None:
            break
        time.sleep(0.05)
    return pid


def start_server(
    app: FastAPI,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> tuple[threading.Thread, Any]:
    """uvicorn サーバーを別スレッドで起動する。戻り値: (スレッド, uvicorn.Server)。

    v1.15: アイドル切断防止のため ws_ping_interval/timeout を延長する（uvicorn 既定の
    20秒/20秒では、ブラウザの省電力やネットワーク揺らぎで長時間開いたままの接続が
    切断されることがある。30秒間隔・120秒タイムアウトに緩める）。

    v1.16: ①起動時に**同一プロセス（run_agent.py）がポートを掴んでいたら自動で終了して
    立て直す**（`kill_stale_web_process`）。Ctrl+Z 等でフリーズした前回実行が残っていても
    起動失敗しなくなる。②バインド失敗（無関係なプロセスが使用中など）を検出して
    `RuntimeError` を投げる。uvicorn は別スレッドで起動するため、バインド失敗はスレッド内で
    ログされるだけで呼び出し側には伝わらない（「開きました」と表示した後に bind エラー、という
    混乱を防ぐ）。
    """
    import time

    import uvicorn

    killed = kill_stale_web_process(port)
    if killed is not None:
        print(f"[lucina] 既存の Web プロセス (PID {killed}) を終了して再起動します", flush=True)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        lifespan="on",
        ws_ping_interval=30.0,
        ws_ping_timeout=120.0,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # バインド完了（server.started）を最大5秒待つ。タイムアウトしたらポート使用中などの失敗。
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not server.started and thread.is_alive():
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(
            f"Web UI サーバーを起動できませんでした: {host}:{port} が使用中か、バインドに失敗しました。"
            f"別のプロセスが起動中でないか確認するか、--port で別ポートを指定してください。"
        )
    return thread, server


# --------------------------------------------------------------------------- #
# フロントエンド（単一HTML・依存なし）
# --------------------------------------------------------------------------- #
_PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Lucina</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Hiragino Kaku Gothic ProN", Meiryo, sans-serif; background:#111; color:#ddd; height:100vh; display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:12px; padding:8px 16px; background:#1a1a1a; border-bottom:1px solid #333; }
  header h1 { font-size:16px; margin:0; color:#7ec8ff; }
  #status { font-size:12px; padding:2px 8px; border-radius:8px; background:#333; }
  #status.on { background:#1e5; color:#000; }
  #mode { font-size:12px; background:#2a2a2a; padding:2px 8px; border-radius:8px; }
  main { flex:1; display:flex; min-height:0; }
  #chat { flex:3; overflow-y:auto; padding:16px; display:flex; flex-direction:column; }
  .msg { margin:6px 0; padding:8px 12px; border-radius:10px; max-width:85%; white-space:pre-wrap; word-break:break-word; font-size:14px; }
  .speech { background:#1e3a5f; align-self:flex-start; }
  .question { background:#4a3b1e; align-self:flex-start; border-left:3px solid #e8b64c; }
  .executor { background:#203020; align-self:flex-start; border-left:3px solid #7ec85e; }
  .executor_result { background:#1a2a1a; align-self:flex-start; border-left:3px solid #4caf50; font-size:12px; color:#a8d8a8; }
  .human_prompt { background:#3a1e1e; align-self:flex-start; border-left:3px solid #e85c5c; }
  .human { background:#3a3a3a; align-self:flex-end; }
  .meta { font-size:11px; color:#888; margin-bottom:2px; }
  #side { flex:2; border-left:1px solid #333; display:flex; flex-direction:column; min-width:280px; max-width:420px; }
  .panel { padding:10px 14px; border-bottom:1px solid #222; }
  .panel h2 { font-size:12px; margin:0 0 8px; color:#888; }
  .gauge { margin:4px 0; font-size:12px; }
  .gauge .bar { background:#222; border-radius:6px; height:10px; overflow:hidden; }
  .gauge .fill { height:100%; border-radius:6px; transition:width .3s; }
  #logs { flex:1; overflow-y:auto; }
  .log { font-size:11px; padding:2px 6px; border-bottom:1px solid #1a1a1a; color:#aaa; white-space:pre-wrap; word-break:break-all; }
  footer { display:flex; padding:10px 16px; background:#1a1a1a; gap:8px; }
  #input { flex:1; background:#222; color:#ddd; border:1px solid #444; border-radius:8px; padding:8px 12px; }
  #send { background:#2a6db5; color:#fff; border:none; border-radius:8px; padding:8px 20px; cursor:pointer; }
  #splash { position:fixed; inset:0; background:#0d0d0d; display:flex; align-items:center; justify-content:center; z-index:100; transition:opacity .4s; }
  #splash.hidden { opacity:0; pointer-events:none; }
  #splash .card { width:min(480px, 80vw); background:#1a1a1a; border:1px solid #333; border-radius:14px; padding:28px 32px; }
  #splash .logo { font-size:24px; color:#7ec8ff; margin-bottom:12px; }
  #splash .msg { font-size:13px; color:#ccc; margin-bottom:12px; }
  #splash .bar { background:#222; border-radius:8px; height:12px; overflow:hidden; }
  #splash .fill { height:100%; width:0%; background:linear-gradient(90deg,#2a6db5,#7ec8ff); border-radius:8px; transition:width .4s; }
  #splash .stage { font-size:11px; color:#888; margin-top:10px; }
</style>
</head>
<body>
<div id="splash">
  <div class="card">
    <div class="logo">Lucina</div>
    <div class="msg" id="splash-msg">モデルをロード中…（初回は数分かかります）</div>
    <div class="bar"><div class="fill" id="splash-bar"></div></div>
    <div class="stage" id="splash-stage"></div>
  </div>
</div>
<header>
  <h1>Lucina</h1>
  <span id="mode">--</span>
  <span id="status">接続中…</span>
</header>
<main>
  <div id="chat"></div>
  <div id="side">
    <div class="panel" id="drives"></div>
    <div class="panel" id="logs">
      <h2>自律イベント・記憶</h2>
    </div>
  </div>
</main>
<footer>
  <input id="input" placeholder="Lucina に話しかける…（Enter で送信）">
  <button id="send">送信</button>
</footer>
<script>
const $ = (id) => document.getElementById(id);
// v1.15: 自動再接続（切断しても指数バックオフで復帰する）
let ws = null;
let wsRetry = 0;
function wsUrl(){ return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws'; }
function esc(s){ return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function addMsg(kind, text, meta){
  const div = document.createElement('div');
  div.className = 'msg ' + kind;
  if (meta) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = meta; div.appendChild(m); }
  div.appendChild(document.createTextNode(text));
  $('chat').appendChild(div);
  $('chat').scrollTop = $('chat').scrollHeight;
}
const GAUGES = ['boredom','loneliness','fatigue','curiosity'];
const LABELS = {boredom:'退屈', loneliness:'寂しさ', fatigue:'疲労', curiosity:'好奇心'};
const COLORS = {boredom:'#e8b64c', loneliness:'#e85c5c', fatigue:'#7ec8ff', curiosity:'#b48ce8'};
function renderDrives(state){
  let html = '<h2>Drive</h2>';
  for (const k of GAUGES){
    const v = Math.max(0, Math.min(100, Math.round((state[k] ?? 0) * 100)));
    html += '<div class="gauge">' + LABELS[k] + ' ' + v + '%<div class="bar"><div class="fill" style="width:' + v + '%;background:' + COLORS[k] + '"></div></div></div>';
  }
  $('drives').innerHTML = html;
}
function addLog(text){
  const div = document.createElement('div');
  div.className = 'log';
  div.textContent = text;
  $('logs').appendChild(div);
  while ($('logs').children.length > 300) $('logs').removeChild($('logs').firstChild);
}
function onMessage(ev){
  let d; try { d = JSON.parse(ev.data); } catch(e){ return; }
  switch (d.type){
    case 'chat': addMsg(d.kind, d.text, {speech:'Lucina', question:'Lucinaの質問', executor:'実行エージェント', executor_result:'実行結果', human_prompt:'応答待ち', human:'あなた'}[d.kind]); break;
    case 'drives': renderDrives(d.state); $('mode').textContent = 'mode: ' + (d.mode || '--'); break;
    case 'memory': addLog('[記憶] ' + (d.event || 'commit') + ' kind=' + (d.kind || '-') + ' ' + esc(d.text || '').slice(0,80)); break;
    case 'autonomy': addLog('[自律] ' + d.event + ' ' + esc(d.reason || '').slice(0,80)); break;
    case 'status':
      if (d.stage === 'done'){
        $('splash').classList.add('hidden');
        $('status').textContent = '起動済み';
      } else {
        if (d.message) $('splash-msg').textContent = d.message;
        if (typeof d.progress === 'number') $('splash-bar').style.width = Math.round(d.progress * 100) + '%';
        if (d.stage) $('splash-stage').textContent = 'stage: ' + d.stage;
      }
      break;
  }
}
function restoreHistory(){
  // v1.15: 再接続時にサーバーが保持するチャット履歴を復元する
  // v1.16: 起動完了済み（ready）ならスプラッシュを閉じる（done は一時イベントのため）
  fetch('/history').then(r => r.json()).then(d => {
    const chat = $('chat');
    chat.innerHTML = '';
    for (const ev of (d.events || [])) addMsg(ev.kind, ev.text, {speech:'Lucina', question:'Lucinaの質問', executor:'実行エージェント', executor_result:'実行結果', human_prompt:'応答待ち', human:'あなた'}[ev.kind]);
    if (d.ready){
      $('splash').classList.add('hidden');
      $('status').textContent = '起動済み';
    }
  }).catch(() => {});
}
function onOpen(){
  wsRetry = 0;
  $('status').textContent = '接続済み';
  $('status').classList.add('on');
  restoreHistory();
}
function onClose(){
  $('status').textContent = '切断・再接続中…';
  $('status').classList.remove('on');
  // 指数バックオフ: 1s→2s→4s→…最大15s
  const delay = Math.min(1000 * Math.pow(2, wsRetry), 15000);
  wsRetry++;
  setTimeout(connect, delay);
}
function connect(){
  ws = new WebSocket(wsUrl());
  ws.onmessage = onMessage;
  ws.onopen = onOpen;
  ws.onclose = onClose;
  ws.onerror = () => { try { ws.close(); } catch(e){} };
}
connect();  // v1.16: 初期接続を必ず確立する（これが無いとページが一切更新されない）
function send(){
  const v = $('input').value.trim();
  if (!v) return;
  fetch('/send', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: v})});
  addMsg('human', v, 'あなた');
  $('input').value = '';
}
$('send').onclick = send;
$('input').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>
"""
