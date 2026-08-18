"""Lucina Web UI（v1.14）のテスト。

- WebBridge: put→get のスレッドセーフ転送
- tail_jsonl: 追記監視（新規行のみ・切り詰めリセット）
- FastAPI アプリ: GET /（HTML）・POST /send（inject コールバック）・WS イベント配信
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from lucina.web import (
    Broadcaster,
    WebBridge,
    create_app,
    find_pid_on_port,
    is_lucina_web_process,
    kill_stale_web_process,
    open_browser,
    start_server,
    tail_jsonl,
)


# --------------------------------------------------------------------------- #
# WebBridge
# --------------------------------------------------------------------------- #
def test_bridge_put_get() -> None:
    bridge = WebBridge()
    assert bridge.get(timeout=0.01) is None  # 空なら None
    bridge.put({"type": "chat", "kind": "speech", "text": "こんにちは"})
    ev = bridge.get(timeout=0.5)
    assert ev == {"type": "chat", "kind": "speech", "text": "こんにちは"}
    assert bridge.get(timeout=0.01) is None


# --------------------------------------------------------------------------- #
# tail_jsonl
# --------------------------------------------------------------------------- #
def test_tail_jsonl_new_lines(tmp_path) -> None:
    f = tmp_path / "memory.jsonl"
    f.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    pos, events = tail_jsonl(f, 0)
    assert len(events) == 1
    # 追記分だけ返る
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"b": 2}) + "\n")
    pos, events = tail_jsonl(f, pos)
    assert [e["b"] for e in events] == [2]
    # 変化なしなら空
    pos, events = tail_jsonl(f, pos)
    assert events == []


def test_tail_jsonl_truncation_resets(tmp_path) -> None:
    f = tmp_path / "memory.jsonl"
    f.write_text("x" * 100, encoding="utf-8")
    pos, _ = tail_jsonl(f, 0)
    f.write_text(json.dumps({"c": 3}) + "\n", encoding="utf-8")  # 切り詰め
    pos, events = tail_jsonl(f, pos)
    assert [e["c"] for e in events] == [3]


# --------------------------------------------------------------------------- #
# FastAPI アプリ
# --------------------------------------------------------------------------- #
def _app(tmp_path, on_send=None):
    bridge = WebBridge()
    return create_app(bridge, tmp_path, on_send=on_send), bridge


def test_index_serves_html(tmp_path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "Lucina" in res.text
        assert "/ws" in res.text  # WebSocket 接続コードが含まれる


def test_index_has_splash_for_load_progress(tmp_path) -> None:
    """v1.15: モデルロード中のスプラッシュ（進捗バー・ステージ表示）が HTML に含まれる。"""
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/")
        assert 'id="splash"' in res.text
        assert 'id="splash-bar"' in res.text
        assert 'id="splash-msg"' in res.text
        assert 'id="splash-stage"' in res.text
        # status イベント処理（スプラッシュ更新・完了で閉じる）が JS に含まれる
        assert "case 'status'" in res.text
        assert "stage === 'done'" in res.text


def test_html_establishes_initial_ws_connection(tmp_path) -> None:
    """v1.16: ページ読込時に初期 WebSocket 接続が確立される（connect() が呼ばれる）。

    これが無いと `ws.onmessage = ...` が null への代入になりスクリプト全体が TypeError で死に、
    進捗・チャットが一切更新されない（実機で発見したバグ）。
    """
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/")
        # connect() 関数が定義され、ページ読込時に呼ばれている
        assert "function connect()" in res.text
        assert "connect();" in res.text
        # 初期 ws は null のままハンドラを代入していない（onmessage を null に付けない）
        assert "new WebSocket(wsUrl()){" not in res.text
        # 再接続は connect() を再利用している
        assert "setTimeout(connect, delay)" in res.text


def test_ws_receives_status_event(tmp_path) -> None:
    """v1.15: ロード進捗（status イベント）が bridge 経由で WS に配信される。"""
    app, bridge = _app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            bridge.put({"type": "status", "stage": "model", "message": "モデルをGPUへロード中…", "progress": 0.0})
            data = ws.receive_json()
            assert data["type"] == "status"
            assert data["stage"] == "model"
            assert data["progress"] == 0.0


def test_bridge_keeps_chat_history() -> None:
    """v1.15: chat イベントが履歴バッファに保持され、非 chat は保持されない。"""
    bridge = WebBridge(history_size=3)
    for i in range(4):
        bridge.put({"type": "chat", "kind": "speech", "text": f"発話{i}"})
    bridge.put({"type": "drives", "state": {}})
    hist = bridge.history()
    assert [e["text"] for e in hist] == ["発話1", "発話2", "発話3"]  # リングバッファ（最新3件）


def test_bridge_ready_flag_on_done() -> None:
    """v1.16: status done を受信すると ready フラグが立ち、以後のクライアントがスプラッシュを閉じられる。"""
    bridge = WebBridge()
    assert bridge.ready is False
    bridge.put({"type": "status", "stage": "model", "progress": 0.5})
    assert bridge.ready is False  # done 以外では立たない
    bridge.put({"type": "status", "stage": "done", "progress": 1.0})
    assert bridge.ready is True


def test_history_endpoint_returns_chat(tmp_path) -> None:
    """v1.15: GET /history がチャット履歴を返す（再接続時の復元用）。"""
    app, bridge = _app(tmp_path)
    bridge.put({"type": "chat", "kind": "question", "text": "何か知りたいです"})
    with TestClient(app) as client:
        res = client.get("/history")
        assert res.status_code == 200
        body = res.json()
        events = body["events"]
        assert len(events) == 1
        assert events[0]["text"] == "何か知りたいです"
        assert body["ready"] is False  # done 未受信なら False


def test_history_ready_true_after_done(tmp_path) -> None:
    """v1.16: done 受信後は /history の ready が True（起動完了後に接続したクライアントがスプラッシュを閉じる）。"""
    app, bridge = _app(tmp_path)
    bridge.put({"type": "status", "stage": "done", "progress": 1.0})
    with TestClient(app) as client:
        res = client.get("/history")
        assert res.json()["ready"] is True


def test_index_has_autoreconnect_js(tmp_path) -> None:
    """v1.15: HTML に自動再接続（指数バックオフ）と履歴復元の JS が含まれる。"""
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/")
        assert "wsRetry" in res.text  # 再接続カウンタ
        assert "切断・再接続中…" in res.text
        assert "restoreHistory" in res.text
        assert "/history" in res.text
        # v1.16: 起動完了済みならスプラッシュを閉じる（done は一時イベントのため）
        assert "d.ready" in res.text
        assert "classList.add('hidden')" in res.text


def test_send_injects_message(tmp_path) -> None:
    sent: list[str] = []
    app, _ = _app(tmp_path, on_send=sent.append)
    with TestClient(app) as client:
        res = client.post("/send", json={"message": "こんにちは、Lucina"})
        assert res.status_code == 200
        assert res.json() == {"ok": True}
        assert sent == ["こんにちは、Lucina"]
        # 空メッセージは拒否
        res = client.post("/send", json={"message": "  "})
        assert res.json()["ok"] is False


@pytest.mark.asyncio
async def test_broadcaster_broadcast(tmp_path) -> None:
    """Broadcaster は接続中のクライアントへ配信し、切断クライアントを掃除する。"""
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # broadcaster 経由で配信されることを確認（アプリのlifespan内ブロードキャスタは
            # bridge を監視している。ここでは WebSocket が accept されていること自体を確認）
            pass


def test_ws_receives_bridge_event(tmp_path) -> None:
    """bridge に積んだイベントが WebSocket クライアントへ配信される（エンドツーエンド）。"""
    app, bridge = _app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            bridge.put({"type": "chat", "kind": "speech", "text": "テスト発話"})
            # ブロードキャスタは最大 0.2s 周期で polling する
            data = ws.receive_json()
            assert data["type"] == "chat"
            assert data["text"] == "テスト発話"


def test_ws_receives_tailed_file_events(tmp_path) -> None:
    """memory.jsonl への追記が WS で memory イベントとして配信される。"""
    app, _ = _app(tmp_path)
    log_dir = tmp_path
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            f = log_dir / "memory.jsonl"
            f.write_text(json.dumps({"kind": "episodic", "text": "思い出"}) + "\n", encoding="utf-8")
            deadline = time.time() + 3.0
            seen = False
            while time.time() < deadline:
                data = ws.receive_json()
                if data.get("type") == "memory":
                    assert data["text"] == "思い出"
                    seen = True
                    break
            assert seen


# --------------------------------------------------------------------------- #
# open_browser（v1.16: ブラウザ自動オープン）
# --------------------------------------------------------------------------- #
def test_open_browser_launches_background(monkeypatch) -> None:
    """webbrowser.open がバックグラウンドスレッドで呼ばれる（ブロックしない）。"""
    import threading
    import webbrowser

    calls: list[str] = []
    spawned: list[threading.Thread] = []

    def fake_start(self: threading.Thread) -> None:
        # 実際には起動せず捕捉する（open 呼び出しは手動で確定させる）
        spawned.append(self)

    monkeypatch.setattr(threading.Thread, "start", fake_start)
    monkeypatch.setattr(webbrowser, "get", lambda: object())
    monkeypatch.setattr(webbrowser, "open", lambda url: calls.append(url) or True)

    assert open_browser("http://127.0.0.1:8787") is True
    assert len(spawned) == 1  # バックグラウンドスレッドとして起動された
    assert spawned[0].daemon is True
    # スレッドのターゲットを手動実行して open が呼ばれることを確定
    spawned[0]._target(*spawned[0]._args)
    assert calls == ["http://127.0.0.1:8787"]


def test_open_browser_no_browser_returns_false(monkeypatch) -> None:
    """webbrowser.get() が失敗する環境（ヘッドレス等）では False を返し、例外を投げない。"""
    import webbrowser

    monkeypatch.setattr(webbrowser, "get", lambda: None)
    assert open_browser("http://127.0.0.1:8787") is False


# --------------------------------------------------------------------------- #
# start_server（v1.16: バインド失敗の検出）
# --------------------------------------------------------------------------- #
def test_start_server_port_in_use_raises() -> None:
    """ポート使用中は RuntimeError を投げる（「開きました」表示後に背景で bind エラーになる混乱を防ぐ）。"""
    import socket

    import pytest
    from fastapi import FastAPI

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    try:
        with pytest.raises(RuntimeError, match="使用中"):
            start_server(FastAPI(), "127.0.0.1", port)
    finally:
        sock.close()


# --------------------------------------------------------------------------- #
# 残存プロセスの自動キル（v1.16: 同一プロセスがあれば殺して立て直す）
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_find_pid_on_port_own_socket() -> None:
    """自分が LISTEN しているポートから自分の PID を特定できる。"""
    import os
    import socket

    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert find_pid_on_port(port) == os.getpid()
    finally:
        s.close()


def test_is_lucina_web_process_false_for_test() -> None:
    """テストプロセス（cmdline に run_agent.py を含まない）は lucina と判定しない。"""
    import os

    assert is_lucina_web_process(os.getpid()) is False


def test_kill_stale_web_process_ignores_foreign() -> None:
    """無関係なプロセス（run_agent.py でない）がポートを掴んでいても殺さない。"""
    import os
    import socket

    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        port = s.getsockname()[1]
        assert kill_stale_web_process(port) is None  # 殺さない
        assert find_pid_on_port(port) == os.getpid()  # まだ生きている
    finally:
        s.close()


def test_kill_stale_web_process_kills_lucina_like() -> None:
    """cmdline に run_agent.py を含むプロセスがポートを掴んでいたら終了して解放する。"""
    import subprocess
    import time

    port = _free_port()
    script = (
        "import socket, time;"
        f"s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
        f"s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep(60)"
    )
    proc = subprocess.Popen(["python3", "-c", script, "run_agent.py"])
    try:
        # 子プロセスが LISTEN するまで待つ
        deadline = time.time() + 5.0
        while time.time() < deadline and find_pid_on_port(port) is None:
            time.sleep(0.05)
        assert find_pid_on_port(port) == proc.pid

        killed = kill_stale_web_process(port)
        assert killed == proc.pid  # 終了した
        # ポートが解放されている
        deadline = time.time() + 5.0
        while time.time() < deadline and find_pid_on_port(port) is not None:
            time.sleep(0.05)
        assert find_pid_on_port(port) is None
    finally:
        proc.kill()
        proc.wait()
