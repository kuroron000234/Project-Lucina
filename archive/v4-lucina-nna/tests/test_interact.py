"""対話モード（--interact）と実行エージェント（v1.13）のテスト。

- OutputChannel: emit→drain の基本動作
- core が発話・質問を output に emit する（_finalize_segment の配線）
- 応答待ち（awaiting）中の inject で「応答→発話」に遷移する（対話ループ）
- ExecutorAdapter: ルーティング（時刻/URL/調査/実行不能）とサンドボックス実行
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lucina.io.executor import ExecutorAdapter
from lucina.io.output import OutputChannel
from lucina.testing import build_mock_core, make_test_config


# --------------------------------------------------------------------------- #
# OutputChannel
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_output_channel_emit_drain() -> None:
    ch = OutputChannel()
    ch.bind()
    ch.emit("speech", "こんにちは")
    ch.emit("question", "今何時ですか")
    assert ch.drain() == [("speech", "こんにちは"), ("question", "今何時ですか")]
    assert ch.drain() == []  # 空になった


# --------------------------------------------------------------------------- #
# core の出力配線
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_core_emits_speech_segment(tmp_path) -> None:
    """通常の発話セグメントは kind=speech で emit される。"""
    cfg = make_test_config(**{"drive.relief.segment.max_tokens": 3})
    core = build_mock_core(cfg, log_dir=tmp_path)
    core.seed_prompt("起動")
    for _ in range(3):
        await core.step_once()
    events = core.output.drain()
    assert any(kind == "speech" for kind, _ in events)
    core.close()


@pytest.mark.asyncio
async def test_core_emits_question(tmp_path) -> None:
    """問いかけセッション（_ask_mode）のセグメントは kind=question で emit される。"""
    cfg = make_test_config(**{"drive.relief.segment.max_tokens": 3})
    core = build_mock_core(cfg, log_dir=tmp_path)
    core.seed_prompt("起動")
    core._ask_mode = True  # noqa: SLF001
    for _ in range(3):
        await core.step_once()
    events = core.output.drain()
    assert any(kind == "question" for kind, _ in events)
    core.close()


@pytest.mark.asyncio
async def test_response_in_awaiting_starts_speaking(tmp_path) -> None:
    """応答待ち中の外部応答で「応答→発話」に遷移する（質問→応答→Lucinaの応答ループ）。"""
    cfg = make_test_config(
        **{
            "drive.relief.segment.max_tokens": 3,
            "drive.scheduling.enabled": True,
            "drive.scheduling.mode": "thinking",
            "drive.scheduling.thinking_mode": "manual",
            "drive.scheduling.inner_interval_sec": 9999,  # タイマー内言を抑止
            "drive.scheduling.introspection_sec": 0.0,
            "drive.scheduling.decide_on_think_end": False,
            "drive.scheduling.decide_on_segment_end": False,
            "drive.scheduling.control_tokens": False,
            "drive.scheduling.idle_boredom_rate": 0.0,
            "drive.scheduling.idle_curiosity_rate": 0.0,
            "drive.scheduling.curiosity_ask_threshold": 0.5,
            "drive.scheduling.await_timeout_sec": 60.0,
            "drive.initial_state.boredom": 0.0,
            "drive.initial_state.loneliness": 0.0,
            "drive.initial_state.fatigue": 0.0,
            "drive.initial_state.curiosity": 0.9,  # 即座に問いかけを発火
        }
    )
    core = build_mock_core(cfg, log_dir=tmp_path)
    core.seed_prompt("起動")
    task = asyncio.create_task(core.run(max_tokens=None, drive_loop=False))
    try:
        await asyncio.sleep(0.2)
        assert core.mode == "awaiting", f"問いかけ→応答待ちになるはず: mode={core.mode}"
        # 質問が emit されている
        assert any(kind == "question" for kind, _ in core.output.drain())

        # モックは即時生成のため応答セッション（最大4セグメント）は瞬時に完了する。
        # 「応答→発話」の遷移は、注入後に発話イベントが emit されることで検証する。
        core.interrupts.inject("答えは42です")
        await asyncio.sleep(0.15)
        events = core.output.drain()
        assert any(kind == "speech" for kind, _ in events), \
            "応答を受けて発話が emit されるはず（応答→Lucinaの応答ループ）"
        assert core.segments_completed > 1, "応答セッションでセグメントが進むはず"
    finally:
        core.stop()
        await task
        core.close()


# --------------------------------------------------------------------------- #
# ExecutorAdapter
# --------------------------------------------------------------------------- #
def test_route_datetime() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    assert ex.route("今何時ですか") == ("sandbox", "date")
    assert ex.route("今の時刻を教えてください") == ("sandbox", "date")


def test_route_url() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    assert ex.route("https://example.com の内容を取得して") == ("sandbox", "https://example.com")


def test_route_opencode() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    assert ex.route("このバグの原因を調べて") == ("opencode", "このバグの原因を調べて")
    assert ex.route("この機能のコードを書いて") == ("opencode", "この機能のコードを書いて")


def test_route_none_for_plain_speech() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    assert ex.route("こんにちは、今日はいい天気ですね") is None
    assert ex.route("") is None


@pytest.mark.asyncio
async def test_sandbox_date_executes() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    result = await ex.run("sandbox", "date")
    assert result is not None and len(result) > 5


@pytest.mark.asyncio
async def test_disabled_executor_returns_none() -> None:
    ex = ExecutorAdapter({"executor": {"enabled": False}})
    assert await ex.run("sandbox", "date") is None
    assert ex.route("今何時ですか") is not None  # ルーティング自体は動作する


def _json_text_event(sid: str, text: str) -> str:
    """Opencode の --format json 出力（textイベント1行）を模する。"""
    return json.dumps({
        "type": "text",
        "sessionID": sid,
        "part": {"type": "text", "text": text},
    }, ensure_ascii=False) + "\n"


@pytest.mark.asyncio
async def test_opencode_model_flag(monkeypatch) -> None:
    """opencode_model 設定時は --model フラグ付きで Opencode CLI を呼び、JSONから回答を抽出する。"""
    ex = ExecutorAdapter({"executor": {"enabled": True, "opencode_model": "opencode/deepseek-v4-flash-free"}})
    captured: dict = {}

    monkeypatch.setattr("lucina.io.executor.shutil.which", lambda cmd: "/usr/bin/opencode")

    class FakeResult:
        stdout = _json_text_event("ses_m1", "調査結果です")
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return FakeResult()

    monkeypatch.setattr("lucina.io.executor.subprocess.run", fake_run)
    result = await ex.run("opencode", "このバグを調べて")
    assert result == "調査結果です"
    assert captured["cmd"] == [
        "opencode", "run", "--format", "json", "--no-replay",
        "--model", "opencode/deepseek-v4-flash-free", "このバグを調べて",
    ]


@pytest.mark.asyncio
async def test_opencode_session_reuse(monkeypatch) -> None:
    """セッション乱立対策: 2回目以降は取得した sessionID で --session を渡す（1プロセス=1セッション）。"""
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    calls: list = []

    monkeypatch.setattr("lucina.io.executor.shutil.which", lambda cmd: "/usr/bin/opencode")

    class FakeResult:
        stdout = _json_text_event("ses_fixed", "ok")
        stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr("lucina.io.executor.subprocess.run", fake_run)
    await ex.run("opencode", "1回目")
    await ex.run("opencode", "2回目")
    assert "--session" not in calls[0]          # 1回目はまだセッションIDがない
    assert "--session" in calls[1]              # 2回目は同じセッションを継続
    assert "ses_fixed" in calls[1]


@pytest.mark.asyncio
async def test_opencode_session_disabled(monkeypatch) -> None:
    """opencode_reuse_session=false なら毎回新セッション（--session を渡さない）。"""
    ex = ExecutorAdapter({"executor": {"enabled": True, "opencode_reuse_session": False}})
    calls: list = []

    monkeypatch.setattr("lucina.io.executor.shutil.which", lambda cmd: "/usr/bin/opencode")

    class FakeResult:
        stdout = _json_text_event("ses_x", "ok")
        stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr("lucina.io.executor.subprocess.run", fake_run)
    await ex.run("opencode", "1回目")
    await ex.run("opencode", "2回目")
    assert all("--session" not in c for c in calls)


@pytest.mark.asyncio
async def test_opencode_close_deletes_session(monkeypatch) -> None:
    """close() で作成したセッションを削除する（セッション乱立の掃除）。"""
    ex = ExecutorAdapter({"executor": {"enabled": True}})
    ex._opencode_session = "ses_cleanup"  # noqa: SLF001
    deleted: list = []

    monkeypatch.setattr("lucina.io.executor.shutil.which", lambda cmd: "/usr/bin/opencode")

    class FakeResult:
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        deleted.append(cmd)
        return FakeResult()

    monkeypatch.setattr("lucina.io.executor.subprocess.run", fake_run)
    ex.close()
    assert deleted == [["opencode", "session", "delete", "ses_cleanup"]]
    assert ex._opencode_session is None


@pytest.mark.asyncio
async def test_url_fetch_failure_is_safe() -> None:
    """URL取得の失敗は例外でなく（実行失敗）… の結果文字列として返る（Lucina に届く）。"""
    ex = ExecutorAdapter({"executor": {"enabled": True, "sandbox_timeout_sec": 5}})
    result = await ex.run("sandbox", "http://127.0.0.1:1/")  # 即時失敗するURL
    assert result is not None and "実行失敗" in result
