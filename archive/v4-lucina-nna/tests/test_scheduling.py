"""発話スケジューリング・内言ループのテスト（仕様書 v1.4 §0 自律思考ループ・v1.7・M2）。

- _should_start_speaking / _should_stop_speaking: Drive閾値による自発的な発話遷移判定（純関数）。
- 自律サイクル: run() の scheduled モードで speech_start / speech_end / inner_thought が
  autonomy ログに記録される（M2「自発的行動選択をログで確認できる」）。
- 内言の分離: 内言は発話・relief・記憶に影響しない（internal フラグ）。
- 外部刺激（interrupt）が発話トリガーになる。
"""

from __future__ import annotations

import asyncio
import json

from lucina.core import _should_start_speaking, _should_stop_speaking  # noqa: SLF001
from lucina.memory.working_buffer import WorkingBuffer
from lucina.testing import build_mock_core, make_test_config


def _cfg(**overrides: float) -> dict:
    base = {
        "speak_start_boredom": 0.6,
        "speak_start_loneliness": 0.5,
        "speak_block_fatigue": 0.8,
        "speak_override_boredom": 0.95,
        "quiet_on_fatigue": 0.85,
        "max_speech_segments": 4,
    }
    base.update(overrides)
    return base


def _events(path) -> list[str]:
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# 発話遷移判定（純関数）
# --------------------------------------------------------------------------- #
def test_should_start_speaking_by_boredom() -> None:
    assert _should_start_speaking({"boredom": 0.61, "loneliness": 0.0, "fatigue": 0.0}, _cfg())
    assert not _should_start_speaking({"boredom": 0.5, "loneliness": 0.0, "fatigue": 0.0}, _cfg())


def test_should_start_speaking_by_loneliness() -> None:
    assert _should_start_speaking({"boredom": 0.0, "loneliness": 0.55, "fatigue": 0.0}, _cfg())


def test_should_start_speaking_blocked_by_fatigue_but_override() -> None:
    # 疲労で発話は抑止される
    assert not _should_start_speaking({"boredom": 0.8, "loneliness": 0.0, "fatigue": 0.9}, _cfg())
    # 我慢の限界（speak_override_boredom）なら疲労でも発話する（永久沈黙の防止）
    assert _should_start_speaking({"boredom": 0.96, "loneliness": 0.0, "fatigue": 0.9}, _cfg())


def test_should_stop_speaking_by_fatigue_and_limit() -> None:
    assert _should_stop_speaking({"fatigue": 0.9}, _cfg(), segments_done=1) == "疲労による沈黙"
    assert _should_stop_speaking({"fatigue": 0.3}, _cfg(), segments_done=1) is None
    assert _should_stop_speaking({"fatigue": 0.0}, _cfg(), segments_done=4) == "発話セグメント上限"


# --------------------------------------------------------------------------- #
# WorkingBuffer の内言分離
# --------------------------------------------------------------------------- #
def test_working_buffer_internal_flag() -> None:
    buf = WorkingBuffer()
    buf.append("こんにちは", n_tokens=2)
    buf.append("[内言] 考え中", n_tokens=3, internal=True)
    assert buf.content() == "こんにちは[内言] 考え中"          # 全文脈（モデルには見える）
    assert buf.spoken_content() == "こんにちは"                 # 発話表示からは内言を除外
    assert buf.items == ["こんにちは", "[内言] 考え中"]


# --------------------------------------------------------------------------- #
# 自律サイクル（モック）
# --------------------------------------------------------------------------- #
async def test_scheduled_run_autonomous_speech_cycle(tmp_path) -> None:
    """scheduled モードで「思考→自発的に発話→沈黙」のサイクルが autonomy ログに記録される。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.mode": "thinking",
        "drive.scheduling.inner_interval_sec": 0.0,
        "drive.scheduling.inner_max_tokens": 3,
        "drive.scheduling.speak_start_boredom": 0.0,   # 直ちに発話開始（動機条件を常時満たす）
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_block_fatigue": 1.0,
        "drive.scheduling.quiet_on_fatigue": 1.0,
        "drive.scheduling.max_speech_segments": 1,     # 1セグメント話したら沈黙
        "drive.relief.segment.max_tokens": 8,          # モックは文末記号を出さないため強制区切り
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.seed_prompt("起動")
        await core.run(max_tokens=60, drive_loop=False)
    finally:
        core.close()

    events = _events(tmp_path / "autonomy.jsonl")
    assert "speech_start" in events
    assert "speech_end" in events
    assert "inner_thought" in events
    assert events.count("speech_start") >= 2  # 複数回の自発サイクルが起きている
    assert events.count("speech_end") >= 2


async def test_inner_thought_not_speech_not_relief(tmp_path) -> None:
    """内言は発話（spoken）として扱われず、relief・記憶コミットに影響しない。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.inner_max_tokens": 6,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["loneliness"] = 0.9  # バイアスで内言に loneliness 語彙が出やすくする
        await core._generate_inner_thought()  # noqa: SLF001
        assert core.thoughts_generated > 0
        assert core.segment.texts == []       # セグメントには入らない
        assert core.relief.pending == {}      # relief は発火しない
        assert core.buffer.spoken_content() == ""   # 発話としては扱われない
        assert core.buffer.content() != ""          # ただしモデルの文脈には残る
    finally:
        core.close()


async def test_interrupt_triggers_speech_in_scheduled_mode(tmp_path) -> None:
    """外部刺激（interrupt）が Drive 閾値より先に発話トリガーになる。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.inner_interval_sec": 100.0,  # 内言なし（時間待ちさせない）
        "drive.scheduling.speak_start_boredom": 1.0,   # Drive閾値では発話しない
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_block_fatigue": 1.0,
        "drive.relief.segment.max_tokens": 4,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=False))
        await asyncio.sleep(0.1)  # run() が bind() を実行するのを待つ（C1）
        core.interrupts.inject("外部刺激テスト")
        await asyncio.sleep(0.5)  # 発話サイクルが進むのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path / "autonomy.jsonl")
        assert "speech_start" in events
        assert "speech_end" in events
    finally:
        core.close()
