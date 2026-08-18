"""scripts/run_agent.py のユニットテスト（v1.16: ロード進捗モニター）。"""

from __future__ import annotations

import time

from run_agent import spawn_load_progress_monitor


def test_load_progress_monitor_reports_elapsed() -> None:
    """経過時間ベースで model ステージの進捗が定期報告され、0→0.33 の範囲で進む。"""
    events: list[tuple[str, str, float]] = []
    stop = spawn_load_progress_monitor(
        lambda stage, message, progress: events.append((stage, message, progress)),
        expected_sec=1.0,
        interval=0.05,
    )
    time.sleep(0.25)
    stop()
    assert len(events) >= 2  # 複数回報告される
    assert all(stage == "model" for stage, _, _ in events)
    # 進捗は 0 < p <= 0.33 の範囲で単調増加（目安時間を超えても 0.33 を超えない）
    assert all(0.0 < p <= 0.33 for _, _, p in events)
    assert events[-1][2] >= events[0][2]
    # メッセージに経過秒が含まれる
    assert "経過" in events[0][1]


def test_load_progress_monitor_stop_halts() -> None:
    """stop() 後は報告が止まる。"""
    events: list[float] = []
    stop = spawn_load_progress_monitor(
        lambda _s, _m, p: events.append(p),
        expected_sec=0.5,
        interval=0.05,
    )
    time.sleep(0.15)
    stop()
    count_after_stop = len(events)
    time.sleep(0.2)
    assert len(events) == count_after_stop  # stop 後に増えない
