"""メモリ健全性のスモークテスト（仕様書 §6タスク8の短期版）。

本番は24時間稼働で「メモリ健全性テストが0件のオーバーフローで通過」を確認する
（v3 §5.3）。本テストはその短期スモーク版として、連続ステップで以下を検証する。
    - WorkingBuffer が context_window を超えない（オーバーフロー0件）
    - Drive値が常に [0,1] にクリップされている（飽和・負値なし）
    - 圧縮イベントが構造化ログに記録されている
"""

from __future__ import annotations

import asyncio

import pytest

from lucina.testing import build_mock_core, make_test_config


def test_memory_health_short_run(tmp_path) -> None:
    """連続ステップでバッファのオーバーフローが発生しない（タスク8の短期スモーク）。"""
    config = make_test_config(context_window=64, log_dir=str(tmp_path))
    core = build_mock_core(config, log_dir=str(tmp_path))
    window = config["model"]["context_window"]

    async def go() -> None:
        for _ in range(2000):
            await core.step_once()
            assert core.buffer.token_count <= window, "WorkingBuffer がオーバーフローした"
            assert all(0.0 <= v <= 1.0 for v in core.drives.values())

    asyncio.run(go())
    assert core.tokens_generated == 2000
    core.close()

    comp_log = tmp_path / "compression.jsonl"
    assert comp_log.exists()
    lines = comp_log.read_text(encoding="utf-8").splitlines()
    assert any("removed_tokens" in line for line in lines)  # 圧縮が実際に発火している


@pytest.mark.asyncio
async def test_drive_loop_stability_short_run(tmp_path) -> None:
    """Driveループを有効にした連続稼働でも、Drive値が常に [0,1] に収まる。"""
    config = make_test_config(context_window=64, log_dir=str(tmp_path))
    core = build_mock_core(config, log_dir=str(tmp_path))

    await core.run(max_tokens=500, drive_loop=True)

    assert core.tokens_generated == 500
    assert all(0.0 <= v <= 1.0 for v in core.drives.values())
    core.close()
