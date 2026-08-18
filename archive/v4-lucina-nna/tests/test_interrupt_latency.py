"""割り込み反映レイテンシの校正実験（仕様書 v1.4 §7 test_interrupt_latency.py の骨格を実装）。

v3 §5.2 の相対式: 閾値 = 平均トークン生成レイテンシ * interrupt_latency_multiplier(1.5)。
後半のテスト群は C1 レース条件（外部スレッドからの最初の inject がキュー未初期化で失敗する問題）の
回帰テスト。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from calibrate_thresholds import measure_avg_token_latency, wait_until_buffer_contains
from lucina.io.interrupts import InterruptChannel


@pytest.mark.asyncio
async def test_interrupt_reflection_latency(slow_core_fixture):
    core = slow_core_fixture
    baseline_latency_ms = await measure_avg_token_latency(core, n=20)
    threshold_ms = baseline_latency_ms * 1.5  # v3 §5.2の相対式

    injected_at = time.monotonic()
    core.interrupts.inject("テスト割り込み")
    reflected_at = await wait_until_buffer_contains(core, "テスト割り込み", timeout_ms=threshold_ms * 3)

    actual_latency_ms = (reflected_at - injected_at) * 1000
    assert actual_latency_ms <= threshold_ms


@pytest.mark.asyncio
async def test_interrupt_reflected_into_buffer(slow_core_fixture):
    """割り込みがバッファ（次ステップのコンテキスト）に反映されることの確認。"""
    core = slow_core_fixture
    core.interrupts.inject("外部刺激テスト")
    for _ in range(5):
        await core.step_once()
        if core.buffer.contains("外部刺激テスト"):
            break
    assert core.buffer.contains("外部刺激テスト")


@pytest.mark.asyncio
async def test_multiple_interrupts_fifo(lucina_core_fixture):
    """複数割り込みが FIFO で drain される。

    C1: inject は loop.call_soon_threadsafe でスケジュールされるため、
    ループが1回処理してから drain する（任意スレッドからの注入を想定した設計）。
    """

    core = lucina_core_fixture
    for msg in ("一つ目", "二つ目", "三つ目"):
        core.interrupts.inject(msg)
    await asyncio.sleep(0)  # スケジュール済みコールバックをループに処理させる
    drained = core.interrupts.drain()
    assert drained == ["一つ目", "二つ目", "三つ目"]


# --------------------------------------------------------------------------- #
# C1 レース条件の回帰テスト
# 従来実装ではキュー初期化が inject() 側にしかなく、外部スレッドが最初の inject を
# 呼ぶと get_running_loop() がループ外で実行され RuntimeError になっていた。
# bind()（core.run 起動直後に呼ぶ）でこの順序保証をコード上に固定する。
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_external_thread_inject_after_bind():
    """bind() 後は外部スレッドからの inject が失敗せず FIFO で drain できる。"""
    channel = InterruptChannel()
    channel.bind()
    errors: list[str] = []

    def worker() -> None:
        try:
            for msg in ("外部1", "外部2"):
                channel.inject(msg)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    await asyncio.sleep(0)  # call_soon_threadsafe コールバックをループに処理させる
    assert channel.drain() == ["外部1", "外部2"]


def test_bind_outside_loop_raises_clear_error():
    """ループ外で bind() を呼ぶと明確なエラーになる（キューは必ずループのスレッドで生成）。"""
    channel = InterruptChannel()
    with pytest.raises(RuntimeError, match="イベントループ"):
        channel.bind()


def test_inject_before_bind_from_external_thread_raises():
    """契約: 外部スレッドからの最初の inject は core.run()/bind() の後に限る。"""
    channel = InterruptChannel()
    errors: list[str] = []

    def worker() -> None:
        try:
            channel.inject("x")
        except RuntimeError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert len(errors) == 1
    assert "イベントループ" in errors[0]


@pytest.mark.asyncio
async def test_drain_inside_loop_binds_channel():
    """ループ内から drain() を呼ぶと初期化され、その後の外部スレッド inject が成功する。"""
    channel = InterruptChannel()
    assert channel.drain() == []  # ループ内: 初期化される
    errors: list[str] = []

    def worker() -> None:
        try:
            channel.inject("外部注入")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    await asyncio.sleep(0)
    assert channel.drain() == ["外部注入"]


def test_drain_outside_loop_returns_empty():
    """ループ外（未起動）からの drain() は従来通り安全に空リストを返す。"""
    channel = InterruptChannel()
    assert channel.drain() == []


@pytest.mark.asyncio
async def test_run_binds_channel_before_external_thread_inject(lucina_core_fixture):
    """統合回帰: run() 起動後は外部スレッドの最初の inject が成功し、バッファに反映される。"""
    core = lucina_core_fixture
    errors: list[str] = []

    def external_inject() -> None:
        try:
            core.interrupts.inject("外部スレッドからの割り込み")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    run_task = asyncio.create_task(core.run(max_tokens=1_000_000, drive_loop=False))
    await asyncio.sleep(0)  # run() 冒頭の bind() を完了させる
    t = threading.Thread(target=external_inject)
    t.start()
    t.join()
    assert errors == []  # 外部スレッドの最初の inject が失敗しない（レース条件の回帰）

    for _ in range(50):  # バッファ反映を待つ
        await asyncio.sleep(0.01)
        if core.buffer.contains("外部スレッドからの割り込み"):
            break
    core.stop()
    await run_task
    assert core.buffer.contains("外部スレッドからの割り込み")
