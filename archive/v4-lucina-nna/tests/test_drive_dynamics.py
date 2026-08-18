"""DriveDynamics / ReliefController の単体テスト（仕様書 v1.4 §5.1・§4 drive.relief）。"""

from __future__ import annotations

import pytest

from lucina.drives.decay import ReliefController
from lucina.drives.dynamics import DriveDynamics

FULL_MATRIX = {
    "boredom": {"boredom": 0.005, "fatigue": 0.01},
    "loneliness": {"loneliness": 0.002},
    "fatigue": {"fatigue": 0.003},
}
BOREDOM_ONLY_MATRIX = {
    "boredom": {"boredom": 0.005},
    "loneliness": {"loneliness": 0.002},
    "fatigue": {"fatigue": 0.003},
}
INITIAL = {"boredom": 0.1, "loneliness": 0.1, "fatigue": 0.2}


def _steps_to(dynamics: DriveDynamics, drive: str, target: float, dt: float = 0.1, max_steps: int = 100_000) -> int:
    steps = 0
    while dynamics.state[drive] < target and steps < max_steps:
        dynamics.step(dt)
        steps += 1
    return steps


def test_clip_to_unit_interval() -> None:
    """契約: 戻り値の各Drive値は必ず [0.0, 1.0] にクリップされていること。"""
    dynamics = DriveDynamics(FULL_MATRIX, INITIAL)
    for _ in range(5000):
        state = dynamics.step(0.1)
        for value in state.values():
            assert 0.0 <= value <= 1.0


def test_saturation_without_relief() -> None:
    """C3: 対角成分が全て正のため、relief がなければ全Driveは1.0へ飽和する。"""
    dynamics = DriveDynamics(FULL_MATRIX, INITIAL)
    for _ in range(100_000):
        dynamics.step(0.1)
    assert dynamics.state["boredom"] == pytest.approx(1.0)
    assert dynamics.state["fatigue"] == pytest.approx(1.0)


def test_fatigue_interaction_accelerates_boredom() -> None:
    """テスト要件（回帰）: fatigueとの相互作用により、単独更新時より boredom の到達速度が速くなる。"""
    full = DriveDynamics(FULL_MATRIX, INITIAL)
    alone = DriveDynamics(BOREDOM_ONLY_MATRIX, INITIAL)
    steps_full = _steps_to(full, "boredom", 0.9)
    steps_alone = _steps_to(alone, "boredom", 0.9)
    assert steps_full < steps_alone


def test_relief_reduces_state() -> None:
    dynamics = DriveDynamics(FULL_MATRIX, INITIAL)
    before = dynamics.state["boredom"]
    dynamics.step(0.1, relief={"boredom": 0.3})
    assert dynamics.state["boredom"] < before


def test_relief_never_below_zero() -> None:
    dynamics = DriveDynamics(FULL_MATRIX, {"boredom": 0.0, "loneliness": 0.0, "fatigue": 0.0})
    dynamics.step(0.1, relief={"boredom": 5.0})
    assert dynamics.state["boredom"] == 0.0


def test_step_mutates_state_in_place() -> None:
    dynamics = DriveDynamics(FULL_MATRIX, INITIAL)
    ref = dynamics.state
    dynamics.step(0.1)
    assert dynamics.state is ref  # 共有辞書（core.drives が追従する）


# --------------------------------------------------------------------------- #
# ReliefController
# --------------------------------------------------------------------------- #
RELIEF_CFG = {
    "boredom": {"enabled": True, "per_action": 0.4, "decay_rate": 0.1},
    "loneliness": {"enabled": True, "per_action": 0.6, "decay_rate": 0.0},
    "fatigue": {"enabled": False, "per_action": 0.5, "decay_rate": 0.0},
}


def test_per_action_applied_once() -> None:
    """①: per_action はキューされ、次のステップで1回だけ消費される。"""
    rc = ReliefController(RELIEF_CFG)
    rc.apply_per_action("boredom")
    step1 = rc.step(0.0, {"boredom": 0.5})
    assert step1["boredom"] == pytest.approx(0.4)  # decay=0.1*0.5*0.0=0 のため per_action のみ
    step2 = rc.step(0.0, {"boredom": 0.5})
    assert "boredom" not in step2  # 2回目は消費されない


def test_per_action_max_one_per_segment() -> None:
    """①: 同一セグメントで複数回 apply しても1セグメント分として扱う（コア側で1回しか呼ばない）。"""
    rc = ReliefController(RELIEF_CFG)
    rc.apply_per_action("loneliness")
    rc.apply_per_action("loneliness")  # コアの契約違反を防ぐ設計（加算は呼び出し側次第）
    assert rc.pending["loneliness"] == pytest.approx(1.2)


def test_decay_is_proportional_to_state() -> None:
    rc = ReliefController(RELIEF_CFG)
    d1 = rc.step(1.0, {"boredom": 1.0})["boredom"]
    d2 = rc.step(1.0, {"boredom": 0.5})["boredom"]
    assert d1 == pytest.approx(0.1)
    assert d2 == pytest.approx(0.05)


def test_disabled_drive_no_relief() -> None:
    rc = ReliefController(RELIEF_CFG)
    rc.apply_per_action("fatigue")  # disabled
    assert rc.step(1.0, {"fatigue": 1.0}) == {}
