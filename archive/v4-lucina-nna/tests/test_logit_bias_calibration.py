"""logit_bias_coefficient 校正ロジックの単体テスト（仕様書 v1.4 §11 次の課題）。

- compute_probability_shift: DriveLogitsProcessor と同一定義の ΔP 計算（純関数）。
- select_coefficient: 最小十分原理の選定ルール（有効域フィルタ・退化除外・最小選択）。
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrate_logit_bias import compute_probability_shift, select_coefficient
from lucina.inference.logits import DriveLogitsProcessor


def _rows(coefs, shifts, conv_rates, reps):
    return [
        {
            "coefficient": c,
            "prob_shift": dp,
            "convergence_rate": r,
            "repetition_ratio": rep,
        }
        for c, dp, r, rep in zip(coefs, shifts, conv_rates, reps)
    ]


# --------------------------------------------------------------------------- #
# compute_probability_shift
# --------------------------------------------------------------------------- #
def test_probability_shift_matches_processor() -> None:
    """ΔP は DriveLogitsProcessor を直接適用した softmax 差と一致する（同一定義の再現）。"""
    rng = np.random.default_rng(1)
    logits = rng.normal(0.0, 1.0, 32)
    vocab = {"loneliness": [[3, 4], [5]], "boredom": [[3]]}  # 先頭トークン3は重複（⑤合算対象）
    c, dv = 2.5, 0.9
    got = compute_probability_shift(logits, vocab, "loneliness", c, dv)

    biased = DriveLogitsProcessor(c).apply(logits, {"loneliness": dv}, vocab)

    def _target_prob(arr: np.ndarray) -> float:
        z = arr - np.max(arr)
        p = np.exp(z)
        p = p / np.sum(p)
        return float(p[[3, 5]].sum())  # loneliness 語彙の先頭トークン集合

    assert got == pytest.approx(_target_prob(biased) - _target_prob(logits))


def test_probability_shift_monotonic_in_coefficient() -> None:
    """ΔP は係数に対して単調非減少（softmax 飽和まで）。"""
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.0, 64)
    vocab = {"loneliness": [[10, 11], [20]]}
    shifts = [
        compute_probability_shift(logits, vocab, "loneliness", c, 0.9)
        for c in [0.0, 1.0, 2.5, 4.0]
    ]
    assert all(b >= a for a, b in zip(shifts, shifts[1:]))
    assert shifts[-1] > shifts[0]


def test_probability_shift_zero_for_empty_vocab() -> None:
    """目的語彙が空なら ΔP = 0（ゼロ除算・IndexError を起こさない）。"""
    logits = np.zeros(16)
    got = compute_probability_shift(logits, {"loneliness": []}, "loneliness", 2.5, 0.9)
    assert got == 0.0


def test_probability_shift_scales_with_drive_value() -> None:
    """同じ係数でも Drive値が大きいほど ΔP が大きい（線形スケーリングの性質）。"""
    rng = np.random.default_rng(2)
    logits = rng.normal(0.0, 1.0, 64)
    vocab = {"loneliness": [[10]]}
    low = compute_probability_shift(logits, vocab, "loneliness", 2.5, 0.3)
    high = compute_probability_shift(logits, vocab, "loneliness", 2.5, 0.9)
    assert high > low


# --------------------------------------------------------------------------- #
# select_coefficient
# --------------------------------------------------------------------------- #
def test_select_coefficient_minimal_sufficient() -> None:
    """最大シフトの80%に達する最小の係数を採用する（最小十分原理）。"""
    rows = _rows(
        coefs=[1.0, 2.0, 3.0, 4.0],
        shifts=[0.02, 0.05, 0.06, 0.06],  # 最大0.06 → ターゲット0.048
        conv_rates=[1.0, 1.0, 1.0, 1.0],
        reps=[0.1, 0.1, 0.2, 0.3],
    )
    chosen, reasoning = select_coefficient(rows, saturation_frac=0.8)
    assert chosen == 2.0  # ΔP 0.05 >= 0.048 を満たす最小
    assert reasoning["chosen_coefficient"] == 2.0
    assert reasoning["effective_count"] == 4


def test_select_coefficient_filters_degenerate() -> None:
    """repetition 過大（退化ループ）はシフトが大きくても除外する。"""
    rows = _rows(
        coefs=[2.0, 3.0],
        shifts=[0.05, 0.05],
        conv_rates=[1.0, 1.0],
        reps=[0.9, 0.2],  # 2.0 は退化 → 除外
    )
    chosen, _ = select_coefficient(rows, max_repetition=0.5)
    assert chosen == 3.0


def test_select_coefficient_requires_convergence() -> None:
    """収束率不足（バイアスが生成を誘導できない）は除外する。"""
    rows = _rows(
        coefs=[1.0, 3.0],
        shifts=[0.05, 0.06],
        conv_rates=[0.4, 1.0],
        reps=[0.1, 0.2],
    )
    chosen, _ = select_coefficient(rows, min_conv_rate=0.8)
    assert chosen == 3.0


def test_select_coefficient_no_effective_returns_none() -> None:
    """有効域が空なら None と理由を返す（config 更新をしない）。"""
    rows = _rows(
        coefs=[1.0, 2.0],
        shifts=[0.0, 0.01],
        conv_rates=[0.0, 0.0],
        reps=[0.1, 0.1],
    )
    chosen, reasoning = select_coefficient(rows)
    assert chosen is None
    assert "有効な係数なし" in reasoning["reason"]


def test_select_coefficient_rejects_negligible_shift() -> None:
    """最大ΔPが min_shift 未満（モデルと語彙の不一致）なら不採用で config 更新しない。"""
    rows = _rows(
        coefs=[1.0, 3.0],
        shifts=[0.01, 0.02],  # 最大でも 0.02 < min_shift(0.05)
        conv_rates=[1.0, 1.0],
        reps=[0.1, 0.1],
    )
    chosen, reasoning = select_coefficient(rows, min_shift=0.05)
    assert chosen is None
    assert "下限" in reasoning["reason"]
    assert reasoning["max_prob_shift"] == 0.02


async def test_run_sweep_mock_integration(tmp_path) -> None:
    """run_sweep の配線（リセット→ΔP→収束→健全性→スイープ）をモックで固定化する。

    select_coefficient 等の純関数テストだけでは、実モデル/モックとの結合部
    （reset_for_trial・run_until_target_vocab・係数差し替え）の破損を検出できない。
    """
    from lucina.testing import build_mock_core, make_test_config

    from calibrate_logit_bias import run_sweep

    cfg = make_test_config(log_dir=str(tmp_path))
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        rows = await run_sweep(
            core, [1.0, 2.0, 3.0],
            trials=2, n_tokens=6, prompt=None, max_tokens=200,
        )
    finally:
        core.close()

    assert [r["coefficient"] for r in rows] == [1.0, 2.0, 3.0]
    keys = {
        "coefficient", "prob_shift", "p90_tokens", "convergence_rate",
        "repetition_ratio", "mean_surprise", "sample_text",
    }
    assert all(keys <= set(r) for r in rows)
    shifts = [r["prob_shift"] for r in rows]
    assert all(b >= a for a, b in zip(shifts, shifts[1:]))  # ΔPは単調非減少
    assert all(0.0 <= r["convergence_rate"] <= 1.0 for r in rows)
