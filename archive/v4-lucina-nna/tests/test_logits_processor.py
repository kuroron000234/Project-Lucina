"""DriveLogitsProcessor / サプライズ計算の単体テスト（仕様書 v1.4 §5.4）。"""

from __future__ import annotations

import numpy as np
import pytest

from lucina.inference.entropy import entropy_from_logits, surprise_from_logits
from lucina.inference.logits import DriveLogitsProcessor
from lucina.testing import MockTokenizer

COEF = 2.5


def _zeros(size: int = 64) -> np.ndarray:
    return np.zeros(size, dtype=np.float64)


def test_first_token_only_bias() -> None:
    """⑤: 複数トークン語彙は先頭トークンにのみバイアスを加算し、後続トークンには適用しない。"""
    vocab = {"loneliness": [[10, 11]], "boredom": [[20]]}
    proc = DriveLogitsProcessor(COEF)
    out = proc.apply(_zeros(), {"loneliness": 1.0, "boredom": 0.0}, vocab)
    assert out[10] == pytest.approx(COEF)
    assert out[11] == 0.0  # 後続トークンには適用しない
    assert out[20] == 0.0


def test_overlap_first_tokens_sum_once() -> None:
    """⑤: 同じ先頭トークンIDが複数Driveで重複する場合は合算（二重カウントしない）。"""
    vocab = {"loneliness": [[10, 11]], "boredom": [[10, 99]]}
    proc = DriveLogitsProcessor(COEF)
    out = proc.apply(_zeros(), {"loneliness": 1.0, "boredom": 1.0}, vocab)
    assert out[10] == pytest.approx(2 * COEF)  # 両Driveの合算


def test_zero_drive_no_bias() -> None:
    proc = DriveLogitsProcessor(COEF)
    vocab = {"loneliness": [[10, 11]]}
    out = proc.apply(_zeros(), {"loneliness": 0.0}, vocab)
    assert out[10] == 0.0


def test_does_not_mutate_input() -> None:
    vocab = {"loneliness": [[10, 11]]}
    logits = _zeros()
    proc = DriveLogitsProcessor(COEF)
    proc.apply(logits, {"loneliness": 1.0}, vocab)
    assert logits[10] == 0.0  # 入力は変更しない


def test_drive_value_scales_bias() -> None:
    """logit_bias_coefficient * Drive値 の線形スケーリング。"""
    vocab = {"loneliness": [[10]]}
    proc = DriveLogitsProcessor(COEF)
    out = proc.apply(_zeros(), {"loneliness": 0.4}, vocab)
    assert out[10] == pytest.approx(COEF * 0.4)


def test_entropy_of_uniform_distribution() -> None:
    n = 8
    logits = np.zeros(n)
    assert entropy_from_logits(logits) == pytest.approx(np.log(n))


def test_surprise_normalization() -> None:
    """A2: surprise = min(1.0, entropy / entropy_scaling)。"""
    logits = np.zeros(5)  # uniform over 5 → entropy = ln5
    assert surprise_from_logits(logits, 5.0) == pytest.approx(np.log(5) / 5.0)
    # scaling が小さいと 1.0 に張り付く
    assert surprise_from_logits(logits, 0.1) == pytest.approx(1.0)


def test_vocab_expansion_integration(lucina_core_fixture) -> None:
    """③/B4: シード語彙から類似語が拡張され、vocab_map に反映されている（バイアス適用とrelief判定で共有）。"""
    core = lucina_core_fixture
    tok = MockTokenizer()
    loneliness_seed = tok.encode("寂しい")
    expanded = tok.encode("孤独")  # FakeEmbedder 上で loneliness 軸に近い語
    assert loneliness_seed in core.vocab_map["loneliness"]
    assert expanded in core.vocab_map["loneliness"]  # sim_threshold を超えて拡張されている
