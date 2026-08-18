"""サプライズ（予測エントロピー近似）計算（仕様書 v1.4 §5.4）。

正規化は統一: surprise = min(1.0, entropy / entropy_scaling)
（A2: スケーリング係数は config の inference.entropy_scaling から与える）。
"""

from __future__ import annotations

import numpy as np


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """数値安定な softmax。"""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)


def entropy_from_logits(logits: np.ndarray) -> float:
    """次トークン分布のエントロピー（nats）。"""
    p = logits_to_probs(logits)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def surprise_from_logits(logits: np.ndarray, entropy_scaling: float) -> float:
    """エントロピーを [0,1] に正規化したサプライズ値。"""
    scaling = float(entropy_scaling)
    if scaling <= 0.0:
        scaling = 1.0
    return min(1.0, entropy_from_logits(logits) / scaling)
