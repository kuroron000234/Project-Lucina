"""設定ファイル（YAML）の読み込み。

仕様書 v1.4 §4: すべてのマジックナンバーは config/default.yaml に集約する。
本モジュールは設定パス解決と、欠損キーに対する既定値マージを担当する。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 既定値（設定ファイルにキーがない場合のフォールバック。マジックナンバーではない運用値のみ）
DEFAULTS: dict[str, Any] = {
    "model": {
        "path": "./models/SELECTED_MODEL.gguf",
        "context_window": 8192,
        "n_gpu_layers": -1,
    },
    "memory": {
        "max_working_tokens_ratio": 0.75,
        "summarizer_model_path": "./models/summarizer-small.gguf",
        "persist_directory": "./data/chroma",
    },
    "drive": {
        "update_interval_sec": 0.1,
        "initial_state": {"boredom": 0.1, "loneliness": 0.1, "fatigue": 0.0},
        "relief": {
            "unit": "segment",
            "segment": {"max_tokens": 256, "boundary_tokens": ["。", "！", "？"]},
            "boredom": {"enabled": True, "per_action": 0.4, "decay_rate": 0.0005},
            "loneliness": {"enabled": True, "per_action": 0.6, "decay_rate": 0.0002},
            "fatigue": {"enabled": True, "per_action": 0.5, "decay_rate": 0.0003},
        },
        "dynamics_matrix": {
            "boredom": {"boredom": 0.005, "loneliness": 0.0, "fatigue": 0.01},
            "loneliness": {"boredom": 0.0, "loneliness": 0.002, "fatigue": 0.0},
            "fatigue": {"boredom": 0.0, "loneliness": 0.0, "fatigue": 0.003},
        },
        "vocab_expansion": {
            "top_k": 30,
            "sim_threshold": 0.45,
            "seed_vocab_path": "./config/seed_vocab.yaml",
            "max_candidates": 0,  # 拡張候補数の上限（0 = 無制限。実モデルの起動速度を制御する運用値）
        },
    },
    "inference": {
        "logit_bias_coefficient": 2.5,
        "entropy_think_threshold": 0.7,
        "surprise_relief_threshold": 0.7,
        "entropy_scaling": 5.0,
        "sampling": "multinomial",
        "temperature": 0.8,
        "seed": 42,
        "think_token_ids": [],
    },
    "embedding": {"model": "intfloat/multilingual-e5-small"},
    "logging": {"log_dir": "./reports", "level": "INFO"},
    "thresholds": {
        "attractor_survival_tokens": 300,
        "attractor_survival_prob": 0.6,
        "interrupt_latency_multiplier": 1.5,
    },
}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def find_config(path: str | Path | None = None) -> Path:
    """設定ファイルのパスを解決する。優先順: 引数 > 環境変数 LUCINA_CONFIG > 既定パス。"""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(_resolve_path(path))
    env = os.environ.get("LUCINA_CONFIG")
    if env:
        candidates.append(_resolve_path(env))
    candidates.append(Path.cwd() / "config" / "default.yaml")
    candidates.append(PROJECT_ROOT / "config" / "default.yaml")
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"設定ファイルが見つかりません: {', '.join(str(c) for c in candidates)}")


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    """YAML設定を読み込み、既定値とマージして返す。

    path を指定した場合は config/default.yaml をベースに重ね合わせる（レイヤリング）。
    戻り値の辞書はマージ済みのため、モジュール側はキー欠損を考慮しなくてよい。
    マジックナンバーは常に設定値（または既定値）経由で参照すること（§9）。
    """
    cfg_path = find_config(path)
    base: dict = copy.deepcopy(DEFAULTS)
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    if default_path.is_file() and default_path.resolve() != cfg_path.resolve():
        with default_path.open("r", encoding="utf-8") as fh:
            base = _deep_merge(base, yaml.safe_load(fh) or {})
    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return _deep_merge(base, loaded)
