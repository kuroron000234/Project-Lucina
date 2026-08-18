"""設定読込・レイヤリングのテスト（仕様書 v1.4 §4）。"""

from __future__ import annotations

from lucina.config import load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert cfg["model"]["context_window"] == 8192
    assert cfg["thresholds"]["attractor_survival_tokens"] == 91  # 校正済み 2026-08-09（Qwen3.5-9B p90=90.9）
    assert cfg["drive"]["relief"]["unit"] == "segment"
    assert cfg["drive"]["scheduling"]["enabled"] is False  # v1.7: 既定は従来の連続生成モード
    assert cfg["drive"]["scheduling"]["inner_prefix"] == "[内言] "
    assert cfg["inference"]["logit_bias_coefficient"] == 3.5  # 校正済み 2026-08-12（Qwen3.5-9B スイープ選定）


def test_load_config_layers_over_default(tmp_path) -> None:
    """指定パスは default.yaml に重ねられる（レイヤリング）。"""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "model:\n"
        "  path: \"./models/my.gguf\"\n"
        "thresholds:\n"
        "  attractor_survival_tokens: 123\n",
        encoding="utf-8",
    )
    cfg = load_config(custom)
    assert cfg["model"]["path"] == "./models/my.gguf"          # 上書き
    assert cfg["model"]["context_window"] == 8192              # default から継承
    assert cfg["thresholds"]["attractor_survival_tokens"] == 123  # 上書き
    assert cfg["drive"]["relief"]["unit"] == "segment"          # default から継承
