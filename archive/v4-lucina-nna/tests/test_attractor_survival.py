"""アトラクタ収束の校正実験（仕様書 v1.4 §7 test_attractor_survival.py の骨格を実装）。

重要な設計判断: このテストは「合否判定」ではなく「閾値を生成するための計測」として
実装する。固定の assert count < 300 のようなテストにはしない（v3指摘事項）。
p90 を reports/calibration_*.json に出力し、config 更新のトリガーとする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrate_thresholds import _update_config_thresholds, run_until_target_vocab, write_calibration_report


@pytest.mark.asyncio
async def test_loneliness_attractor_convergence(lucina_core_fixture, tmp_path):
    """loneliness=0.9固定時、目的語彙への収束をトークン数で計測する。"""
    core = lucina_core_fixture
    core.drives["loneliness"] = 0.9
    tokens_to_convergence: list[int] = []

    for _trial in range(50):  # v3 §5.1の校正手順: 50試行
        core.reset_working_buffer()
        count, converged = await run_until_target_vocab(core, target_kind="loneliness", max_tokens=1000)
        if converged:
            tokens_to_convergence.append(count)

    assert tokens_to_convergence, "全試行で収束しなかった（語彙マップ・バイアス適用を確認）"

    p90 = float(np.percentile(tokens_to_convergence, 90))
    # この p90 が新しい閾値になる。アサーションで固定値と比較するのではなく、
    # 結果をレポートファイルに出力し、config更新のトリガーとする。
    report = write_calibration_report("attractor_survival_tokens", p90, output_dir=str(tmp_path))
    assert Path(report).exists()
    assert 0 < p90 <= 1000

    # モックバックエンドではバイアスが強く効くため、収束は極めて速いはず（配線確認）
    assert p90 <= 20


@pytest.mark.asyncio
async def test_convergence_requires_drive_bias(lucina_core_fixture):
    """対照実験: Drive値が0ならバイアスが効かず、単独試行では収束が遅い（配線の健全性確認）。"""
    core = lucina_core_fixture
    core.drives["loneliness"] = 0.0
    core.reset_working_buffer()
    count, converged = await run_until_target_vocab(core, target_kind="loneliness", max_tokens=200)
    assert not converged  # 0.001オーダーの摂動のみでは lonelinness 語彙に収束しない


def test_update_config_thresholds_preserves_comments(tmp_path):
    """config の行置換で値だけが変わり、コメント・整形（# 前の空白）が保持される（§4）。"""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "thresholds:\n"
        "  attractor_survival_tokens: 300     # PLACEHOLDER — 実測前\n"
        "  attractor_survival_prob: 0.6       # PLACEHOLDER — 実測前\n",
        encoding="utf-8",
    )
    _update_config_thresholds(str(cfg), {"attractor_survival_tokens": 91, "attractor_survival_prob": 1.0})
    text = cfg.read_text(encoding="utf-8")
    assert "attractor_survival_tokens: 91" in text
    assert "attractor_survival_prob: 1.0" in text
    assert "# PLACEHOLDER — 実測前" in text          # コメントは保持される
    assert "91# " not in text and "1.0# " not in text  # 値とコメントがくっつかない
