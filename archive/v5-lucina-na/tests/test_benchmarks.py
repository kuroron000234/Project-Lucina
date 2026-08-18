"""
Phase 3 (M15-M17): ベンチマークハーネスのテスト

- サプライズ検証（M14の実クラス検証）がレポートを生成する
- アブレーション検証（M15）がレポートを生成する
- 記憶保持ベンチマーク（M16）がレポートを生成する
- run_all（M17）が3本のJSONレポートを書き出す
"""

import json
import os

from benchmarks.ablation import run_ablation_validation
from benchmarks.memory_persistence import run_memory_benchmark
from benchmarks.surprise_validation import run_surprise_validation


def test_surprise_validation_report(tmp_path):
    """サプライズ検証が全セクションPASSのレポートを返す。"""
    r = run_surprise_validation(str(tmp_path))
    assert r.name == "surprise_validation"
    assert r.all_passed
    names = {s.name for s in r.sections}
    assert {"surprise_math", "drive_integration", "learning_integration"} <= names


def test_ablation_validation_report(tmp_path):
    """アブレーション検証が全セクションPASSのレポートを返す。"""
    r = run_ablation_validation(str(tmp_path))
    assert r.name == "ablation"
    assert r.all_passed
    names = {s.name for s in r.sections}
    assert {"learning_on_off", "memory_on_off", "evaluation_modes"} <= names


def test_memory_benchmark_report(tmp_path):
    """記憶保持ベンチマークが全セクションPASSのレポートを返す。"""
    r = run_memory_benchmark(str(tmp_path))
    assert r.name == "memory_persistence"
    assert r.all_passed
    names = {s.name for s in r.sections}
    assert {"recall_day0", "recall_paraphrase", "recall_day3", "forget_loss"} <= names


def test_memory_benchmark_does_not_touch_real_data(tmp_path):
    """ベンチマーク実行前後で実データ (data/episodes) が変化しない。"""
    real_dir = "data/episodes"
    before = set(os.listdir(real_dir)) if os.path.isdir(real_dir) else set()
    run_memory_benchmark(str(tmp_path))
    run_ablation_validation(str(tmp_path))
    after = set(os.listdir(real_dir)) if os.path.isdir(real_dir) else set()
    assert before == after


def test_run_all_generates_three_reports(tmp_path):
    """M17: run_all が3本のJSONレポートを書き出し、全PASSを返す。"""
    from benchmarks.run_all import run_all
    ok = run_all(str(tmp_path))
    assert ok is True
    files = set(os.listdir(tmp_path))
    assert {"ablation.json", "memory_persistence.json", "surprise_validation.json"} <= files

    for name in ["ablation", "memory_persistence", "surprise_validation"]:
        with open(os.path.join(tmp_path, f"{name}.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["all_passed"] is True
        assert data["generated_at"]


def test_report_serialization_roundtrip():
    """BenchmarkReport が辞書化・JSON化できる。"""
    from benchmarks.interface import BenchmarkReport, BenchmarkSection
    rep = BenchmarkReport(
        name="test",
        sections=[BenchmarkSection(name="s", passed=True, metrics={"a": 1})],
    )
    d = rep.to_dict()
    assert d["name"] == "test"
    assert d["all_passed"] is True
    assert d["sections"][0]["metrics"] == {"a": 1}
