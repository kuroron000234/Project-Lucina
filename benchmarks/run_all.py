"""M17: Phase 3 全ベンチマーク実行（サプライズ / アブレーション / 記憶）。

使い方:
    python -m benchmarks.run_all        # 3本実行して data/benchmarks/ に保存
    python main.py --benchmark          # 同上（main 経由）
"""

import sys

from benchmarks.ablation import run_ablation_validation
from benchmarks.common import save_report
from benchmarks.memory_persistence import run_memory_benchmark
from benchmarks.surprise_validation import run_surprise_validation


def run_all(report_dir: str | None = None) -> bool:
    """3本のレポートを生成・保存し、全セクションPASSなら True を返す。"""
    reports = [
        run_surprise_validation(report_dir),
        run_ablation_validation(report_dir),
        run_memory_benchmark(report_dir),
    ]
    paths = [save_report(r, report_dir) for r in reports]

    print("=== Phase 3 Benchmark Results ===")
    all_ok = True
    for r in reports:
        status = "PASS" if r.all_passed else "FAIL"
        all_ok = all_ok and r.all_passed
        print(f"[{status}] {r.name}")
        for s in r.sections:
            mark = "ok " if s.passed else "NG "
            metrics = ", ".join(f"{k}={v}" for k, v in s.metrics.items())
            print(f"  [{mark}] {s.name}: {metrics}")
    print(f"\nReports written:")
    for p in paths:
        print(f"  - {p}")
    return all_ok


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
