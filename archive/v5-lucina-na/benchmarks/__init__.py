"""
Phase 3: 検証と本物化（外部レビュー対応）のベンチマークハーネス。

- surprise_validation: サプライズ層（本物のFEPコンポーネント）の挙動検証（M14）
- ablation: 学習/記憶/評価の層貢献をON/OFF比較で証明（M15）
- memory_persistence: 記憶保持ベンチマーク Recall@k（M16）
- run_all: 3本まとめて実行しレポートを自動生成（M17）

設計原則（外部レビュー対応）:
- 自己評価スコアは循環のため成功基準にしない（外部から検証できる事実のみ使用）
- 実LLM・実環境は使わない（決定論的で高速・無料）
- 実データ (data/episodes/ など) には一切触れない（一時ディレクトリで実行）
"""

from benchmarks.interface import BenchmarkReport, BenchmarkSection

__all__ = ["BenchmarkReport", "BenchmarkSection"]
