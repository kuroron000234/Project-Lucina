"""ベンチマークレポートのデータ構造（interface.py 規約に従う）。"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BenchmarkSection:
    """ベンチマークの1セクション（検証項目1つ）。"""

    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "details": self.details,
        }


@dataclass
class BenchmarkReport:
    """ベンチマーク1本分のレポート。JSON化して data/benchmarks/ に保存する。"""

    name: str
    sections: list[BenchmarkSection]
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat(timespec="seconds")

    @property
    def all_passed(self) -> bool:
        return bool(self.sections) and all(s.passed for s in self.sections)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "all_passed": self.all_passed,
            "sections": [s.to_dict() for s in self.sections],
        }
