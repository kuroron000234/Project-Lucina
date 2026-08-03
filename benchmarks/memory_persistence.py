"""記憶保持ベンチマーク（M16）。

プローブエピソードを注入し、経過日数（0/3/7日）ごとの Recall@k を測定する。
「動いているのか単に出力しているだけなのか」に数字で答える最初の能力指標。

実データ (data/episodes/) は使わない。一時ディレクトリで完結する。
キーワード検索の特性を正直に定量化する:
- 同一語句クエリ: 日数が経ってもヒットし続ける（時刻非依存の検索）
- 言い換えクエリ: ヒットしない（= 実際の弱点。表記揺れ・類義語に弱い）
- 重要度低下 + forget(): 記憶が実際に失われる（喪失率）
"""

import os
import tempfile
from datetime import datetime, timedelta

from core.memory.memory import Memory
from core.memory.interface import Episode, MemoryInput

from benchmarks.common import save_report
from benchmarks.interface import BenchmarkReport, BenchmarkSection

# プローブ定義: (イベント, タグ, 重要度) — 検索困難な語彙を含める
PROBES = [
    {
        "id": "probe_quantum",
        "event": "goal=量子もつれの仕組みを調査する",
        "context": "量子もつれと重ね合わせ、観測問題について調べた",
        "tags": ["探検", "量子"],
        "importance": 0.9,
    },
    {
        "id": "probe_stars",
        "event": "goal=ユーザーと天体観測について話す",
        "context": "オリオン座と火星、秋の星空の話題",
        "tags": ["会話", "星"],
        "importance": 0.7,
    },
]

EXACT_QUERIES = ["量子もつれ", "天体観測"]
# 言い換えクエリ（完全一致しないが意味的に近い表現）:
# - "量子もつれの概要": バイグラムが大部分共有 → ハイブリッドでヒットする
# - "星を眺める話": 共有バイグラムなし → ハイブリッドでもヒットしない（正直な限界）
PARAPHRASE_QUERIES = ["量子もつれの概要", "星を眺める話"]


def _inject(mem: Memory) -> set[str]:
    """プローブエピソードを注入し、ID集合を返す。"""
    ids = set()
    for p in PROBES:
        ep = Episode(
            id=p["id"],
            timestamp=datetime.now(),
            event=p["event"],
            context=p["context"],
            emotion="",
            result="",
            importance=p["importance"],
            tags=list(p["tags"]),
            source="benchmark",
            driving_drive="exploration",
        )
        mem.save(ep)
        ids.add(ep.id)
    return ids


def _recall(mem: Memory, queries: list[str], probe_ids: set[str], top_k: int = 5,
            use_hybrid: bool = True) -> float:
    """クエリごとに search() し、プローブIDが結果に含まれる割合（Recall@k）。"""
    if not queries:
        return 0.0
    hits = 0
    for q in queries:
        out = mem.search(MemoryInput(query=q, top_k=top_k, use_hybrid=use_hybrid))
        if {e.id for e in out.episodes} & probe_ids:
            hits += 1
    return hits / len(queries)


def _age(mem: Memory, days: int):
    """全エピソードのタイムスタンプを days 日前に書き換えて再読込する。"""
    for ep in mem.episodes:
        ep.timestamp = datetime.now() - timedelta(days=days)
        mem._save_single(ep)
    mem.episodes = []
    mem._load()


def run_memory_benchmark(report_dir: str | None = None) -> BenchmarkReport:
    base = report_dir or tempfile.mkdtemp(prefix="lucina_bench_")
    storage = os.path.join(base, "episodes")
    os.makedirs(storage, exist_ok=True)

    mem = Memory(storage_path=storage)
    probe_ids = _inject(mem)

    # day 0: 同一語句 + 言い換え（キーワードのみ vs ハイブリッドを比較）
    recall_exact = _recall(mem, EXACT_QUERIES, probe_ids)
    recall_para_kw = _recall(mem, PARAPHRASE_QUERIES, probe_ids, use_hybrid=False)
    recall_para_hybrid = _recall(mem, PARAPHRASE_QUERIES, probe_ids, use_hybrid=True)
    sections = [
        BenchmarkSection(
            name="recall_day0",
            passed=(recall_exact == 1.0),
            metrics={"recall@5_exact": recall_exact},
            details=["同一語句クエリは注入直後に全ヒットする（キーワード検索の強み）"],
        ),
        BenchmarkSection(
            name="recall_paraphrase",
            passed=(recall_para_hybrid > recall_para_kw),
            metrics={
                "recall@5_keyword_only": recall_para_kw,
                "recall@5_hybrid": recall_para_hybrid,
                "improvement": round(recall_para_hybrid - recall_para_kw, 2),
            },
            details=[
                "完全一致しない言い換えクエリで Recall@k が向上（n-gram類似度）",
                "共有バイグラムがゼロの真の言い換えは依然ヒットしない（正直な限界）",
            ],
        ),
    ]

    # day 3: タイムスタンプを3日前に → 検索は時刻非依存なので維持されるはず
    _age(mem, 3)
    recall_day3 = _recall(mem, EXACT_QUERIES, probe_ids)
    sections.append(BenchmarkSection(
        name="recall_day3",
        passed=(recall_day3 == 1.0),
        metrics={"recall@5_exact_day3": recall_day3},
        details=["キーワード検索は時刻に依存しないため3日後も想起できる（保持の強み）"],
    ))

    # day 7: 重要度を下げて forget() を実行 → 喪失率を測定
    _age(mem, 7)
    for ep in mem.episodes:
        ep.importance = 0.05
        mem._save_single(ep)
    mem.episodes = []
    mem._load()
    before = len(mem.episodes)
    mem.forget(threshold=0.6)
    after = len(mem.episodes)
    loss = (before - after) / before if before else 0.0
    recall_day7 = _recall(mem, EXACT_QUERIES, probe_ids)
    sections.append(BenchmarkSection(
        name="forget_loss",
        passed=(loss > 0.0),
        metrics={
            "episodes_before_forget": before,
            "episodes_after_forget": after,
            "loss_rate": round(loss, 3),
            "recall@5_exact_day7": recall_day7,
        },
        details=[
            "重要度が閾値を下回ると forget() が記憶を削除する（喪失の実測）",
            "→ 重要度更新（学習層）が記憶の生存を左右する",
        ],
    ))

    return BenchmarkReport(name="memory_persistence", sections=sections)


if __name__ == "__main__":
    report = run_memory_benchmark()
    print(save_report(report))
