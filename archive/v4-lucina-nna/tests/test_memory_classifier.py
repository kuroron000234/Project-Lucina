"""記憶分類器（RuleBasedMemoryClassifier）のテスト（仕様書 v1.4 §5.3・v1.11）。

- EPISODIC（出来事・経験）/ SEMANTIC（知識・事実）/ PROCEDURAL（手順・方法）/
  EMOTIONAL（感情語彙）を日本語のパターンマッチで分類する。
- 未一致は SEMANTIC（既定）。同点は優先順位 EMOTIONAL > EPISODIC > PROCEDURAL > SEMANTIC。
- ストア連携: commit に分類器を配線すると kind が付与される。Drive変化大（>=0.3）の
  EMOTIONAL 強制ルールは分類器より優先される。
"""

from __future__ import annotations

import asyncio

from lucina.memory.classifier import RuleBasedMemoryClassifier
from lucina.memory.schema import MemoryKind
from lucina.memory.store import HierarchicalMemoryStore, InMemoryVectorStore
from lucina.testing import build_mock_core, make_test_config


def test_classify_episodic_experience() -> None:
    """時間語＋過去形の出来事・経験は EPISODIC に分類される。"""
    clf = RuleBasedMemoryClassifier()
    text = "今日は静かな夜でした。月がとても綺麗に見えました。"
    assert clf.classify(text) == MemoryKind.EPISODIC


def test_classify_semantic_definition() -> None:
    """定義文（〜とは）は SEMANTIC に分類される。"""
    clf = RuleBasedMemoryClassifier()
    text = "記憶とは、過去の経験を保持し再現する心の仕組みです。"
    assert clf.classify(text) == MemoryKind.SEMANTIC


def test_classify_procedural_steps() -> None:
    """手順（まず〜次に〜）は PROCEDURAL に分類される。"""
    clf = RuleBasedMemoryClassifier()
    text = "新しい記憶を保存するには、まずベクトル化して、次にストアに追加します。"
    assert clf.classify(text) == MemoryKind.PROCEDURAL


def test_classify_emotional_lexicon() -> None:
    """強い感情語彙は EMOTIONAL に分類される（補助シグナル）。"""
    clf = RuleBasedMemoryClassifier()
    text = "とても悲しくて、胸が張り裂けそうでした。"
    assert clf.classify(text) == MemoryKind.EMOTIONAL


def test_classify_default_semantic() -> None:
    """パターン未一致は SEMANTIC（既定）。"""
    clf = RuleBasedMemoryClassifier()
    assert clf.classify("なるほど、そういうことですね。") == MemoryKind.SEMANTIC
    assert clf.classify("") == MemoryKind.SEMANTIC


def test_classify_definition_beats_emotional_word() -> None:
    """定義文は感情語彙より優先される（定義パターンの重みが高い）。"""
    clf = RuleBasedMemoryClassifier()
    text = "寂しさとは、他者とのつながりを求める心の動きです。"
    assert clf.classify(text) == MemoryKind.SEMANTIC


def test_classify_tie_broken_by_priority() -> None:
    """同点は優先順位 EMOTIONAL > EPISODIC > PROCEDURAL > SEMANTIC。"""
    clf = RuleBasedMemoryClassifier()
    # 「手順」(PROCEDURAL 1.2) と「とは」(SEMANTIC 1.2) が同点 → PROCEDURAL が優先
    assert clf.classify("手順とは何ですか。") == MemoryKind.PROCEDURAL


def test_store_uses_classifier() -> None:
    """分類器を配線したストアの commit は分類結果を kind に付与する。"""

    async def go() -> MemoryKind:
        store = HierarchicalMemoryStore(InMemoryVectorStore(), classifier=RuleBasedMemoryClassifier())
        record = await store.commit("今日は雨が降っていました。", {"boredom": 0.1})
        return record.kind

    assert asyncio.run(go()) == MemoryKind.EPISODIC


def test_store_drive_change_still_forces_emotional() -> None:
    """Drive変化大（abs(delta) >= 0.3）の EMOTIONAL 強制は分類器より優先される。"""

    async def go() -> MemoryKind:
        store = HierarchicalMemoryStore(InMemoryVectorStore(), classifier=RuleBasedMemoryClassifier())
        await store.commit("初回", {"boredom": 0.1})
        # 定義文（SEMANTIC判定のはず）でも Drive が大きく動けば EMOTIONAL になる
        record = await store.commit("記憶とは心の仕組みです。", {"boredom": 0.5})
        return record.kind

    assert asyncio.run(go()) == MemoryKind.EMOTIONAL


def test_core_memory_commit_logged_with_kind(tmp_path) -> None:
    """コア統合: セグメント完了で記憶コミットが分類結果つきで構造化ログに残る。"""
    cfg = make_test_config(log_dir=str(tmp_path))
    core = build_mock_core(cfg, log_dir=str(tmp_path))

    async def go() -> None:
        core.seed_prompt("起動")
        core.drives["boredom"] = 0.3
        core.segment.texts = ["今日は", "静かな", "夜でした。"]
        core.segment.token_ids = [1, 2, 3]
        core.segment.surprises = [0.2, 0.2, 0.2]
        await core._finalize_segment()  # noqa: SLF001

    asyncio.run(go())
    core.close()
    mem_log = tmp_path / "memory.jsonl"
    assert mem_log.exists()
    lines = mem_log.read_text(encoding="utf-8").splitlines()
    assert lines, "memory.jsonl にコミットログが書かれるべき"
    import json

    payload = json.loads(lines[-1])
    assert payload["kind"] == "episodic"  # 「今日は〜でした」→ EPISODIC
    assert payload["importance"] >= 0.0
