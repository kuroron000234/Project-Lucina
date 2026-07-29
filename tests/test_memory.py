"""
記憶層 (Memory) の単体テスト
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from core.memory.interface import Episode, MemoryInput, MemoryOutput
from core.memory.memory import Memory


class TestMemory:
    def setup_method(self):
        """各テスト前に一時ディレクトリでMemoryを初期化"""
        self.tmpdir = tempfile.mkdtemp()
        self.memory = Memory(storage_path=self.tmpdir)

    def _create_episode(self, event: str, importance: float = 0.5,
                        tags: list[str] | None = None, result: str = "success",
                        context: str = "test context") -> Episode:
        return Episode(
            id=f"test_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            event=event,
            context=context,
            emotion="neutral",
            result=result,
            importance=importance,
            tags=tags or ["test"],
        )

    # --- 正常系テスト ---

    def test_save_and_search(self):
        """save → search で同じエピソードが取得できる"""
        ep = self._create_episode("目標達成: ファイル探索")
        saved_id = self.memory.save(ep)
        assert saved_id == ep.id

        result = self.memory.search(MemoryInput(query="ファイル探索"))
        assert len(result.episodes) == 1
        assert result.episodes[0].id == ep.id
        assert result.episodes[0].event == ep.event
        assert result.total_count == 1

    def test_search_returns_empty_for_no_match(self):
        """クエリにマッチしない場合、空リストが返る"""
        ep = self._create_episode("目標達成: ファイル探索")
        self.memory.save(ep)

        result = self.memory.search(MemoryInput(query="存在しないクエリ"))
        assert len(result.episodes) == 0
        assert result.summary == "まだ記憶がありません"

    def test_search_empty_query_returns_recent(self):
        """空クエリ検索で最近のエピソードが返る"""
        ep1 = self._create_episode("最初の行動", importance=0.3)
        ep2 = self._create_episode("最近の行動", importance=0.5)
        self.memory.save(ep1)
        self.memory.save(ep2)

        result = self.memory.search(MemoryInput(query=""))
        assert len(result.episodes) > 0

    def test_importance_sorting(self):
        """重要度順にソートされている"""
        ep_low = self._create_episode("低重要度", importance=0.2)
        ep_high = self._create_episode("高重要度", importance=0.9)
        ep_med = self._create_episode("中重要度", importance=0.5)
        self.memory.save(ep_low)
        self.memory.save(ep_med)
        self.memory.save(ep_high)

        result = self.memory.search(MemoryInput(query="重要度", top_k=3))
        importances = [ep.importance for ep in result.episodes]
        assert importances == sorted(importances, reverse=True)

    def test_top_k_limits_results(self):
        """top_k で結果数が制限される"""
        for i in range(10):
            ep = self._create_episode(f"エピソード{i}")
            self.memory.save(ep)

        result = self.memory.search(MemoryInput(query="エピソード", top_k=3))
        assert len(result.episodes) == 3

    # --- 永続化テスト ---

    def test_persistence(self):
        """save後に再起動してもロードできる"""
        ep = self._create_episode("永続化テスト")
        self.memory.save(ep)

        # 新しいメモリインスタンスで読み込み
        memory2 = Memory(storage_path=self.tmpdir)
        result = memory2.search(MemoryInput(query="永続化テスト"))
        assert len(result.episodes) == 1
        assert result.episodes[0].event == ep.event

    # --- エッジケーステスト ---

    def test_save_without_id_generates_one(self):
        """IDなしで保存すると自動生成される"""
        ep = Episode(
            id="",
            timestamp=datetime.now(),
            event="自動IDテスト",
            context="",
            emotion="",
            result="",
            importance=0.5,
        )
        saved_id = self.memory.save(ep)
        assert saved_id  # 空でない
        assert ep.id == saved_id  # インプレイス更新される

    def test_search_with_min_importance(self):
        """min_importance フィルタが機能する"""
        ep_low = self._create_episode("低重要度", importance=0.1)
        ep_high = self._create_episode("高重要度", importance=0.8)
        self.memory.save(ep_low)
        self.memory.save(ep_high)

        result = self.memory.search(MemoryInput(
            query="重要度", min_importance=0.5
        ))
        assert len(result.episodes) == 1
        assert result.episodes[0].importance >= 0.5

    def test_forget_removes_low_importance(self):
        """forget() で低重要度エピソードが削除される"""
        ep_low = self._create_episode("忘れられる", importance=0.05)
        ep_high = self._create_episode("残る", importance=0.9)
        self.memory.save(ep_low)
        self.memory.save(ep_high)

        self.memory.forget(threshold=0.1)

        result = self.memory.search(MemoryInput(query="", top_k=10))
        events = [ep.event for ep in result.episodes]
        assert "残る" in events
        assert "忘れられる" not in events

    def test_update_importance(self):
        """update_importance() で重要度が更新される"""
        ep = self._create_episode("更新テスト", importance=0.5)
        self.memory.save(ep)

        self.memory.update_importance(ep.id, 0.9)

        result = self.memory.search(MemoryInput(query="更新テスト"))
        assert result.episodes[0].importance == 0.9

    def test_get_statistics_empty(self):
        """空のメモリで統計が取得できる"""
        stats = self.memory.get_statistics()
        assert stats["total_episodes"] == 0
        assert stats["avg_importance"] == 0.0

    def test_get_statistics_with_data(self):
        """エピソードがある場合の統計"""
        self.memory.save(self._create_episode("行動1", importance=0.8, tags=["探索"]))
        self.memory.save(self._create_episode("行動2", importance=0.3, tags=["休息"]))
        stats = self.memory.get_statistics()

        assert stats["total_episodes"] == 2
        assert 0.5 <= stats["avg_importance"] <= 0.6
        assert "探索" in stats["tag_distribution"]
        assert "休息" in stats["tag_distribution"]

    def test_no_episodes_returns_empty_summary(self):
        """エピソード0件の場合の要約"""
        result = self.memory.search(MemoryInput(query="何か"))
        assert result.summary == "まだ記憶がありません"
        assert result.total_count == 0
