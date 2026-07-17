"""
記憶システム — 埋め込みベースの類似度検索

SQLite バックエンド（storage.py）に保存。
Zen API 埋め込み（trigramフォールバック）を使用したコサイン類似度検索。
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from .llm_client import embed as _zen_embed, trigram_hash

logger = logging.getLogger(__name__)


def _local_embed(text: str, dim: int = 64) -> list[float]:
    return trigram_hash(text, dim=dim)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Memory:
    """記憶システム — 埋め込みベースの検索・保存"""

    def __init__(self, path: str = "data/memory_store.json"):
        self.path = Path(path)
        self.dim = 64
        self.entries: list[dict] = []

        # JSONからの初回読み込みを試行（移行用）
        if self.path.exists():
            self.load()

    def _embed(self, text: str) -> list[float]:
        emb = _zen_embed(text[:1000])
        return emb if emb else _local_embed(text, self.dim)

    def add(self, text: str, activity_type: str = "other",
            tags: list[str] | None = None, importance: int = 5):
        """記憶を追加"""
        embedding = self._embed(text)
        timestamp = datetime.now().isoformat()

        # SQLite優先
        try:
            from .storage import memory_add
            memory_add(
                text=text,
                activity_type=activity_type,
                tags=tags or [],
                importance=importance,
                embedding=embedding,
                timestamp=timestamp,
            )
            return
        except Exception as e:
            logger.debug(f"SQLite memory_add failed, using JSON: {e}")

        # JSON フォールバック
        entry = {
            "text": text,
            "embedding": embedding,
            "timestamp": timestamp,
            "activity_type": activity_type,
            "tags": tags or [],
            "importance": importance,
        }
        self.entries.append(entry)
        self._trim(500)
        self._save_json()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """検索クエリに類似した記憶を返す"""
        if not self.entries and not self._has_sqlite():
            return []

        # SQLite優先
        try:
            from .storage import memory_search
            q_emb = self._embed(query)
            results = memory_search(query_embedding=q_emb, k=k)
            if results:
                return results
        except Exception as e:
            logger.debug(f"SQLite memory_search failed, using JSON: {e}")

        return self._search_json(query, k)

    def _has_sqlite(self) -> bool:
        try:
            from .storage import memory_count
            return memory_count() > 0
        except Exception:
            return False

    def _search_json(self, query: str, k: int) -> list[dict]:
        if not self.entries:
            return []
        q_emb = self._embed(query)
        scored = [(_cosine(q_emb, e["embedding"]), e) for e in self.entries]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def context(self, query: str, k: int = 5) -> str:
        """検索結果を文字列コンテキストに整形"""
        entries = self.search(query, k)
        parts = []
        for e in entries:
            ts = e.get("timestamp", "")
            if len(ts) > 16:
                ts = ts[:16]
            parts.append(f"[{ts}] ({e.get('activity_type', '?')}) {e.get('text', '')}")
        return "\n".join(parts)

    def _trim(self, max_entries: int = 500):
        if len(self.entries) > max_entries:
            self.entries = self.entries[-max_entries:]

    def _save_json(self):
        """JSONフォールバック保存"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"dim": self.dim, "entries": self.entries}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def save(self):
        """永続化（SQLite優先）"""
        # SQLiteの場合は即時保存されているので何もしない
        # JSONフォールバック用
        if self.entries:
            self._save_json()

    def load(self):
        """JSONからの読み込み（移行用）"""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.dim = data.get("dim", 64)
            self.entries = data.get("entries", [])
        except Exception:
            self.entries = []

    def compress(self, max_entries: int = 200):
        """古い記憶を圧縮・削除（重要度の低いものを優先的に削除）"""
        try:
            from .storage import memory_trim
            memory_trim(max_entries)
            return
        except Exception:
            pass

        # JSONフォールバック
        if len(self.entries) > max_entries:
            # 重要度でソートし、上位のみ残す
            self.entries.sort(key=lambda e: (
                e.get("importance", 5),
                e.get("timestamp", ""),
            ), reverse=True)
            self.entries = self.entries[:max_entries]
            self._save_json()
