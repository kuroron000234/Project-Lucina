"""エピソード記憶 — 会話の要約と長期保存"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class EpisodicMemory:
    """会話のエピソードを要約して保存する長期記憶"""

    def __init__(self, data_dir: str = "monica/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.data_dir / "episodes.jsonl"
        self._episodes: list[dict] = []
        self._load()

    def add_episode(self, summary: str, topics: list[str], importance: float = 0.5):
        """エピソードを追加"""
        episode = {
            "summary": summary,
            "topics": topics,
            "importance": importance,
            "timestamp": time.time(),
            "date": datetime.now(timezone.utc).isoformat(),
        }
        self._episodes.append(episode)

        with open(self.episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 5) -> list[dict]:
        """直近n件のエピソードを取得"""
        return self._episodes[-n:] if self._episodes else []

    def get_important(self, threshold: float = 0.6, n: int = 5) -> list[dict]:
        """重要度が高いエピソードを取得"""
        important = sorted(
            [e for e in self._episodes if e["importance"] >= threshold],
            key=lambda x: x["timestamp"],
            reverse=True,
        )
        return important[:n]

    def search(self, query: str, n: int = 3) -> list[dict]:
        """トピックでエピソードを検索（部分一致）"""
        query = query.lower()
        matched = [
            e for e in self._episodes
            if any(query in t.lower() for t in e["topics"])
            or query in e["summary"].lower()
        ]
        return matched[-n:]

    def context_string(self, n: int = 3) -> str:
        """LLMに渡す用のエピソード要約文字列"""
        episodes = self.get_recent(n)
        if not episodes:
            return ""
        lines = ["【これまでの会話の記憶】"]
        for e in episodes:
            topics_str = ", ".join(e["topics"][:3])
            lines.append(f"  • {e['summary']} (話題: {topics_str})")
        return "\n".join(lines)

    def _load(self):
        if self.episodes_path.exists():
            with open(self.episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._episodes.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    @property
    def count(self) -> int:
        return len(self._episodes)
