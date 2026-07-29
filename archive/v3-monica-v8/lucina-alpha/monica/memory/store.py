"""記憶モジュール — 会話履歴の保存と復元"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MemoryStore:
    """会話履歴を JSON Lines で保存・管理"""

    def __init__(self, data_dir: str = "monica/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_path = self.data_dir / "conversations.jsonl"
        self._history: list[dict] = []
        self._load()

    # ── 基本操作 ──

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """メッセージを履歴に追加"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        self._history.append(entry)

        # ファイルにも追記
        with open(self.conversations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 10) -> list[dict]:
        """直近n件の会話を取得（role, content のリスト）"""
        return self._history[-n:] if self._history else []

    def get_context_messages(self, max_exchanges: int = 10) -> list[dict]:
        """LLMに渡す用のメッセージリストを取得"""
        recent = self.get_recent(max_exchanges * 2)  # user + assistant で2倍
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent
        ]

    def clear(self):
        """履歴をクリア"""
        self._history = []
        if self.conversations_path.exists():
            self.conversations_path.unlink()

    # ── 統計 ──

    @property
    def total_messages(self) -> int:
        return len(self._history)

    @property
    def total_exchanges(self) -> int:
        return len(self._history) // 2

    # ── 内部 ──

    def _load(self):
        """ファイルから履歴を復元"""
        if not self.conversations_path.exists():
            return

        with open(self.conversations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
