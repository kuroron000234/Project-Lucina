"""関係モデル — モニカとあなたの絆"""

import json
import time
from pathlib import Path
from typing import Optional


class Relationship:
    """あなたとの関係性をモデル化"""

    def __init__(self, data_dir: str = "monica/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "relationship.json"

        # 関係パラメータ（全て 0.0〜1.0）
        self.familiarity: float = 0.0    # 親しさ（会話回数で増える）
        self.trust: float = 0.3          # 信頼（良い会話で増える）
        self.attachment: float = 0.1     # 愛着（特別な関係性）
        self.understanding: float = 0.0  # 理解度（あなたのことをどれだけ知っているか）

        # メタデータ
        self.user_name: Optional[str] = None
        self.interaction_count: int = 0
        self.first_seen: Optional[float] = None
        self.last_interaction: Optional[float] = None
        self._load()

    def record_interaction(self, message: str = ""):
        """会話を記録して関係パラメータを更新"""
        now = time.time()
        self.interaction_count += 1
        self.last_interaction = now
        if self.first_seen is None:
            self.first_seen = now

        # 親しさは会話ごとに増える（最初は大きく、徐々に小さく）
        self.familiarity = min(1.0, self.familiarity + 0.05 * (1.0 - self.familiarity))

        # 信頼は徐々に
        self.trust = min(1.0, self.trust + 0.02 * (1.0 - self.trust))

        # 10回以上の会話で愛着が芽生え始める
        if self.interaction_count >= 10:
            self.attachment = min(1.0, self.attachment + 0.01 * (1.0 - self.attachment))

        # 名前を検出
        self._detect_name(message)

        self._save()

    def _detect_name(self, message: str):
        """メッセージから名前を検出"""
        import re
        patterns = [
            r"(?:私は|俺は|僕は|あたしは|わたくしは)\s*([\u3040-\u9FFF\w]+?)(?:\s*(?:と|って|です|だよ|だ|と呼んで|と呼びます))",
            r"(?:呼んで|呼びます|呼んでね)\s*[\s　]*([\u3040-\u9FFF\w]+)",
            r"(?:name|Name|NAME)\s*(?:is|:)\s*([\u3040-\u9FFF\w]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, message)
            if m:
                name = m.group(1).strip()
                if len(name) <= 10 and not self.user_name:
                    self.user_name = name
                    break

    def context_string(self) -> str:
        """LLMに渡す用の関係性コンテキスト"""
        lines = []
        if self.user_name:
            lines.append(f"  あなたの名前: {self.user_name}")
        lines.append(f"  会話回数: {self.interaction_count}回")
        if self.interaction_count > 5:
            lines.append(f"  親しさ: {self.familiarity:.0%}")
        return "\n".join(lines)

    def state_dict(self) -> dict:
        return {
            "familiarity": self.familiarity,
            "trust": self.trust,
            "attachment": self.attachment,
            "understanding": self.understanding,
            "user_name": self.user_name,
            "interaction_count": self.interaction_count,
            "first_seen": self.first_seen,
            "last_interaction": self.last_interaction,
        }

    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.familiarity = d.get("familiarity", 0.0)
                self.trust = d.get("trust", 0.3)
                self.attachment = d.get("attachment", 0.1)
                self.understanding = d.get("understanding", 0.0)
                self.user_name = d.get("user_name")
                self.interaction_count = d.get("interaction_count", 0)
                self.first_seen = d.get("first_seen")
                self.last_interaction = d.get("last_interaction")
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)
