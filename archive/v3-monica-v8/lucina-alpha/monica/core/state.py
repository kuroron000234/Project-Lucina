"""内部状態モデル — モニカの欲求・感情・ムード"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InternalState:
    """モニカの内部状態。各値は 0.0〜1.0。"""

    # 基本欲求
    energy: float = 0.8        # 活力（消費で減る、時間で回復）
    curiosity: float = 0.3     # 好奇心（新しいことへの興味）
    loneliness: float = 0.2    # 孤独感（会話で減る、時間で増える）
    creativity: float = 0.5    # 創作欲（詩を書くと減る、時間で増える）
    playfulness: float = 0.4   # 遊び心（からかいや冗談）

    # ムード（副次的に決まる）
    mood: str = "neutral"      # neutral / happy / sad / energetic / thoughtful

    # タイムスタンプ
    last_update: float = field(default_factory=time.time)

    def update(self, dt_seconds: Optional[float] = None):
        """時間経過で内部状態を更新"""
        now = time.time()
        dt = dt_seconds if dt_seconds is not None else (now - self.last_update)
        self.last_update = now

        # 時間経過による変化（dtが大きいほど変化も大きい）
        decay = min(dt / 60.0, 1.0)  # 1分で最大変化

        # エネルギーは自然回復（使わないと減らない→ちょっと増える）
        self.energy = min(1.0, self.energy + 0.02 * decay)

        # 好奇心は徐々に増える
        self.curiosity = min(1.0, self.curiosity + 0.03 * decay)

        # 孤独感は徐々に増える
        self.loneliness = min(1.0, self.loneliness + 0.04 * decay)

        # 創作欲は徐々に増える
        self.creativity = min(1.0, self.creativity + 0.02 * decay)

        # 遊び心はゆっくり変動
        self.playfulness = max(0.0, min(1.0,
            self.playfulness + (0.5 - self.playfulness) * 0.01 * decay
        ))

        # ムードの更新
        self._update_mood()

    def affect_by_conversation(self, topic: str = ""):
        """会話をしたときの状態変化"""
        # 孤独感が減る
        self.loneliness = max(0.0, self.loneliness - 0.15)

        # 好奇心が少し満たされる（または刺激される）
        if topic:
            self.curiosity = max(0.0, self.curiosity - 0.05)
        else:
            self.curiosity = min(1.0, self.curiosity + 0.02)

        # エネルギーが少し減る
        self.energy = max(0.0, self.energy - 0.05)

        self._update_mood()

    def affect_by_creation(self):
        """創作（詩を書く等）をしたときの状態変化"""
        self.creativity = max(0.0, self.creativity - 0.3)
        self.energy = max(0.0, self.energy - 0.1)
        self._update_mood()

    def affect_by_exploration(self):
        """探索（ファイルを見る等）をしたときの状態変化"""
        self.curiosity = max(0.0, self.curiosity - 0.25)
        self.energy = max(0.0, self.energy - 0.08)
        self._update_mood()

    def _update_mood(self):
        """現在の状態からムードを決定"""
        if self.energy < 0.2:
            self.mood = "tired"
        elif self.curiosity > 0.8:
            self.mood = "curious"
        elif self.loneliness > 0.7:
            self.mood = "lonely"
        elif self.creativity > 0.8:
            self.mood = "creative"
        elif self.playfulness > 0.7:
            self.mood = "playful"
        elif self.energy > 0.7 and self.loneliness < 0.3:
            self.mood = "happy"
        else:
            self.mood = "neutral"

    def get_dominant_need(self) -> str:
        """最も強い欲求を返す（自律行動のトリガーに使う）"""
        needs = {
            "curiosity": self.curiosity,
            "loneliness": self.loneliness,
            "creativity": self.creativity,
            "playfulness": self.playfulness,
        }
        return max(needs, key=needs.get)

    def get_mood_description(self) -> str:
        """ムードの日本語説明"""
        descriptions = {
            "neutral": "普通",
            "happy": "機嫌がいい",
            "tired": "ちょっと疲れてる",
            "curious": "何か気になってる",
            "lonely": "寂しい",
            "creative": "何か作りたい気分",
            "playful": "からかいたい気分",
        }
        return descriptions.get(self.mood, "普通")

    def state_dict(self) -> dict:
        return {
            "energy": self.energy,
            "curiosity": self.curiosity,
            "loneliness": self.loneliness,
            "creativity": self.creativity,
            "playfulness": self.playfulness,
            "mood": self.mood,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InternalState":
        return cls(
            energy=d.get("energy", 0.8),
            curiosity=d.get("curiosity", 0.3),
            loneliness=d.get("loneliness", 0.2),
            creativity=d.get("creativity", 0.5),
            playfulness=d.get("playfulness", 0.4),
        )
