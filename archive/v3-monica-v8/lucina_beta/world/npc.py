"""NPC: 内部状態を持つ他者エージェント

Phase 5 で Other Model との相互作用をテストするための最小NPC。
単純な内部状態と行動パターンを持つ。
"""

import random


class NPC:
    """内部状態を持つ他者エージェント。

    予測可能な行動パターンを持ち、Lucina が Other Model を学習するための対象。
    """

    def __init__(self, name: str, personality: str = "friendly", seed: int = 42):
        self.name = name
        self.personality = personality  # friendly, neutral, hostile
        self._rng = random.Random(seed)
        self.mood: float = 50.0  # 0=hostile, 100=friendly
        self.energy: float = 100.0

        # 性格に基づく行動確率
        if personality == "friendly":
            self._response_probs = {"positive": 0.7, "neutral": 0.25, "negative": 0.05}
            self.mood = 70.0
        elif personality == "hostile":
            self._response_probs = {"positive": 0.1, "neutral": 0.3, "negative": 0.6}
            self.mood = 30.0
        else:  # neutral
            self._response_probs = {"positive": 0.3, "neutral": 0.5, "negative": 0.2}
            self.mood = 50.0

    def respond(self, action: str) -> str:
        """与えられた行動に対するNPCの応答を生成する。

        Parameters
        ----------
        action : str
            エージェントからの行動（"greet", "talk", "help", "insult", "ignore" など）。

        Returns
        -------
        str
            NPC の応答（positive / neutral / negative）。
        """
        # 行動に基づく気分変化
        if action == "insult":
            self.mood = max(0.0, self.mood - 15.0)
            self.energy = max(0.0, self.energy - 10.0)
            return "negative"
        elif action == "help":
            self.mood = min(100.0, self.mood + 10.0)
            return "positive" if self._rng.random() < 0.8 else "neutral"
        elif action == "greet":
            self.mood = min(100.0, self.mood + 2.0)
        elif action == "talk":
            self.energy = max(0.0, self.energy - 5.0)
            if self.energy < 20:
                return "negative"  # 疲れているときは不機嫌

        # 性格 + 気分で応答を調整
        mood_bonus = (self.mood - 50.0) / 100.0  # -0.5 〜 +0.5
        probs = dict(self._response_probs)
        if mood_bonus > 0:
            probs["positive"] = min(1.0, probs["positive"] + mood_bonus * 0.3)
            probs["negative"] = max(0.0, probs["negative"] - mood_bonus * 0.2)
        else:
            probs["positive"] = max(0.0, probs["positive"] + mood_bonus * 0.2)
            probs["negative"] = min(1.0, probs["negative"] - mood_bonus * 0.3)

        # 正規化
        total = sum(probs.values())
        r = self._rng.random() * total
        cumulative = 0.0
        for outcome, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return outcome
        return "neutral"

    def known_actions(self) -> list[str]:
        """このNPCが応答可能な行動。"""
        return ["greet", "talk", "help", "insult", "ignore"]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "personality": self.personality,
            "mood": round(self.mood, 1),
            "energy": round(self.energy, 1),
        }
