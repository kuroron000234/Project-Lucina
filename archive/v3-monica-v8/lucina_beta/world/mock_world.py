"""MockWorld: 確率的な3地点環境"""

import random


class MockWorld:
    """確率的な環境。Phase 0 では A/B/C の3地点、Phase 1 以降では rest/explore も追加。

    各行動の真の確率:
      A: food 80%, nothing 10%, danger 10%
      B: food 40%, nothing 40%, danger 20%
      C: food 20%, nothing 20%, danger 60%
      rest: 100% nothing（エネルギー回復用）
      explore: food 20%, nothing 30%, danger 50%（高リスク高リターン）
    """

    OUTCOMES = ["food", "nothing", "danger"]

    def __init__(self, seed: int | None = None, phase: int = 0):
        if seed is not None:
            random.seed(seed)
        self.phase = phase
        self.locations = {
            "A": {"food": 0.80, "nothing": 0.10, "danger": 0.10},
            "B": {"food": 0.40, "nothing": 0.40, "danger": 0.20},
            "C": {"food": 0.20, "nothing": 0.20, "danger": 0.60},
        }
        if phase >= 1:
            self.locations["rest"] = {"food": 0.0, "nothing": 1.0, "danger": 0.0}
            self.locations["explore"] = {"food": 0.20, "nothing": 0.30, "danger": 0.50}

    def step(self, action: str) -> str:
        """行動を実行し、結果を返す。"""
        probs = self.locations[action]
        r = random.random()
        cumulative = 0.0
        for outcome, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return outcome
        return "nothing"

    def true_probabilities(self, action: str) -> dict[str, float]:
        """真の確率を返す（比較用）。"""
        return dict(self.locations[action])

    def actions(self) -> list[str]:
        return list(self.locations.keys())
