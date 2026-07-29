"""Value System: 価値観形成

価値観は単純な強化ではない。
「何を選ぶ傾向があるか」だけでなく「なぜそれを重要だと思うようになったか」まで含む。

個性は単一のパラメータではなく、価値同士の関係から生まれる。
"""

import math


class ValueSystem:
    """経験から形成される価値観。

    外部報酬 (Phase 0) とは違い、価値観は個体の内部状態と経験から形成される。
    同じ成功でも個体によって価値の変化は異なる。

    Value と Preference は異なる：
      Value:      「探索は重要だ」という価値判断
      Preference: 「私は検索を選びやすい」という行動傾向
    """

    def __init__(self):
        # 価値観の重み [0, 1] — 高いほどその価値を重視
        self.weights: dict[str, float] = {
            "exploration": 0.3,       # 探索の価値
            "safety": 0.5,            # 安全の価値
            "social_bond": 0.3,       # 社会的絆の価値
            "knowledge": 0.4,         # 知識獲得の価値
            "efficiency": 0.4,        # 効率の価値
            "novelty": 0.3,           # 新奇性の価値
        }

        # Preferences (行動傾向) — Value から形成される
        self.preferences: dict[str, float] = {}  # action → preference weight

        # 更新履歴（価値観の変化を追跡）
        self.update_history: list[dict[str, float]] = []

    # --- Value Update ---

    def update_from_experience(
        self,
        action: str,
        outcome: str,
        pe: float,
        internal_state: dict | None = None,
    ) -> None:
        """経験に基づいて価値観を更新する。

        Parameters
        ----------
        action : str
            取った行動。
        outcome : str
            結果。
        pe : float
            予測誤差 (0-1)。
        internal_state : dict | None
            行動時の内部状態（更新の変調に使用）。
        """
        # 価値観の更新は PE と結果の両方に依存する
        if outcome == "danger":
            # 危険な結果 → 安全の価値が上がり、探索の価値が下がる
            self.weights["safety"] = min(1.0, self.weights["safety"] + 0.03 * pe)
            self.weights["exploration"] = max(0.05, self.weights["exploration"] - 0.02 * pe)
        elif outcome == "food":
            # 良い結果 → その行動タイプの価値が上がる
            if action in ("explore",):
                self.weights["exploration"] = min(1.0, self.weights["exploration"] + 0.02)
                self.weights["novelty"] = min(1.0, self.weights["novelty"] + 0.02)
            elif action in ("A", "B", "C"):
                self.weights["efficiency"] = min(1.0, self.weights["efficiency"] + 0.01)
        elif outcome == "positive":
            # 社会的に良い結果 → 社会的絆の価値が上がる
            self.weights["social_bond"] = min(1.0, self.weights["social_bond"] + 0.03)

        # 高PE事象は価値観に大きな影響を与える
        if pe > 0.7:
            for key in self.weights:
                # 驚くような結果は全ての価値観を微調整
                self.weights[key] = max(0.05, min(1.0, self.weights[key] + 0.01 * (0.5 - self.weights[key])))

        # 内部状態による変調
        if internal_state:
            curiosity = internal_state.get("curiosity", 50)
            if curiosity > 70:
                self.weights["exploration"] = min(1.0, self.weights["exploration"] + 0.01)
            safety = internal_state.get("safety", 100)
            if safety < 30:
                self.weights["safety"] = min(1.0, self.weights["safety"] + 0.02)

        # 履歴に保存
        self.update_history.append(dict(self.weights))

    # --- Preference Formation ---

    def update_preference(self, action: str, value_contribution: float) -> None:
        """行動の価値貢献に基づいて Preference を更新する。"""
        alpha = 0.05
        current = self.preferences.get(action, 0.0)
        self.preferences[action] = current * (1 - alpha) + value_contribution * alpha

    def get_preference(self, action: str) -> float:
        """その行動に対する Preference weight を返す。"""
        return self.preferences.get(action, 0.0)

    # --- Query ---

    def value_bonus(self, action: str) -> float:
        """価値観に基づく行動のボーナス値。

        EFE の Need Satisfaction 項に加算するための値。
        """
        bonus = 0.0
        if action in ("explore",):
            bonus += self.weights["exploration"] * 0.3
            bonus += self.weights["novelty"] * 0.2
        if action in ("A", "B", "C", "rest"):
            bonus += self.weights["safety"] * 0.2
            bonus += self.weights["efficiency"] * 0.1
        if action in ("greet", "talk", "help"):
            bonus += self.weights["social_bond"] * 0.3

        # Preference bonus
        bonus += self.get_preference(action) * 0.2

        return bonus

    def dominant_values(self, top_n: int = 3) -> list[tuple[str, float]]:
        """最も強い価値観を返す。"""
        sorted_vals = sorted(
            self.weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_vals[:top_n]

    def summary(self) -> dict:
        return {
            "values": {k: round(v, 3) for k, v in self.weights.items()},
            "preferences": {k: round(v, 3) for k, v in
                           sorted(self.preferences.items(), key=lambda x: x[1], reverse=True)[:5]},
            "dominant": self.dominant_values(),
        }
