"""WorldModel: エージェントの信念 — カウントベース確率推定

MockWorld は ["food", "nothing", "danger"] の3種類のoutcomeを返すが、
DDLCWorld は任意の文字列（"poem", "dialogue", "system_access" など）を返す。
そのため WorldModel は動的にoutcomeを追加できるようにする。
"""

import math


class WorldModel:
    """各 (action, outcome) の共起カウントから確率を推定する。

    Prediction:     P(outcome | action) = count / total
    Learning:       surprise = 1 - P(observed | action), count += 1
    Action Value:   EV = Σ P(outcome) * utility(outcome)
    """

    # デフォルトのOutcomeセット（必要に応じて動的に拡張される）
    UTILITIES = {"food": 1.0, "nothing": 0.0, "danger": -1.0}

    def __init__(self, actions: list[str] | None = None):
        self.actions = actions or ["A", "B", "C"]
        self.counts: dict[str, dict[str, int]] = {
            a: {o: 0 for o in ("food", "nothing", "danger")} for a in self.actions
        }
        self.totals: dict[str, int] = {a: 0 for a in self.actions}
        self._all_outcomes: set[str] = {"food", "nothing", "danger"}

    # --- Dynamic outcome management ---

    def _ensure_outcome(self, action: str, outcome: str) -> None:
        """未知のoutcomeが来た場合、そのアクションにのみoutcomeカウンターを追加する。

        以前は全アクションに追加していたが、それだと「rest」に「poem」カウンターが
        生まれるなど不正確だった。アクション単位で管理することで確率を正確に保つ。
        """
        if outcome not in self._all_outcomes:
            self._all_outcomes.add(outcome)
        # このアクションにoutcomeがなければ追加
        if outcome not in self.counts[action]:
            self.counts[action][outcome] = 0

    # --- Action management ---

    def add_action(self, action: str):
        """新しい行動を動的に追加する。"""
        if action not in self.counts:
            self.counts[action] = {o: 0 for o in self._all_outcomes}
            self.totals[action] = 0
            self.actions.append(action)

    # --- Prediction ---

    def predict(self, action: str) -> dict[str, float]:
        """P(outcome | action) を返す。未経験なら一様分布。"""
        total = self.totals[action]
        if total == 0:
            uniform = 1.0 / len(self._all_outcomes)
            return {o: uniform for o in self._all_outcomes}
        return {o: self.counts[action].get(o, 0) / total for o in self._all_outcomes}

    def expected_value(self, action: str) -> float:
        """既知のutilityに基づく期待値。
        未知のoutcomeはutility=0として扱う。"""
        probs = self.predict(action)
        total = 0.0
        for outcome, prob in probs.items():
            utility = self.UTILITIES.get(outcome, 0.0)
            total += prob * utility
        return total

    # --- Learning ---

    def update(self, action: str, outcome: str):
        """カウントを1増やし、信念を更新する。"""
        self._ensure_outcome(action, outcome)
        self.counts[action][outcome] = self.counts[action].get(outcome, 0) + 1
        self.totals[action] += 1

    def surprise(self, action: str, outcome: str) -> float:
        """PE = 1 - P(observed | action)。"""
        self._ensure_outcome(action, outcome)
        probs = self.predict(action)
        return 1.0 - probs.get(outcome, 1.0 / len(self._all_outcomes))

    # --- Uncertainty ---

    def confidence(self, action: str) -> int:
        """その行動を試行した回数。"""
        return self.totals[action]

    def load_true_probabilities(self, action: str, probs: dict[str, float], samples: int = 100):
        """真の確率を直接設定する（実験用）。"""
        self.add_action(action)
        for outcome, prob in probs.items():
            self._ensure_outcome(action, outcome)
            self.counts[action][outcome] = int(prob * samples)
        self.totals[action] = samples

    def uncertainty(self, action: str) -> float:
        """不確実性指標。0=確信, 1=全く不明。"""
        n = self.totals[action]
        if n == 0:
            return 1.0
        return 1.0 / (1.0 + math.log1p(n))

    # --- Evaluation ---

    def l1_error(self, action: str, true_probs: dict[str, float]) -> float:
        """推定確率と真の確率のL1距離。"""
        pred = self.predict(action)
        return sum(abs(pred.get(o, 0) - true_probs.get(o, 0)) for o in self._all_outcomes)

    def summary(self) -> dict:
        """現在の信念の要約。"""
        return {
            a: {
                "predict": self.predict(a),
                "ev": round(self.expected_value(a), 3),
                "samples": self.confidence(a),
            }
            for a in self.actions
        }
