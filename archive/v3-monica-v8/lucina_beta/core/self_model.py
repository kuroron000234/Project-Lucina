"""Self Model: 自己モデル — 自分自身を予測する

自己モデルは自己紹介文ではない。未来の自分の行動・結果を予測するモデルである。

順序:
  行動履歴 → 統計的自己モデル → 自己理解 → 言語化された Identity

Ability:      自分は何ができるか（検索成功率、予測精度）
Limitation:   自分は何ができないか（曖昧タスクの失敗率）
History:      行動履歴の統計
Preference:   選択傾向
Prediction:   自分の予測はどの程度当たるか
"""

import math


class SelfModel:
    """自己モデル — 行動履歴から統計的に自己をモデル化する。"""

    def __init__(self):
        # 能力推定
        self.abilities: dict[str, float] = {
            "prediction": 0.0,        # 予測の的中率 [0, 1]
            "social": 0.0,            # 社会的相互作用の成功率
            "exploration": 0.0,       # 探索行動の成果率
        }
        # 制限
        self.limitations: dict[str, float] = {
            "uncertainty_tolerance": 1.0,  # 不確実性耐性
            "failure_sensitivity": 0.5,    # 失敗への敏感さ
        }
        # 行動履歴の統計
        self.action_counts: dict[str, int] = {}
        self.action_success_rate: dict[str, float] = {}
        # 予測精度追跡
        self._total_predictions = 0
        self._correct_predictions = 0
        self._prediction_history: list[float] = []  # 直近の的中/不的中

    # --- Learning ---

    def record_action(self, action: str, outcome: str, success: bool) -> None:
        """行動と結果を記録する。

        Parameters
        ----------
        action : str
            取った行動。
        outcome : str
            結果（food / nothing / danger / positive / neutral / negative）。
        success : bool
            成功かどうか（food=成功, danger=失敗, positive=成功 など）。
        """
        # カウント
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        total = self.action_counts[action]

        # 成功確率の更新（指数移動平均）
        prev_sr = self.action_success_rate.get(action, 0.5)
        alpha = 1.0 / (1.0 + total * 0.1)  # 徐々に学習率を下げる
        self.action_success_rate[action] = prev_sr * (1 - alpha) + float(success) * alpha

        # 能力推定の更新
        if action in ("greet", "talk", "help"):
            # 社会的行動
            old = self.abilities["social"]
            self.abilities["social"] = old * 0.95 + float(success) * 0.05
        elif action in ("explore",):
            old = self.abilities["exploration"]
            self.abilities["exploration"] = old * 0.95 + float(success) * 0.05

    def record_prediction(self, correct: bool) -> None:
        """予測の正誤を記録する。"""
        self._total_predictions += 1
        if correct:
            self._correct_predictions += 1
        self._prediction_history.append(1.0 if correct else 0.0)
        if len(self._prediction_history) > 50:
            self._prediction_history.pop(0)

        # 予測能力の更新
        rate = self._correct_predictions / max(1, self._total_predictions)
        self.abilities["prediction"] = rate

    def record_prediction_outcome(
        self, action: str, outcome: str, predicted_prob: float
    ) -> None:
        """予測結果に基づいて自己モデルを更新する。"""
        # 予測確率が高かった → 自信があった → 外れたら大きな衝撃
        success = outcome in ("food", "positive")
        self.record_action(action, outcome, success)

        # 予測精度: 自信があった予測が外れたら大きくペナルティ
        if predicted_prob > 0.7 and not success:
            self.limitations["uncertainty_tolerance"] = max(
                0.0, self.limitations["uncertainty_tolerance"] - 0.02
            )
            self.limitations["failure_sensitivity"] = min(
                1.0, self.limitations["failure_sensitivity"] + 0.03
            )

    # --- Query ---

    def predict_self_success(self, action: str) -> float:
        """この行動を自分がやったら成功する確率を予測する。"""
        # 行動の成功確率 × 全体的な予測能力
        action_sr = self.action_success_rate.get(action, 0.5)
        pred_ability = self.abilities["prediction"]
        # 統合: weighted
        return action_sr * 0.7 + pred_ability * 0.3

    def confidence_in_prediction(self, predicted_prob: float) -> float:
        """予測に対する自信度。"""
        ability = self.abilities["prediction"]
        # 高い能力 + 高い確率 = 高い自信
        return ability * 0.5 + predicted_prob * 0.5

    @property
    def prediction_accuracy(self) -> float:
        """全体的な予測的中率。"""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    @property
    def recent_prediction_accuracy(self) -> float:
        """最近の予測的中率（直近50件）。"""
        if not self._prediction_history:
            return 0.0
        return sum(self._prediction_history) / len(self._prediction_history)

    # --- Summary ---

    def summary(self) -> dict:
        return {
            "abilities": {k: round(v, 3) for k, v in self.abilities.items()},
            "limitations": {k: round(v, 3) for k, v in self.limitations.items()},
            "prediction_accuracy": round(self.prediction_accuracy, 3),
            "recent_accuracy": round(self.recent_prediction_accuracy, 3),
            "actions_taken": len(self.action_counts),
        }
