"""Relationship Model: 他者との関係モデル

Other Model が「他者の行動予測」を担当するのに対し、
Relationship Model は「自己と他者の関係」をモデル化する。

Relationship は相手の属性ではなく、相互作用の履歴から生まれる。
"""

import math


class Relationship:
    """他者との関係をモデル化する。

    Parameters
    ----------
    entity_id : str
        関係の相手。
    """

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.trust: float = 0.5        # [0, 1] 信頼度
        self.familiarity: float = 0.0   # [0, 1] 親密度（相互作用回数に応じて上昇）
        self.attachment: float = 0.0    # [0, 1] 愛着度（ポジティブな相互作用の蓄積）
        self.conflict: float = 0.0      # [0, 1] 対立度（ネガティブな相互作用の蓄積）

        # 相互作用の記録
        self._total_interactions = 0
        self._positive_count = 0
        self._negative_count = 0
        self._recent_mood: list[float] = []  # 最近の感情（window=10）

    def update(self, action: str, response: str, prediction_was_correct: bool = True) -> None:
        """相互作用に基づいて関係を更新する。

        Parameters
        ----------
        action : str
            自分が取った行動。
        response : str
            相手の応答（positive / neutral / negative）。
        prediction_was_correct : bool
            自分の予測が当たったか（Other Model の信頼性）。
        """
        self._total_interactions += 1

        # Familiarity: 相互作用回数に応じて上昇（対数スケール）
        self.familiarity = min(1.0, math.log1p(self._total_interactions) / 10.0)

        # Trust: 予測が当たると上昇、外れると低下
        trust_delta = 0.05 if prediction_was_correct else -0.05
        self.trust = max(0.0, min(1.0, self.trust + trust_delta))

        # Attachment / Conflict: 応答の質に基づく
        if response == "positive":
            self._positive_count += 1
            self.attachment = min(1.0, self.attachment + 0.08)
            self.conflict = max(0.0, self.conflict - 0.03)
        elif response == "negative":
            self._negative_count += 1
            self.conflict = min(1.0, self.conflict + 0.12)
            self.attachment = max(0.0, self.attachment - 0.05)
        # neutral: 変化なし

        # 感情の記録（最近10件）
        mood_value = 1.0 if response == "positive" else (-1.0 if response == "negative" else 0.0)
        self._recent_mood.append(mood_value)
        if len(self._recent_mood) > 10:
            self._recent_mood.pop(0)

    @property
    def recent_sentiment(self) -> float:
        """最近の感情スコア [-1.0, +1.0]。"""
        if not self._recent_mood:
            return 0.0
        return sum(self._recent_mood) / len(self._recent_mood)

    @property
    def interaction_value(self) -> float:
        """この関係の総合的な価値（EFE の Need Satisfaction で使用）。"""
        return (self.trust * 0.3 + self.attachment * 0.4 +
                self.familiarity * 0.2 - self.conflict * 0.1)

    def summary(self) -> dict:
        return {
            "entity": self.entity_id,
            "trust": round(self.trust, 3),
            "familiarity": round(self.familiarity, 3),
            "attachment": round(self.attachment, 3),
            "conflict": round(self.conflict, 3),
            "interactions": self._total_interactions,
            "sentiment": round(self.recent_sentiment, 3),
            "value": round(self.interaction_value, 3),
        }
