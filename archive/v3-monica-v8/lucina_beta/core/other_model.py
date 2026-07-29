"""Other Model: 他者の行動予測モデル

他者モデルは他者の本当の心ではない。
Lucinaが持つのは「相手はこう考えているだろう」という予測モデル。
つまり他者モデルも誤る。この「相手を誤解する可能性」が他者理解の核心。
"""

from typing import Optional


class OtherModel:
    """他者の行動を予測するモデル。

    他者ごとに以下の情報を保持する：
    - 行動履歴に基づく応答予測
    - 意図の推定
    - 信頼性評価
    - 感情状態の推定
    """

    def __init__(self):
        # entity_id → { action: { response: count } }
        self._history: dict[str, dict[str, dict[str, int]]] = {}
        # entity_id → reliability score [0, 1]
        self._reliability: dict[str, float] = {}
        # entity_id → total interactions
        self._total: dict[str, int] = {}

    def observe(self, entity_id: str, action: str, response: str) -> None:
        """他者の応答を観測し、モデルを更新する。

        Parameters
        ----------
        entity_id : str
            他者の識別子。
        action : str
            自分が取った行動。
        response : str
            他者の応答。
        """
        if entity_id not in self._history:
            self._history[entity_id] = {}
            self._reliability[entity_id] = 0.5  # 初期値: 中立
            self._total[entity_id] = 0

        if action not in self._history[entity_id]:
            self._history[entity_id][action] = {}

        # カウント更新
        counts = self._history[entity_id][action]
        counts[response] = counts.get(response, 0) + 1
        self._total[entity_id] += 1

        # 予測可能性の更新
        # 過去の予測と実際の応答の一致度から信頼性を計算
        total_responses = sum(counts.values())
        if total_responses > 0:
            max_count = max(counts.values())
            # reliability = 最も頻繁な応答の割合
            self._reliability[entity_id] = max_count / total_responses

    def predict(self, entity_id: str, action: str) -> dict:
        """他者の応答を予測する。

        Parameters
        ----------
        entity_id : str
            他者の識別子。
        action : str
            自分が取ろうとしている行動。

        Returns
        -------
        dict
            {"prediction": str, "confidence": float, "uncertainty": float}
        """
        if entity_id not in self._history or action not in self._history[entity_id]:
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "uncertainty": 1.0,
            }

        counts = self._history[entity_id][action]
        total = sum(counts.values())
        if total == 0:
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "uncertainty": 1.0,
            }

        # 最も頻繁な応答を予測
        best_response = max(counts, key=counts.get)
        confidence = counts[best_response] / total

        return {
            "prediction": best_response,
            "confidence": round(confidence, 3),
            "uncertainty": round(1.0 / (1.0 + total), 3),
        }

    def reliability(self, entity_id: str) -> float:
        """他者の信頼性を返す。"""
        return self._reliability.get(entity_id, 0.5)

    def interaction_count(self, entity_id: str) -> int:
        """他者との累積相互作用回数。"""
        return self._total.get(entity_id, 0)

    def known_entities(self) -> list[str]:
        """これまで観測した他者のリスト。"""
        return list(self._history.keys())

    def summary(self, entity_id: str) -> dict:
        """他者モデルの要約。"""
        if entity_id not in self._history:
            return {"entity": entity_id, "known": False}
        return {
            "entity": entity_id,
            "known": True,
            "reliability": round(self.reliability(entity_id), 3),
            "total_interactions": self.interaction_count(entity_id),
            "actions_observed": len(self._history[entity_id]),
        }
