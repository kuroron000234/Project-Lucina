"""Metacognition: メタ認知 — 自分の認知を観測する

「私は今、どう考えているのか」を扱う。

これがないと、内部状態はただの数値で終わる。
メタ認知があると以下のループになる：
  内部状態 → 認知を変える → 認知の変化を観測する → 「自分は今こういう状態だ」と理解する
"""

import math


class Metacognition:
    """メタ認知モジュール。

    自分の信念・自信・バイアスを観測し、認知の歪みを検出する。
    """

    def __init__(self):
        # 現在の認知状態
        self.confidence: float = 0.5      # [0, 1] 全体的な自信
        self.cognitive_load: float = 0.3  # [0, 1] 認知負荷
        self.mood_bias: float = 0.0       # [-1, 1] 気分によるバイアス
        self.overconfidence: float = 0.0  # [0, 1] 過信度

        # メタ認知の記録
        self._insights: list[dict] = []

    # --- Update ---

    def update(
        self,
        prediction_accuracy: float,
        recent_accuracy: float,
        internal_state: dict | None = None,
    ) -> None:
        """現在の認知状態を更新する。

        Parameters
        ----------
        prediction_accuracy : float
            全体的な予測的中率。
        recent_accuracy : float
            最近の予測的中率。
        internal_state : dict | None
            現在の内部状態（energy, safety, curiosity etc.）。
        """
        # 自信の更新
        # 最近の精度が高い → 自信が上がる
        # しかし過信とのバランスを取る
        self.confidence = self.confidence * 0.9 + recent_accuracy * 0.1

        # 過信度: 自信 > 実際の精度
        self.overconfidence = max(0.0, self.confidence - prediction_accuracy)

        # 内部状態による変調
        if internal_state:
            safety = internal_state.get("safety", 100)
            energy = internal_state.get("energy", 100)
            curiosity = internal_state.get("curiosity", 50)

            # 安全が低い → 認知負荷が高い（警戒モード）
            if safety < 30:
                self.cognitive_load = min(1.0, self.cognitive_load + 0.05)
            else:
                self.cognitive_load = max(0.1, self.cognitive_load - 0.02)

            # エネルギーが低い → 認知負荷が高い
            if energy < 30:
                self.cognitive_load = min(1.0, self.cognitive_load + 0.03)

            # 好奇心が高い → 気分バイアスがポジティブに
            if curiosity > 70:
                self.mood_bias = min(1.0, self.mood_bias + 0.02)
            else:
                self.mood_bias = max(-1.0, self.mood_bias - 0.01)

        # 自然減衰
        self.cognitive_load = max(0.1, self.cognitive_load - 0.01)
        self.mood_bias *= 0.99

    # --- Insights ---

    def generate_insight(self) -> dict | None:
        """現在の認知状態から洞察を生成する。

        バイアスや歪みが検出された場合のみ返す。
        """
        insights = []

        if self.overconfidence > 0.2:
            insights.append(f"I may be overconfident (gap: {self.overconfidence:.2f})")
        if self.cognitive_load > 0.7:
            insights.append(f"High cognitive load ({self.cognitive_load:.2f}) — considering simpler actions")
        if abs(self.mood_bias) > 0.3:
            direction = "positive" if self.mood_bias > 0 else "negative"
            insights.append(f"My judgment may be affected by {direction} mood bias")

        if insights:
            entry = {
                "confidence": round(self.confidence, 3),
                "cognitive_load": round(self.cognitive_load, 3),
                "mood_bias": round(self.mood_bias, 3),
                "overconfidence": round(self.overconfidence, 3),
                "insights": insights,
            }
            self._insights.append(entry)
            return entry
        return None

    # --- Query ---

    def should_simplify(self) -> bool:
        """認知負荷が高すぎる場合、行動を単純化すべきか。"""
        return self.cognitive_load > 0.7

    def is_confident(self) -> bool:
        """現在の判断に自信があるか。"""
        return self.confidence > 0.6 and self.overconfidence < 0.2

    # --- Summary ---

    def summary(self) -> dict:
        latest_insight = self._insights[-1] if self._insights else None
        return {
            "confidence": round(self.confidence, 3),
            "cognitive_load": round(self.cognitive_load, 3),
            "mood_bias": round(self.mood_bias, 3),
            "overconfidence": round(self.overconfidence, 3),
            "should_simplify": self.should_simplify(),
            "is_confident": self.is_confident(),
            "latest_insight": latest_insight["insights"] if latest_insight else None,
        }
