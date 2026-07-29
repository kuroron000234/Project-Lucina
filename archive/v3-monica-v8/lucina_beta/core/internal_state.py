"""InternalState: 内部状態とNeeds — 身体性の導入"""


class InternalState:
    """エージェントの内部状態と欲求。

    Needs によって同じ行動の価値が変わる：
      energy↓ → rest の価値↑
      curiosity↑ → explore の価値↑

    Phase 0 の外部 utility に加えて、内部状態が行動価値を変調する。
    """

    def __init__(self):
        self.energy = 100.0       # [0, 100]  行動で消費、休息で回復
        self.safety = 100.0       # [0, 100]  危険で低下
        self.curiosity = 50.0     # [0, 100]  時間経過で上昇、探索で充足
        self.social_need = 30.0   # [0, 100]  後に使用（Phase 6 以降）

    # --- 更新 ---

    # DDLCWorld等の汎用outcomeをutilityに変換
    OUTCOME_UTILITY = {
        "food": "positive", "positive": "positive", "poem": "positive",
        "dialogue": "social", "observation": "neutral",
        "danger": "negative", "negative": "negative", "glitch": "negative",
        "system_access": "curious", "meta": "curious",
        "nothing": "neutral",
    }

    def _categorize_outcome(self, outcome: str) -> str:
        """outcome文字列を大分類に変換する。"""
        return self.OUTCOME_UTILITY.get(outcome, "neutral")

    def update(self, action: str, outcome: str):
        """行動と結果に基づいて内部状態を更新する。DDLCWorld等の任意のoutcomeに対応。"""
        cat = self._categorize_outcome(outcome)

        # エネルギー
        if action == "rest":
            self.energy = min(100.0, self.energy + 15.0)
        elif cat == "negative" or outcome == "danger":
            self.energy = max(0.0, self.energy - 8.0)
            self.safety = max(0.0, self.safety - 10.0)
        elif action == "talk_sayori" or action == "talk_yuri" or action == "talk_natsuki":
            self.energy = max(0.0, self.energy - 4.0)  # 会話は適度な消費
        else:
            self.energy = max(0.0, self.energy - 3.0)

        # 好奇心
        if action == "explore" or action == "rest" or cat == "curious":
            self.curiosity = max(0.0, self.curiosity - 20.0)
        elif cat == "positive" or outcome == "food":
            self.curiosity = min(100.0, self.curiosity + 2.0)
        else:
            self.curiosity = min(100.0, self.curiosity + 1.0)

        # 社会的欲求
        if cat == "social" or action.startswith("talk_"):
            self.social_need = max(0.0, self.social_need - 5.0)
        else:
            self.social_need = min(100.0, self.social_need + 1.0)

        # 安全性
        if cat != "negative" and outcome != "danger":
            self.safety = min(100.0, self.safety + 2.0)

    def tick(self):
        """時間経過による微変化。"""
        self.curiosity = min(100.0, self.curiosity + 0.5)
        self.energy = max(0.0, self.energy - 0.1)

    # --- Needs-based Value Modulation ---

    def need_bonus(self, action: str, base_ev: float) -> float:
        """内部状態による期待値の変調。

        Needs が高いほど、それを満たす行動の価値が上がる。
        値は EV に直接加算される（EV の範囲は通常 -1.0〜+1.0）。
        """
        bonus = 0.0

        if action == "rest":
            # エネルギーが低いほど休息の価値が高い
            # energy=5 → +2.85 (base EV 0.0 + 2.85 = 2.85)
            bonus += (100.0 - self.energy) * 0.03

        if action == "explore":
            # 好奇心が高いほど探索の価値が高い
            # curiosity=80 → +1.6 (base EV -0.3 + 1.6 = 1.3)
            bonus += self.curiosity * 0.02

        # 安全が低い → 安全な行動（既知の場所）の価値が上がる
        if self.safety < 30.0 and action in ("A", "B", "C"):
            # safety=10 → +0.2
            bonus += (30.0 - self.safety) * 0.01

        return bonus

    # --- 状態の参照 ---

    def summary(self) -> dict:
        return {
            "energy": round(self.energy, 1),
            "safety": round(self.safety, 1),
            "curiosity": round(self.curiosity, 1),
            "social_need": round(self.social_need, 1),
        }
