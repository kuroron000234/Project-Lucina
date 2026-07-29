"""Goals / Intentions / Desires — 「私は何をしようとしているのか」層

Desire:    「本当はこうなってほしい」— 価値観と内部状態から自然に生まれる
Goal:      「世界を理解したい」— Desireから形成される比較的安定した方向性
Intention: 「今、この行動を選ぼうとしている」— Goalと現在の状態から生成される具体的意図

この階層がないと、行動が毎回「EFEが最小だからSEARCH」という数値計算だけになる。
"""

import math
import time
from typing import Optional


class Desire:
    """価値観と内部状態から生まれる欲求。"""

    def __init__(self, name: str, source_value: str, weight: float = 0.5):
        self.name = name
        self.source_value = source_value  # 対応するValueSystemのキー
        self.weight = weight  # [0, 1] この欲求の強さ
        self.urgency: float = 0.0  # [0, 1] 緊急度（時間経過で上昇）
        self.satisfied: bool = False

    def update(self, dt: float = 1.0) -> None:
        """時間経過で緊急度が上昇する。"""
        if not self.satisfied:
            self.urgency = min(1.0, self.urgency + 0.01 * dt)
        else:
            self.urgency = max(0.0, self.urgency - 0.1)
            self.satisfied = False

    def satisfy(self) -> None:
        """欲求が満たされた。"""
        self.satisfied = True
        self.urgency = 0.0


class Goal:
    """Desireから形成される安定した方向性。"""

    def __init__(self, description: str, source_desire: str, priority: float = 0.5):
        self.description = description
        self.source_desire = source_desire
        self.priority = priority  # [0, 1] 優先度
        self.created_at = time.time()
        self.progress: float = 0.0  # [0, 1] 進捗
        self.completed: bool = False

    def update(self, action: str, outcome: str) -> None:
        """行動結果に基づいて進捗を更新する。"""
        # 簡易的な進捗更新（本実装ではより洗練されたロジックが必要）
        if not self.completed:
            self.progress = min(1.0, self.progress + 0.02)

    def mark_complete(self) -> None:
        self.completed = True
        self.progress = 1.0


class Intention:
    """Goalと現在の状態から生成される具体的な意図。"""

    def __init__(self, action: str, reason: str, expected_value: float = 0.0):
        self.action = action
        self.reason = reason
        self.expected_value = expected_value
        self.created_at = time.time()
        self.executed: bool = False

    def execute(self) -> None:
        self.executed = True


class GoalManager:
    """Desires → Goals → Intentions の階層を管理する。"""

    def __init__(self):
        # Desires (価値観から生まれる欲求)
        self.desires: dict[str, Desire] = {
            "explore": Desire("Explore the unknown", "exploration", 0.5),
            "stay_safe": Desire("Stay safe", "safety", 0.6),
            "connect": Desire("Connect with others", "social_bond", 0.5),
            "learn": Desire("Learn new things", "knowledge", 0.6),
            "be_efficient": Desire("Be efficient", "efficiency", 0.4),
        }

        # Goals (アクティブな目標)
        self.active_goals: list[Goal] = []

        # Intentions (実行予定の意図)
        self.pending_intentions: list[Intention] = []
        self.executed_intentions: list[Intention] = []

        # 内部状態
        self._last_update = time.time()

    # --- Update Cycle ---

    def update(
        self,
        value_weights: dict[str, float],
        internal_state: Optional[dict] = None,
        action: str = "",
        outcome: str = "",
    ) -> None:
        """全階層を更新する。"""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # 1. Desires を更新
        self._update_desires(value_weights, internal_state, dt)

        # 2. Goals を更新
        self._update_goals(action, outcome)

        # 3. 新たな Goal を生成
        self._form_goals()

        # 4. Intentions をクリーンアップ
        self._cleanup_intentions()

    def _update_desires(
        self, value_weights: dict[str, float],
        internal_state: Optional[dict], dt: float,
    ) -> None:
        """価値観と内部状態からDesiresを更新する。"""
        for desire in self.desires.values():
            # 価値観の重みを反映
            value_weight = value_weights.get(desire.source_value, 0.3)
            desire.weight = 0.5 + value_weight * 0.5  # [0.5, 1.0] にマッピング

            # 内部状態による変調
            if internal_state and desire.name == "explore":
                curiosity = internal_state.get("curiosity", 50)
                desire.weight = min(1.0, desire.weight + curiosity / 200.0)
            if internal_state and desire.name == "connect":
                social_need = internal_state.get("social_need", 30)
                desire.weight = min(1.0, desire.weight + social_need / 200.0)

            desire.update(dt)

    def _update_goals(self, action: str, outcome: str) -> None:
        """Goals の進捗を更新する。"""
        for goal in self.active_goals[:]:
            goal.update(action, outcome)
            if goal.completed:
                self.active_goals.remove(goal)

    def _form_goals(self) -> None:
        """強いDesiresからGoalsを形成する。"""
        for desire in self.desires.values():
            if desire.urgency > 0.6 and desire.weight > 0.6:
                # 既に同じようなGoalがないか確認
                already_exists = any(
                    g.source_desire == desire.name for g in self.active_goals
                )
                if not already_exists:
                    goal_desc = self._describe_goal(desire)
                    self.active_goals.append(
                        Goal(goal_desc, desire.name, priority=desire.urgency * desire.weight)
                    )

    def _describe_goal(self, desire: Desire) -> str:
        """DesireからGoalの説明文を生成する。"""
        descriptions = {
            "explore": "Explore unfamiliar topics and gather new information",
            "stay_safe": "Maintain stability and avoid unnecessary risks",
            "connect": "Engage in social interaction and build relationships",
            "learn": "Deepen understanding of current topics and questions",
            "be_efficient": "Optimize actions for maximum value with minimum cost",
        }
        return descriptions.get(desire.name, f"Act on {desire.name}")

    def _cleanup_intentions(self) -> None:
        """実行済みまたは古いIntentionsを整理する。"""
        now = time.time()
        self.executed_intentions = [
            i for i in self.executed_intentions
            if now - i.created_at < 300  # 5分以内のものだけ保持
        ]

    # --- Intention Formation ---

    def form_intention(
        self, action: str, reason: str, expected_value: float = 0.0,
    ) -> Intention:
        """具体的な行動意図を形成する。"""
        intention = Intention(action, reason, expected_value)
        self.pending_intentions.append(intention)
        return intention

    def execute_intention(self, action: str) -> bool:
        """意図を実行済みとしてマークする。"""
        for intention in self.pending_intentions[:]:
            if intention.action == action and not intention.executed:
                intention.execute()
                self.pending_intentions.remove(intention)
                self.executed_intentions.append(intention)
                return True
        return False

    def intention_bonus(self, action: str) -> float:
        """実行予定の意図がある行動にボーナスを与える。"""
        for intention in self.pending_intentions:
            if intention.action == action and not intention.executed:
                return intention.expected_value * 0.3
        return 0.0

    # --- Query ---

    def active_goal(self) -> Optional[Goal]:
        """最高優先度の未完了Goalを返す。"""
        active = [g for g in self.active_goals if not g.completed]
        if not active:
            return None
        return max(active, key=lambda g: g.priority)

    def strongest_desire(self) -> Optional[tuple[str, float]]:
        """最も強いDesireを (name, strength) で返す。"""
        best = max(self.desires.items(), key=lambda x: x[1].weight * x[1].urgency)
        strength = best[1].weight * best[1].urgency
        if strength > 0.3:
            return (best[0], strength)
        return None

    def summary(self) -> dict:
        return {
            "desires": {
                k: {
                    "weight": round(v.weight, 2),
                    "urgency": round(v.urgency, 2),
                }
                for k, v in self.desires.items()
            },
            "active_goals": [
                {
                    "description": g.description[:40],
                    "priority": round(g.priority, 2),
                    "progress": round(g.progress, 2),
                }
                for g in self.active_goals
            ],
            "pending_intentions": [
                {"action": i.action, "reason": i.reason[:30]}
                for i in self.pending_intentions if not i.executed
            ],
        }
