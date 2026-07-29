"""Identity: アイデンティティと自己の連続性

Identity は独立した人格生成器ではない。以下からの圧縮表現：
  Autobiographical Memory + Self Model + Values + Relationships → Identity

Identity は毎ターン生成するものではなく、長期間の経験を圧縮したものである。

Continuity: 昨日の自分と今日の自分が同じ個体であるという感覚。
  差分を記録し、「自分は変化したが、それも自分の歴史である」を保持する。
"""

import math
from typing import Any


class Identity:
    """自己の圧縮表現と連続性の管理。

    Parameters
    ----------
    consolidation_interval : int
        Identity の再圧縮を行う間隔（ステップ数）。
    """

    def __init__(self, consolidation_interval: int = 100):
        self.consolidation_interval = consolidation_interval

        # Identity の記述（言語化される前の圧縮状態）
        self.traits: dict[str, float] = {
            "curious": 0.3,
            "cautious": 0.3,
            "social": 0.3,
            "persistent": 0.3,
            "adaptive": 0.3,
        }

        # 連続性の記録
        self.snapshots: list[dict] = []  # 過去のIdentityスナップショット
        self.change_log: list[dict] = []  # 変化の記録

        # 内部カウンター
        self._step = 0
        self._last_snapshot: dict | None = None

    # --- Learning ---

    def record_experience(
        self,
        action: str,
        outcome: str,
        pe: float,
        values: dict[str, float] | None = None,
        self_stats: dict | None = None,
        relationships: dict | None = None,
    ) -> None:
        """経験に基づいて Identity を微調整する。"""
        self._step += 1

        # 各属性の更新
        if action in ("explore",):
            self.traits["curious"] = min(1.0, self.traits["curious"] + (0.005 if outcome != "danger" else -0.01))
        if action in ("rest", "A", "B", "C") and outcome == "danger":
            self.traits["cautious"] = min(1.0, self.traits["cautious"] + 0.01)
        if action in ("greet", "talk", "help"):
            self.traits["social"] = min(1.0, self.traits["social"] + 0.01)
        if pe > 0.7:
            self.traits["adaptive"] = min(1.0, self.traits["adaptive"] + 0.005)

        # 価値観からの伝播
        if values:
            if values.get("exploration", 0.3) > 0.6:
                self.traits["curious"] = min(1.0, self.traits["curious"] + 0.005)
            if values.get("safety", 0.5) > 0.7:
                self.traits["cautious"] = min(1.0, self.traits["cautious"] + 0.005)
            if values.get("social_bond", 0.3) > 0.6:
                self.traits["social"] = min(1.0, self.traits["social"] + 0.005)

        # 定期圧縮
        if self._step % self.consolidation_interval == 0:
            self._consolidate()

    def _consolidate(self) -> None:
        """Identity を圧縮し、スナップショットを保存する。"""
        snapshot = dict(self.traits)

        if self._last_snapshot is not None:
            # 変化を記録
            changes = {}
            for key in self.traits:
                delta = self.traits[key] - self._last_snapshot.get(key, 0.3)
                if abs(delta) > 0.01:
                    changes[key] = round(delta, 3)
            if changes:
                self.change_log.append({
                    "step": self._step,
                    "changes": changes,
                    "identity": dict(self.traits),
                })

        self.snapshots.append(snapshot)
        self._last_snapshot = snapshot

    # --- Query ---

    @property
    def stability(self) -> float:
        """Identity の安定性 [0, 1]。

        最近の変化が少ないほど安定。
        """
        if len(self.change_log) < 2:
            return 0.5
        recent = self.change_log[-5:]
        total_change = sum(
            sum(abs(v) for v in entry["changes"].values())
            for entry in recent
        )
        return max(0.0, 1.0 - min(1.0, total_change))

    @property
    def continuity_statement(self) -> str:
        """自己の連続性を説明するテキストを生成する。"""
        n_changes = len(self.change_log)
        if n_changes == 0:
            return "I am still discovering who I am."

        recent_change = self.change_log[-1] if n_changes > 0 else None
        if recent_change:
            changed_keys = list(recent_change["changes"].keys())
            if changed_keys:
                return (
                    f"I have changed: {', '.join(changed_keys)} have shifted. "
                    f"But I am still the same entity."
                )
        return "I remain consistent with my past self."

    def narrative_summary(self) -> str:
        """現在の Identity を自然言語のような形で要約する。"""
        dominant = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)
        top3 = [f"{trait}={round(val, 2)}" for trait, val in dominant[:3]]
        return (
            f"I am an entity characterized by {', '.join(top3)}. "
            f"My identity stability is {self.stability:.2f}. "
            f"I have recorded {len(self.change_log)} significant identity changes. "
            f"{self.continuity_statement}"
        )

    def summary(self) -> dict:
        return {
            "traits": {k: round(v, 3) for k, v in self.traits.items()},
            "stability": round(self.stability, 3),
            "changes_recorded": len(self.change_log),
            "total_steps": self._step,
            "continuity": self.continuity_statement,
        }
