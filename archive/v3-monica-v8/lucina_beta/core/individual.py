"""Individual Genesis: 個体生成システム

Lucina Core + Initial Conditions + World + Time → 一個体

同じCore + 異なるConfig → 異なる個体
人格を直接書かずに、個体差を生み出す。
"""

from typing import Any, Optional

from .agent import Agent
from .self_model import SelfModel
from .values import ValueSystem
from .identity import Identity
from .memory import Memory
from .development import Development
from .metacognition import Metacognition
from .goals import GoalManager
from .persistence import Persistence, IdleCycle


class IndividualConfig:
    """一個体の初期条件を定義する。"""

    def __init__(
        self,
        name: str = "Individual",
        initial_abilities: dict[str, float] | None = None,
        initial_values: dict[str, float] | None = None,
        initial_traits: dict[str, float] | None = None,
        temperature: float = 1.0,
        exploration_bias: float = 0.3,
        social_bias: float = 0.3,
    ):
        self.name = name
        self.initial_abilities = initial_abilities or {
            "prediction": 0.3,
            "social": 0.3,
            "exploration": 0.3,
        }
        self.initial_values = initial_values or {
            "exploration": 0.3,
            "safety": 0.5,
            "social_bond": 0.3,
            "knowledge": 0.4,
            "efficiency": 0.4,
            "novelty": 0.3,
        }
        self.initial_traits = initial_traits or {
            "curious": 0.3,
            "cautious": 0.3,
            "social": 0.3,
            "persistent": 0.3,
            "adaptive": 0.3,
        }
        self.temperature = temperature
        self.exploration_bias = exploration_bias
        self.social_bias = social_bias


class Individual:
    """一個体を表す。全モジュールを統合したオーケストレーション層。

    Lucina Core + IndividualConfig から生成される。
    CLI や REPL はこのクラスをラップする。
    """

    def __init__(
        self,
        config: IndividualConfig,
        agent: Agent | None = None,
        world: Any = None,
    ):
        self.config = config
        self.name = config.name
        self.world = world

        # Core components
        self.self_model = SelfModel()
        self.values = ValueSystem()
        self.goals = GoalManager()
        self.memory = Memory()
        self.identity = Identity()
        self.metacognition = Metacognition()
        self.development = Development()

        # Agent (with module references for extended EFE)
        self.agent = agent or Agent(
            temperature=config.temperature,
            use_needs=True,
            use_efe=True,
            goal_manager=self.goals,
            values=self.values,
            self_model=self.self_model,
        )
        self.persistence = Persistence()
        self.idle_cycle = IdleCycle(self.memory)

        # Apply initial config
        self._apply_config(config)

        # Internal state
        self._step_count = 0

    def _apply_config(self, config: IndividualConfig) -> None:
        """初期条件を各モジュールに適用する。"""
        for key, val in config.initial_abilities.items():
            if key in self.self_model.abilities:
                self.self_model.abilities[key] = val
        for key, val in config.initial_values.items():
            if key in self.values.weights:
                self.values.weights[key] = val
        for key, val in config.initial_traits.items():
            if key in self.identity.traits:
                self.identity.traits[key] = val

    def step(self) -> dict:
        """1サイクル: 全モジュールを統合更新する。"""
        self._step_count += 1

        # Agent step
        entry = self.agent.step(self.world)

        # Development
        self.development.add_experience()

        # Memory
        self.memory.store_from_history(entry)
        
        # Memory Consolidation (Idle Cycle)
        self.idle_cycle.step()

        # Self Model
        success = entry.get("outcome") in ("food", "positive")
        self.self_model.record_action(entry["action"], entry["outcome"], success)
        prediction = entry.get("prediction", {})
        if prediction:
            max_prob = max(prediction.values()) if prediction else 0.5
            self.self_model.record_prediction_outcome(
                entry["action"], entry["outcome"], max_prob
            )

        # Values
        self.values.update_from_experience(
            entry["action"], entry["outcome"], entry["pe"],
            entry.get("internal"),
        )
        ev = entry.get("ev", 0)
        self.values.update_preference(entry["action"], max(0, min(1, ev + 0.5)))

        # Goals / Intentions / Desires
        self.goals.update(
            value_weights=self.values.weights,
            internal_state=entry.get("internal"),
            action=entry["action"],
            outcome=entry["outcome"],
        )
        self.goals.execute_intention(entry["action"])

        # Identity
        self.identity.record_experience(
            entry["action"], entry["outcome"], entry["pe"],
            values=self.values.weights,
            self_stats=self.self_model.summary(),
        )

        # Metacognition
        self.metacognition.update(
            self.self_model.prediction_accuracy,
            self.self_model.recent_prediction_accuracy,
            entry.get("internal"),
        )

        # Full log entry
        full_entry = dict(entry)
        full_entry["step"] = self._step_count
        full_entry["development"] = self.development.summary()
        full_entry["memory"] = self.memory.summary()
        full_entry["self_model"] = self.self_model.summary()
        full_entry["values"] = self.values.summary()
        full_entry["identity"] = self.identity.summary()
        full_entry["metacognition"] = self.metacognition.summary()
        full_entry["goals"] = self.goals.summary()
        full_entry["idle"] = self.idle_cycle.summary()

        return full_entry

    def run(self, n_steps: int = 10) -> list[dict]:
        """指定されたステップ数だけ自律実行する。"""
        logs = []
        for _ in range(n_steps):
            entry = self.step()
            logs.append(entry)
        return logs

    def summary(self) -> dict:
        return {
            "name": self.name,
            "temperature": self.config.temperature,
            "development": self.development.summary(),
            "self_model": self.self_model.summary(),
            "values": self.values.summary(),
            "identity": self.identity.summary(),
            "memory": self.memory.summary(),
            "metacognition": self.metacognition.summary(),
            "total_steps": self._step_count,
        }
