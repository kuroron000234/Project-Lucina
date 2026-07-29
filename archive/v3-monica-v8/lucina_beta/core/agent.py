"""Agent: 予測→行動→観測→学習のメインループ

Phase 0:  最小ループ（InternalState なし）
Phase 1+: InternalState を統合し、Needs が行動選択に影響
Phase 2+: EFE (Active Inference) による行動選択
Phase 3+: LLM Cognitive Layer による候補生成・予測・解釈
"""

import math
import random

from .world_model import WorldModel
from .internal_state import InternalState
from .inference import compute_efe, select_action_efe
from .llm import LLMCognitiveLayer


class Agent:
    """予測学習エージェント。"""

    def __init__(
        self,
        actions: list[str] | None = None,
        temperature: float = 1.0,
        use_needs: bool = False,
        use_llm: bool = False,
        use_efe: bool = False,
        goal_manager=None,
        values=None,
        self_model=None,
    ):
        self.world_model = WorldModel(actions)
        self.internal_state = InternalState() if use_needs else None
        self.llm = LLMCognitiveLayer(backend="ollama") if use_llm else None
        self.temperature = temperature
        self.use_needs = use_needs
        self.use_efe = use_efe
        self.use_llm = use_llm
        self.goal_manager = goal_manager
        self.values = values
        self.self_model = self_model
        self.history: list[dict] = []

    # --- Action Selection ---

    def select_action(self, context: str = "") -> str:
        """LLM 候補 + 拡張EFE で行動選択。

        全モジュール（Values/Goals/SelfModel）からの寄与をEFEに反映する。
        """
        return self._select_action_with_state(
            self.world_model, self.internal_state, self.temperature,
            self.use_efe, self.llm, context,
            goal_manager=self.goal_manager,
            values=self.values,
            self_model=self.self_model,
        )

    @staticmethod
    def _select_action_with_state(
        wm, internal_state, temperature,
        use_efe: bool = False, llm=None, context: str = "",
        goal_manager=None, values=None, self_model=None,
    ):
        """内部状態と全モジュールを考慮した行動選択（静的メソッド）。"""
        actions = list(wm.actions)

        # Phase 3+: LLM が候補を生成する
        if llm is not None and llm.is_available() and context:
            llm_candidates = llm.generate_candidates(context, available_actions=actions)
            if llm_candidates:
                filtered = [a for a in llm_candidates if a in actions or a in wm.counts]
                if filtered:
                    actions = filtered

        # Phase 2+: 拡張EFE による選択（Values/Goals/SelfModel を考慮）
        if use_efe:
            return select_action_efe(
                wm, internal_state, temperature,
                candidate_actions=actions,
                goal_manager=goal_manager,
                values=values,
                self_model=self_model,
            )

        # Phase 0-1: EV + NeedBonus によるソフトマックス選択
        values = []
        for a in actions:
            base_ev = wm.expected_value(a)
            if internal_state is not None:
                base_ev += internal_state.need_bonus(a, base_ev)
            values.append(base_ev)

        exp_values = [math.exp(v / temperature) for v in values]
        total = sum(exp_values)
        if total == 0 or not math.isfinite(total):
            return random.choice(actions)
        probs = [ev / total for ev in exp_values]
        return random.choices(actions, weights=probs)[0]

    def _softmax_select(self) -> str:
        """内部互換用。select_action() に委譲。"""
        return self.select_action()

    # --- Main Loop ---

    @staticmethod
    def _resolve_outcome(raw_outcome):
        """World の出力を統一された文字列に解決する。

        MockWorld は文字列を返し、DDLCWorld は dict を返すので、
        両方に対応する。
        """
        if isinstance(raw_outcome, dict):
            return raw_outcome.get("outcome", str(raw_outcome))
        return str(raw_outcome)

    def step(self, world) -> dict:
        """1サイクル: predict → act → observe → learn → update internal state"""
        for a in world.actions():
            self.world_model.add_action(a)

        # Build context for LLM
        context = self._build_context(world)

        action = self.select_action(context=context)
        prediction = self.world_model.predict(action)
        raw_outcome = world.step(action)
        outcome = self._resolve_outcome(raw_outcome)
        pe = self.world_model.surprise(action, outcome)
        ev_before = self.world_model.expected_value(action)
        if self.internal_state is not None:
            ev_before += self.internal_state.need_bonus(action, ev_before)

        self.world_model.update(action, outcome)
        if self.internal_state is not None:
            self.internal_state.update(action, outcome)
            self.internal_state.tick()

        # Phase 3+: LLM で結果を解釈
        interpretation = ""
        if self.llm is not None and self.llm.is_available():
            interpretation = self.llm.interpret_result(action, outcome, prediction, context)

        entry = {
            "action": action,
            "outcome": outcome,
            "raw_outcome": raw_outcome if isinstance(raw_outcome, dict) else None,
            "prediction": dict(prediction),
            "pe": pe,
            "ev": ev_before,
            "interpretation": interpretation,
            "internal": self.internal_state.summary() if self.internal_state else None,
        }
        self.history.append(entry)
        return entry

    def _build_context(self, world) -> str:
        """LLM に渡す現在の状況説明を生成する。"""
        if not self.use_llm or self.llm is None:
            return ""

        summary = self.world_model.summary()
        parts = ["Current state:"]
        for action, data in summary.items():
            parts.append(
                f"  {action}: EV={data['ev']}, samples={data['samples']}, "
                f"pred={data['predict']}"
            )
        if self.internal_state:
            istate = self.internal_state.summary()
            parts.append(
                f"  Internal: energy={istate['energy']}, "
                f"curiosity={istate['curiosity']}, "
                f"safety={istate['safety']}"
            )
        return "\n".join(parts)

    def run(self, world, n_steps: int) -> list[dict]:
        """n_steps 回ループを回す。"""
        for _ in range(n_steps):
            self.step(world)
        return self.history

    # --- Forced Experience (実験用) ---

    def force_experience(self, world, experiences: list[str]):
        """指定された順序で行動を強制実行する。"""
        for action in experiences:
            raw_outcome = world.step(action)
            outcome = self._resolve_outcome(raw_outcome)
            pred_before = self.world_model.predict(action)
            actual_pe = 1.0 - pred_before[outcome]
            ev_before = self.world_model.expected_value(action)
            self.world_model.update(action, outcome)
            if self.internal_state is not None:
                self.internal_state.update(action, outcome)
                self.internal_state.tick()
            self.history.append({
                "action": action,
                "outcome": outcome,
                "prediction": dict(pred_before),
                "pe": actual_pe,
                "ev": ev_before,
                "forced": True,
            })

    def reset_history(self):
        """履歴のみリセット（信念は保持）。"""
        self.history = []
