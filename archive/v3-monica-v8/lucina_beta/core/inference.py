"""Active Inference: EFE (Expected Free Energy) による行動選択

EFE = Pragmatic Value + Risk - Epistemic Value - Need Satisfaction

  Pragmatic Value:  期待される実用的価値（base EV）
  Risk:             予測の不確実性（未知の行動ほど高い）
  Epistemic Value:  行動によって得られる情報量（未知ほど高い）
  Need Satisfaction: 内部状態の欲求を満たす価値

Phase 2 では EFE が最も低い行動を選ぶことで、
「既知の安全な行動」と「未知だが情報価値が高い行動」の
トレードオフを制御する。
"""

import math

from .world_model import WorldModel
from .internal_state import InternalState


def compute_efe(
    action: str,
    world_model: WorldModel,
    internal_state: InternalState | None = None,
    goal_manager=None,
    values=None,
    self_model=None,
    epistemic_weight: float = 2.0,
) -> float:
    """行動の EFE (Expected Free Energy) を計算する。

    値が低いほど良い行動。

    EFE = Risk - Pragmatic - epistemic_weight * Epistemic - Need - Value - Goal - SelfConsistency

    Risk:        不確実な行動のコスト（未知ほど高い）
    Pragmatic:   期待される実用的価値
    Epistemic:   情報獲得量（未知ほど価値が高い）
    Need:        内部状態の欲求
    Value:       価値観に基づくボーナス（Values module）
    Goal:        目標に対する意図ボーナス（Goals module）
    SelfConsistency: 自己モデルとの一貫性（Self Model）
    """
    # Pragmatic Value
    pragmatic = world_model.expected_value(action)

    # Uncertainty
    uncertainty = world_model.uncertainty(action)
    risk = uncertainty
    epistemic = uncertainty * epistemic_weight

    # Need Satisfaction
    need_value = 0.0
    if internal_state is not None:
        need_value = internal_state.need_bonus(action, pragmatic)

    # Value Bonus (Phase 8+: 価値観からの寄与)
    value_bonus = 0.0
    if values is not None:
        value_bonus = values.value_bonus(action)

    # Goal / Intention Bonus (Phase Goals層)
    goal_bonus = 0.0
    if goal_manager is not None:
        goal_bonus = goal_manager.intention_bonus(action)

    # Self Consistency Bonus (Phase 7+: 自己モデルと一致する行動)
    self_bonus = 0.0
    if self_model is not None:
        predicted_success = self_model.predict_self_success(action)
        self_bonus = predicted_success * 0.2  # 自信がある行動ほど選ばれやすい

    # 拡張EFE
    efe = risk - pragmatic - epistemic - need_value - value_bonus - goal_bonus - self_bonus

    return efe


def select_action_efe(
    world_model: WorldModel,
    internal_state: InternalState | None = None,
    temperature: float = 1.0,
    epistemic_weight: float = 2.0,
    candidate_actions: list[str] | None = None,
    goal_manager=None,
    values=None,
    self_model=None,
) -> str:
    """拡張EFEで行動選択。全モジュールからの寄与を考慮する。

    Parameters
    ----------
    candidate_actions : list[str] | None
        評価対象の行動リスト。None の場合は world_model.actions を使う。
    goal_manager : GoalManager | None
        Goals/Intentions層からのボーナス。
    values : ValueSystem | None
        価値観からのボーナス。
    self_model : SelfModel | None
        自己モデルからの一貫性ボーナス。
    """
    actions = candidate_actions if candidate_actions is not None else world_model.actions
    efes = [
        compute_efe(a, world_model, internal_state, goal_manager, values, self_model, epistemic_weight)
        for a in actions
    ]

    exp_values = [math.exp(-e / temperature) for e in efes]
    total = sum(exp_values)
    if total == 0 or not math.isfinite(total):
        import random
        return random.choice(actions)
    probs = [ev / total for ev in exp_values]
    import random
    return random.choices(actions, weights=probs)[0]




def efe_summary(
    world_model: WorldModel,
    internal_state: InternalState | None = None,
    epistemic_weight: float = 2.0,
    goal_manager=None,
    values=None,
    self_model=None,
) -> list[dict]:
    """全行動の拡張EFE内訳を表示用に返す（compute_efeと完全に一致）。"""
    rows = []
    for a in world_model.actions:
        efe = compute_efe(a, world_model, internal_state, goal_manager, values, self_model, epistemic_weight)
        pragmatic = world_model.expected_value(a)
        uncertainty = world_model.uncertainty(a)
        rows.append({
            "action": a,
            "pragmatic": round(pragmatic, 3),
            "risk": round(uncertainty, 3),
            "epistemic": round(uncertainty * epistemic_weight, 3),
            "efe": round(efe, 3),
        })
    return rows
