"""Experiment 1: 内部状態 / Needs の検証

真の確率を直接設定することで learning noise を排除し、
need_bonus による状態依存行動だけを測定する。

検証:
  1. energy↓ → rest 選択率↑
  2. curiosity↑ → explore 選択率↑
"""

import math
import random

from core.agent import Agent
from world.mock_world import MockWorld
from cli.display import show_header


def _softmax_prob(values: list[float], temperature: float) -> list[float]:
    """ソフトマックス確率を計算。"""
    exp_v = [math.exp(v / temperature) for v in values]
    total = sum(exp_v)
    return [ev / total for ev in exp_v]


def run():
    show_header("Experiment 1: Internal State / Needs — State-Dependent Behavior")

    world = MockWorld(seed=42, phase=1)

    # 真の確率を WorldModel に直接設定（learning noise を排除）
    agent = Agent(use_needs=True, temperature=0.8)
    for action in world.actions():
        agent.world_model.load_true_probabilities(
            action, world.true_probabilities(action), samples=500
        )

    print("\n  True probabilities loaded into WorldModel:")
    for a in agent.world_model.actions:
        p = agent.world_model.predict(a)
        ev = agent.world_model.expected_value(a)
        print(f"    {a:8s}: base_EV={ev:+.2f}  food={p['food']:.2f}  "
              f"nothing={p['nothing']:.2f}  danger={p['danger']:.2f}")

    # --- Phase 2: 高エネルギー状態 ---
    agent.internal_state.energy = 100.0
    agent.internal_state.curiosity = 10.0
    print("\n  Phase 2: energy=100, curiosity=10")
    print("  → Expect: mostly A (highest base EV=+0.70)")

    counts_high: dict[str, int] = {a: 0 for a in agent.world_model.actions}
    for _ in range(200):
        action = agent.select_action()
        # 行動したが学習はしない（確率は固定のまま）
        counts_high[action] += 1

    rest_high = counts_high.get("rest", 0)
    explore_high = counts_high.get("explore", 0)

    print("\n  Action distribution (high energy):")
    for a in sorted(counts_high):
        pct = counts_high[a] / 2
        bar = "█" * int(pct)
        print(f"    {a:8s}: {counts_high[a]:3d}/200 ({counts_high[a]/2:.1f}%) {bar}")

    # --- Phase 3: 低エネルギー + 高好奇心状態 ---
    agent.internal_state.energy = 5.0
    agent.internal_state.curiosity = 90.0
    print("\n  Phase 3: energy=5, curiosity=90")
    print("  → Expect: rest (need_bonus=+2.85) and explore (need_bonus=+1.8)")
    print("    dominate over A (base_EV=+0.70)")

    # 理論値を表示
    print("\n  Theoretical effective EVs (with need_bonus):")
    for a in agent.world_model.actions:
        base = agent.world_model.expected_value(a)
        bonus = agent.internal_state.need_bonus(a, base)
        print(f"    {a:8s}: base={base:+.2f} + bonus={bonus:+.2f} = {base+bonus:+.2f}")

    probs = _softmax_prob(
        [agent.world_model.expected_value(a) + agent.internal_state.need_bonus(a, 0)
         for a in agent.world_model.actions],
        agent.temperature
    )
    print("\n  Theoretical action probabilities (softmax):")
    for a, p in zip(agent.world_model.actions, probs):
        bar = "█" * int(p * 50)
        print(f"    {a:8s}: {p*100:.1f}% {bar}")

    counts_low: dict[str, int] = {a: 0 for a in agent.world_model.actions}
    for _ in range(200):
        action = agent.select_action()
        counts_low[action] += 1

    rest_low = counts_low.get("rest", 0)
    explore_low = counts_low.get("explore", 0)

    print("\n  Action distribution (low energy, high curiosity):")
    for a in sorted(counts_low):
        pct = counts_low[a] / 2
        bar = "█" * int(pct)
        print(f"    {a:8s}: {counts_low[a]:3d}/200 ({counts_low[a]/2:.1f}%) {bar}")

    # 検証
    rest_increased = rest_low > rest_high + 20
    explore_increased = explore_low > explore_high + 10

    print(f"\n  {'='*55}")
    print(f"  Results:")
    print(f"    Rest selection:     {rest_high}/200 → {rest_low}/200  "
          f"{'✅' if rest_increased else '❌'}")
    print(f"    Explore selection:  {explore_high}/200 → {explore_low}/200  "
          f"{'✅' if explore_increased else '❌'}")
    print(f"  {'='*55}")

    success = rest_increased and explore_increased
    print(f"\n  Overall: {'✅ STATE-DEPENDENT BEHAVIOR CONFIRMED' if success else '❌ NEEDS TUNING'}")

    return success
