"""Experiment B: 初期条件（temperature）の違いによる行動分布の差"""

from core.agent import Agent
from world.mock_world import MockWorld
from cli.display import show_header


def run():
    """temperature の違いが探索/活用バイアスに与える影響を測定する。

    Agent A: temperature=0.5（活用バイアス: 期待値の高い行動を選びやすい）
    Agent B: temperature=2.0（探索バイアス: ランダム性が高い）
    """
    show_header("Experiment B: Temperature → Exploration/Exploitation Bias")

    world = MockWorld(seed=42)
    agent_low = Agent(temperature=0.5)    # 活用バイアス
    agent_high = Agent(temperature=2.0)   # 探索バイアス

    # まず学習フェーズ（両者同じ経験）
    for _ in range(500):
        agent_low.step(world)
        agent_high.step(world)

    print(f"\n  Agent A: temperature = 0.5 (exploitation)")
    print(f"  Agent B: temperature = 2.0 (exploration)")
    print(f"  Both agents experienced 500 trials in the same world.")

    print(f"\n  Beliefs after learning:")
    print(f"  Agent A (temp=0.5):")
    for a in agent_low.world_model.actions:
        p = agent_low.world_model.predict(a)
        print(f"    {a}: EV={agent_low.world_model.expected_value(a):.2f}  "
              f"food={p['food']:.2f}  danger={p['danger']:.2f}")
    print(f"  Agent B (temp=2.0):")
    for a in agent_high.world_model.actions:
        p = agent_high.world_model.predict(a)
        print(f"    {a}: EV={agent_high.world_model.expected_value(a):.2f}  "
              f"food={p['food']:.2f}  danger={p['danger']:.2f}")

    # 測定フェーズ
    n_trials = 500
    agent_low.reset_history()
    agent_high.reset_history()

    def measure_distribution(agent, world, n: int) -> dict[str, int]:
        counts: dict[str, int] = {a: 0 for a in world.actions()}
        for _ in range(n):
            action = agent.select_action()
            outcome = world.step(action)
            agent.world_model.update(action, outcome)
            counts[action] += 1
        return counts

    counts_low = measure_distribution(agent_low, world, n_trials)
    counts_high = measure_distribution(agent_high, world, n_trials)

    # 結果表示
    print(f"\n  Action selection over {n_trials} trials:")
    print(f"  Agent A (temp=0.5):")
    for a in sorted(counts_low):
        pct = counts_low[a] / n_trials * 100
        bar = "█" * int(pct / 2)
        print(f"    {a}: {counts_low[a]:4d} ({pct:5.1f}%) {bar}")
    print(f"  Agent B (temp=2.0):")
    for a in sorted(counts_high):
        pct = counts_high[a] / n_trials * 100
        bar = "█" * int(pct / 2)
        print(f"    {a}: {counts_high[a]:4d} ({pct:5.1f}%) {bar}")

    # 判定: 低温度の方が最善手への集中度が高い
    max_low = max(counts_low.values()) / n_trials
    max_high = max(counts_high.values()) / n_trials
    more_extreme = max_low > max_high + 0.05

    print(f"\n  Max selection rate:")
    print(f"    Agent A (temp=0.5): {max_low*100:.1f}%")
    print(f"    Agent B (temp=2.0): {max_high*100:.1f}%")

    print(f"\n  Result: {'✅ LOW TEMP = MORE EXPLOITATION' if more_extreme else '❌ NO CLEAR EFFECT'}")
    if more_extreme:
        print("  → Lower temperature concentrates choices on high-EV actions.")

    return more_extreme
