"""Experiment A: 経験履歴による行動分布の差"""

from core.agent import Agent
from world.mock_world import MockWorld
from cli.display import show_header, show_beliefs


def run():
    """異なる経験分布 → 異なる行動分布 を検証する。

    Agent A: Aを50回, Bを10回, Cを10回 強制経験
    Agent B: Bを50回, Aを10回, Cを10回 強制経験
    → 自由選択時の行動分布を比較
    """
    show_header("Experiment A: Experience History → Action Distribution")

    # 各Agentに独立した世界インスタンスを割り当て
    agent_a = Agent()
    agent_b = Agent()
    world_a = MockWorld(seed=99)
    world_b = MockWorld(seed=199)

    # Phase 1: 異なる経験分布
    exp_a = ["A"] * 50 + ["B"] * 10 + ["C"] * 10
    exp_b = ["B"] * 50 + ["A"] * 10 + ["C"] * 10

    print(f"  Agent A: A=50, B=10, C=10")
    print(f"  Agent B: B=50, A=10, C=10")

    for action in exp_a:
        outcome = world_a.step(action)
        agent_a.world_model.update(action, outcome)
    for action in exp_b:
        outcome = world_b.step(action)
        agent_b.world_model.update(action, outcome)

    print("\n  After forced experiences:")
    print("  Agent A:")
    show_beliefs(agent_a.world_model)
    print("  Agent B:")
    show_beliefs(agent_b.world_model)

    # Phase 2: 自由選択（各Agent独立した世界で行動）
    n_trials = 200
    print(f"\n  Free choice phase ({n_trials} trials, independent worlds)...")

    counts_a: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    counts_b: dict[str, int] = {"A": 0, "B": 0, "C": 0}

    for _ in range(n_trials):
        act_a = agent_a.select_action()
        out_a = world_a.step(act_a)
        agent_a.world_model.update(act_a, out_a)
        counts_a[act_a] += 1

        act_b = agent_b.select_action()
        out_b = world_b.step(act_b)
        agent_b.world_model.update(act_b, out_b)
        counts_b[act_b] += 1

    # 結果表示
    def fmt_counts(c: dict[str, int], total: int) -> str:
        parts = [f"{k}={v} ({v/total*100:.0f}%)" for k, v in sorted(c.items())]
        return "  ".join(parts)

    print(f"\n  Action distributions:")
    print(f"    Agent A: {fmt_counts(counts_a, n_trials)}")
    print(f"    Agent B: {fmt_counts(counts_b, n_trials)}")

    # 差の検証
    n_a = counts_a["A"]
    n_b_from_a = counts_b["A"]
    diff = abs(n_a - n_b_from_a)
    significant = diff > n_trials * 0.1

    print(f"\n  Agent A chose A: {n_a}/{n_trials} ({n_a/n_trials*100:.0f}%)")
    print(f"  Agent B chose A: {n_b_from_a}/{n_trials} ({n_b_from_a/n_trials*100:.0f}%)")
    print(f"  Difference: {diff} ({diff/n_trials*100:.0f}%)")

    result = significant
    print(f"\n  Result: {'✅ DIFFERENT DISTRIBUTIONS' if result else '❌ SIMILAR'}")
    if result:
        print("  → Experience history caused different action preferences.")

    return result
