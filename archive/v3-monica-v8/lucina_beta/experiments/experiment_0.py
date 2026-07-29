"""Experiment 0: 基本学習 — L1 Error の収束測定"""

import random

from core.agent import Agent
from world.mock_world import MockWorld
from cli.display import show_header, show_l1_errors, show_final_beliefs_vs_true


def run(seeds: int = 3):
    """複数seedでL1 Errorの収束を測定する。"""
    show_header("Experiment 0: Basic Learning — L1 Error Convergence")

    checkpoints = [10, 100, 1000, 10000]
    all_results = {}

    for seed in range(seeds):
        world = MockWorld(seed=seed + 42)
        agent = Agent()
        errors = {}

        for step in range(1, 10001):
            agent.step(world)
            if step in checkpoints:
                total_error = sum(
                    agent.world_model.l1_error(a, world.true_probabilities(a))
                    for a in world.actions()
                )
                errors[step] = total_error

        all_results[seed] = errors
        print(f"\n  Seed {seed}:")
        for s, err in errors.items():
            print(f"    {s:5d} steps: L1 = {err:.4f}")

    # 平均
    print(f"\n  Average across {seeds} seeds:")
    for step in checkpoints:
        avg = sum(all_results[s][step] for s in range(seeds)) / seeds
        print(f"    {step:5d} steps: L1 (avg) = {avg:.4f}")

    # 収束判定
    initial = sum(all_results[s][10] for s in range(seeds)) / seeds
    final = sum(all_results[s][10000] for s in range(seeds)) / seeds
    converged = final < initial * 0.5

    print(f"\n  Result: {'✅ CONVERGED' if converged else '❌ NOT CONVERGED'}")
    print(f"    L1 at 10 steps:    {initial:.4f}")
    print(f"    L1 at 10000 steps: {final:.4f}")
    print(f"    Reduction: {(1 - final/initial)*100:.1f}%")

    # 最終状態の表示（最後のseedのAgentを使用）
    final_world = MockWorld(seed=seeds + 42)
    final_agent = Agent()
    final_agent.run(final_world, 10000)
    show_final_beliefs_vs_true(final_agent.world_model, final_world)
    show_l1_errors(final_agent.world_model, final_world)

    return converged
