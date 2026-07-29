"""Experiment 2: Active Inference — EFE による探索行動

検証:
  EFE を使うと「既知の安全な行動」だけでなく、
  「未知だが情報価値が高い行動」も選ばれるようになる。
"""

import math

from core.world_model import WorldModel
from core.internal_state import InternalState
from core.inference import compute_efe, select_action_efe, efe_summary
from world.mock_world import MockWorld
from cli.display import show_header


def run_exploration_comparison() -> bool:
    """EFE と EV-only の探索率比較実験。
    Returns: EFEが探索を促進したか（bool）"""
    show_header("Experiment 2: Active Inference — EFE-Driven Exploration")

    world = MockWorld(seed=42, phase=1)
    wm = WorldModel()
    state = InternalState()

    # まず A/B/C だけを学習（rest/explore は未経験のまま残す）
    print("\n  Phase 1: Learn A=50, B=50, C=50 (leave rest/explore unknown)")
    for a in ["A"] * 50 + ["B"] * 50 + ["C"] * 50:
        obs = world.step(a)
        wm.update(a, obs)

    print("\n  Learned states:")
    for a in wm.actions:
        p = wm.predict(a)
        n = wm.confidence(a)
        unc = wm.uncertainty(a)
        print(f"    {a}: EV={wm.expected_value(a):.2f}  samples={n}  uncertainty={unc:.2f}")

    # rest と explore を WorldModel に追加（未経験のまま）
    for a in ["rest", "explore"]:
        wm.add_action(a)

    print("\n  After adding rest/explore (unvisited):")
    for a in wm.actions:
        p = wm.predict(a)
        n = wm.confidence(a)
        unc = wm.uncertainty(a)
        print(f"    {a}: EV={wm.expected_value(a):.2f}  samples={n}  uncertainty={unc:.2f}")

    # Phase 2: EV のみの選択（Phase 0 方式）
    print("\n  Phase 2: Selection by EV only (Phase 0 way)")
    print("  → Will always pick A (known best). Never explores rest/explore.")

    counts_ev: dict[str, int] = {a: 0 for a in wm.actions}
    for _ in range(200):
        action = _select_by_ev(wm, state, temperature=0.5)
        counts_ev[action] += 1

    for a in sorted(counts_ev):
        pct = counts_ev[a] / 2
        bar = "█" * int(pct)
        print(f"    {a:8s}: {counts_ev[a]:3d}/200 ({counts_ev[a]/2:.1f}%) {bar}")

    # Phase 3: EFE による選択
    print("\n  Phase 3: Selection by EFE (Active Inference)")
    print("  → Should explore rest/explore despite low base EVs,")
    print("    because they have high Epistemic Value.")

    print("\n  EFE breakdown:")
    for row in efe_summary(wm, state):
        print(f"    {row['action']:8s}: EFE={row['efe']:+.3f}  "
              f"Prag={row['pragmatic']:+.2f}  Risk={row['risk']:.2f}  "
              f"Epi={row['epistemic']:.2f}")

    counts_efe: dict[str, int] = {a: 0 for a in wm.actions}
    for _ in range(200):
        action = select_action_efe(wm, state, temperature=0.5)
        counts_efe[action] += 1

    print("\n  Action distribution (EFE):")
    for a in sorted(counts_efe):
        pct = counts_efe[a] / 2
        bar = "█" * int(pct)
        print(f"    {a:8s}: {counts_efe[a]:3d}/200 ({counts_efe[a]/2:.1f}%) {bar}")

    # 検証: EFE では未知の行動（rest/explore）が選択される
    explored_ev = counts_ev.get("rest", 0) + counts_ev.get("explore", 0)
    explored_efe = counts_efe.get("rest", 0) + counts_efe.get("explore", 0)

    print(f"\n  {'='*55}")
    print(f"  Unknown actions selected:")
    print(f"    EV selection:  {explored_ev}/200")
    print(f"    EFE selection: {explored_efe}/200")
    efe_increases = explored_efe > explored_ev + 20
    print(f"  {'✅ EFE ENABLES EXPLORATION' if efe_increases else '❌ NO SIGNIFICANT DIFFERENCE'}")
    print(f"  {'='*55}")

    return efe_increases


def run():
    """Experiment 2 のエントリポイント。"""
    run_exploration_comparison()


def _select_by_ev(wm: WorldModel, state: InternalState | None, temperature: float) -> str:
    """EV のみで行動選択（Phase 0 方式、EFE なし）。"""
    actions = wm.actions
    values = [wm.expected_value(a) for a in actions]
    if state is not None:
        values = [v + state.need_bonus(a, v) for a, v in zip(actions, values)]
    exp_v = [math.exp(v / temperature) for v in values]
    total = sum(exp_v)
    if total == 0 or not math.isfinite(total):
        import random
        return random.choice(actions)
    probs = [ev / total for ev in exp_v]
    import random
    return random.choices(actions, weights=probs)[0]
