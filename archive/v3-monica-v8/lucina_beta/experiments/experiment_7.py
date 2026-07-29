"""Phase 7-10: Self Model + Values + Identity — 実験

検証項目:
1. Self Model が行動履歴から能力を学習する
2. Value System が経験から価値観を形成する
3. Identity が長期経験から圧縮される
4. Continuity が変化を記録する
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.self_model import SelfModel
from core.values import ValueSystem
from core.identity import Identity
from world.mock_world import MockWorld


def test_self_model_learning():
    """Self Model が行動履歴から能力を学習するか。"""
    print("Phase 7 — Test: Self Model Learning")
    print("=" * 60)

    sm = SelfModel()

    # 探索で成功
    for _ in range(20):
        sm.record_action("explore", "food", success=True)
    print(f"  After 20 successful explores:")
    print(f"    Exploration ability: {sm.abilities['exploration']:.3f}")
    assert sm.abilities["exploration"] > 0.5, "Should learn exploration ability"

    # 予測記録
    for _ in range(30):
        sm.record_prediction(correct=True)
    for _ in range(10):
        sm.record_prediction(correct=False)
    print(f"  Prediction accuracy: {sm.prediction_accuracy:.3f}")
    assert sm.prediction_accuracy > 0.7, "Should have >70% accuracy"
    print(f"  ✅ Self Model learning OK\n")


def test_value_formation_from_experience():
    """Value System が経験から価値観を形成するか。"""
    print("Phase 8 — Test: Value Formation")
    print("=" * 60)

    vs = ValueSystem()

    # 危険な経験を繰り返す → safety↑, exploration↓
    for _ in range(20):
        vs.update_from_experience("explore", "danger", pe=0.8)

    print(f"  After repeated danger:")
    print(f"    Safety value: {vs.weights['safety']:.3f}")
    print(f"    Exploration value: {vs.weights['exploration']:.3f}")
    assert vs.weights["safety"] > 0.55, "Safety should increase after danger"
    assert vs.weights["exploration"] < 0.3, "Exploration should decrease after danger"

    # 社会的に良い経験を繰り返す → social_bond↑
    for _ in range(20):
        vs.update_from_experience("talk", "positive", pe=0.3)

    print(f"\n  After repeated positive social interactions:")
    print(f"    Social bond value: {vs.weights['social_bond']:.3f}")
    assert vs.weights["social_bond"] > 0.5, "Social bond should increase"
    print(f"  ✅ Value formation OK\n")


def test_preference_formation():
    """Value から Preference が形成されるか。"""
    print("Phase 8 — Test: Preference Formation")
    print("=" * 60)

    vs = ValueSystem()

    # explore が高い価値を生んだことを繰り返し記録
    for _ in range(20):
        vs.update_preference("explore", value_contribution=0.5)

    pref_explore = vs.get_preference("explore")
    pref_safety = vs.get_preference("rest")
    print(f"  Explore preference: {pref_explore:.3f}")
    print(f"  Rest preference: {pref_safety:.3f}")
    assert pref_explore > pref_safety, "Explore should have higher preference"
    print(f"  ✅ Preference formation OK\n")


def test_identity_compression():
    """Identity が長期経験から圧縮されるか。"""
    print("Phase 9 — Test: Identity Compression")
    print("=" * 60)

    identity = Identity(consolidation_interval=50)
    world = MockWorld(seed=42, phase=1)

    # 探索中心の経験
    for i in range(200):
        action = "explore" if i % 3 == 0 else world.actions()[i % len(world.actions())]
        outcome = world.step(action) if action in ("A", "B", "C", "rest", "explore") else "nothing"
        identity.record_experience(
            action=action,
            outcome=outcome,
            pe=0.3,
        )

    print(f"  Identity after 200 steps:")
    print(f"    Traits: {identity.summary()['traits']}")
    print(f"    Stability: {identity.stability:.3f}")
    print(f"    Changes: {identity.summary()['changes_recorded']}")
    print(f"    Narrative: {identity.narrative_summary()[:80]}...")

    # 探索行動の結果、curious trait が変化しているはず
    # （dangerの多いexploreではcuriousが下がることもあるが、変化は必ず発生する）
    assert identity.traits["curious"] != 0.3, "Curiosity should have changed with experience"
    print(f"  ✅ Identity compression OK (curious={identity.traits['curious']:.2f})\n")


def test_continuity():
    """Continuity が変化を記録するか。"""
    print("Phase 10 — Test: Continuity")
    print("=" * 60)

    identity = Identity(consolidation_interval=10)

    # Phase 1: 探索中心
    for _ in range(100):
        identity.record_experience("explore", "food", pe=0.3)

    phase1_curious = identity.traits["curious"]

    # Phase 2: 安全中心
    for _ in range(100):
        identity.record_experience("rest", "nothing", pe=0.1, values={"safety": 0.8})

    phase2_cautious = identity.traits["cautious"]

    print(f"  After exploration phase:")
    print(f"    Curiosity: {phase1_curious:.3f}")
    print(f"  After safety phase:")
    print(f"    Cautious: {phase2_cautious:.3f}")
    print(f"  Changes recorded: {len(identity.change_log)}")

    assert len(identity.change_log) >= 1, "Should have recorded at least one change"
    assert identity.continuity_statement != "", "Continuity statement should exist"
    print(f"  Continuity: {identity.continuity_statement}")
    print(f"  ✅ Continuity OK\n")


def run():
    test_self_model_learning()
    test_value_formation_from_experience()
    test_preference_formation()
    test_identity_compression()
    test_continuity()

    print("=" * 60)
    print("✅ Phase 7-10 complete: Self Model + Values + Identity + Continuity")
    print("   • Self Model learning: OK")
    print("   • Value formation: OK")
    print("   • Preference formation: OK")
    print("   • Identity compression: OK")
    print("   • Continuity: OK")


if __name__ == "__main__":
    run()
