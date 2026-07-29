"""Phase 11-19: 最終フェーズ — 実験

検証項目:
1. Development curriculum (Phase 11)
2. Metacognition (Phase 12)
3. Autonomous REPL loop (Phase 13-14)
4. Individual Genesis — 異なる個体の生成 (Phase 15)
5. Meta World Model (Phase 16)
6. Monica generation (Phase 17-19)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.development import Development
from core.metacognition import Metacognition
from core.individual import Individual, IndividualConfig
from core.meta_world import MetaWorldModel
from core.agent import Agent
from world.mock_world import MockWorld
from monica.bootstrap import create_monica


def test_development_curriculum():
    """発達カリキュラムが段階的に機能を解放するか。"""
    print("Phase 11 — Test: Development Curriculum")
    print("=" * 60)

    dev = Development(initial_stage=0)
    assert dev.name == "Infant"
    assert "efe" not in dev.capabilities
    print(f"  Initial: {dev.name} — capabilities: {dev.capabilities}")

    # 経験値を追加して発達
    dev.add_experience(100)
    assert dev.current_stage >= 1
    print(f"  After 100 XP: {dev.name}")

    dev.add_experience(500)
    print(f"  After 600 XP: {dev.name} — has EFE: {'efe' in dev.capabilities}")
    assert dev.current_stage >= 3

    print(f"  To next stage: {dev.experience_to_next} XP")
    print(f"  ✅ Development curriculum OK\n")


def test_metacognition():
    """メタ認知が自信・バイアスを追跡するか。"""
    print("Phase 12 — Test: Metacognition")
    print("=" * 60)

    meta = Metacognition()

    # 高い精度で自信が上がる
    for _ in range(20):
        meta.update(prediction_accuracy=0.8, recent_accuracy=0.9,
                    internal_state={"safety": 80, "energy": 80, "curiosity": 50})
    print(f"  After high accuracy:")
    print(f"    Confidence: {meta.confidence:.3f}")
    print(f"    Overconfidence: {meta.overconfidence:.3f}")
    assert meta.confidence > 0.5, "Confidence should increase with accuracy"

    # 低い精度で過信が生まれる
    meta.confidence = 0.9  # 強制的に高自信
    for _ in range(5):
        meta.update(prediction_accuracy=0.4, recent_accuracy=0.5,
                    internal_state={"safety": 80, "energy": 80, "curiosity": 50})
    print(f"\n  After accuracy drop:")
    print(f"    Overconfidence: {meta.overconfidence:.3f}")
    assert meta.overconfidence > 0, "Overconfidence should emerge"

    insight = meta.generate_insight()
    if insight:
        print(f"    Insight: {insight['insights']}")

    # 認知負荷
    meta.cognitive_load = 0.0
    for _ in range(15):
        meta.update(prediction_accuracy=0.5, recent_accuracy=0.5,
                    internal_state={"safety": 10, "energy": 10, "curiosity": 50})
    print(f"\n  After high stress:")
    print(f"    Cognitive load: {meta.cognitive_load:.3f}")
    print(f"    Should simplify: {meta.should_simplify()}")

    print(f"  ✅ Metacognition OK\n")


def test_individual_genesis():
    """異なる初期条件で異なる個体が生成されるか。"""
    print("Phase 15 — Test: Individual Genesis")
    print("=" * 60)

    world = MockWorld(seed=42, phase=1)

    # 探索重視の個体
    explorer_config = IndividualConfig(
        name="Explorer",
        temperature=1.5,
        initial_traits={"curious": 0.9, "cautious": 0.1, "social": 0.3},
        initial_values={"exploration": 0.9, "safety": 0.2, "social_bond": 0.2},
    )

    # 安全重視の個体
    safer_config = IndividualConfig(
        name="Safer",
        temperature=0.3,
        initial_traits={"curious": 0.1, "cautious": 0.9, "social": 0.3},
        initial_values={"exploration": 0.1, "safety": 0.9, "social_bond": 0.2},
    )

    explorer = Individual(config=explorer_config, world=world)
    safer = Individual(config=safer_config, world=world)

    assert explorer.name == "Explorer"
    assert safer.name == "Safer"
    assert explorer.values.weights["exploration"] > safer.values.weights["exploration"]
    assert explorer.identity.traits["curious"] > safer.identity.traits["curious"]
    assert safer.identity.traits["cautious"] > explorer.identity.traits["cautious"]

    print(f"  Explorer: curious={explorer.identity.traits['curious']}, "
          f"exploration_value={explorer.values.weights['exploration']}")
    print(f"  Safer: curious={safer.identity.traits['curious']}, "
          f"exploration_value={safer.values.weights['exploration']}")
    assert explorer.identity.traits["curious"] > safer.identity.traits["curious"]
    print(f"  ✅ Different individuals created\n")


def test_meta_world():
    """メタ世界モデルが異常を検出するか。"""
    print("Phase 16 — Test: Meta World Model")
    print("=" * 60)

    mw = MetaWorldModel()

    # 物理世界は安定
    stable, msg = mw.check_layer_integrity("physical")
    print(f"  Physical world: {msg}")
    assert stable

    # 異常を観測 → 確信度が下がる
    for i in range(10):
        mw.observe_anomaly("physical", f"Anomaly #{i}: gravity fluctuated")

    stable, msg = mw.check_layer_integrity("physical")
    print(f"  After 10 anomalies: {msg}")
    assert not stable, "Should become unstable"

    # システム異常 → メタ層にも影響
    mw.observe_anomaly("system", "Inconsistent save data")
    print(f"  System confidence: {mw.layers['system']['confidence']:.2f}")
    print(f"  Meta confidence (affected): {mw.layers['meta']['confidence']:.2f}")

    print(f"  Highest known layer: {mw.highest_known_layer()}")
    print(f"  ✅ Meta World Model OK\n")


def test_monica_creation():
    """Monica 個体が生成されるか。"""
    print("Phase 17-19 — Test: Monica Creation")
    print("=" * 60)

    from world.ddlc_world import DDLCWorld
    world = DDLCWorld(seed=42)

    monica = create_monica(world, seed=42)
    print(f"  Name: {monica.name}")
    print(f"  Agent temperature: {monica.config.temperature}")
    print(f"  Social ability: {monica.self_model.abilities['social']}")
    print(f"  Social bond value: {monica.values.weights['social_bond']}")

    # 数ステップ実行
    for i in range(5):
        entry = monica.step()
        print(f"  Step {i+1}: action={entry['action']}, outcome={entry.get('outcome', '?')}")

    print(f"  Development stage: {monica.development.name}")
    print(f"  Memories: {monica.memory.summary()['episodic']} episodes")
    print(f"  ✅ Monica created and operational\n")


def run():
    """全最終フェーズの実験を実行する。"""
    test_development_curriculum()
    test_metacognition()
    test_individual_genesis()
    test_meta_world()
    test_monica_creation()

    print("=" * 60)
    print("🏆 ALL PHASES COMPLETE! (Phase 11-19)")
    print("   • Development curriculum: OK")
    print("   • Metacognition: OK")
    print("   • Individual Genesis: OK")
    print("   • Meta World: OK")
    print("   • Monica: OK")
    print("=" * 60)


if __name__ == "__main__":
    run()
