"""Phase 4: Memory System — 実験

検証項目:
1. エピソード記憶の保存と検索
2. 意味記憶への圧縮（反復による知識の形成）
3. 自伝的記憶のフィルタリング（高PE事象のみ保存）
4. 記憶の容量管理と忘却
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import Memory, Episode
from core.world_model import WorldModel
from world.mock_world import MockWorld


def test_episodic_storage_and_recall():
    """エピソード記憶の保存と検索をテストする。"""
    print("Phase 4 — Test: Episodic Storage & Recall")
    print("=" * 60)

    memory = Memory(capacity=100)
    world = MockWorld(seed=42, phase=1)

    # 経験をエピソードとして保存
    n_episodes = 20
    for i in range(n_episodes):
        action = world.actions()[i % len(world.actions())]
        outcome = world.step(action)
        ep = Episode(
            timestamp=float(i),
            action=action,
            outcome=outcome,
            pe=abs(hash((action, outcome)) % 100) / 100.0,
        )
        memory.store(ep)

    # 検索テスト
    recalled = memory.recall_episodic(n=5)
    assert len(recalled) <= 5, f"Should recall max 5, got {len(recalled)}"
    assert len(recalled) > 0, "Should recall at least 1"

    print(f"  Stored: {n_episodes} episodes")
    print(f"  Recalled (top 5): {len(recalled)} episodes")
    print(f"  Most recent: action={recalled[0].action}, outcome={recalled[0].outcome}")
    print(f"  ✅ Episodic storage & recall OK\n")


def test_semantic_consolidation():
    """反復経験による意味記憶の形成をテストする。"""
    print("Phase 4 — Test: Semantic Consolidation")
    print("=" * 60)

    memory = Memory(capacity=1000)
    world = MockWorld(seed=42, phase=0)

    # A を 50回経験 → 意味記憶に A|food/food/nothing のパターンが形成される
    for i in range(50):
        outcome = world.step("A")
        ep = Episode(
            timestamp=float(i),
            action="A",
            outcome=outcome,
            pe=0.2,
        )
        memory.store(ep)

    # 意味記憶の確認
    food_info = memory.query_semantic("A", "food")
    nothing_info = memory.query_semantic("A", "nothing")

    print(f"  Semantic memory for A|food: count={food_info.get('count', 0)}")
    print(f"  Semantic memory for A|nothing: count={nothing_info.get('count', 0)}")

    assert food_info.get("count", 0) > 0, "Should have consolidated A|food"
    assert memory.action_frequency("A") == 50, "Should count 50 A actions"

    # 意味記憶からの確率推定
    probs = memory.outcome_probs_from_memory("A")
    print(f"  Estimated probs from memory: {probs}")
    assert abs(probs.get("food", 0) - 0.80) < 0.3, "food prob should be ~0.80"
    print(f"  ✅ Semantic consolidation OK\n")


def test_autobiographical_filtering():
    """自伝的記憶のフィルタリングをテストする。"""
    print("Phase 4 — Test: Autobiographical Filtering")
    print("=" * 60)

    memory = Memory(capacity=100)

    # 低PEエピソード（自己関連性低）
    for i in range(10):
        memory.store(Episode(
            timestamp=float(i), action="A", outcome="nothing", pe=0.1, self_relevant=False
        ))

    # 高PEエピソード（自己関連性高）
    for i in range(5):
        memory.store(Episode(
            timestamp=float(100 + i), action="B", outcome="danger", pe=0.9, self_relevant=True
        ))

    recalled_auto = memory.recall_autobiographical(n=10)
    print(f"  Episodic: {memory.summary()['episodic']} total")
    print(f"  Autobiographical: {memory.summary()['autobiographical']} (should be ~5)")

    assert len(recalled_auto) <= 6, f"Auto memory should be limited, got {len(recalled_auto)}"
    print(f"  ✅ Autobiographical filtering OK\n")


def test_forgetting():
    """容量超過時の忘却をテストする。"""
    print("Phase 4 — Test: Forgetting")
    print("=" * 60)

    memory = Memory(capacity=10)  # 小さな容量

    # 20エピソード保存 → 10に絞られる
    for i in range(20):
        memory.store(Episode(
            timestamp=float(i), action="A", outcome="nothing",
            pe=0.1, importance=0.1,
        ))

    print(f"  Stored 20 episodes (capacity=10)")
    print(f"  Episodic after forgetting: {memory.summary()['episodic']}")

    assert memory.summary()["episodic"] <= 10, "Should forget excess episodes"
    print(f"  ✅ Forgetting OK\n")


def run():
    """Phase 4 実験を全て実行する。"""
    test_episodic_storage_and_recall()
    test_semantic_consolidation()
    test_autobiographical_filtering()
    test_forgetting()

    print("=" * 60)
    print("✅ Phase 4 complete: Memory system")
    print("   • Episodic storage & recall: OK")
    print("   • Semantic consolidation: OK")
    print("   • Autobiographical filtering: OK")
    print("   • Forgetting: OK")


if __name__ == "__main__":
    run()
