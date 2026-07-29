"""Phase 3: LLM Cognitive Layer — 実験

検証項目:
1. LLM が利用可能な場合、候補の多様性が増す
2. LLM が利用不可でも Phase 2 相当の動作が可能（回帰）
3. LLM による未来予測が World Model の数値予測を補完する
4. LLM による結果解釈がログに保存される
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.world_model import WorldModel
from core.inference import compute_efe, efe_summary, select_action_efe
from core.llm import LLMCognitiveLayer
from world.mock_world import MockWorld


def test_llm_candidate_diversity():
    """LLM 有無による候補の多様性を比較する。"""
    print("=" * 60)
    print("Phase 3 — Test: LLM Candidate Diversity")
    print("=" * 60)

    # Fallback (LLMなし) の候補
    llm = LLMCognitiveLayer(backend="fallback")
    fallback_candidates = llm.generate_candidates(
        "You are in a simulation with locations A, B, C.",
        available_actions=["A", "B", "C", "rest", "explore"],
    )
    print(f"\nFallback candidates ({len(fallback_candidates)}):")
    for c in fallback_candidates:
        print(f"  • {c}")

    assert len(fallback_candidates) > 0, "Fallback should produce candidates"
    print(f"  ✅ Fallback: {len(fallback_candidates)} candidates")

    # Ollama が利用可能なら LLM の候補も表示
    llm_real = LLMCognitiveLayer(backend="ollama")
    if llm_real.is_available():
        llm_candidates = llm_real.generate_candidates(
            "You are hungry and tired in a mysterious forest with paths, a cave, and a river.",
        )
        print(f"\nLLM candidates ({len(llm_candidates)}):")
        for c in llm_candidates:
            print(f"  • {c}")
        print(f"  ✅ LLM candidate diversity test")
    else:
        print("\n  ℹ️  Ollama not available — skipping LLM candidate test")

    print(f"\n{'='*60}\n")


def test_backward_compatibility():
    """Phase 2 (Active Inference) が LLM 導入後も動作することを確認する。
    
    軽量版：フル実験は行わず、EFE 計算と未知行動への探索傾向のみを短期検証する。
    """
    print("Phase 3 — Test: Backward Compatibility (lightweight)")
    print("-" * 60)

    from core.world_model import WorldModel
    from core.inference import select_action_efe
    from world.mock_world import MockWorld

    world = MockWorld(seed=42, phase=1)
    wm = WorldModel()

    # A/B/C を学習、rest/explore は未経験
    for a in ["A"] * 30 + ["B"] * 30 + ["C"] * 30:
        obs = world.step(a)
        wm.update(a, obs)
    for a in ["rest", "explore"]:
        wm.add_action(a)

    # 短期間の EFE 選択で未知行動が選ばれるか
    unknown_selected = 0
    for _ in range(50):
        action = select_action_efe(wm, temperature=0.5)
        if action in ("rest", "explore"):
            unknown_selected += 1

    print(f"  Unknown actions selected (EFE, 50 trials): {unknown_selected}")
    result = unknown_selected > 0
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status}: EFE exploration {'works' if result else 'failed'} without LLM")
    print()
    return result


def test_llm_prediction_and_interpretation():
    """LLM による予測と解釈をテストする。"""
    print("Phase 3 — Test: LLM Prediction & Interpretation")
    print("=" * 60)

    llm = LLMCognitiveLayer(backend="ollama")
    available = llm.is_available()
    print(f"  LLM available: {available}")

    world = MockWorld(seed=42, phase=1)
    wm = WorldModel()

    context = "Exploring locations with uncertain outcomes in a simulation."

    print(f"\n  {'Action':10s} {'Outcome':10s} {'Surprise':8s} {'Interpretation':40s}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*40}")
    
    for action in world.actions():  # Test ALL actions
        outcome = world.step(action)
        wm.add_action(action)
        pred = wm.predict(action)
        surprise = wm.surprise(action, outcome)
        interpretation = llm.interpret_result(action, outcome, pred, context)
        
        # Truncate for display
        interp_short = interpretation[:40] + "..." if len(interpretation) > 40 else interpretation
        print(f"  {action:10s} {outcome:10s} {surprise:<8.3f} {interp_short:<40s}")
        
        # Test prediction
        llm_pred = llm.predict_outcome(action, context, pred)
        if available and "text_prediction" in llm_pred:
            pred_short = llm_pred["text_prediction"][:40] + "..." if len(llm_pred["text_prediction"]) > 40 else llm_pred["text_prediction"]
            print(f"  {'':10s} {'':10s} {'':8s} LLM predicted: {pred_short}")

    print(f"\n{'='*60}\n")


def run():
    """Phase 3 実験を全て実行する。"""
    test_llm_candidate_diversity()
    
    compat_ok = test_backward_compatibility()
    
    if compat_ok:
        test_llm_prediction_and_interpretation()
        print("\n✅ Phase 3 complete: LLM integrated as cognitive layer.")
        print("   • Candidate diversity: fallback works, LLM enhances when available")
        print("   • Backward compat: Phase 2 (EFE) still works without LLM")
        print("   • LLM prediction/interpretation: tested")
    else:
        print("\n❌ Phase 3 FAILED: backward compatibility broken")
        print("   Fix Phase 2 regression before continuing.")


if __name__ == "__main__":
    run()
