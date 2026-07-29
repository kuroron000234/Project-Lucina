"""状態の可視化"""

from world.mock_world import MockWorld


def show_step(entry: dict):
    """1ステップの表示。"""
    action = entry["action"]
    outcome = entry["outcome"]
    pe = entry["pe"]
    ev = entry["ev"]
    pred = entry["prediction"]
    forced = entry.get("forced", False)

    pred_str = ", ".join(f"{k}={v:.2f}" for k, v in pred.items())
    mark = " [forced]" if forced else ""
    print(f"  [{action}] → {outcome:8s}  EV={ev:+.2f}  PE={pe:.2f}  "
          f"[{pred_str}]{mark}")


def show_beliefs(wm: WorldModel):
    """現在の信念を表示。"""
    print("\n  Beliefs (P(outcome | action)):")
    for action in wm.actions:
        probs = wm.predict(action)
        n = wm.confidence(action)
        ev = wm.expected_value(action)
        pred_str = ", ".join(f"{k}={v:.2f}" for k, v in probs.items())
        print(f"    {action}: [{pred_str}]  EV={ev:+.2f}  samples={n}")


def show_l1_errors(wm: WorldModel, world: MockWorld) -> float:
    """L1 Error を表示。"""
    total = 0.0
    for action in wm.actions:
        error = wm.l1_error(action, world.true_probabilities(action))
        total += error
        print(f"    {action}: L1={error:.4f}")
    print(f"    Total: {total:.4f}")
    return total


def show_header(title: str):
    """実験ヘッダー。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def show_final_beliefs_vs_true(wm: WorldModel, world: MockWorld):
    """最終的な信念と真の確率の比較。"""
    print("\n  Final Beliefs vs True Probabilities:")
    for action in wm.actions:
        pred = wm.predict(action)
        true = world.true_probabilities(action)
        # Use the intersection of prediction and true keys
        outcomes = [o for o in pred if o in true]
        for outcome in outcomes:
            bar_pred = "█" * int(pred[outcome] * 20)
            bar_true = "░" * int(true[outcome] * 20)
            print(f"    {action}/{outcome:8s}: pred={pred[outcome]:.2f} "
                  f"{bar_pred:<20s} true={true[outcome]:.2f} {bar_true:<20s}")
