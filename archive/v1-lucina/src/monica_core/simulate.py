from .vital_os import VitalOS


def simulate(days: int = 1, tick_minutes: int = 15, show_model: bool = False):
    os = VitalOS()
    start_day = os.time.day

    print(f"MonicaCore Phase1: 予測+学習モデル シミュレーション ({days}日分)\n")

    while True:
        if not os.current_activity and os.activity_remaining <= 0:
            action, duration = os.decide_next()
            os.start_activity(action, duration)

        os.tick(tick_minutes)

        elapsed_days = os.time.day - start_day
        if elapsed_days >= days and os.time.hour >= 7:
            break

    print(f"\n最終状態: {os.summary()}")
    print(f"\n行動ログ ({len(os.history)} イベント):")
    for e in os.history:
        print(f"  {e.time} {e.activity}: {e.state_before} → {e.state_after}")

    print(f"\n生活ログ:")
    for line in os.day_log:
        print(f"  {line}")

    if show_model:
        print(f"\n{os.model_report()}")


def show_learning(days: int = 7, tick_minutes: int = 15):
    os = VitalOS()
    start_day = os.time.day

    print(f"MonicaCore: 長期学習 ({days}日)\n")

    for day in range(days):
        while True:
            if not os.current_activity and os.activity_remaining <= 0:
                action, duration = os.decide_next()
                os.start_activity(action, duration)
            os.tick(tick_minutes)
            elapsed = os.time.day - start_day
            if elapsed > day and os.time.hour >= 7:
                break

        print(f"--- {day+1}日目 終了 ---")
        print(f"  状態: {os.summary()}")
        print(f"  活動数: {sum(os.model.counts.values())}")
        print(f"  モデル:")
        for name in sorted(os.model.deltas):
            c = os.model.counts[name]
            if c > 0:
                d = os.model.deltas[name]
                print(f"    {name:12s} x{c:2d}  {d['energy']:+.1f} {d['hunger']:+.1f} {d['loneliness']:+.1f} {d['spirit']:+.1f}")
        print()

    print(f"\n=== {days}日後 最終モデル ===")
    print(os.model_report())


if __name__ == "__main__":
    import sys
    if "--learn" in sys.argv:
        show_learning(days=7)
    else:
        simulate(days=1, show_model="--model" in sys.argv)
