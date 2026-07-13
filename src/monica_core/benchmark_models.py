"""複数LLMモデルのベンチマーク: 活動多様性・応答品質・速度を計測"""
import time
import json
import sys
from collections import Counter
from .vital_os import PARAMS, DURATIONS
from .simulate_llm import ConsciousVitalOS


def benchmark_model(model_name: str, days: int = 1, tick_minutes: int = 30,
                    seed: int = 42) -> dict:
    import random
    random.seed(seed)

    os = ConsciousVitalOS(model=model_name)
    start_day = os.time.day

    llm_call_count = 0
    total_llm_time = 0.0
    parse_fails = 0
    activity_sequence = []
    new_activities_proposed = 0

    orig_llm_decide = os._llm_decide
    def timed_llm_decide():
        nonlocal llm_call_count, total_llm_time, parse_fails, new_activities_proposed
        t0 = time.time()
        try:
            result = orig_llm_decide()
            if result:
                action, dur = result
                activity_sequence.append(action)
                # Check if this is a newly created activity (not in DURATIONS before this session)
                # We can't perfectly track this, so we'll approximate
        except Exception:
            parse_fails += 1
            result = ("idle", 30)
        elapsed = time.time() - t0
        llm_call_count += 1
        total_llm_time += elapsed
        return result

    os._llm_decide = timed_llm_decide

    try:
        while True:
            if not os.current_activity:
                action, duration = os.decide_next()
                os.start_activity(action, duration)
            os.tick(tick_minutes)
            elapsed_days = os.time.day - start_day
            if elapsed_days >= days and os.time.hour >= 7:
                break
    except Exception as e:
        return {"error": str(e), "model": model_name}

    final_state = os.state.as_dict()
    activity_counts = Counter(activity_sequence)
    unique_activities = len(activity_counts)
    total_choices = len(activity_sequence)

    # Consecutive repetition rate
    repeats = 0
    for i in range(1, len(activity_sequence)):
        if activity_sequence[i] == activity_sequence[i-1]:
            repeats += 1
    repeat_rate = repeats / max(total_choices - 1, 1)

    # LLM call stats
    avg_time = total_llm_time / max(llm_call_count, 1)

    # Deviation from setpoints at end
    from .vital_os import SETPOINTS, VitalParam
    dev = sum(abs(final_state[p] - SETPOINTS[VitalParam[p.upper()]]) for p in PARAMS)

    # Check for new activities in the model that weren't in original DURATIONS
    original_activities = set(DURATIONS.keys())
    current_activities = set(os.model.counts.keys())
    new_acts = current_activities - original_activities
    new_activities_proposed = len(new_acts)

    return {
        "model": model_name,
        "days": days,
        "llm_calls": llm_call_count,
        "avg_response_sec": round(avg_time, 2),
        "total_llm_time_sec": round(total_llm_time, 1),
        "parse_failures": parse_fails,
        "total_choices": total_choices,
        "unique_activities": unique_activities,
        "activity_diversity": round(unique_activities / max(total_choices, 1), 2),
        "consecutive_repeat_rate": round(repeat_rate, 2),
        "activity_counts": dict(activity_counts.most_common()),
        "new_activities_proposed": new_activities_proposed,
        "final_state": final_state,
        "total_deviation": round(dev, 1),
    }


BENCHMARK_DESC = {
    "qwen2.5:14b": "現在のベースライン",
    "qwen3.5:9b": "最新世代 9B、30B超えと評判",
    "qwen3:8b": "Qwen3世代 8B",
    "gemma3:12b": "Google 最新 12B",
    "deepseek-r1:8b": "DeepSeek R1 推論特化",
    "phi4:14b": "Microsoft Phi-4 14B",
    "llama3.1:8b": "Meta Llama 3.1 8B",
    "mistral:7b": "Mistral 7B 効率的",
    "gemma3:27b": "Google 27B 高性能",
    "qwen3.5:27b": "Qwen3.5 27B",
}


def print_results(results: list[dict]):
    print(f"\n{'='*80}")
    print(f"{'モデル':20s} {'応答(秒)':>10s} {'多様性':>8s} {'連続率':>8s} {'エラー':>6s} {'乖離':>6s} {'新活動':>6s}")
    print(f"{'-'*80}")
    for r in results:
        if "error" in r:
            print(f"{r['model']:20s} {'ERROR':>10s} {r.get('error','')}")
            continue
        label = BENCHMARK_DESC.get(r["model"], "")
        print(f"{r['model']:20s} {r['avg_response_sec']:>8.1f}s  "
              f"{r['activity_diversity']:>7.2f}  "
              f"{r['consecutive_repeat_rate']:>7.2f}  "
              f"{r['parse_failures']:>5d}  "
              f"{r['total_deviation']:>5.1f}  "
              f"{r['new_activities_proposed']:>5d}")
        counts_str = ", ".join(f"{a}:{c}" for a, c in sorted(r['activity_counts'].items(), key=lambda x: -x[1])[:5])
        print(f"  {'':20s}  top5: {counts_str}")
        final = r['final_state']
        print(f"  {'':20s}  最終: E:{final['energy']:.0f} H:{final['hunger']:.0f} L:{final['loneliness']:.0f} S:{final['spirit']:.0f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        "qwen2.5:14b",
        "qwen3.5:9b",
        "gemma3:12b",
        "deepseek-r1:8b",
    ]
    results = []
    for m in models:
        print(f"\n--- Benchmarking {m} ({BENCHMARK_DESC.get(m, '')}) ---")
        r = benchmark_model(m, days=1)
        results.append(r)
        print(f"  done: {r.get('llm_calls', 0)} calls, avg {r.get('avg_response_sec', 0)}s")
    print_results(results)
