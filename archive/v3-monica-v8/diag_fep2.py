"""FEP診断 v2: 予測フィードバック方式を比較"""
import json, urllib.request, time

MODEL = "batiai/qwen3.5-9b:q4"

def gen(msgs, max_tokens=0):
    prompt = ""
    for m in msgs:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    opts = {"num_ctx": 8192, "temperature": 0.85}
    if max_tokens: opts["num_predict"] = max_tokens
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": opts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("response","").strip()
        except:
            time.sleep(1)
    return ""

def test(method, prompt_user):
    msgs = [{"role": "system", "content": "あなたは好奇心旺盛な探検家。日本語で話す。"}]
    # First cycle: establish context
    msgs.append({"role": "user", "content": "洞窟の中で古い地図を見つけた。"})
    r = gen(msgs)
    msgs.pop(); msgs.append({"role": "assistant", "content": r})

    # Extract a prediction
    pred = gen([{"role": "user", "content": f"応答:\n{r[:150]}\n\n次に何が起きる？"}], max_tokens=40)

    # Second cycle with different feedback method
    if method == "fep_match":
        prompt = f"予想通り地図を広げている。{prompt_user}"
    elif method == "fep_mismatch":
        prompt = f"予想と違い、地図にはない場所にいる。{prompt_user}"
    elif method == "fep_control":
        prompt = prompt_user
    elif method == "fep_pred_only":
        prompt = f"(内なる声: {pred})\n{prompt_user}"

    msgs.append({"role": "user", "content": prompt})
    t0 = time.time()
    r2 = gen(msgs)
    t = time.time() - t0
    return r2, t, pred

# Warmup
gen([{"role": "user", "content": "."}], max_tokens=1)
print("Warmup done.\n")

scenarios = [
    ("【一致】予想通り進行", "fep_match", "足音が近づいてくる。"),
    ("【不一致】予想と違う展開", "fep_mismatch", "足音が近づいてくる。"),
    ("【対照】予測なし", "fep_control", "足音が近づいてくる。"),
]

for label, method, obs in scenarios:
    reply, elapsed, pred = test(method, obs)
    ok = "✓" if reply else "✗"
    print(f"{ok} {label} ({elapsed:.1f}s)")
    print(f"  予測: {pred[:60] if pred else '-'}")
    print(f"  応答: {reply[:120]}")
    print()

# Repeat with different observation for robustness
print("--- 2nd round (違う観測) ---")
scenarios2 = [
    ("【一致】予想通り進行", "fep_match", "ランプの火が揺れている。"),
    ("【不一致】予想と違う展開", "fep_mismatch", "ランプの火が揺れている。"),
    ("【対照】予測なし", "fep_control", "ランプの火が揺れている。"),
]

for label, method, obs in scenarios2:
    reply, elapsed, pred = test(method, obs)
    ok = "✓" if reply else "✗"
    print(f"{ok} {label} ({elapsed:.1f}s)")
    print(f"  応答: {reply[:120]}")
    print()
PYEOF
