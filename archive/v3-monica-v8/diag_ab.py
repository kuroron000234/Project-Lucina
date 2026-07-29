"""FEP検証: 予測フィードバックが行動に与える影響をAB比較"""
import json, urllib.request, time

MODEL = "batiai/qwen3.5-9b:q4"
BASE = "http://localhost:11434"

def gen(msgs, max_tokens=0):
    prompt = ""
    for m in msgs:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    opts = {"num_ctx": 8192, "temperature": 0.7}
    if max_tokens: opts["num_predict"] = max_tokens
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": opts}).encode()
    req = urllib.request.Request(f"{BASE}/api/generate", data=data, headers={"Content-Type": "application/json"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("response","").strip()
        except: time.sleep(1)
    return ""

# Warmup
gen([{"role": "user", "content": "warm"}], max_tokens=1)

SYSTEM = "あなたは日本語で会話するAI。"


def run_cycle(method, msg_count=0):
    conv = [{"role": "system", "content": SYSTEM}]
    pred = ""
    history = []

    for i in range(3):
        if method == "fep" and pred:
            conv.append({"role": "user", "content": f"(直前の予感: {pred}) 状況:{history[-1] if history else '開始'}"})
        else:
            conv.append({"role": "user", "content": history[-1] if history else "あなたは部屋にいる。何かが始まろうとしている。"})

        reply = gen(conv)
        conv.pop()
        conv.append({"role": "assistant", "content": reply})
        history.append(reply[:100])

        if method == "fep":
            pred = gen([
                {"role": "user", "content": f"現在の状況に基づき、次に何が起きるか予想してください。返答は30字以内。"}
            ], max_tokens=60)

    return history, conv


print("=== A: FEPモード ===")
hist_a, _ = run_cycle("fep")
for i, h in enumerate(hist_a):
    print(f"  step{i+1}: {h[:80]}")

print("\n=== B: 通常モード（予測なし）===")
hist_b, _ = run_cycle("plain")
for i, h in enumerate(hist_b):
    print(f"  step{i+1}: {h[:80]}")

print(f"\n=== 比較 ===")
print(f"  FEPモード: {[h[:40] for h in hist_a]}")
print(f"  通常モード: {[h[:40] for h in hist_b]}")
print(f"  差があるか: {hist_a != hist_b}")
PYEOF
