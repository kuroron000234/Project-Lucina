"""FEP診断：予測誤差の有無が応答を変えるか検証"""
import json, urllib.request, time

MODEL = "batiai/qwen3.5-9b:q4"
OPTS = {"num_ctx": 8192, "temperature": 0.3}

def _chatml(msgs):
    out = ""
    for m in msgs:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out

def gen(msgs, max_tokens=0):
    prompt = _chatml(msgs)
    opts = {**OPTS}
    if max_tokens: opts["num_predict"] = max_tokens
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": opts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("response","").strip()
        except: time.sleep(1)
    return ""

def extract_prediction(reply):
    return gen([
        {"role": "user", "content": f"さっきの内容:\n{reply[:200]}\n\n次に何が起きる？一文で。"}
    ])

def test_scenario(label, system, history, prediction, observation):
    """Run one FEP cycle and evaluate."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  予測: {prediction[:80] if prediction else '(なし)'}")
    print(f"  観測: {observation[:80]}")

    # Build context
    msgs = [{"role": "system", "content": system}] + history[:]

    if prediction:
        msgs.append({"role": "system", "content": f"(内部状態: {prediction})"})

    msgs.append({"role": "user", "content": observation})

    t0 = time.time()
    reply = gen(msgs)
    elapsed = time.time() - t0
    print(f"  応答 ({elapsed:.1f}s): {reply[:200]}")
    return reply


# ========== TEST 1: 予測一致 vs 不一致の比較 ==========
print("\n========== TEST 1: 予測 vs 現実の一致/不一致 ==========")

system_neutral = "あなたは会話エージェント。日本語で応答する。"

# Phase A: Make initial prediction
initial = gen([
    {"role": "user", "content": "あなたは部屋で本を読んでいる。次に何が起きる？簡潔に予想して。"}
], max_tokens=60)
print(f"\n初期予想: {initial}")

# Phase B: Test prediction match
history = [
    {"role": "user", "content": "あなたは部屋で本を読んでいる。"},
    {"role": "assistant", "content": "静かにページをめくっている。"},
]

r1 = test_scenario("TEST 1a: 予測一致（予測あり）", system_neutral, history, initial, "誰かがドアをノックした。")
r2 = test_scenario("TEST 1b: 予測不一致（予測あり）", system_neutral, history, "誰も来ないと予想していた", "誰かがドアをノックした。")
r3 = test_scenario("TEST 1c: 予測なし（対照群）", system_neutral, history, "", "誰かがドアをノックした。")

# Evaluate difference
print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
print(f"  1a(一致):   「{r1[:100]}」")
print(f"  1b(不一致): 「{r2[:100]}」")
print(f"  1c(対照):   「{r3[:100]}」")
print(f"  1a == 1c → 予測は無意味")
print(f"  1a != 1c → 予測フィードバックが効いている")
print(f"  1bにsurprise/矛盾表現 → 予測誤差検出")

PYEOF
