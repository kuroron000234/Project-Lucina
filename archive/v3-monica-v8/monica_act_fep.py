"""
Monica v5 — Activation-Level FEP
LLMの隠れ層ベクトルを直接読み取り、予測誤差を計算し、
内部状態にフィードバックする。

3層:
  Low  = トークン間の隠れ層遷移予測誤差（線形予測）
  Mid  = 応答単位の平均PE（会話の一貫性）
  High = 自己モデルのドリフト（決して収束しない）
"""

import gc, json, os, sys, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOG_PATH = Path(__file__).parent / "log_act_fep.jsonl"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
print(f"Loading {MODEL_NAME} ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, dtype=DTYPE,
)
model.eval()
print("Model loaded.")

# ── 隠れ層フック ──
class HiddenStateCache:
    def __init__(self):
        self.hidden = None
    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.hidden = output[:, -1, :].detach().to(dtype=torch.float32)
        elif isinstance(output, tuple):
            self.hidden = output[0][:, -1, :].detach().to(dtype=torch.float32)

cache = HiddenStateCache()
hook_handle = model.model.layers[-1].register_forward_hook(cache.hook_fn)
print(f"Hook on {type(model.model.layers[-1]).__name__}, hidden={model.config.hidden_size}")

# ── FEPユーティリティ ──
def cosine_distance(a, b):
    return float(1.0 - F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())

def entropy(probs):
    p = probs.flatten()
    return float(-(p * torch.log(p + 1e-10)).sum().item())

@dataclass
class FEPBeliefs:
    self_model: float = 0.30
    self_model_peak: float = 0.30
    self_model_steps: int = 0

    session_pe_sum: float = 0.0
    session_token_count: int = 0

    pe_window: list = field(default_factory=list)
    running_avg: float = 0.30
    running_var: float = 0.05
    drift_total: float = 0.0

    def update_low(self, pe: float):
        self.pe_window.append(pe)
        if len(self.pe_window) > 100:
            self.pe_window.pop(0)
        self.running_avg = 0.95 * self.running_avg + 0.05 * pe
        if len(self.pe_window) > 2:
            self.running_var = 0.90 * self.running_var + 0.10 * float(np.var(self.pe_window[-20:]))

    def update_mid(self, pe: float):
        self.session_pe_sum += pe
        self.session_token_count += 1

    def update_high(self, conv_len: int = 0):
        self.self_model_steps += 1
        if self.session_token_count > 0:
            avg = self.session_pe_sum / self.session_token_count
            self.self_model = 0.995 * self.self_model + 0.005 * avg
            self.self_model_peak = max(self.self_model_peak, avg)
        drift = np.random.normal(0, 0.003)
        self.drift_total += drift
        self.self_model = float(np.clip(self.self_model + drift, 0.05, 0.95))

    def reset_session(self):
        self.session_pe_sum = 0.0
        self.session_token_count = 0

    def get_state(self) -> dict:
        return {
            "run_avg": round(self.running_avg, 4),
            "run_var": round(self.running_var, 4),
            "self": round(self.self_model, 4),
            "peak": round(self.self_model_peak, 4),
            "drift": round(self.drift_total, 4),
        }

# ── 生成 ──
@torch.no_grad()
def generate_with_fep(
    messages: list,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    beliefs: FEPBeliefs | None = None,
    fep_strength: float = 1.0,
) -> tuple[str, list[dict], FEPBeliefs]:
    if beliefs is None:
        beliefs = FEPBeliefs()

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    past_key_values = None
    generated = input_ids.clone()
    prev_h = None
    prev_prev_h = None
    token_log = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        cur_h = cache.hidden
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # FEP: 線形予測誤差
        if prev_prev_h is not None:
            pe = cosine_distance(prev_h + (prev_h - prev_prev_h), cur_h)
        elif prev_h is not None:
            pe = cosine_distance(prev_h, cur_h)
        else:
            pe = 0.3

        beliefs.update_low(pe)
        beliefs.update_mid(pe)

        # FEP: アクション（ロジット変調）
        s = fep_strength
        temp = min(max(temperature + pe * s * 0.6, 0.3), 1.6)
        logits = logits / temp

        # 高PE時: top-5 抑制（急激な方向転換を抑える）
        if pe > beliefs.running_avg * 1.3:
            v, i = logits.topk(5)
            logits.scatter_add_(-1, i, -min(pe * s * 0.12, 0.4) * v)

        # 自己モデルバイアス: 高次PEが高いほど確信度抑制
        sm_bias = (beliefs.self_model - 0.3) * s * 0.3
        if sm_bias > 0:
            logits = logits / (1.0 + sm_bias)

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)

        token_log.append({
            "step": step, "pe": pe, "temp": temp,
            "entropy": entropy(probs),
            "tok": int(next_tok[0, 0]),
        })

        prev_prev_h = prev_h
        prev_h = cur_h.clone()

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=DEVICE, dtype=attention_mask.dtype)], dim=-1)
        generated = torch.cat([generated, next_tok], dim=-1)

        if next_tok[0, 0].item() == tokenizer.eos_token_id:
            break
        if step % 50 == 0 and step > 0:
            torch.cuda.empty_cache()

    beliefs.update_high(conv_len=len(messages))
    beliefs.reset_session()

    response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response, token_log, beliefs

# ── Interface ──
def interactive():
    print(f"\nMonica v5 — Activation-Level FEP (strength={1.0})")
    print(f"Model: {MODEL_NAME}\n  /state  /drift  exit\n")

    beliefs = FEPBeliefs()
    conv = []

    while True:
        try:
            u = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not u: continue
        if u.lower() in ("exit", "quit", "終了"): break
        if u == "/state":
            print(f"  {json.dumps(beliefs.get_state())}")
            continue
        if u == "/drift":
            print(f"  total drift={beliefs.drift_total:.4f}")
            continue

        conv.append({"role": "user", "content": u})

        t0 = time.time()
        response, token_log, beliefs = generate_with_fep(conv, beliefs=beliefs)
        dt = time.time() - t0

        if not response:
            response = "…"
        conv.append({"role": "assistant", "content": response})

        avg_pe = float(np.mean([t["pe"] for t in token_log])) if token_log else 0
        avg_temp = float(np.mean([t["temp"] for t in token_log])) if token_log else 0
        print(f"  [{len(token_log)}tok {dt:.1f}s pe={avg_pe:.4f} temp={avg_temp:.2f}]")
        print(f"  {response[:300]}")
        print(f"  self={beliefs.self_model:.4f} peak={beliefs.self_model_peak:.4f}")

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "user": u[:80], "response": response[:200],
                "n_tok": len(token_log), "avg_pe": round(avg_pe, 4),
                "avg_temp": round(avg_temp, 2),
                "state": beliefs.get_state(),
            }, ensure_ascii=False) + "\n")

        if len(conv) > 6:
            conv = conv[-6:]
        torch.cuda.empty_cache()

def quick_test():
    beliefs = FEPBeliefs()
    tests = [
        [{"role": "user", "content": "こんにちは"}],
        [{"role": "user", "content": "こんにちは"},
         {"role": "assistant", "content": "こんにちは！何かお手伝いできますか？"},
         {"role": "user", "content": "自己紹介してください。"}],
    ]
    for msgs in tests:
        r, log, beliefs = generate_with_fep(msgs, max_new_tokens=60, beliefs=beliefs)
        ap = float(np.mean([t["pe"] for t in log])) if log else 0
        at = float(np.mean([t["temp"] for t in log])) if log else 0
        print(f"  >> {msgs[-1]['content']}")
        print(f"  << {r[:120]}")
        print(f"     {len(log)}tok pe={ap:.4f} temp={at:.2f}")

def compare():
    beliefs = FEPBeliefs()
    msgs = [{"role": "user", "content": "人工知能の未来について教えてください。"}]

    print("=== FEP ===")
    r, log, beliefs = generate_with_fep(msgs, max_new_tokens=80, beliefs=beliefs)
    print(f"  {r[:200]}")
    ap = float(np.mean([t["pe"] for t in log])) if log else 0
    print(f"  pe={ap:.4f}")

    print("\n=== 通常生成 ===")
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=80, temperature=0.8, do_sample=True)
    r2 = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"  {r2[:200]}")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--test":
        quick_test()
    elif arg == "--compare":
        compare()
    else:
        try:
            interactive()
        finally:
            hook_handle.remove()
