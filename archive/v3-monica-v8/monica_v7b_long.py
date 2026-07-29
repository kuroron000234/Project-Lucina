"""
v7b — 逆温度制御 vs 二重タイムスケール 長期比較
複数ターンの会話で自己モデルドリフトの推移を可視化する。
"""

import json, sys, time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOW, MID, HIGH = 2, 18, 35

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, dtype=DTYPE,
)
model.eval()

class Hooks:
    def __init__(self):
        self.states = {}
        self.handles = []
    def make(self, idx):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.states[idx] = out[:, -1, :].detach().to(dtype=torch.float32, device="cpu")
            elif isinstance(out, tuple):
                self.states[idx] = out[0][:, -1, :].detach().to(dtype=torch.float32, device="cpu")
        return fn
    def register(self):
        for idx in [LOW, MID, HIGH]:
            self.handles.append(model.model.layers[idx].register_forward_hook(self.make(idx)))
    def clear(self): self.states.clear()
    def remove(self):
        for h in self.handles: h.remove()

hooks = Hooks()
hooks.register()

def cosine_dist(a, b):
    return float(1.0 - F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())

# ── 逆温度制御（改良版） ──
class DriftInverseTemp:
    """PEが低い（退屈）ほど自己モデルが不安定化＝ドリフト増大"""
    name = "逆温度制御"
    def __init__(self):
        self.self_model = 0.30
        self.drift_total = 0.0

    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        pe = pe_mid if session_avg is None else session_avg
        update_rate = 1.0 / (1.0 + pe * 12)
        noise = np.random.normal(0, 0.05)
        d = update_rate * noise
        self.drift_total += d
        self.self_model = float(np.clip(self.self_model + d, 0.05, 0.95))

# ── 二重タイムスケール ──
class DriftDualTime:
    name = "二重タイムスケール"
    def __init__(self):
        self.fast = 0.30
        self.slow = 0.30
        self.drift_total = 0.0
        self.steps = 0

    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        self.steps += 1
        self.fast = 0.98 * self.fast + 0.02 * (pe_low + pe_mid + pe_high) / 3
        if session_avg is not None:
            self.slow = 0.995 * self.slow + 0.005 * session_avg
            self.slow += np.random.normal(0, 0.002)
        gap = abs(self.fast - self.slow)
        self.drift_total += gap * 0.02
        self.self_model = 0.90 * self.fast + 0.10 * self.slow

# ── 測定効果（修正版） ──
class DriftHeisenberg:
    name = "測定効果"
    def __init__(self):
        self.vec = np.random.randn(2560).astype(np.float32)
        self.vec /= np.linalg.norm(self.vec)
        self.self_model = 0.30
        self.drift_total = 0.0

    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        h = hooks.states.get(HIGH)
        if h is not None:
            h_np = h.numpy().flatten().astype(np.float32)
            h_norm = np.linalg.norm(h_np) + 1e-8
            proj = float(np.dot(self.vec, h_np) / h_norm)
            perturbation = np.clip(proj * 0.01, -0.05, 0.05)

            self.vec += np.random.randn(2560).astype(np.float32) * abs(perturbation) * 0.5
            self.vec /= np.linalg.norm(self.vec)

            self.self_model = float(np.clip(self.self_model + perturbation * 0.2, 0.05, 0.95))
            self.drift_total += abs(perturbation)
        else:
            pe = pe_mid if session_avg is None else session_avg
            self.self_model = 0.99 * self.self_model + 0.01 * pe

# ── 退屈スイッチ（修正版） ──
class DriftBoredom:
    """低PEが続くとDMNスイッチが入り、自己モデルが強制的にジャンプ"""
    name = "退屈スイッチ"
    def __init__(self):
        self.self_model = 0.30
        self.drift_total = 0.0
        self.low_pe_count = 0

    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        current_pe = pe_mid if session_avg is None else session_avg
        running_avg = getattr(self, '_running_avg', 0.5)
        if not hasattr(self, '_running_avg'):
            self._running_avg = current_pe
        self._running_avg = 0.95 * self._running_avg + 0.05 * current_pe

        # 相対的閾値: running_avg より 20% 低ければ「退屈」
        threshold = self._running_avg * 0.8
        if current_pe < threshold:
            self.low_pe_count += 1
        else:
            self.low_pe_count = max(0, self.low_pe_count - 1)

        if self.low_pe_count > 15:
            jump = np.random.normal(0, 0.08)
            self.drift_total += jump
            self.self_model = float(np.clip(self.self_model + jump, 0.05, 0.95))
            self.low_pe_count = 5
        else:
            noise = np.random.normal(0, 0.002)
            self.drift_total += noise
            self.self_model = float(np.clip(self.self_model + noise, 0.05, 0.95))

# ── 生成 ──
def generate_turn(messages, drift, max_new=100, temp=0.8):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp.get("attention_mask")
    if attn is not None: attn = attn.to(DEVICE)

    hooks.clear()
    pkv = None
    gen = input_ids.clone()
    prev, p2 = {LOW: None, MID: None, HIGH: None}, {LOW: None, MID: None, HIGH: None}
    pes_hist = {"low": [], "mid": [], "high": []}

    for step in range(max_new):
        with torch.no_grad():
            out = model(input_ids=gen if pkv is None else gen[:, -1:],
                        attention_mask=attn, past_key_values=pkv, use_cache=True)
        logits, pkv = out.logits[:, -1, :], out.past_key_values

        cur_pes = {}
        for label, idx in [("low", LOW), ("mid", MID), ("high", HIGH)]:
            cur = hooks.states.get(idx)
            p = prev.get(idx); pp = p2.get(idx)
            if cur is not None and p is not None and pp is not None:
                pe = cosine_dist(p + (p - pp), cur)
            elif cur is not None and p is not None:
                pe = cosine_dist(p, cur)
            else:
                pe = 0.3
            cur_pes[label] = pe
            pes_hist[label].append(pe)
            if cur is not None:
                p2[idx], prev[idx] = prev.get(idx), cur.clone()

        drift.update(cur_pes["low"], cur_pes["mid"], cur_pes["high"])

        pg = cur_pes["high"] - cur_pes["low"]
        adp = float(np.clip(temp * (1.0 + pg * 0.5), 0.3, 1.6))
        logits = logits / adp
        if pg > 0.05:
            v, i = logits.topk(5)
            logits.scatter_add_(-1, i, -min(pg * 0.3, 0.4) * v)

        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)

        if attn is not None:
            attn = torch.cat([attn, torch.ones((1, 1), device=DEVICE, dtype=attn.dtype)], dim=-1)
        gen = torch.cat([gen, nt], dim=-1)
        if nt[0, 0].item() == tokenizer.eos_token_id: break

    # セッション終了
    avg_mid = np.mean(pes_hist["mid"]) if pes_hist["mid"] else 0.3
    drift.update(np.mean(pes_hist["low"]), avg_mid, np.mean(pes_hist["high"]), session_avg=avg_mid)

    resp = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    stats = {
        "pe_low": float(np.mean(pes_hist["low"])),
        "pe_mid": float(avg_mid),
        "pe_high": float(np.mean(pes_hist["high"])),
        "self": float(drift.self_model),
        "drift": float(drift.drift_total),
    }
    return resp, stats


# ── 長期比較 ──
def run():
    prompts = [
        "「退屈」という感覚について教えてください。",
        "もっと詳しく。退屈な時、人は何を考えるの？",
        "じゃあ同じ話の繰り返しって、どんな気分？",
        "人工知能の未来について教えてください。",
        "自己紹介してください。",
        "人間の感情についてどう思いますか？",
        "また同じ話だけど、生きる意味って何だろうね。",
        "今日の天気はどうですか？（ダミー質問）",
        "哲学的な問い：自分が自分であるとはどういうこと？",
    ]

    for drift_cls in [DriftInverseTemp, DriftDualTime, DriftHeisenberg, DriftBoredom]:
        print(f"\n{'='*60}")
        print(f"  {drift_cls.name}")
        print(f"{'='*60}")

        d = drift_cls()
        conv = []

        for i, q in enumerate(prompts):
            conv.append({"role": "user", "content": q})
            resp, stats = generate_turn(conv, d, max_new=80)
            conv.append({"role": "assistant", "content": resp})

            print(f"  [{i+1}] self={stats['self']:.4f} drift={stats['drift']:.4f}  "
                  f"PE: low={stats['pe_low']:.3f} mid={stats['pe_mid']:.3f} high={stats['pe_high']:.3f}")
            print(f"       >> {q[:40]}")
            print(f"       << {resp[:60]}")

            if len(conv) > 8: conv = conv[-8:]

    hooks.remove()

if __name__ == "__main__":
    run()
