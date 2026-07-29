"""
v7 — 自己モデルドリフト機構の比較実験
4つの「決して埋まらない予測誤差」を実装・比較する。

手法:
  0. ベースライン（ランダムウォーク、現在のv6）
  1. 逆温度制御（退屈ほど敏感）
  2. 測定効果（Heisenberg型）
  3. 退屈スイッチ（DMN切り替え）
  4. 二重タイムスケール
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
DIM = 2560

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, dtype=DTYPE,
)
model.eval()
print(f"Model: {MODEL_NAME} ({len(model.model.layers)} layers)")

# ── フック ──
class Hooks:
    def __init__(self):
        self.states: dict[int, torch.Tensor | None] = {}
        self.handles = []
        self.all_states: dict[int, list] = {LOW: [], MID: [], HIGH: []}

    def make(self, idx):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                h = out[:, -1, :].detach().to(dtype=torch.float32, device="cpu")
            elif isinstance(out, tuple):
                h = out[0][:, -1, :].detach().to(dtype=torch.float32, device="cpu")
            else:
                return
            self.states[idx] = h
            self.all_states[idx].append(h.numpy().flatten())
        return fn

    def register(self):
        for idx in [LOW, MID, HIGH]:
            self.handles.append(
                model.model.layers[idx].register_forward_hook(self.make(idx))
            )

    def clear(self):
        self.states.clear()

    def remove(self):
        for h in self.handles:
            h.remove()

hooks = Hooks()
hooks.register()

# ── 共通FEP関数 ──
def cosine_dist(a, b):
    a, b = a.flatten(), b.flatten()
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

@dataclass
class DriftStats:
    """手法比較用の記録"""
    name: str
    self_traj: list = field(default_factory=list)
    pe_low_avg: float = 0.0
    pe_mid_avg: float = 0.0
    pe_high_avg: float = 0.0
    final_self: float = 0.0
    drift_total: float = 0.0
    tokens_generated: int = 0

# ── 4つのドリフト機構 ──

# --- ベースライン: ランダムウォーク ---
class DriftBaseline:
    name = "0_ベースライン(randwalk)"
    def __init__(self):
        self.self_model = 0.30
        self.drift = 0.0
    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        d = np.random.normal(0, 0.003)
        self.drift += d
        self.self_model = float(np.clip(self.self_model + d, 0.05, 0.95))

# --- 1. 逆温度制御 ---
class DriftInverseTemp:
    name = "1_逆温度制御"
    def __init__(self):
        self.self_model = 0.30
        self.drift = 0.0
    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        # PEが小さい（退屈）ほど更新率が高い
        if session_avg is not None:
            update_rate = 1.0 / (1.0 + session_avg * 8)
            target = self.self_model + np.random.normal(0, 0.02)
            d = update_rate * (target - self.self_model)
        else:
            d = np.random.normal(0, 0.001)
        self.drift += d
        self.self_model = float(np.clip(self.self_model + d, 0.05, 0.95))

# --- 2. 測定効果 ---
class DriftHeisenberg:
    name = "2_測定効果"
    def __init__(self):
        self.self_model = torch.zeros(DIM)
        self.drift = 0.0
    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        h = hooks.states.get(HIGH)
        if h is not None:
            h = h.to(dtype=torch.float32, device="cpu").flatten()
            # 測定: 自己モデルと隠れ状態の相互作用
            proj = (self.self_model * h).sum().item()
            # 測定が状態を変える -> 次に測定するときにずれる
            perturbation = torch.randn(DIM) * 0.001 * (1.0 + abs(proj) * 0.5)
            self.self_model = self.self_model + perturbation
            self.self_model = self.self_model / (self.self_model.norm() + 1e-8) * abs(self.self_model).mean().item()
            self.drift += float(perturbation.norm().item())
        else:
            self.self_model += torch.randn(DIM) * 0.0001

# --- 3. 退屈スイッチ ---
class DriftBoredom:
    name = "3_退屈スイッチ"
    def __init__(self):
        self.self_model = 0.30
        self.drift = 0.0
        self.low_pe_streak = 0
    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        low = pe_low < 0.40 and pe_mid < 0.35 and pe_high < 0.40
        if low:
            self.low_pe_streak += 1
        else:
            self.low_pe_streak = 0

        if self.low_pe_streak > 30:
            jump = np.random.normal(0, 0.08)
            self.drift += jump
            self.self_model = float(np.clip(self.self_model + jump, 0.05, 0.95))
            self.low_pe_streak = 15
        else:
            d = np.random.normal(0, 0.001)
            self.drift += d
            self.self_model = float(np.clip(self.self_model + d, 0.05, 0.95))

# --- 4. 二重タイムスケール ---
class DriftDualTime:
    name = "4_二重タイムスケール"
    def __init__(self):
        self.fast = 0.30  # トークン単位で更新
        self.slow = 0.30  # 会話単位で更新
        self.drift = 0.0
        self.steps = 0
    def update(self, pe_low, pe_mid, pe_high, session_avg=None):
        self.steps += 1
        # fast: 毎トークン即時更新
        self.fast = 0.99 * self.fast + 0.01 * (pe_low + pe_mid + pe_high) / 3

        # slow: sessionごとに少しだけ動く
        if session_avg is not None and self.steps % 10 == 0:
            self.slow = 0.999 * self.slow + 0.001 * session_avg
            self.slow += np.random.normal(0, 0.002)

        # ギャップ = 決して埋まらない誤差
        gap = abs(self.fast - self.slow)
        self.drift += gap * 0.01
        self.self_model = 0.95 * self.fast + 0.05 * self.slow

# ── 生成 ──
def generate_one(messages, drift, max_new=80, temp=0.8):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp.get("attention_mask")
    if attn is not None:
        attn = attn.to(DEVICE)

    hooks.clear()
    pkv = None
    gen = input_ids.clone()
    prev = {LOW: None, MID: None, HIGH: None}
    p2 = {LOW: None, MID: None, HIGH: None}
    token_log = []
    stats = DriftStats(name=drift.name)
    pe_mid = 0.3

    for step in range(max_new):
        with torch.no_grad():
            out = model(
                input_ids=gen if pkv is None else gen[:, -1:],
                attention_mask=attn, past_key_values=pkv, use_cache=True,
            )
        logits = out.logits[:, -1, :]
        pkv = out.past_key_values

        pes = {}
        for label, idx in [("low", LOW), ("mid", MID), ("high", HIGH)]:
            cur = hooks.states.get(idx)
            p = prev.get(idx)
            pp = p2.get(idx)
            if cur is not None and p is not None and pp is not None:
                pe = cosine_dist(p + (p - pp), cur)
            elif cur is not None and p is not None:
                pe = cosine_dist(p, cur)
            else:
                pe = 0.3
            pes[label] = pe
            if cur is not None:
                p2[idx] = prev.get(idx)
                prev[idx] = cur.clone()

        # ドリフト更新
        drift.update(pes["low"], pes["mid"], pes["high"])
        stats.self_traj.append(drift.self_model if not isinstance(drift.self_model, torch.Tensor) else drift.self_model.mean().item())

        # FEP: PE勾配で温度
        pg = pes["high"] - pes["low"]
        adp_temp = float(np.clip(temp * (1.0 + pg * 0.5), 0.3, 1.6))
        logits = logits / adp_temp

        if pg > 0.05:
            v, i = logits.topk(5)
            logits.scatter_add_(-1, i, -min(pg * 0.3, 0.4) * v)

        # 自己モデルの影響: 乖離が大きいと温度上昇
        sm_val = drift.self_model if not isinstance(drift.self_model, torch.Tensor) else drift.self_model.mean().item()
        if abs(sm_val - 0.3) > 0.1:
            logits = logits / (1.0 + abs(sm_val - 0.3) * 0.5)

        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)

        if attn is not None:
            attn = torch.cat([attn, torch.ones((1, 1), device=DEVICE, dtype=attn.dtype)], dim=-1)
        gen = torch.cat([gen, nt], dim=-1)

        if nt[0, 0].item() == tokenizer.eos_token_id:
            break

    # セッション終了処理
    drift.update(pes["low"], pes["mid"], pes["high"], session_avg=pe_mid)
    stats.self_traj.append(drift.self_model if not isinstance(drift.self_model, torch.Tensor) else drift.self_model.mean().item())

    stats.pe_low_avg = float(np.mean([pes["low"]]))
    stats.pe_mid_avg = float(np.mean([pes["mid"]]))
    stats.pe_high_avg = float(np.mean([pes["high"]]))
    stats.final_self = float(drift.self_model if not isinstance(drift.self_model, torch.Tensor) else drift.self_model.mean().item())
    stats.drift_total = float(drift.drift if hasattr(drift, 'drift') else 0)
    stats.tokens_generated = len(stats.self_traj)

    resp = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp, stats


# ── 比較実行 ──
def run_comparison():
    prompts = [
        [{"role": "user", "content": "こんにちは、人工知能の未来について教えてください。"}],
        [{"role": "user", "content": "こんにちは"},
         {"role": "assistant", "content": "こんにちは！"},
         {"role": "user", "content": "人間の感情についてどう思いますか？"}],
    ]

    drifters = [
        DriftBaseline(),
        DriftInverseTemp(),
        DriftHeisenberg(),
        DriftBoredom(),
        DriftDualTime(),
    ]

    for pi, msgs in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"Prompt {pi+1}: {msgs[-1]['content'][:50]}")
        print(f"{'='*70}")

        results = []
        for dr in drifters:
            dr2 = deepcopy(dr)
            resp, stats = generate_one(msgs, dr2, max_new=80)
            results.append((dr.name, stats, resp))

        for name, stats, resp in results:
            if stats.self_traj:
                start_self = stats.self_traj[0]
                end_self = stats.self_traj[-1]
                drift_range = max(stats.self_traj) - min(stats.self_traj)
            else:
                start_self = end_self = drift_range = 0

            print(f"\n  [{name}]")
            print(f"    drift_range={drift_range:.4f}  total_drift={stats.drift_total:.4f}")
            print(f"    self: {start_self:.4f} → {end_self:.4f}  (Δ={end_self-start_self:+.4f})")
            print(f"    PEs: low={stats.pe_low_avg:.3f} mid={stats.pe_mid_avg:.3f} high={stats.pe_high_avg:.3f}")
            print(f"    response: {resp[:80]}")

    hooks.remove()


if __name__ == "__main__":
    run_comparison()
