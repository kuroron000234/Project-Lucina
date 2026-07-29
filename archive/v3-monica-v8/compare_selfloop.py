"""
自己ループ比較: LLMの出力を次の入力としてfeed。
FEPなし → モード崩壊するか？
FEPあり → 崩壊を回避できるか？
"""
import sys, time, torch, numpy as np, re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_full = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, dtype=DTYPE)
model_full.eval()
print("Model loaded")

from monica_v6 import (
    MultiHook, SteeringVector, MultiFEPBeliefs,
    LAYER_ORDER,
    FEP_SELF_MODEL_BASELINE, STEER_BUFFER_MAX,
    STEER_STRENGTH_COEFF, STEER_ACTIVATION_THRESHOLD,
    TEMP_GRADIENT_COEFF, TEMP_SELF_MODEL_COEFF, TEMP_MIN, TEMP_MAX,
    PE_GRADIENT_PENALTY_THRESHOLD, PENALTY_TOP_K, PENALTY_COEFF, PENALTY_COEFF_UPPER,
)

TOKENS_PER_LOOP = 30
MAX_LOOPS = 30
INITIAL_PROMPT = "私は人工知能です。自分の考えを自由に書きます。"

def diversity_score(texts):
    """n-gramベースの多様性指標: 0=完全一致, 1=完全に異なる"""
    if len(texts) < 2:
        return 1.0
    texts = [t[:100] for t in texts]
    n_match = sum(1 for i in range(len(texts)-1) for j in range(i+1, len(texts))
                  if texts[i] == texts[j])
    n_pairs = len(texts) * (len(texts) - 1) / 2
    return 1.0 - n_match / n_pairs


def run_bare_selfloop():
    """FEPなし自己ループ"""
    gen = torch.tensor([[tokenizer.bos_token_id or 1]], device=DEVICE)
    initial = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(DEVICE)
    gen = torch.cat([gen, initial["input_ids"]], dim=-1)
    outputs = []
    past_kv = None

    for loop in range(MAX_LOOPS):
        with torch.no_grad():
            out = model_full.generate(
                gen, max_new_tokens=TOKENS_PER_LOOP,
                do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0, gen.shape[1]:], skip_special_tokens=True).strip()
        if not text:
            text = "."
        outputs.append(text)

        # 出力を入力に追加（自己ループ）
        new_toks = out[:, gen.shape[1]:]
        gen = torch.cat([gen, new_toks], dim=-1)

        # 崩壊検出: 3回連続で出力が一致
        if len(outputs) >= 3:
            last3 = outputs[-3:]
            if all(a[:40] == b[:40] for a, b in zip(last3, last3[1:])):
                return outputs, loop + 1

        if gen.shape[1] > 1500:
            gen = gen[:, -1000:]

        torch.cuda.empty_cache()
    return outputs, None


def run_fep_selfloop():
    """FEPあり自己ループ"""
    hook = MultiHook()
    hook.register(model_full, [2, 18, 35])
    steer = SteeringVector()
    beliefs = MultiFEPBeliefs()
    prev_h = {2: None, 18: None, 35: None}
    prev2_h = {2: None, 18: None, 35: None}

    initial = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(DEVICE)
    gen = torch.cat([
        torch.tensor([[tokenizer.bos_token_id or 1]], device=DEVICE),
        initial["input_ids"]
    ], dim=-1)
    outputs = []

    for loop in range(MAX_LOOPS):
        hook.clear()
        past_kv = None
        g = gen.clone()

        for step in range(TOKENS_PER_LOOP):
            with torch.no_grad():
                base_out = model_full.model(
                    input_ids=g if past_kv is None else g[:, -1:],
                    past_key_values=past_kv, use_cache=True,
                )
            past_kv = base_out.past_key_values
            last_hidden = base_out.last_hidden_state[:, -1:, :]

            sm_delta = beliefs.self_model - FEP_SELF_MODEL_BASELINE
            if steer.vector is not None and abs(sm_delta) > STEER_ACTIVATION_THRESHOLD:
                sv_t = torch.from_numpy(steer.vector).to(device=last_hidden.device, dtype=last_hidden.dtype)
                last_hidden = last_hidden + sv_t.unsqueeze(0) * sm_delta * STEER_STRENGTH_COEFF

            pred_list, cur_list, label_list = [], [], []
            for label, idx in LAYER_ORDER:
                cur = hook.get(idx)
                p = prev_h.get(idx)
                p2 = prev2_h.get(idx)
                if cur is not None and p is not None and p2 is not None:
                    pred_list.append((p + (p - p2)).flatten())
                    cur_list.append(cur.flatten())
                    label_list.append(label)
                elif cur is not None and p is not None:
                    pred_list.append(p.flatten())
                    cur_list.append(cur.flatten())
                    label_list.append(label)
                prev2_h[idx] = prev_h.get(idx)
                prev_h[idx] = cur.clone() if cur is not None else prev_h.get(idx)

            pe_low = pe_mid = pe_high = FEP_SELF_MODEL_BASELINE
            if pred_list:
                pred_t = torch.stack(pred_list)
                cur_t = torch.stack(cur_list)
                pes = 1.0 - F.cosine_similarity(pred_t, cur_t, dim=1)
                for lab, pe in zip(label_list, pes.tolist()):
                    if lab == "low": pe_low = pe
                    elif lab == "mid": pe_mid = pe
                    elif lab == "high": pe_high = pe

            beliefs.update_mid_session(pe_mid)
            beliefs.drift_update(pe_mid)

            pe_grad = pe_high - pe_low
            temp_off = pe_grad * TEMP_GRADIENT_COEFF + abs(sm_delta) * TEMP_SELF_MODEL_COEFF
            adaptive_temp = float(np.clip(0.8 * (1.0 + temp_off), TEMP_MIN, TEMP_MAX))

            hh = hook.get(35)
            if hh is not None and len(steer.high_pe_buffer) < STEER_BUFFER_MAX:
                steer.observe(hh.cpu().numpy().flatten(), pe_high)

            logits = model_full.lm_head(last_hidden.to(dtype=model_full.lm_head.weight.dtype))
            logits = logits[:, -1, :] / adaptive_temp

            if pe_grad > PE_GRADIENT_PENALTY_THRESHOLD:
                v, i = logits.topk(PENALTY_TOP_K)
                logits.scatter_add_(-1, i, -min(pe_grad * PENALTY_COEFF, PENALTY_COEFF_UPPER) * v)

            probs = F.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, 1)
            g = torch.cat([g, nt], dim=-1)

            if nt[0, 0].item() == tokenizer.eos_token_id:
                break

        beliefs.finalize_session()
        steer.update()

        text = tokenizer.decode(g[0, gen.shape[1]:], skip_special_tokens=True).strip()
        if not text:
            text = "."
        outputs.append(text)

        gen = g.clone()
        if gen.shape[1] > 1500:
            gen = gen[:, -1000:]

        if len(outputs) >= 3:
            last3 = outputs[-3:]
            if all(a[:40] == b[:40] for a, b in zip(last3, last3[1:])):
                hook.remove()
                return outputs, loop + 1

        torch.cuda.empty_cache()

    hook.remove()
    return outputs, None


# ── 実行 ──
print(f"自己ループ比較")
print(f"初期コンテキスト: '{INITIAL_PROMPT}'")
print(f"1ループ{TOKENS_PER_LOOP}トークン追加 × 最大{MAX_LOOPS}ループ\n")

print("【BARE: FEPなし自己ループ】")
t0 = time.time()
bare_out, bare_collapse = run_bare_selfloop()
print(f"  時間: {time.time()-t0:.1f}s")
if bare_collapse:
    print(f"  >>> {bare_collapse}ループ目で崩壊!")
else:
    print(f"  >>> {MAX_LOOPS}ループ持続")

print("\n【FEP: 多層FEP+ステアリング】")
t0 = time.time()
fep_out, fep_collapse = run_fep_selfloop()
print(f"  時間: {time.time()-t0:.1f}s")
if fep_collapse:
    print(f"  >>> {fep_collapse}ループ目で崩壊!")
else:
    print(f"  >>> {MAX_LOOPS}ループ持続")

print("\n" + "─"*60)
print("出力推移（先頭60文字）:")
for i in range(max(len(bare_out), len(fep_out))):
    b = (bare_out[i] if i < len(bare_out) else "-")[:60]
    f = (fep_out[i] if i < len(fep_out) else "-")[:60]
    m = " ← COLLAPSE" if (bare_collapse and i+1 == bare_collapse) or (fep_collapse and i+1 == fep_collapse) else ""
    print(f"  [{i+1:2d}] BARE: {b}")
    print(f"       FEP:  {f}{m}")
print("─"*60)

# 多様性スコア（最終10ループ）
bd = diversity_score(bare_out[-10:])
fd = diversity_score(fep_out[-10:])
print(f"最終10ループの多様性: BARE={bd:.2f}  FEP={fd:.2f}  (1=完全多様, 0=全部同一)")
print(f"崩壊: BARE={'@'+str(bare_collapse) if bare_collapse else 'なし'}  "
      f"FEP={'@'+str(fep_collapse) if fep_collapse else 'なし'}")
