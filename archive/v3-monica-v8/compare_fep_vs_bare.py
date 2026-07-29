"""
FEPあり/なし 空リクエストループ比較。
どちらが長く多様な出力を維持できるか。
"""
import sys, time, torch, numpy as np
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

TOKENS_PER_LOOP = 40
MAX_LOOPS = 25
FIXED_PROMPT = [{"role": "user", "content": "人工知能について教えてください。"}]

def tokens_are_too_similar(a: str, b: str) -> bool:
    a, b = a[:80], b[:80]
    if len(a) < 10 or len(b) < 10:
        return False
    return a == b


def run_bare():
    """固定プロンプトでループ（毎回同じユーザー入力を与える）"""
    text = tokenizer.apply_chat_template(FIXED_PROMPT, tokenize=False, add_generation_prompt=True)
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    if not text.startswith(tokenizer.bos_token or ""):
        pass  # chat template に BOS が含まれる想定

    outputs = []
    for loop in range(MAX_LOOPS):
        inp = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model_full.generate(
                **inp, max_new_tokens=TOKENS_PER_LOOP,
                do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not text:
            text = ".."
        outputs.append(text)

        if len(outputs) >= 3:
            if all(tokens_are_too_similar(outputs[-i], outputs[-i-1]) for i in range(1, 3)):
                return outputs, loop + 1
        torch.cuda.empty_cache()
    return outputs, None


def run_fep():
    """per-token生成 + FEP制御 をループ"""
    hook = MultiHook()
    hook.register(model_full, [2, 18, 35])
    steer = SteeringVector()
    beliefs = MultiFEPBeliefs()
    prev_h = {2: None, 18: None, 35: None}
    prev2_h = {2: None, 18: None, 35: None}

    text = tokenizer.apply_chat_template(FIXED_PROMPT, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True)
    prompt_ids = inp["input_ids"].to(DEVICE)

    outputs = []
    for loop in range(MAX_LOOPS):
        hook.clear()
        gen = prompt_ids.clone()
        past_kv = None

        for step in range(TOKENS_PER_LOOP):
            with torch.no_grad():
                base_out = model_full.model(
                    input_ids=gen if past_kv is None else gen[:, -1:],
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
            gen = torch.cat([gen, nt], dim=-1)

            if nt[0, 0].item() == tokenizer.eos_token_id:
                break

        beliefs.finalize_session()
        steer.update()

        text = tokenizer.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        if not text:
            text = ".."
        outputs.append(text)

        if len(outputs) >= 3:
            if all(tokens_are_too_similar(outputs[-i], outputs[-i-1]) for i in range(1, 3)):
                hook.remove()
                return outputs, loop + 1
        torch.cuda.empty_cache()

    hook.remove()
    return outputs, None


# ── 実行 ──
print(f"固定プロンプトループ比較: {FIXED_PROMPT[0]['content']}")
print(f"1ループ{TOKENS_PER_LOOP}トークン × 最大{MAX_LOOPS}ループ")
print("毎回同じプロンプトから生成→リセット→繰り返し\n")

print("【BARE: 素の generate() (temp=0.8)】")
t0 = time.time()
bare_out, bare_collapse = run_bare()
print(f"  時間: {time.time()-t0:.1f}s")
if bare_collapse:
    print(f"  >>> {bare_collapse}ループ目で崩壊 (3回連続一致)")
else:
    print(f"  >>> {MAX_LOOPS}ループ持続（崩壊なし）")

print("\n【FEP: 多層FEP+ステアリング】")
t0 = time.time()
fep_out, fep_collapse = run_fep()
print(f"  時間: {time.time()-t0:.1f}s")
if fep_collapse:
    print(f"  >>> {fep_collapse}ループ目で崩壊 (3回連続一致)")
else:
    print(f"  >>> {MAX_LOOPS}ループ持続（崩壊なし）")

print("\n" + "─"*60)
print("各ループの出力先頭40文字:")
for i in range(max(len(bare_out), len(fep_out))):
    b = (bare_out[i] if i < len(bare_out) else "-")[:40]
    f = (fep_out[i] if i < len(fep_out) else "-")[:40]
    print(f"  [{i+1:2d}] BARE: {b}")
    print(f"       FEP:  {f}")
print("─"*60)
print(f"BARE: {len(bare_out)}ループ {'崩壊@'+str(bare_collapse) if bare_collapse else '持続'}")
print(f"FEP:  {len(fep_out)}ループ {'崩壊@'+str(fep_collapse) if fep_collapse else '持続'}")
