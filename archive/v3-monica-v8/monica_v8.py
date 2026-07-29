"""
Monica v8 — 3層FEP + 行動選択（自律エージェント）
Meta層が自己モデルの状態に応じてCHAT/THINK/IDLEを切り替える。
ユーザー入力がなくても内部的に活動し続ける。
"""
import json, sys, time, threading, torch, numpy as np, queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOG_PATH = Path(__file__).parent / "log_v8.jsonl"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LAYER_ORDER = [("low", 2), ("mid", 18), ("high", 35)]
FEP_SELF_MODEL_BASELINE = 0.30

DRIFT_INV_TEMP_COEFF = 6.0
DRIFT_NOISE_STD = 0.03
DRIFT_RESTORING_COEFF = 0.03
SELF_MODEL_MIN = 0.05
SELF_MODEL_MAX = 0.95

STEER_BUFFER_MAX = 200
STEER_HIGH_PE_THRESHOLD = 0.65
STEER_LOW_PE_THRESHOLD_FACTOR = 0.6
STEER_MIN_SAMPLES = 10
STEER_STRENGTH_COEFF = 0.50
STEER_ACTIVATION_THRESHOLD = 0.01

TEMP_GRADIENT_COEFF = 0.50
TEMP_SELF_MODEL_COEFF = 0.80
TEMP_MIN = 0.3
TEMP_MAX = 1.8

PE_GRADIENT_PENALTY_THRESHOLD = 0.05
PENALTY_TOP_K = 5
PENALTY_COEFF = 0.25
PENALTY_COEFF_UPPER = 0.35

FEP_AVG_DECAY = 0.95
FEP_AVG_UPDATE = 0.05
FEP_VAR_DECAY = 0.90
FEP_VAR_UPDATE = 0.10
FEP_HISTORY_MAX = 100
FEP_VAR_WINDOW = 20

DEFAULT_TEMPERATURE = 0.8
CONTEXT_MAX_LENGTH = 4096
CONVERSATION_HISTORY_LIMIT = 10
THINK_TOKENS = 40
CHAT_TIMEOUT = 8.0
IDLE_DRIFT_STEPS = 5

print(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
print(f"Loading {MODEL_NAME} ...")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, dtype=DTYPE)
model.eval()
print(f"Model loaded. {len(model.model.layers)} layers, hidden={model.config.hidden_size}")

# ── フック ──
class MultiHook:
    def __init__(self):
        self.states = {}
        self.handles = []
    def make_hook(self, idx):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                h = output[:, -1, :].detach().to(dtype=torch.float32)
            elif isinstance(output, tuple):
                h = output[0][:, -1, :].detach().to(dtype=torch.float32)
            else:
                return
            self.states[idx] = h
        return fn
    def register(self, model, indices):
        for idx in indices:
            self.handles.append(model.model.layers[idx].register_forward_hook(self.make_hook(idx)))
    def get(self, idx):
        return self.states.get(idx)
    def clear(self):
        self.states.clear()
    def remove(self):
        for h in self.handles:
            h.remove()

hooks = MultiHook()
hooks.register(model, [2, 18, 35])
print(f"Hooks on layers 2, 18, 35")

# ── ステアリング ──
class SteeringVector:
    def __init__(self):
        self.high_buffer = []
        self.low_buffer = []
        self.vector = None
    def observe(self, hidden, pe):
        if pe > STEER_HIGH_PE_THRESHOLD and len(self.high_buffer) < STEER_BUFFER_MAX:
            self.high_buffer.append(hidden)
        elif pe < STEER_HIGH_PE_THRESHOLD * STEER_LOW_PE_THRESHOLD_FACTOR and len(self.low_buffer) < STEER_BUFFER_MAX:
            self.low_buffer.append(hidden)
    def update(self):
        if len(self.high_buffer) < STEER_MIN_SAMPLES or len(self.low_buffer) < STEER_MIN_SAMPLES:
            return False, 0.0
        h = np.mean(self.high_buffer, axis=0)
        l = np.mean(self.low_buffer, axis=0)
        d = h - l
        n = np.linalg.norm(d)
        if n > 1e-8:
            self.vector = d / n
            return True, n
        return False, 0.0

steer = SteeringVector()

def entropy(probs):
    p = probs.flatten()
    return float(-(p * torch.log(p + 1e-10)).sum().item())

# ── FEP Beliefs ──
@dataclass
class LayerFEP:
    running_avg: float = FEP_SELF_MODEL_BASELINE
    running_var: float = 0.05
    history: list = field(default_factory=list)
    def update(self, pe):
        self.history.append(pe)
        if len(self.history) > FEP_HISTORY_MAX:
            self.history.pop(0)
        self.running_avg = FEP_AVG_DECAY * self.running_avg + FEP_AVG_UPDATE * pe
        if len(self.history) > 2:
            self.running_var = FEP_VAR_DECAY * self.running_var + FEP_VAR_UPDATE * float(np.var(self.history[-FEP_VAR_WINDOW:]))
    def zscore(self, pe):
        s = np.sqrt(self.running_var + 1e-6)
        return (pe - self.running_avg) / s if s > 0.01 else 0.0

@dataclass
class MultiFEPBeliefs:
    low: LayerFEP = field(default_factory=LayerFEP)
    mid: LayerFEP = field(default_factory=LayerFEP)
    high: LayerFEP = field(default_factory=LayerFEP)
    session_pe_sum: float = 0.0
    session_tokens: int = 0
    self_model: float = FEP_SELF_MODEL_BASELINE
    self_drift: float = 0.0

    def drift_update(self, pe_mid):
        update_rate = 1.0 / (1.0 + pe_mid * DRIFT_INV_TEMP_COEFF)
        noise = np.random.normal(0, DRIFT_NOISE_STD)
        restoring = (FEP_SELF_MODEL_BASELINE - self.self_model) * DRIFT_RESTORING_COEFF
        d = update_rate * noise + restoring
        self.self_drift += d
        self.self_model = float(np.clip(self.self_model + d, SELF_MODEL_MIN, SELF_MODEL_MAX))

    def finalize_session(self):
        if self.session_tokens > 0:
            self.session_pe_sum = 0.0
            self.session_tokens = 0

    def state(self):
        return {"low_avg": round(self.low.running_avg, 3),
                "mid_avg": round(self.mid.running_avg, 3),
                "high_avg": round(self.high.running_avg, 3),
                "self": round(self.self_model, 3),
                "drift": round(self.self_drift, 3)}

# ── 生成関数（対話用） ──
@torch.no_grad()
def generate_chat(messages, max_new=200, temp=DEFAULT_TEMPERATURE, beliefs=None):
    if beliefs is None:
        beliefs = MultiFEPBeliefs()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX_LENGTH)
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp.get("attention_mask")
    if attn is not None:
        attn = attn.to(DEVICE)
    hooks.clear()
    pkv = None
    gen = input_ids.clone()
    prev_h = {2: None, 18: None, 35: None}
    prev2_h = {2: None, 18: None, 35: None}
    token_log = []

    for step in range(max_new):
        with torch.no_grad():
            base_out = model.model(
                input_ids=gen if pkv is None else gen[:, -1:],
                attention_mask=attn, past_key_values=pkv, use_cache=True,
            )
        pkv = base_out.past_key_values
        last_hidden = base_out.last_hidden_state[:, -1:, :]

        sm_delta = beliefs.self_model - FEP_SELF_MODEL_BASELINE
        if steer.vector is not None and abs(sm_delta) > STEER_ACTIVATION_THRESHOLD:
            sv_t = torch.from_numpy(steer.vector).to(dtype=last_hidden.dtype, device=last_hidden.device)
            last_hidden = last_hidden + sv_t.unsqueeze(0) * sm_delta * STEER_STRENGTH_COEFF

        logits = model.lm_head(last_hidden.to(dtype=model.lm_head.weight.dtype))
        logits = logits[:, -1, :]

        pred_list, cur_list, label_list = [], [], []
        for lab, idx in LAYER_ORDER:
            cur = hooks.get(idx)
            p = prev_h.get(idx)
            p2 = prev2_h.get(idx)
            if cur is not None and p is not None and p2 is not None:
                pred_list.append((p + (p - p2)).flatten())
                cur_list.append(cur.flatten())
                label_list.append(lab)
            elif cur is not None and p is not None:
                pred_list.append(p.flatten())
                cur_list.append(cur.flatten())
                label_list.append(lab)
            prev2_h[idx] = prev_h.get(idx)
            prev_h[idx] = cur.clone() if cur is not None else prev_h.get(idx)

        pe_low = pe_mid = pe_high = FEP_SELF_MODEL_BASELINE
        if pred_list:
            pt = torch.stack(pred_list)
            ct = torch.stack(cur_list)
            pes = 1.0 - F.cosine_similarity(pt, ct, dim=1)
            for lab, pe in zip(label_list, pes.tolist()):
                if lab == "low": pe_low = pe; beliefs.low.update(pe)
                elif lab == "mid": pe_mid = pe; beliefs.mid.update(pe)
                elif lab == "high": pe_high = pe; beliefs.high.update(pe)

        beliefs.session_pe_sum += pe_mid
        beliefs.session_tokens += 1
        pe_grad = pe_high - pe_low
        temp_off = pe_grad * TEMP_GRADIENT_COEFF + abs(sm_delta) * TEMP_SELF_MODEL_COEFF
        adaptive_temp = float(np.clip(temp * (1.0 + temp_off), TEMP_MIN, TEMP_MAX))
        logits = logits / adaptive_temp

        if pe_grad > PE_GRADIENT_PENALTY_THRESHOLD:
            v, i = logits.topk(PENALTY_TOP_K)
            logits.scatter_add_(-1, i, -min(pe_grad * PENALTY_COEFF, PENALTY_COEFF_UPPER) * v)

        hh = hooks.get(35)
        if hh is not None and len(steer.high_buffer) < STEER_BUFFER_MAX:
            steer.observe(hh.cpu().numpy().flatten(), pe_high)

        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)
        token_log.append({"pe_low": pe_low, "pe_mid": pe_mid, "pe_high": pe_high,
                          "pe_grad": pe_grad, "temp": adaptive_temp})

        if attn is not None:
            attn = torch.cat([attn, torch.ones((1, 1), device=DEVICE, dtype=attn.dtype)], dim=-1)
        gen = torch.cat([gen, nt], dim=-1)
        if nt[0, 0].item() == tokenizer.eos_token_id:
            break
        if step % 40 == 0 and step > 0:
            torch.cuda.empty_cache()

    beliefs.drift_update(pe_mid)
    beliefs.finalize_session()
    steered, norm = steer.update()
    if steered:
        print(f"  [steer] updated norm={norm:.1f}")

    resp = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp if resp else "…", token_log, beliefs

# ── 思考生成（自己ループ） ──
@torch.no_grad()
def generate_think(context_text, max_new=THINK_TOKENS, temp=0.9, beliefs=None):
    """自己ループ: 短い内部思考を生成"""
    if beliefs is None:
        beliefs = MultiFEPBeliefs()
    inp = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX_LENGTH)
    input_ids = inp["input_ids"].to(DEVICE)
    hooks.clear()
    pkv = None
    gen = input_ids.clone()
    prev_h = {2: None, 18: None, 35: None}
    prev2_h = {2: None, 18: None, 35: None}

    for step in range(max_new):
        with torch.no_grad():
            base_out = model.model(
                input_ids=gen if pkv is None else gen[:, -1:],
                past_key_values=pkv, use_cache=True,
            )
        pkv = base_out.past_key_values
        last_hidden = base_out.last_hidden_state[:, -1:, :]

        sm_delta = beliefs.self_model - FEP_SELF_MODEL_BASELINE
        if steer.vector is not None and abs(sm_delta) > STEER_ACTIVATION_THRESHOLD:
            sv_t = torch.from_numpy(steer.vector).to(dtype=last_hidden.dtype, device=last_hidden.device)
            last_hidden = last_hidden + sv_t.unsqueeze(0) * sm_delta * STEER_STRENGTH_COEFF

        logits = model.lm_head(last_hidden.to(dtype=model.lm_head.weight.dtype))
        logits = logits[:, -1, :]

        pred_list, cur_list, label_list = [], [], []
        for lab, idx in LAYER_ORDER:
            cur = hooks.get(idx)
            p = prev_h.get(idx)
            p2 = prev2_h.get(idx)
            if cur is not None and p is not None and p2 is not None:
                pred_list.append((p + (p - p2)).flatten())
                cur_list.append(cur.flatten())
                label_list.append(lab)
            elif cur is not None and p is not None:
                pred_list.append(p.flatten())
                cur_list.append(cur.flatten())
                label_list.append(lab)
            prev2_h[idx] = prev_h.get(idx)
            prev_h[idx] = cur.clone() if cur is not None else prev_h.get(idx)

        pe_low = pe_mid = pe_high = FEP_SELF_MODEL_BASELINE
        if pred_list:
            pt = torch.stack(pred_list)
            ct = torch.stack(cur_list)
            pes = 1.0 - F.cosine_similarity(pt, ct, dim=1)
            for lab, pe in zip(label_list, pes.tolist()):
                if lab == "low": pe_low = pe; beliefs.low.update(pe)
                elif lab == "mid": pe_mid = pe; beliefs.mid.update(pe)
                elif lab == "high": pe_high = pe; beliefs.high.update(pe)

        beliefs.session_pe_sum += pe_mid
        beliefs.session_tokens += 1
        pe_grad = pe_high - pe_low

        think_temp = float(np.clip(temp * (1.0 + pe_grad * 0.2 + abs(sm_delta) * 0.3), 0.5, 1.5))
        logits = logits / think_temp

        hh = hooks.get(35)
        if hh is not None and len(steer.high_buffer) < STEER_BUFFER_MAX:
            steer.observe(hh.cpu().numpy().flatten(), pe_high)

        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)
        gen = torch.cat([gen, nt], dim=-1)

        if nt[0, 0].item() == tokenizer.eos_token_id:
            break
        if step % 40 == 0:
            torch.cuda.empty_cache()

    beliefs.drift_update(pe_mid)
    text = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return text if text else None, beliefs

# ── 行動選択Meta層 ──
def decide_mode(beliefs, recent_diversity, user_active, idle_cycles):
    """自己モデルと履歴から行動モードを決定"""
    sm = beliefs.self_model

    if sm > 0.55:
        return "CHAT"   # 混乱→外部入力で安定化
    elif sm < 0.25:
        return "THINK"  # 退屈→内部探索
    else:
        if user_active:
            return "CHAT"
        elif recent_diversity < 0.3:
            return "THINK"  # 出力が単調→思考
        elif idle_cycles > 10 and sm < 0.35:
            return "THINK"  # 長時間アイドル＋やや退屈→思考
        else:
            return "IDLE"   # 安定→待機

# ── メインループ ──
def autonomous_loop():
    print(f"\nMonica v8 — Autonomous FEP Agent")
    print(f"Modes: CHAT / THINK / IDLE\n")

    beliefs = MultiFEPBeliefs()
    conv = []
    internal_log = []
    input_queue = queue.Queue()
    running = True

    # 入力読み取りスレッド
    def read_input():
        while running:
            try:
                line = sys.stdin.readline()
                if line:
                    input_queue.put(line.strip())
            except:
                break
    threading.Thread(target=read_input, daemon=True).start()

    recent_outputs = []
    mode = "IDLE"
    last_activity = time.time()
    idle_cycles = 0

    while running:
        sm = beliefs.self_model
        user_active = not input_queue.empty()
        recent_div = 1.0

        # 最近の出力から多様性を計算
        if len(recent_outputs) >= 3:
            recent_div = diversity_score(recent_outputs[-5:])

        new_mode = decide_mode(beliefs, recent_div, user_active, idle_cycles)
        if new_mode != mode:
            print(f"  [mode] {mode} → {new_mode}  (self={sm:.2f} idle={idle_cycles})")
            mode = new_mode
            idle_cycles = 0

        # ─── CHAT ───
        if mode == "CHAT":
            if user_active:
                idle_cycles = 0
                u = input_queue.get_nowait()
                if u.lower() in ("exit", "quit", "終了"):
                    running = False
                    break
                if u == "/s":
                    print(f"  {json.dumps(beliefs.state())}")
                    continue

                conv.append({"role": "user", "content": u})
                t0 = time.time()
                resp, log, beliefs = generate_chat(conv, beliefs=beliefs)
                dt = time.time() - t0
                if not resp or resp == "…":
                    resp = "…"
                conv.append({"role": "assistant", "content": resp})
                recent_outputs.append(resp)
                avg_pe = np.mean([t["pe_low"] for t in log]) if log else 0
                print(f"  [{len(log)}tok {dt:.1f}s pe={avg_pe:.3f}]")
                print(f"  {resp[:200]}")
                log_state(beliefs, "chat", u, resp, len(log))
                if len(conv) > CONVERSATION_HISTORY_LIMIT:
                    conv = conv[-CONVERSATION_HISTORY_LIMIT:]
                last_activity = time.time()
            else:
                time.sleep(0.1)
                idle_cycles += 1

        # ─── THINK ───
        elif mode == "THINK":
            think_msgs = list(conv)
            think_msgs.append({"role": "user", "content": "自由に思いつくことを書いてみて。自分の内なる考えを探る感じで。"})
            context = tokenizer.apply_chat_template(think_msgs, tokenize=False, add_generation_prompt=True)

            t0 = time.time()
            thought, beliefs = generate_think(context, beliefs=beliefs)
            if thought and len(thought) > 5:
                recent_outputs.append(thought)
                internal_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")
                print(f"  [think] {thought[:150]}")
                log_state(beliefs, "think", "", thought, 0)
                last_activity = time.time()
            else:
                print(f"  [think] (empty)")
            torch.cuda.empty_cache()
            idle_cycles = 0

        # ─── IDLE ───
        elif mode == "IDLE":
            if user_active:
                continue
            for _ in range(IDLE_DRIFT_STEPS):
                beliefs.drift_update(beliefs.mid.running_avg)
            idle_cycles += 1
            time.sleep(0.5)

        if time.time() - last_activity > 30 and mode == "IDLE":
            print(f"  [idle→think] 30s inactivity")
            mode = "THINK"

    print("\nShutting down.")
    hooks.remove()

def diversity_score(texts):
    if len(texts) < 2:
        return 1.0
    t = [x[:80] for x in texts[-5:]]
    m = sum(1 for i in range(len(t)) for j in range(i+1, len(t)) if t[i] == t[j])
    return 1.0 - m / (len(t) * (len(t)-1) / 2) if len(t) > 1 else 1.0

def log_state(beliefs, mode, user_in, resp, n_tok):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "mode": mode, "user": user_in[:40], "resp": resp[:100],
            "n_tok": n_tok, "state": beliefs.state(),
        }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    try:
        autonomous_loop()
    finally:
        hooks.remove()
