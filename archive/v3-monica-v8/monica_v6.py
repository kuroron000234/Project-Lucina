"""
Monica v6 — Multi-Layer FEP + Representation Steering
3箇所にフックを仕掛け、層間の予測誤差伝播を観測。
PEが高いとき／低いときの隠れベクトルからステアリング方向を学習し、
次の生成にフィードバックする。
"""

import json, sys, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOG_PATH = Path(__file__).parent / "log_v6.jsonl"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ── フック対象レイヤー（0-indexed、36層中） ──
# L2: 低次（トークン遷移）、L18: 中次（イベント/文脈）、L35: 高次（自己モデル）
# 選定理由: 36層の2/18/35 ≒ 5%/50%/97% 。低層=構文、中層=意味、高層=抽象。
LOW_LAYER_IDX = 2
MID_LAYER_IDX = 18
HIGH_LAYER_IDX = 35
LAYER_ORDER = [("low", LOW_LAYER_IDX), ("mid", MID_LAYER_IDX), ("high", HIGH_LAYER_IDX)]

# ── FEP Beliefs パラメータ ──
# 自己モデルの基準値（ニュートラル）。低PEなら自己モデルに追従、高PEなら逸脱。
FEP_SELF_MODEL_BASELINE = 0.30

# ドリフト：逆温度制御（PEが低い = 退屈ほど更新率大）
#   update_rate = 1 / (1 + pe_mid * DRIFT_INV_TEMP_COEFF)
#   - pe_mid=0.3(低) → rate=0.36（敏感）
#   - pe_mid=0.6(高) → rate=0.22（鈍感）
#   差は約1.6倍。緩やかな差だが、累積で効く。
DRIFT_INV_TEMP_COEFF = 6.0

# ドリフト：ランダムウォーク成分の標準偏差
#   1トークンあたり平均|drift| = update_rate * noise_std * sqrt(2/π) ≈ 0.27*0.03*0.8 ≈ 0.0065
#   200トークンで期待累積 = 0.0065*sqrt(200) ≈ 0.092
#   → 復元力と釣り合い、0.30±0.10の範囲で振動する設計。
DRIFT_NOISE_STD = 0.03

# ドリフト：復元力（自己モデルが基準値から乖離するほど強く引き戻す）
#   自己モデルが0.50に達した場合の復元力 = (0.30 - 0.50) * 0.03 = -0.006/トークン
#   これは平均ノイズ成分(≈0.007)と同程度。発散を防ぎつつ振動を許容。  
DRIFT_RESTORING_COEFF = 0.03

# 自己モデルの上限/下限
SELF_MODEL_MIN = 0.05
SELF_MODEL_MAX = 0.95

# ── ステアリングベクトル ──
# 高PE時 / 低PE時の隠れ状態を収集し、その差分をステアリング方向とする。
# 収集するサンプル数の上限
STEER_BUFFER_MAX = 200
# 高PEと判定するしきい値（cosine distance 0.65 ≈ 35°の角度差）
STEER_HIGH_PE_THRESHOLD = 0.65
# 低PEと判定するしきい値 = 上記の 0.6倍
STEER_LOW_PE_THRESHOLD_FACTOR = 0.6
# ベクトルを確定するのに最低限必要なサンプル数
STEER_MIN_SAMPLES = 10
# 自己モデル乖離をステアリング強度に変換する係数
#   乖離0.10 × 0.50 = 0.05倍ベクトル加算（hidden_norm≈30のとき1.5単位のシフト）
STEER_STRENGTH_COEFF = 0.50
# ステアリング介入を開始する自己モデル乖離の最小値
STEER_ACTIVATION_THRESHOLD = 0.01

# ── 温度変調 ──
# PE勾配(高-低)が温度に与える影響の強さ
#   勾配+0.10 → 温度 +5%上昇
TEMP_GRADIENT_COEFF = 0.50
# 自己モデル乖離が温度に与える影響の強さ
#   乖離0.10 → 温度 +8%上昇（自己不一致は探索へ）
TEMP_SELF_MODEL_COEFF = 0.80
# 温度の上限・下限
TEMP_MIN = 0.3
TEMP_MAX = 1.8

# ── PE勾配トップkペナルティ ──
# 高層のPEが低層より高い（gradient>0）= 抽象層で不確実性増大 → 安全策としてトップ候補を抑制
# ペナルティ発動する勾配の最小値
PE_GRADIENT_PENALTY_THRESHOLD = 0.05
# 抑制する候補数
PENALTY_TOP_K = 5
# ペナルティ係数（ペナルティ上限でクリップ）
PENALTY_COEFF_UPPER = 0.35
PENALTY_COEFF = 0.25

# ── FEPフィルタ係数（EMA平滑化） ──
# running_avg のEMA減衰率 （0.95×prev_avg + 0.05×current_pe）
FEP_AVG_DECAY = 0.95
FEP_AVG_UPDATE = 0.05  # = 1 - FEP_AVG_DECAY
# running_var のEMA減衰率
FEP_VAR_DECAY = 0.90
FEP_VAR_UPDATE = 0.10  # = 1 - FEP_VAR_DECAY
# 履歴バッファサイズ（分散計算用）
FEP_HISTORY_MAX = 100
FEP_VAR_WINDOW = 20

# ── 生成 ──
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
CONTEXT_MAX_LENGTH = 4096
CONVERSATION_HISTORY_LIMIT = 6
LOG_RESPONSE_TRIM = 150
LOG_USER_TRIM = 60

# ── メモリ管理 ──
CACHE_CLEAN_INTERVAL = 40

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
print(f"Model loaded. {len(model.model.layers)} layers, hidden={model.config.hidden_size}")

# ── 多層フック ──
class MultiHook:
    def __init__(self):
        self.states: dict[int, torch.Tensor | None] = {}
        self.handles = []

    def make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                h = output[:, -1, :].detach().to(dtype=torch.float32)
            elif isinstance(output, tuple):
                h = output[0][:, -1, :].detach().to(dtype=torch.float32)
            else:
                return
            self.states[layer_idx] = h
        return hook_fn

    def register(self, model, indices: list[int]):
        for idx in indices:
            handle = model.model.layers[idx].register_forward_hook(self.make_hook(idx))
            self.handles.append(handle)

    def get(self, idx: int) -> torch.Tensor | None:
        return self.states.get(idx)

    def clear(self):
        self.states.clear()

    def remove(self):
        for h in self.handles:
            h.remove()

hooks = MultiHook()
hooks.register(model, [LOW_LAYER_IDX, MID_LAYER_IDX, HIGH_LAYER_IDX])
print(f"Hooks on layers {LOW_LAYER_IDX}, {MID_LAYER_IDX}, {HIGH_LAYER_IDX}")

# ── ステアリングベクトル ──
class SteeringVector:
    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.high_pe_buffer: list[np.ndarray] = []
        self.low_pe_buffer: list[np.ndarray] = []
        self.vector: np.ndarray | None = None
        self.buffer_max = STEER_BUFFER_MAX

    def observe(self, hidden: np.ndarray, pe: float, threshold: float = STEER_HIGH_PE_THRESHOLD):
        if pe > threshold and len(self.high_pe_buffer) < self.buffer_max:
            self.high_pe_buffer.append(hidden)
        elif pe < threshold * STEER_LOW_PE_THRESHOLD_FACTOR and len(self.low_pe_buffer) < self.buffer_max:
            self.low_pe_buffer.append(hidden)

    def update(self):
        if len(self.high_pe_buffer) < STEER_MIN_SAMPLES or len(self.low_pe_buffer) < STEER_MIN_SAMPLES:
            return
        high_mean = np.mean(self.high_pe_buffer, axis=0)
        low_mean = np.mean(self.low_pe_buffer, axis=0)
        diff = high_mean - low_mean
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            self.vector = diff / norm
            print(f"  [steering] updated: |high|={len(self.high_pe_buffer)} |low|={len(self.low_pe_buffer)} norm={norm:.4f}")

steer = SteeringVector()

# ── FEPユーティリティ ──
def entropy(probs: torch.Tensor) -> float:
    p = probs.flatten()
    return float(-(p * torch.log(p + 1e-10)).sum().item())

# ── FEP Beliefs（層ごと） ──
@dataclass
class LayerFEP:
    running_avg: float = FEP_SELF_MODEL_BASELINE
    running_var: float = 0.05
    history: list = field(default_factory=list)

    def update(self, pe: float):
        self.history.append(pe)
        if len(self.history) > FEP_HISTORY_MAX:
            self.history.pop(0)
        self.running_avg = FEP_AVG_DECAY * self.running_avg + FEP_AVG_UPDATE * pe
        if len(self.history) > 2:
            self.running_var = FEP_VAR_DECAY * self.running_var + FEP_VAR_UPDATE * float(np.var(self.history[-FEP_VAR_WINDOW:]))

@dataclass
class MultiFEPBeliefs:
    low: LayerFEP = field(default_factory=LayerFEP)
    mid: LayerFEP = field(default_factory=LayerFEP)
    high: LayerFEP = field(default_factory=LayerFEP)

    session_pe_sum: float = 0.0
    session_tokens: int = 0
    self_model: float = FEP_SELF_MODEL_BASELINE
    self_model_peak: float = FEP_SELF_MODEL_BASELINE
    self_drift: float = 0.0

    def update_low(self, pe: float): self.low.update(pe)
    def update_mid(self, pe: float): self.mid.update(pe)
    def update_high(self, pe: float): self.high.update(pe)

    def update_mid_session(self, pe: float):
        self.session_pe_sum += pe
        self.session_tokens += 1

    def drift_update(self, pe_mid: float):
        """
        逆温度制御 + 復元力:
        PEが低い（退屈）→ 敏感にドリフト（自己モデル再構築）
        PEが高い（混乱）→ 硬直（現状維持）
        復元力が自壊を防ぎ、振動を持続させる。
        """
        update_rate = 1.0 / (1.0 + pe_mid * DRIFT_INV_TEMP_COEFF)
        noise = np.random.normal(0, DRIFT_NOISE_STD)
        restoring = (FEP_SELF_MODEL_BASELINE - self.self_model) * DRIFT_RESTORING_COEFF
        d = update_rate * noise + restoring
        self.self_drift += d
        self.self_model = float(np.clip(self.self_model + d, SELF_MODEL_MIN, SELF_MODEL_MAX))

    def finalize_session(self):
        if self.session_tokens > 0:
            avg = self.session_pe_sum / self.session_tokens
            self.self_model_peak = max(self.self_model_peak, avg)
        self.session_pe_sum = 0.0
        self.session_tokens = 0

    def state(self, steer_ready: bool = False) -> dict:
        return {
            "low_avg": round(self.low.running_avg, 4),
            "mid_avg": round(self.mid.running_avg, 4),
            "high_avg": round(self.high.running_avg, 4),
            "self": round(self.self_model, 4),
            "drift": round(self.self_drift, 4),
            "steer": "ready" if steer_ready else "none",
        }


# ── 生成 ──
@torch.no_grad()
def generate_with_fep(
    messages: list,
    max_new: int = DEFAULT_MAX_NEW_TOKENS,
    temp: float = DEFAULT_TEMPERATURE,
    beliefs: MultiFEPBeliefs | None = None,
) -> tuple[str, list[dict], MultiFEPBeliefs]:
    if beliefs is None:
        beliefs = MultiFEPBeliefs()

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX_LENGTH)
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp.get("attention_mask")
    if attn is not None:
        attn = attn.to(DEVICE)

    hooks.clear()

    past_kv = None
    generated = input_ids.clone()
    prev_h = {LOW_LAYER_IDX: None, MID_LAYER_IDX: None, HIGH_LAYER_IDX: None}
    prev2_h = {LOW_LAYER_IDX: None, MID_LAYER_IDX: None, HIGH_LAYER_IDX: None}
    token_log = []

    for step in range(max_new):
        with torch.no_grad():
            # transformer backbone（LM headなし）
            base_out = model.model(
                input_ids=generated if past_kv is None else generated[:, -1:],
                attention_mask=attn,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = base_out.past_key_values

        # 最終層の隠れ状態（フックとは別に直接取得）
        last_hidden = base_out.last_hidden_state[:, -1:, :]

        # ── FEP: 自己モデル×ステアリングで隠れ状態を直接操作 ──
        # perception（hookの生状態）→ action（LM head直前の隠れ状態を操作）
        sm_delta = beliefs.self_model - FEP_SELF_MODEL_BASELINE
        if steer.vector is not None and abs(sm_delta) > STEER_ACTIVATION_THRESHOLD:
            sv_t = torch.from_numpy(steer.vector).to(device=last_hidden.device, dtype=last_hidden.dtype)
            sv_t = sv_t.unsqueeze(0)
            last_hidden = last_hidden + sv_t * sm_delta * STEER_STRENGTH_COEFF

        # LM head でロジットに変換
        logits = model.lm_head(last_hidden.to(dtype=model.lm_head.weight.dtype))
        logits = logits[:, -1, :]

        pe_low = pe_mid = pe_high = FEP_SELF_MODEL_BASELINE

        # 各層の予測誤差 — GPUで一括計算
        pred_list, cur_list, label_list = [], [], []
        for label, idx in LAYER_ORDER:
            cur = hooks.get(idx)
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

        if pred_list:
            pred_t = torch.stack(pred_list)
            cur_t = torch.stack(cur_list)
            pes = 1.0 - F.cosine_similarity(pred_t, cur_t, dim=1)
            for label, pe in zip(label_list, pes.tolist()):
                if label == "low":
                    pe_low = pe
                    beliefs.update_low(pe)
                elif label == "mid":
                    pe_mid = pe
                    beliefs.update_mid(pe)
                elif label == "high":
                    pe_high = pe
                    beliefs.update_high(pe)

        # 中次PE = mid層の今の値でセッション更新とドリフト
        beliefs.update_mid_session(pe_mid)
        beliefs.drift_update(pe_mid)

        # ── FEP: ロジット操作（自己モデル影響は隠れ状態に反映済み） ──
        pe_gradient = pe_high - pe_low

        temp_offset = pe_gradient * TEMP_GRADIENT_COEFF + abs(sm_delta) * TEMP_SELF_MODEL_COEFF
        adaptive_temp = float(np.clip(temp * (1.0 + temp_offset), TEMP_MIN, TEMP_MAX))
        logits = logits / adaptive_temp

        if pe_gradient > PE_GRADIENT_PENALTY_THRESHOLD:
            v, i = logits.topk(PENALTY_TOP_K)
            logits.scatter_add_(-1, i, -min(pe_gradient * PENALTY_COEFF, PENALTY_COEFF_UPPER) * v)

        hh = hooks.get(HIGH_LAYER_IDX)
        if hh is not None and len(steer.high_pe_buffer) < STEER_BUFFER_MAX:
            steer.observe(hh.cpu().numpy().flatten(), pe_high)

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)

        token_log.append({
            "step": step, "pe_low": pe_low, "pe_mid": pe_mid, "pe_high": pe_high,
            "pe_grad": pe_high - pe_low, "temp": adaptive_temp,
            "tok": int(next_tok[0, 0]),
        })

        if attn is not None:
            attn = torch.cat([attn, torch.ones((1, 1), device=DEVICE, dtype=attn.dtype)], dim=-1)
        generated = torch.cat([generated, next_tok], dim=-1)

        if next_tok[0, 0].item() == tokenizer.eos_token_id:
            break
        if step % CACHE_CLEAN_INTERVAL == 0 and step > 0:
            torch.cuda.empty_cache()

    beliefs.finalize_session()
    steer.update()

    resp = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp, token_log, beliefs


# ── チャットループ ──
def interactive():
    print(f"\nMonica v6 — Multi-Layer FEP + Steering")
    print(f"Layers: L{LOW_LAYER_IDX} M{MID_LAYER_IDX} H{HIGH_LAYER_IDX}\n")

    beliefs = MultiFEPBeliefs()
    conv = []

    while True:
        try:
            u = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not u: continue
        if u.lower() in ("exit", "quit", "終了"): break
        if u == "/s":
            print(f"  {json.dumps(beliefs.state(steer_ready=steer.vector is not None))}")
            continue
        if u == "/steer":
            print(f"  steer vector: {'ready' if steer.vector is not None else 'none'} "
                  f"(high={len(steer.high_pe_buffer)} low={len(steer.low_pe_buffer)})")
            continue

        conv.append({"role": "user", "content": u})

        t0 = time.time()
        response, token_log, beliefs = generate_with_fep(conv, beliefs=beliefs)
        dt = time.time() - t0

        if not response:
            response = "…"
        conv.append({"role": "assistant", "content": response})

        if token_log:
            pe_l = np.mean([t["pe_low"] for t in token_log])
            pe_m = np.mean([t["pe_mid"] for t in token_log])
            pe_h = np.mean([t["pe_high"] for t in token_log])
            pg = np.mean([t["pe_grad"] for t in token_log])
            at = np.mean([t["temp"] for t in token_log])
            print(f"  [{len(token_log)}tok {dt:.1f}s | "
                  f"low={pe_l:.3f} mid={pe_m:.3f} high={pe_h:.3f} "
                  f"grad={pg:.3f} temp={at:.2f}]")
            print(f"  {response[:300]}")

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "user": u[:LOG_USER_TRIM], "response": response[:LOG_RESPONSE_TRIM],
                "n_tok": len(token_log),
                "pe_low": round(float(pe_l), 4) if token_log else 0,
                "pe_mid": round(float(pe_m), 4) if token_log else 0,
                "pe_high": round(float(pe_h), 4) if token_log else 0,
                "state": beliefs.state(steer_ready=steer.vector is not None),
            }, ensure_ascii=False) + "\n")

        if len(conv) > CONVERSATION_HISTORY_LIMIT:
            conv = conv[-CONVERSATION_HISTORY_LIMIT:]
        torch.cuda.empty_cache()


def quick_test():
    beliefs = MultiFEPBeliefs()
    tests = [
        [{"role": "user", "content": "こんにちは"}],
        [{"role": "user", "content": "こんにちは"},
         {"role": "assistant", "content": "こんにちは！"},
         {"role": "user", "content": "人工知能の未来は？"}],
    ]
    for msgs in tests:
        r, log, beliefs = generate_with_fep(msgs, max_new=60 if len(msgs) < 3 else DEFAULT_MAX_NEW_TOKENS, beliefs=beliefs)
        if log:
            pe_l = np.mean([t["pe_low"] for t in log])
            pe_m = np.mean([t["pe_mid"] for t in log])
            pe_h = np.mean([t["pe_high"] for t in log])
            print(f"  low={pe_l:.3f} mid={pe_m:.3f} high={pe_h:.3f} | {r[:80]}")

if __name__ == "__main__":
    try:
        if "--test" in sys.argv:
            quick_test()
        else:
            interactive()
    finally:
        hooks.remove()
        print("Hooks removed.")
