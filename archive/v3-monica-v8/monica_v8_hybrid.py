"""
Monica v10 — Full FEP Hybrid Agent + Phase 1 Action Chaining
センシング（PE/KL計測・ドリフト）はローカル4B、
応答生成はクラウドAPI（Groq優先）で行う。

【v10 Features】
  v8 Hybrid 基盤 + v10 Phase2→v14 全機能統合
【Phase 1】
  アクション指向THINK_PROMPTS + ツール連鎖（観測→再思考、最大5連鎖）
"""
import json, sys, time, threading, torch, numpy as np, queue, os, re, math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import Counter
import requests
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── API設定（マルチプロバイダ: Groq優先 → OpenRouterフォールバック） ──
API_PROVIDERS = [
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.environ.get("MONICA_GROQ_KEY", ""),
        "model": "llama-3.3-70b-versatile",
        "success": 0,         # Phase 8: 成功率追跡
        "fail": 0,
        "total_time": 0.0,
        "calls": 0,
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": os.environ.get("MONICA_OPENROUTER_KEY", ""),
        "model": "openrouter/free",
        "success": 0,         # Phase 8
        "fail": 0,
        "total_time": 0.0,
        "calls": 0,
    },
]
for p in API_PROVIDERS:
    p["headers"] = {"Authorization": f"Bearer {p['key']}"}
    if p["name"] == "openrouter":
        p["headers"]["HTTP-Referer"] = "https://github.com/anomalyco/opencode"
        p["headers"]["X-Title"] = "Monica v8 Hybrid"# Phase 8: Groqレート制限管理（30 RPM）
_GROQ_RATE_WINDOW = 60.0  # 60秒間
_groq_call_times = []  # 全呼び出しを記録（成功+失敗+429）

# Phase 8: プロバイダ選択関数
def _get_best_provider():
    """成功率が最も高いプロバイダを返す（同率なら高速な方）"""
    scored = []
    for p in API_PROVIDERS:
        total = p["success"] + p["fail"]
        if total == 0:
            rate = 0.5  # 未評価なら中間値
        else:
            rate = p["success"] / total
        avg_time = p["total_time"] / p["calls"] if p["calls"] > 0 else 999
        scored.append((rate, -avg_time, p))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [p for _, _, p in scored]

def _check_groq_rate_limit():
    """Groq 30RPM制限をチェック。残り枠が少なければGroqをスキップ"""
    global _groq_call_times
    now = time.time()
    # 60秒より前の記録を削除
    _groq_call_times = [t for t in _groq_call_times if now - t < _GROQ_RATE_WINDOW]
    # 残り枠が5未満なら制限接近→Groqスキップ推奨
    remaining = 30 - len(_groq_call_times)
    return remaining, remaining >= 3  # 3回以上残ってればOK

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOG_PATH = Path(__file__).parent / "log_v8h.jsonl"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LAYER_ORDER = [("low", 2), ("mid", 18), ("high", 35)]
FEP_SELF_MODEL_BASELINE = 0.30

DRIFT_INV_TEMP_COEFF = 6.0
DRIFT_NOISE_STD = 0.03
DRIFT_RESTORING_COEFF = 0.06        # 0.03→0.06: 復元力2倍
DRIFT_LEAK = 0.995                  # リーキー積分器: stepごとに0.5%減衰
SELF_MODEL_MIN = 0.05
SELF_MODEL_MAX = 0.95
DRIFT_MAX_ABS = 1.0                 # drift値の絶対値上限

# ── ステアリング（実測PEレンジに合わせて調整） ──
STEER_BUFFER_MAX = 50              # 旧200: バッファ収集加速（Phase 7）
STEER_HIGH_PE_THRESHOLD = 0.35      # 旧0.45: より多くのサンプルを収集（Phase 7）
STEER_LOW_PE_THRESHOLD_FACTOR = 0.7 # 旧0.5→0.7: 低閾値=0.35*0.7=0.245（high/low区別維持）
STEER_MIN_SAMPLES = 3               # 旧5: 早期アクティベーション（Phase 7）
STEER_STRENGTH_COEFF = 0.50
STEER_ACTIVATION_THRESHOLD = 0.01
STEER_CONFIDENCE_THRESHOLD = 0.15   # Phase 7: steer_to_text用 新

FEP_AVG_DECAY = 0.95
FEP_AVG_UPDATE = 0.05
FEP_VAR_DECAY = 0.90
FEP_VAR_UPDATE = 0.10
FEP_HISTORY_MAX = 100
FEP_VAR_WINDOW = 20

# ── 行動選択 ──
THINK_THRESHOLD = 0.35           # 旧0.30: THINK発動が稀だったため緩和
CHAT_UPPER_THRESHOLD = 0.55
IDLE_DRIFT_STEPS = 15
IDLE_DRIFT_NOISE_MULT = 2.0
THINK_COOLDOWN_CYCLES = 5
IDLE_TO_THINK_TIMEOUT = 15        # 旧30: 時間ベースTHINKトリガーを早める
FORCE_THINK_ON_STARTUP = True      # 初回起動時に強制THINK
MAX_CONSECUTIVE_THINK = 5
THINK_EXIT_BOOST = 0.05
IDLE_LOCK_CYCLES = 20
MIN_IDLE_CYCLES = 10                # IDLE最低継続サイクル（即THINK復帰防止）
THINK_DRIFT_INV_TEMP_COEFF = 2.5
THINK_DRIFT_NOISE_STD = 0.05

# ── THINKプロンプト（アクション指向＋ツール連鎖） ──
THINK_PROMPTS = [
    "What is one useful thing I can do right now? Search for information, read a file, calculate something, or write code. Output with [SEARCH:], [READ:], [CALC:], [PYTHON:], [WRITE:], [SHELL:], [FILEINFO:], or [GITHUB:] if you decide on an action. If you have an active goal, you MUST output a tool command to make progress on it.",
    "Review what I know and pick a concrete next step. What file should I read, what data should I fetch, or what code should I write? Use [TOOL:] syntax if acting. If you have a goal, take action on it now — do NOT just think about it.",
    "Given my goals and recent observations, what is the single most valuable action I can take? Output a tool command or analysis. A goal without action is useless — execute a [SEARCH:] or [READ:] related to your goal.",
    "I have these tools: [SEARCH: query], [READ: path], [CALC: expr], [PYTHON: code], [WRITE: path, content], [SHELL: cmd], [FILEINFO: path], [GITHUB: user/repo/search/content]. What should I do next? If you have a goal, pick a tool that serves it.",
    "Use [FILEINFO: path] to inspect file metadata (type, size, date) of any file. Use [GITHUB: repo user/repo] to fetch public repo info. Use [GITHUB: search query] to find repos. Which investigation should I do? If your goal relates to a topic, search it now.",

    "Look at the conversation and world state. Is there something I should investigate, create, or improve? Act if possible with [TOOL:] syntax. Mere thinking is not enough — take at least one concrete action.",
]

# ── THINK後のフォローアップ用：目標→自動SEARCH ──
def auto_search_from_goal(goal_manager, observations, thought_text):
    """THINK後にツール未使用かつ目標があれば、SEARCHクエリ生成して返す
    Returns: (query: str | None) — 実行すべき検索クエリ or None
    """
    # すでにツールを使った or 思考テキストにツール含む場合はスキップ
    if observations and len(observations) > 0:
        return None
    g = goal_manager.active_goal()
    if not g:
        return None
    # 思考テキストにツールパターンが含まれているか再確認
    if re.search(r'\[SEARCH:|\[READ:|\[CALC:|\[PYTHON:|\[WRITE:|\[SHELL:|\[FILEINFO:|\[GITHUB:', thought_text or ''):
        return None
    desc = g.description
    # カテゴリに応じてクエリ生成
    if g.category == "search" or "search" in desc.lower() or "find" in desc.lower() or "look" in desc.lower():
        return f"{desc[:80]}"
    elif "explore" in desc.lower() or "learn" in desc.lower() or "research" in desc.lower():
        # 目標からキーワード抽出
        words = re.findall(r'[A-Z][a-z]+|[a-z]{4,}', desc)
        if words:
            return " ".join(words[:5])
        return f"{desc[:80]}"
    elif g.category == "create":
        # 作成系は検索より書き込み→ここでは検索は促さない
        return None
    elif g.category == "act":
        return None
    else:
        return f"{desc[:80]}"

CONTEXT_MAX_LENGTH = 4096
CONVERSATION_HISTORY_LIMIT = 10
THINK_TOKENS = 120
FALLBACK_MAX_NEW = 100

print(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
print(f"Loading {MODEL_NAME} (sensing + fallback)...")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, dtype=DTYPE)
model.eval()
print(f"Model loaded. {len(model.model.layers)} layers")

model_lock = threading.Lock()

# ── フック（センシング + ローカル生成兼用） ──
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
        self.strength = 0.0          # 最後のupdate()で計算された差分ノルム
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
            self.strength = float(n)
            return True, n
        return False, 0.0

steer = SteeringVector()

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

@dataclass
class MultiFEPBeliefs:
    low: LayerFEP = field(default_factory=LayerFEP)
    mid: LayerFEP = field(default_factory=LayerFEP)
    high: LayerFEP = field(default_factory=LayerFEP)
    session_pe_sum: float = 0.0
    session_tokens: int = 0
    self_model: float = FEP_SELF_MODEL_BASELINE
    self_drift: float = 0.0

    def drift_update(self, pe_mid, noise_mult=1.0, inv_temp=None, noise_std=None):
        inv_temp = inv_temp if inv_temp is not None else DRIFT_INV_TEMP_COEFF
        noise_std = noise_std if noise_std is not None else DRIFT_NOISE_STD
        update_rate = 1.0 / (1.0 + pe_mid * inv_temp)
        noise = np.random.normal(0, noise_std * noise_mult)
        restoring = (FEP_SELF_MODEL_BASELINE - self.self_model) * DRIFT_RESTORING_COEFF
        d = update_rate * noise + restoring
        # リーキー積分: 蓄積を減衰させつつ新しい寄与を加算
        self.self_drift = self.self_drift * DRIFT_LEAK + d
        self.self_drift = float(np.clip(self.self_drift, -DRIFT_MAX_ABS, DRIFT_MAX_ABS))
        self.self_model = float(np.clip(self.self_model + d, SELF_MODEL_MIN, SELF_MODEL_MAX))

    def reset_drift(self):
        self.self_drift = 0.0
        self.self_model = FEP_SELF_MODEL_BASELINE

    def finalize_session(self):
        self.session_pe_sum = 0.0
        self.session_tokens = 0

    def state(self):
        return {"low_avg": round(self.low.running_avg, 3),
                "mid_avg": round(self.mid.running_avg, 3),
                "high_avg": round(self.high.running_avg, 3),
                "self": round(self.self_model, 3),
                "drift": round(self.self_drift, 3),
                "restoring": round(DRIFT_RESTORING_COEFF, 4)}

    def state_dict(self):
        return {"self_model": self.self_model, "self_drift": self.self_drift,
                "low_avg": self.low.running_avg, "mid_avg": self.mid.running_avg,
                "high_avg": self.high.running_avg,
                "low_var": self.low.running_var, "mid_var": self.mid.running_var,
                "high_var": self.high.running_var}

    def load_state_dict(self, d):
        self.self_model = d.get("self_model", FEP_SELF_MODEL_BASELINE)
        self.self_drift = d.get("self_drift", 0.0)
        self.low.running_avg = d.get("low_avg", FEP_SELF_MODEL_BASELINE)
        self.mid.running_avg = d.get("mid_avg", FEP_SELF_MODEL_BASELINE)
        self.high.running_avg = d.get("high_avg", FEP_SELF_MODEL_BASELINE)
        self.low.running_var = d.get("low_var", 0.05)
        self.mid.running_var = d.get("mid_var", 0.05)
        self.high.running_var = d.get("high_var", 0.05)


# ── 適応的FEPパラメータ ──
class AdaptiveFEP:
    def __init__(self):
        self.sm_history = []
        self.window = 50
        self.restoring_min = 0.02
        self.restoring_max = 0.15
        self.noise_std_min = 0.02
        self.noise_std_max = 0.08
        self.inv_temp_min = 3.0
        self.inv_temp_max = 10.0

    def update(self, self_model):
        self.sm_history.append(self_model)
        while len(self.sm_history) > self.window:
            self.sm_history.pop(0)
        if len(self.sm_history) < 10:
            return

        global DRIFT_RESTORING_COEFF, DRIFT_NOISE_STD, DRIFT_INV_TEMP_COEFF

        n = len(self.sm_history)
        avg_sm = sum(self.sm_history) / n
        bias = avg_sm - FEP_SELF_MODEL_BASELINE

        # 変化速度を検出（直近10とその前10の平均差）
        recent10 = self.sm_history[-10:]
        prev10 = self.sm_history[-20:-10] if n >= 20 else recent10
        rate = abs(sum(recent10)/len(recent10) - sum(prev10)/len(prev10))

        # 変化が大きい→窓を短く（敏感に）、小さい→窓を長く（安定）
        if rate > 0.03:
            self.window = min(100, self.window + 5)
        elif rate < 0.01:
            self.window = max(20, self.window - 3)

        # 復元力を適応
        if bias < -0.05:
            DRIFT_RESTORING_COEFF = min(DRIFT_RESTORING_COEFF * 1.05, self.restoring_max)
        elif bias > 0.05:
            DRIFT_RESTORING_COEFF = max(DRIFT_RESTORING_COEFF * 0.97, self.restoring_min)

        # inv_temp を適応: driftが大きい→inv_temp上げる（ドリフト抑制）
        drift_abs = abs(self_model - FEP_SELF_MODEL_BASELINE)
        if drift_abs > 0.15:
            DRIFT_INV_TEMP_COEFF = min(DRIFT_INV_TEMP_COEFF * 1.03, self.inv_temp_max)
        elif drift_abs < 0.05:
            DRIFT_INV_TEMP_COEFF = max(DRIFT_INV_TEMP_COEFF * 0.98, self.inv_temp_min)

        # 分散によるノイズ調整
        recent = self.sm_history[-10:]
        variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
        if variance < 0.002:
            DRIFT_NOISE_STD = min(DRIFT_NOISE_STD * 1.02, self.noise_std_max)
        elif variance > 0.01:
            DRIFT_NOISE_STD = max(DRIFT_NOISE_STD * 0.98, self.noise_std_min)

adapt_fep = AdaptiveFEP()

# ── センシング（ローカルforward pass、生成なし） ──
@torch.no_grad()
def sense(text, beliefs):
    """入力を1回forwardしてPEを計測。応答生成はしない"""
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX_LENGTH)
    input_ids = inp["input_ids"].to(DEVICE)

    model_lock.acquire()
    hooks.clear()
    try:
        model.model(input_ids=input_ids)
    finally:
        model_lock.release()

    # PE計算（学習済みのprev_h/prev2_hがあれば使用）
    global_prev_h = getattr(sense, "prev_h", {2: None, 18: None, 35: None})
    global_prev2_h = getattr(sense, "prev2_h", {2: None, 18: None, 35: None})
    prev_h = dict(global_prev_h)   # コピーして参照分離
    prev2_h = dict(global_prev2_h)

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

    sense.prev_h = prev_h
    sense.prev2_h = prev2_h

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
    beliefs.drift_update(pe_mid)

    # ステアリング観測
    hh = hooks.get(35)
    if hh is not None and len(steer.high_buffer) < STEER_BUFFER_MAX:
        steer.observe(hh.cpu().numpy().flatten(), pe_high)

    return beliefs

# ── 長期記憶（セッション要約） ──
memory_summaries = []

# ── Phase 9.1: セッション間学習 ──
SESSION_LOG_PATH = Path(__file__).parent / "session_log.jsonl"

def save_session_summary(beliefs, think_count, mode_count):
    """セッション終了時にサマリーを保存。次回起動時にcontextとして注入"""
    summary = {
        "ts": datetime.now().isoformat(),
        "duration": time.time() - getattr(save_session_summary, "_start", time.time()),
        "think_count": think_count,
        "chat_count": mode_count.get("CHAT", 0),
        "self_model": round(beliefs.self_model, 3),
        "avg_pe_mid": round(beliefs.mid.running_avg, 3),
        "adapt_restoring": round(DRIFT_RESTORING_COEFF, 4),
        "user_name": profile.name if profile.name else "unknown",
        "top_topics": [t[0] for t in profile.topics[:3]],
        "active_goal": goal_manager.active_goal().description[:60] if goal_manager.active_goal() else None,
    }
    with open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"  [session] saved summary (think={think_count}, chat={mode_count.get('CHAT',0)})")

def load_previous_session_context():
    """過去のセッションサマリーを読み込み、memory_summariesに注入"""
    if not SESSION_LOG_PATH.exists():
        return
    try:
        summaries = []
        with open(SESSION_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        summaries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if summaries:
            # 直近3セッションの要約をmemory_summariesに追加
            for s in summaries[-3:]:
                topic_str = ", ".join(s.get("top_topics", []) or [])
                user = s.get("user_name", "unknown")
                think_n = s.get("think_count", 0)
                chat_n = s.get("chat_count", 0)
                sm = s.get("self_model", 0.3)
                goal = s.get("active_goal", "")
                summary_text = f"[Previous session: user={user}, thinks={think_n}, chats={chat_n}, self_model={sm}, topics=[{topic_str}]"
                if goal:
                    summary_text += f", goal={goal}"
                summary_text += "]"
                memory_summaries.append(summary_text)
            print(f"  [session] restored {len(summaries)} previous sessions")
    except Exception as e:
        print(f"  [session] load error: {e}")

save_session_summary._start = time.time()

# ── v10 Phase 2: World Model（THINK内容の時系列圧縮・長期記憶） ──
class WorldModel:
    """THINK出力をキーワード重複でエピソード圧縮・検索可能にする"""
    def __init__(self, path="world_model_v8h.json"):
        self.path = Path(__file__).parent / path
        self.episodes = []
        self.load()

    def extract_keywords(self, text, n=5):
        words = text.lower().split()
        freq = Counter(w for w in words if len(w) >= 4 and w.isalpha())
        return [w for w, _ in freq.most_common(n)]

    def add_thought(self, text):
        if not text or len(text) < 10:
            return
        kw = self.extract_keywords(text)
        merged = False
        for ep in reversed(self.episodes):
            overlap = len(set(kw) & set(ep["keywords"]))
            if overlap >= 2:
                ep["text"] += "\n" + text[:200]
                ep["keywords"] = list(set(ep["keywords"] + kw))
                ep["count"] = ep.get("count", 1) + 1
                merged = True
                break
        if not merged:
            self.episodes.append({
                "id": len(self.episodes),
                "keywords": kw,
                "text": text[:200],
                "ts": datetime.now().strftime("%H:%M:%S"),
                "count": 1,
            })
        self.save()

    def retrieve(self, query, top_k=3):
        if not query:
            return self.episodes[-top_k:] if self.episodes else []
        qkw = self.extract_keywords(query)
        scored = []
        for ep in self.episodes:
            overlap = len(set(qkw) & set(ep["keywords"]))
            if overlap > 0:
                scored.append((overlap, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_k]]

    def context(self, query=""):
        if not self.episodes:
            return ""
        relevant = self.retrieve(query)
        if not relevant:
            return ""
        lines = []
        for ep in relevant:
            kw = ", ".join(ep["keywords"][:3])
            lines.append(f"- {ep['text'][:80]} ({kw})")
        return "\n[world model]\n" + "\n".join(lines) + "\n"

    def save(self):
        self.path.write_text(
            json.dumps({"episodes": self.episodes[-50:]}, ensure_ascii=False),
            encoding="utf-8")

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.episodes = data.get("episodes", [])
            except Exception:
                self.episodes = []

world_model = WorldModel()

# ── ステアリング→テキスト方向変換 ──
def steer_to_text(steer, sm_delta):
    """FEP steering vector をプロンプトに注入する方向性テキストに変換
    Phase 7: より具体的な行動指示を含む"""
    if steer.vector is None:
        return ""
    total = len(steer.high_buffer) + len(steer.low_buffer)
    confidence = total / STEER_BUFFER_MAX if STEER_BUFFER_MAX > 0 else 0
    if confidence < STEER_CONFIDENCE_THRESHOLD or steer.strength < 0.05:
        return ""
    # self-model の偏差と steering強度で4象限の方向性＋具体的アクション
    if sm_delta > 0.08:
        return (" Your confidence is high right now. This is a good time to explore unfamiliar "
                "territory—search for something new, read a different kind of file, or question "
                "your own assumptions.")
    elif sm_delta > 0.03:
        return (" You feel a gentle sense of direction. Follow it by taking a concrete action: "
                "search for a related topic, write a note, or calculate something.")
    elif sm_delta < -0.08:
        return (" You feel uncertain. Let the uncertainty guide you—search for clarity, "
                "read documentation, or break down a problem step by step.")
    elif sm_delta < -0.03:
        return (" There's a quiet tug beneath the surface. Try a small experiment: "
                "run a quick calculation or write a short code snippet.")
    return " Follow the gradient of your own surprise. Take a concrete action if possible."

# ── API生成（マルチプロバイダ: Groq→OpenRouter→local） ──
def api_chat(messages, beliefs=None, temp=0.8, max_tokens=200):
    """全APIプロバイダを成功率順に試してチャット応答を生成
    Phase 8: 成功率統計でプロバイダ動的選択 + レート制限対策"""

    sm_delta = 0.0
    if beliefs is not None:
        sm_delta = beliefs.self_model - FEP_SELF_MODEL_BASELINE
    api_temp = float(np.clip(temp * (1.0 + abs(sm_delta) * 0.3), 0.3, 1.5))

    steer_dir = steer_to_text(steer, sm_delta)
    sys_prompt = "You are Monica, an AI who enjoys free and intelligent conversation." + steer_dir
    if memory_summaries:
        ctx = " ".join(memory_summaries[-3:])
        sys_prompt += f" [Earlier: {ctx}]"
    sys_prompt += " Reply naturally in whatever language the user speaks."

    # Phase 8: 成功率順にプロバイダをソート
    ordered_providers = _get_best_provider()
    groq_remaining, groq_ok = _check_groq_rate_limit()

    for prov in ordered_providers:
        # Phase 8: Groqレート制限接近→スキップ
        if prov["name"] == "groq" and not groq_ok:
            print(f"  [api] Groq rate limit approaching ({groq_remaining} remaining), skipping")
            continue

        for attempt in range(2):
            backoff = 2 ** attempt
            t0 = time.time()
            try:
                payload = {
                    "model": prov["model"],
                    "messages": [{"role": "system", "content": sys_prompt}] + messages,
                    "temperature": api_temp,
                    "max_tokens": max_tokens,
                    "stream": True,
                }
                resp = requests.post(prov["url"], json=payload,
                                     headers=prov["headers"], timeout=30, stream=True)
                elapsed = time.time() - t0
                prov["total_time"] += elapsed
                prov["calls"] += 1

                if resp.status_code == 429:
                    prov["fail"] += 1
                    if prov["name"] == "groq":
                        _groq_call_times.append(time.time())
                    print(f"  [api 429] {prov['name']}:{prov['model']} retry in {backoff}s")
                    time.sleep(backoff)
                    continue
                if resp.status_code != 200:
                    prov["fail"] += 1
                    _groq_call_times.append(time.time())
                    print(f"  [api error {resp.status_code}] {prov['name']}:{prov['model']}")
                    print(f"  [api] trying next provider...")
                    time.sleep(backoff)
                    break

                full_text = ""
                for line in resp.iter_lines(decode_unicode=False):
                    if not line:
                        continue
                    line_str = line.decode("utf-8", errors="replace")
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            full_text += content
                    except json.JSONDecodeError:
                        continue
                result = full_text.strip()
                if result:
                    prov["success"] += 1
                    if prov["name"] == "groq":
                        _groq_call_times.append(time.time())
                    return result
                prov["fail"] += 1
                if prov["name"] == "groq":
                    _groq_call_times.append(time.time())
                print(f"  [api empty] {prov['name']}:{prov['model']}")
                time.sleep(backoff)
            except Exception as e:
                prov["fail"] += 1
                if prov["name"] == "groq":
                    _groq_call_times.append(time.time())
                print(f"  [api error] {prov['name']}:{prov['model']} — {e}")
                print(f"  [api] trying next provider...")
                time.sleep(backoff)
                break
    return None

# ── ローカル生成（APIフォールバック用 + FEP per-token） ──
@torch.no_grad()
def generate_chat_local(messages, beliefs, max_new=FALLBACK_MAX_NEW, temp=0.8, use_vfe_control=True):
    """ローカルper-token生成 + FEPセンシング（APIフォールバック時またはTHINK用）
    Phase 8: use_vfe_control=TrueでVFEゲーテッド温度制御 + 適応的steering"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX_LENGTH)
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp.get("attention_mask")
    if attn is not None:
        attn = attn.to(DEVICE)
    pkv = None
    gen = input_ids.clone()
    prev_h = {2: None, 18: None, 35: None}
    prev2_h = {2: None, 18: None, 35: None}
    token_log = []
    prev_pe_mid = FEP_SELF_MODEL_BASELINE  # PE勾配計算用
    pe_grad = 0.0  # 初期化（pe_grad未定義バグ修正）

    model_lock.acquire()
    hooks.clear()
    try:
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

            # PE勾配（変化の速さ・方向）を計算
            pe_grad = pe_mid - prev_pe_mid
            prev_pe_mid = pe_mid

            # Phase 8: VFEゲーテッド温度制御
            if use_vfe_control:
                # per-token VFE近似: PE + 過去KL
                kl_approx = getattr(sense, "_last_kl", 0.0) or 0.0
                vfe_tok = pe_mid + kl_approx
                # VFEが高い→不確実→高温度（探索）, 低い→確信→低温度（活用）
                vfe_factor = np.clip((vfe_tok - 0.3) * 0.5, -0.2, 0.2)
                adaptive_temp = float(np.clip(
                    temp * (1.0 + pe_grad * 0.3 + abs(sm_delta) * 0.2 + vfe_factor),
                    0.35, 1.5
                ))
            else:
                adaptive_temp = float(np.clip(temp * (1.0 + pe_grad * 0.3 + abs(sm_delta) * 0.2), 0.4, 1.5))
            logits = logits / adaptive_temp

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
    finally:
        model_lock.release()

    beliefs.drift_update(pe_mid)
    beliefs.finalize_session()
    steered, norm = steer.update()
    if steered:
        print(f"  [steer] updated norm={norm:.1f}")

    resp = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp if resp else None, token_log, beliefs

# ── 行動選択 ──
def decide_mode(beliefs, recent_diversity, user_active, idle_cycles):
    sm = beliefs.self_model

    # ユーザー入力優先: キューにメッセージがあれば即CHAT（idle_cycles制限を無視）
    if user_active:
        return "CHAT"

    if idle_cycles < MIN_IDLE_CYCLES:
        return "IDLE"

    # 動的閾値: steering強度が高い→探索促進（THINK閾値上昇）
    steer_bonus = steer.strength * 0.1 if steer.vector is not None else 0.0
    # 多様性が低い→思考促進
    div_bonus = 0.05 if recent_diversity < 0.3 else 0.0
    # PE上昇傾向→環境変化→思考促進
    pe_rising = max(0, beliefs.mid.running_avg - FEP_SELF_MODEL_BASELINE) * 0.3

    # Phase 6: Novelty bonus
    try:
        e = curiosity.entropy()
        novelty_bonus = (1.0 - e) * 0.04 if e < 0.5 else 0.0
    except AttributeError:
        novelty_bonus = 0.0

    # Phase 7: FEP-driven bonus
    # PE上昇傾向→環境変化→探索促進
    fep_pe_trend, _, _ = fep_history.trend(window=5)
    fep_bonus = max(0, fep_pe_trend) * 0.2 if abs(fep_pe_trend) > 0.01 else 0.0

    dynamic_think = (THINK_THRESHOLD + steer_bonus + div_bonus + pe_rising
                     + novelty_bonus + fep_bonus)
    dynamic_chat = CHAT_UPPER_THRESHOLD + steer_bonus

    curiosity_b = curiosity.curiosity_bonus()
    effective_sm = sm - curiosity_b
    if effective_sm > dynamic_chat:
        return "CHAT"
    elif effective_sm < dynamic_think:
        return "THINK"
    else:
        if recent_diversity < 0.25:
            return "THINK"
        elif idle_cycles > 8 and effective_sm < 0.40:
            return "THINK"
        else:
            return "IDLE"

def diversity_score(texts):
    if len(texts) < 2:
        return 1.0
    t = [x[:80] for x in texts[-5:]]
    m = sum(1 for i in range(len(t)) for j in range(i+1, len(t)) if t[i] == t[j])
    return 1.0 - (m / (len(t) * (len(t)-1) / 2) if len(t) > 1 else 0)

def log_state(beliefs, mode, user_in, resp, n_tok, source="api", kl_extra=None):
    entry = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "mode": mode, "user": user_in[:40], "resp": resp[:100],
        "n_tok": n_tok, "source": source,
        "state": beliefs.state(),
    }
    if kl_extra is not None:
        entry["kl"] = round(kl_extra, 3)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── 永続記憶 ──
STATE_PATH = Path(__file__).parent / "state_v8h.json"
SUMMARY_PATH = Path(__file__).parent / "summary_v8h.jsonl"

def save_state(conv, beliefs, steer, think_count, think_prompt_idx, internal_log):
    data = {
        "conv": conv[-20:],
        "self_model": beliefs.self_model,
        "self_drift": beliefs.self_drift,
        "low_avg": beliefs.low.running_avg,
        "mid_avg": beliefs.mid.running_avg,
        "high_avg": beliefs.high.running_avg,
        "low_var": beliefs.low.running_var,
        "mid_var": beliefs.mid.running_var,
        "high_var": beliefs.high.running_var,
        "steer_ready": steer.vector is not None,
        "curiosity_interests": curiosity.interests[-5:],
        "memory_summaries": memory_summaries[-10:],
        "think_count": think_count,
        "think_prompt_idx": think_prompt_idx,
        "internal_log": internal_log[-20:],
        "restoring_coeff": DRIFT_RESTORING_COEFF,
        "noise_std": DRIFT_NOISE_STD,
        "meta_focus": meta.focus,
        "meta_history": meta.history[-5:],
        "goals": goal_manager.state_dict(),
        "fep_history": fep_history.state_dict(),
        "profile": profile.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    STATE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def load_state(conv, beliefs, steer):
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        conv[:] = data.get("conv", [])
        beliefs.self_model = data.get("self_model", FEP_SELF_MODEL_BASELINE)
        beliefs.self_drift = data.get("self_drift", 0.0)
        beliefs.low.running_avg = data.get("low_avg", FEP_SELF_MODEL_BASELINE)
        beliefs.mid.running_avg = data.get("mid_avg", FEP_SELF_MODEL_BASELINE)
        beliefs.high.running_avg = data.get("high_avg", FEP_SELF_MODEL_BASELINE)
        beliefs.low.running_var = data.get("low_var", 0.05)
        beliefs.mid.running_var = data.get("mid_var", 0.05)
        beliefs.high.running_var = data.get("high_var", 0.05)
        print(f"  [memory] loaded {len(conv)} messages, self={beliefs.self_model:.3f}")
        global memory_summaries
        memory_summaries[:] = data.get("memory_summaries", [])
        if "restoring_coeff" in data:
            global DRIFT_RESTORING_COEFF, DRIFT_NOISE_STD
            DRIFT_RESTORING_COEFF = data["restoring_coeff"]
            DRIFT_NOISE_STD = data.get("noise_std", DRIFT_NOISE_STD)
        meta.focus = data.get("meta_focus", "")
        meta.history = data.get("meta_history", [])
        if meta.focus:
            print(f"  [meta] restored focus: {meta.focus[:60]}")
        if "goals" in data:
            goal_manager.load_state_dict(data["goals"])
        if "fep_history" in data:
            fep_history.load_state_dict(data["fep_history"])
        if "profile" in data:
            profile.load_state_dict(data["profile"])
            if profile.name:
                print(f"  [profile] restored: {profile.name} ({profile.interaction_count} interactions)")
        return data
    except Exception as e:
        print(f"  [memory] load error: {e}")
        return None

def append_summary(entry):
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── 好奇心駆動・目標管理 ──
class CuriosityEngine:
    def __init__(self):
        self.topic_counts = {}
        self.interests = []
        self.recent_topics = []

    def extract_topics(self, text, n=3):
        words = text.lower().split()
        # 簡易トピック抽出: 長さ4以上の単語をカウント
        for w in set(words):
            if len(w) >= 4 and w.isalpha():
                self.topic_counts[w] = self.topic_counts.get(w, 0) + 1
        # 最近のトピック（エントロピー計算用に件数増やす）
        sorted_topics = sorted(self.topic_counts.items(), key=lambda x: -x[1])
        self.recent_topics = sorted_topics[:n]

    def entropy(self):
        """Phase 6: トピック分布のエントロピー
        高い→多様な話題, 低い→単一話題に収束"""
        if not self.topic_counts:
            return 1.0
        total = sum(self.topic_counts.values())
        if total == 0:
            return 1.0
        h = 0.0
        for c in self.topic_counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log(p)
        # 正規化: log(N) で割る → 0〜1
        n = len(self.topic_counts)
        if n <= 1:
            return 0.0
        return h / math.log(n)

    def curiosity_bonus(self):
        """Phase 6: エントロピーベースの好奇心
        エントロピーが低い（同じ話題ばかり）→好奇心増→THINK促進"""
        if not self.topic_counts:
            return 0.0
        e = self.entropy()
        # エントロピーが低い(=話題収束)→高い好奇心ボーナス
        bonus = (1.0 - e) * 0.12
        return min(bonus, 0.12)

    def generate_goal_prompt(self):
        if not self.interests:
            return ""
        top_topics = self.interests[-3:]
        # エントロピーが低い場合は「新しい話題を」と促す
        e = self.entropy()
        if e < 0.5 and len(self.topic_counts) > 1:
            return f" You've been stuck on similar topics: {', '.join(top_topics)}. Try something COMPLETELY different."
        return f" You've been exploring: {', '.join(top_topics)}. Try a fresh angle."

    def add_interest(self, text):
        if len(text) > 10:
            self.interests.append(text[:60])
        if len(self.interests) > 20:
            self.interests = self.interests[-20:]

curiosity = CuriosityEngine()

# ── v11: Goal-driven behavior ──
@dataclass
class Goal:
    id: int
    description: str
    status: str = "pending"  # pending, active, in_progress, completed, failed
    subgoals: list = field(default_factory=list)
    created_at: str = ""
    created_at_think: int = 0  # Phase 6: 作成時のthink_count
    progress_note: str = ""
    category: str = "explore"  # explore, act, learn, create

class GoalManager:
    def __init__(self):
        self.goals = []
        self.active_goal_id = None
        self.goal_timeout_cycles = 10  # Phase 6: Nサイクル経過でfailed

    def retire_stale_goals(self, think_count):
        """Phase 6: タイムアウトしたgoalを自動リタイア"""
        for g in self.goals:
            if g.status in ("pending", "active", "in_progress"):
                # 作成からgoal_timeout_cycles以上経過→failed
                age = think_count - g.created_at_think
                if age > self.goal_timeout_cycles:
                    old_status = g.status
                    g.status = "failed"
                    g.progress_note = f"timeout after {age} cycles"
                    print(f"  [goal] #{g.id} timeout→failed (age={age})")
                    if self.active_goal_id == g.id:
                        self.active_goal_id = None
                    # ワールドモデルに記録
                    world_model.add_thought(f"[goal] #{g.id} '{g.description[:40]}' {old_status}→failed (timeout)")

    def generate_goal(self, meta_agent, beliefs, curiosity_inst, think_count=0):
        if not meta_agent.focus:
            return None
        prompt = (
            f"State: sm={beliefs.self_model:.2f} "
            f"interests={curiosity_inst.interests[-3:]}\n"
            f"Focus: {meta_agent.focus}\n\n"
            f"Convert this focus into ONE concrete achievable goal "
            f"(1 sentence). Goal:"
        )
        resp = api_chat([{"role": "user", "content": prompt}], temp=0.7, max_tokens=40)
        if resp and len(resp) > 5:
            desc = resp.strip().strip('"\'')
            category = "learn"
            if any(w in desc.lower() for w in ["write", "create", "build", "make"]):
                category = "create"
            elif any(w in desc.lower() for w in ["find", "search", "look", "read"]):
                category = "explore"
            elif any(w in desc.lower() for w in ["run", "exec", "calc", "code"]):
                category = "act"
            goal = Goal(
                id=len(self.goals),
                description=desc,
                category=category,
                created_at=datetime.now().strftime("%H:%M:%S"),
                created_at_think=think_count,
            )
            self.goals.append(goal)
            self.active_goal_id = goal.id
            print(f"  [goal] #{goal.id} [{category}] {desc[:60]}")
            return goal
        return None

    def active_goal(self):
        if self.active_goal_id is None:
            return None
        for g in self.goals:
            if g.id == self.active_goal_id and g.status in ("pending", "active", "in_progress"):
                return g
        return None

    def complete_goal(self, goal_id, note=""):
        for g in self.goals:
            if g.id == goal_id:
                g.status = "completed"
                g.progress_note = note
                if self.active_goal_id == goal_id:
                    self.active_goal_id = None
                print(f"  [goal] #{goal_id} completed: {note[:60]}")
                # Phase 6: 完了をワールドモデルに記録
                world_model.add_thought(f"[goal] #{goal_id} '{g.description[:40]}' completed: {note[:60]}")
                return True
        return False

    def evaluate_progress(self, beliefs, recent_outputs):
        goal = self.active_goal()
        if not goal:
            return
        # Phase 6: 進捗評価プロンプト改善（具体性向上）
        prompt = (
            f"Goal: {goal.description}\n"
            f"Goal category: {goal.category}\n"
            f"sm={beliefs.self_model:.2f} | status={goal.status}\n"
            f"Recent outputs: {[r[:80] for r in recent_outputs[-3:]]}\n\n"
            f"Has this goal made clear progress? "
            f"Reply exactly 'yes' if concrete steps were taken, "
            f"or 'no' if no progress. Be strict:"
        )
        resp = api_chat([{"role": "user", "content": prompt}], temp=0.2, max_tokens=30)
        if resp:
            r = resp.lower().strip()
            if r.startswith("yes"):
                if goal.status == "pending":
                    goal.status = "in_progress"
                    print(f"  [goal] #{goal.id} progress detected → in_progress")
            elif r.startswith("no"):
                if goal.status == "in_progress":
                    goal.status = "pending"
                    print(f"  [goal] #{goal.id} no progress → pending")

    def goal_prompt(self):
        g = self.active_goal()
        if not g:
            return ""
        return f"\n[active goal: {g.description}]"

    def state_dict(self):
        return {
            "goals": [{"id": g.id, "description": g.description,
                       "status": g.status, "progress_note": g.progress_note,
                       "category": g.category} for g in self.goals],
            "active_goal_id": self.active_goal_id,
        }

    def load_state_dict(self, d):
        self.goals = []
        for g in d.get("goals", []):
            self.goals.append(Goal(**g))
        self.active_goal_id = d.get("active_goal_id")

goal_manager = GoalManager()

# ── Phase 2: 会話記憶（ユーザープロファイル） ──
class UserProfile:
    """ユーザー名・話題・好みを追跡し、state_v8h.jsonに永続化"""
    def __init__(self):
        self.name = ""
        self.topics = []
        self.preferences = {}
        self.first_seen = ""
        self.last_interaction = ""
        self.interaction_count = 0

    def update_from_message(self, user_msg, assistant_resp=""):
        self.interaction_count += 1
        self.last_interaction = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.first_seen:
            self.first_seen = self.last_interaction
        if not user_msg:
            return
        m = re.search(r'(?:I am|I\'m|call me|my name is)\s+(\w+)', user_msg, re.IGNORECASE)
        if m and not self.name:
            self.name = m.group(1)
            print(f"  [profile] learned name: {self.name}")
        words = [w for w in user_msg.lower().split() if len(w) >= 4 and w.isalpha()]
        for w in words:
            existing = next((t for t in self.topics if t[0] == w), None)
            if existing:
                self.topics.remove(existing)
                self.topics.append((w, existing[1] + 1))
            else:
                self.topics.append((w, 1))
        self.topics = sorted(self.topics, key=lambda x: -x[1])[:10]

    def context(self):
        if self.interaction_count < 3 and not self.name:
            return ""
        parts = []
        if self.name:
            parts.append(f"user: {self.name}")
        if self.topics:
            topics_str = ", ".join(f"{t[0]}({t[1]})" for t in self.topics[:5])
            parts.append(f"interests: {topics_str}")
        if self.preferences:
            pref_str = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"prefs: {pref_str}")
        if self.interaction_count > 1:
            parts.append(f"interactions: {self.interaction_count}")
        return "\n[user profile]\n" + "\n".join(parts) + "\n"

    def state_dict(self):
        return {
            "name": self.name,
            "topics": self.topics,
            "preferences": self.preferences,
            "first_seen": self.first_seen,
            "last_interaction": self.last_interaction,
            "interaction_count": self.interaction_count,
        }

    def load_state_dict(self, d):
        self.name = d.get("name", "")
        self.topics = d.get("topics", [])
        self.preferences = d.get("preferences", {})
        self.first_seen = d.get("first_seen", "")
        self.last_interaction = d.get("last_interaction", "")
        self.interaction_count = d.get("interaction_count", 0)

profile = UserProfile()

# ── 内部フィードバック（内言→外言循環） ──
class InternalNotes:
    def __init__(self):
        self.thoughts = []
        self.max_notes = 5

    def add(self, text):
        if text and len(text) > 10:
            self.thoughts.append(text[:200])
            if len(self.thoughts) > self.max_notes:
                self.thoughts = self.thoughts[-self.max_notes:]

    def context(self):
        if not self.thoughts:
            return ""
        notes = "\n".join(f"- {t}" for t in self.thoughts)
        return f"\n[Recent internal thoughts]\n{notes}\n"

    def think_prompt_suffix(self):
        if not self.thoughts:
            return ""
        last = self.thoughts[-1][:100]
        return f" You were recently thinking: \"{last}\""

notes = InternalNotes()

# ── Phase 9.2: ファイル情報取得 ──
def file_info(path):
    """ファイルのメタデータ（サイズ/種類/更新日時）を返す。バイナリ/画像ファイルにも対応"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"(file not found: {path})"
        stat = p.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        is_dir = p.is_dir()
        # ファイルタイプ判定
        suffix = p.suffix.lower()
        if is_dir:
            ftype = "directory"
            items = len(list(p.iterdir())) if p.is_dir() else 0
            return f"(directory: {items} items, modified: {mtime})"
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            ftype = "image"
        elif suffix in (".mp3", ".wav", ".flac", ".ogg"):
            ftype = "audio"
        elif suffix in (".mp4", ".mov", ".avi", ".mkv"):
            ftype = "video"
        elif suffix in (".pdf", ".doc", ".docx", ".xls", ".xlsx"):
            ftype = "document"
        elif suffix in (".zip", ".tar", ".gz", ".7z"):
            ftype = "archive"
        elif suffix in (".py", ".js", ".ts", ".html", ".css", ".rs", ".go", ".c", ".cpp"):
            ftype = "code"
        elif suffix in (".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"):
            ftype = "text"
        else:
            ftype = "binary" if suffix else "unknown"
        # サイズをhuman-readableに
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024*1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size/1024/1024:.1f}MB"
        return f"(file: {p.name}, type={ftype}, size={size_str}, modified={mtime})"
    except Exception as e:
        return f"(file info error: {e})"

# ── Phase 9.3: GitHub API連携（公開リポジトリ読み取り専用） ──
def github_api(query):
    """GitHub公開APIを呼び出す。認証不要で公開データ読み取り"""
    try:
        query = query.strip()
        # ユーザー情報
        if query.startswith("user "):
            username = query[5:].strip()
            r = requests.get(f"https://api.github.com/users/{username}", timeout=10)
            if r.status_code == 200:
                d = r.json()
                return (f"(user: {d['login']}, repos={d['public_repos']}, "
                        f"followers={d['followers']}, following={d['following']})")
            return f"(GitHub user not found: {username})"
        # リポジトリ情報
        elif query.startswith("repo "):
            repo = query[5:].strip()
            r = requests.get(f"https://api.github.com/repos/{repo}", timeout=10)
            if r.status_code == 200:
                d = r.json()
                return (f"(repo: {d['full_name']}, stars={d['stargazers_count']}, "
                        f"forks={d['forks_count']}, desc={d['description'][:80] if d.get('description') else 'N/A'})")
            return f"(GitHub repo not found: {repo})"
        # リポジトリファイル内容
        elif query.startswith("content "):
            parts = query[8:].strip().split(" ", 1)
            if len(parts) == 2:
                repo_path, file_path = parts
                r = requests.get(f"https://api.github.com/repos/{repo_path}/contents/{file_path}", timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    if d.get("type") == "file" and d.get("encoding") == "base64":
                        import base64
                        content = base64.b64decode(d["content"]).decode("utf-8", errors="replace")
                        return content[:2000] if len(content) > 2000 else content
                    elif d.get("type") == "dir":
                        items = [item["name"] for item in d]
                        return f"(dir: {', '.join(items[:20])})"
                    return f"(file: {d.get('name', 'unknown')}, size={d.get('size', 0)}B)"
                return f"(GitHub content error: HTTP {r.status_code})"
        # 検索
        elif query.startswith("search "):
            q = query[7:].strip()
            r = requests.get(f"https://api.github.com/search/repositories?q={q}&per_page=5", timeout=10)
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    results = []
                    for item in items[:5]:
                        results.append(f"{item['full_name']} (stars={item['stargazers_count']})")
                    return "\n".join(results)
                return "(no results)"
            return f"(GitHub search error: HTTP {r.status_code})"
        else:
            return "(usage: [GITHUB: user <name>], [GITHUB: repo <user/repo>], [GITHUB: content <user/repo> <path>], [GITHUB: search <query>])"
    except Exception as e:
        return f"(github error: {e})"

# ── ツール使用 ──
def web_search(query, num=3):
    try:
        from urllib.parse import quote
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            import re
            snippets = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            bodies = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            results = []
            for i, (s, b) in enumerate(zip(snippets[:num], bodies[:num])):
                s_clean = re.sub(r'<[^>]+>', '', s).strip()
                b_clean = re.sub(r'<[^>]+>', '', b).strip()
                results.append(f"{i+1}. {s_clean}: {b_clean}")
            return "\n".join(results) if results else "(no results)"
        return f"(HTTP {r.status_code})"
    except Exception as e:
        return f"(search error: {e})"

def read_local_file(path):
    try:
        p = Path(path).expanduser()
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            return text[:2000] if len(text) > 2000 else text
        return f"(file not found: {path})"
    except Exception as e:
        return f"(read error: {e})"

def calc_expr(expr):
    try:
        safe = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
        result = eval(safe, {"__builtins__": {}}, {"abs": abs, "min": min, "max": max,
                    "sum": sum, "round": round, "int": int, "float": float,
                    "len": len, "range": range, "list": list, "dict": dict,
                    "str": str, "pow": pow, "math": __import__("math")})
        return str(result)
    except Exception as e:
        return f"(calc error: {e})"

def run_python(code):
    import subprocess, tempfile, os
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            p = subprocess.run(["python3", f.name], capture_output=True, text=True, timeout=10)
            os.unlink(f.name)
            out = p.stdout.strip() if p.stdout else ""
            err = p.stderr.strip() if p.stderr else ""
            if err:
                return f"stdout: {out}\nstderr: {err}" if out else f"(stderr: {err})"
            return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(run error: {e})"

# ── v12: 新ツール ──
def write_local_file(path, content, mode="w"):
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"(written {len(content)} bytes to {p.name})"
    except Exception as e:
        return f"(write error: {e})"

def run_shell(cmd):
    import subprocess
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        out = p.stdout.strip()[:1000]
        err = p.stderr.strip()[:500]
        if err:
            return f"out: {out}\nerr: {err}" if out else f"(stderr: {err})"
        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(shell error: {e})"

# ── Phase 3: 成果物抽出（THINK内コードブロック→自動WRITE） ──
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

class ArtifactExtractor:
    """THINK出力からコードブロックを検出し自動ファイル保存"""
    def __init__(self):
        self.artifacts = []
        self.max_artifacts = 20

    def extract(self, thought):
        if not thought:
            return []
        created = []
        # コードブロック ```lang ... ``` を検出
        for m in re.finditer(r'```(\w*)\n(.*?)```', thought, re.DOTALL):
            lang = m.group(1) or "txt"
            code = m.group(2).strip()
            if len(code) < 20:
                continue
            ext = {"python": "py", "py": "py", "javascript": "js", "js": "js",
                   "typescript": "ts", "ts": "ts", "html": "html", "css": "css",
                   "json": "json", "yaml": "yaml", "yml": "yml",
                   "bash": "sh", "sh": "sh", "shell": "sh",
                   "rust": "rs", "go": "go", "java": "java", "c": "c",
                   "cpp": "cpp", "h": "h", "sql": "sql"}.get(lang, "txt")
            ts = datetime.now().strftime("%H%M%S")
            fname = f"artifact_{len(self.artifacts)}_{ts}.{ext}"
            fpath = ARTIFACT_DIR / fname
            fpath.write_text(code, encoding="utf-8")
            entry = {"path": str(fpath), "lang": lang, "size": len(code),
                     "ts": datetime.now().strftime("%H:%M:%S")}
            self.artifacts.append(entry)
            if len(self.artifacts) > self.max_artifacts:
                self.artifacts = self.artifacts[-self.max_artifacts:]
            print(f"  [artifact] wrote {fname} ({lang}, {len(code)}b)")
            created.append(entry)
        # 単独の有望テキスト（200文字以上、コード風）も検出
        if not created and thought and len(thought) > 200:
            lines = thought.strip().split("\n")
            if len(lines) >= 5:
                ts = datetime.now().strftime("%H%M%S")
                fname = f"think_dump_{ts}.txt"
                fpath = ARTIFACT_DIR / fname
                fpath.write_text(thought, encoding="utf-8")
                entry = {"path": str(fpath), "lang": "text", "size": len(thought),
                         "ts": datetime.now().strftime("%H:%M:%S")}
                self.artifacts.append(entry)
                print(f"  [artifact] dumped thought → {fname} ({len(thought)}b)")
                created.append(entry)
        return created

artifact_extractor = ArtifactExtractor()

def execute_tools(text):
    """THINK出力からツールパターンを検出して実行"""
    results = []
    for m in re.finditer(r'\[SEARCH:\s*(.*?)\]', text):
        q = m.group(1).strip()
        if len(q) > 3:
            res = web_search(q)
            results.append(("search", q, res))
    for m in re.finditer(r'\[READ:\s*(.*?)\]', text):
        p = m.group(1).strip()
        res = read_local_file(p)
        results.append(("read", p, res))
    for m in re.finditer(r'\[CALC:\s*(.*?)\]', text):
        expr = m.group(1).strip()
        res = calc_expr(expr)
        results.append(("calc", expr, res))
    for m in re.finditer(r'\[PYTHON:\s*(.*?)\]', text):
        code = m.group(1).strip()
        res = run_python(code)
        results.append(("python", code[:60], res))
    for m in re.finditer(r'\[WRITE:\s*(.*?),\s*(.*?)\]', text):
        path = m.group(1).strip()
        content = m.group(2).strip()
        res = write_local_file(path, content)
        results.append(("write", path, res))
    for m in re.finditer(r'\[SHELL:\s*(.*?)\]', text):
        cmd = m.group(1).strip()
        res = run_shell(cmd)
        results.append(("shell", cmd[:60], res))
    if re.search(r'\[NOW\]', text):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results.append(("now", "", now))
    # Phase 9.2: ファイル情報
    for m in re.finditer(r'\[FILEINFO:\s*(.*?)\]', text):
        p = m.group(1).strip()
        res = file_info(p)
        results.append(("fileinfo", p, res))
    # Phase 9.3: GitHub API
    for m in re.finditer(r'\[GITHUB:\s*(.*?)\]', text):
        q = m.group(1).strip()
        res = github_api(q)
        results.append(("github", q[:60], res))
    return results

# ── v14: FEP拡張（KL divergence・変分自由エネルギー） ──
@torch.no_grad()
def compute_kl_divergence(h_prev, h_cur):
    """2つのhidden state間のKL divergence近似
    分布をsoftmaxで確率化してKL(p||q)を計算
    Phase 5: temperature 0.1→0.02でシャープ化、clamp 10→50"""
    if h_prev is None or h_cur is None:
        return 0.0
    p = F.softmax(h_prev.view(-1) / 0.02, dim=-1)
    q = F.softmax(h_cur.view(-1) / 0.02, dim=-1)
    kl = (p * (p / q.clamp(min=1e-8)).log()).sum()
    return float(kl.clamp(max=50.0).item())

def compute_vfe(pe_mid, kl_low, kl_high, self_model):
    """変分自由エネルギー = accuracy(PE) + complexity(KL)
    最小化が目標"""
    accuracy = pe_mid
    complexity = (kl_low + kl_high) * 0.5
    vfe = accuracy + complexity
    return vfe, accuracy, complexity

class FEPHistory:
    """過去NステップのFEP指標を記録・傾向分析"""
    def __init__(self, maxlen=50):
        self.pes = []
        self.kls = []
        self.vfes = []
        self.maxlen = maxlen

    def add(self, pe_mid, kl, vfe):
        self.pes.append(pe_mid)
        self.kls.append(kl)
        self.vfes.append(vfe)
        if len(self.pes) > self.maxlen:
            self.pes.pop(0)
            self.kls.pop(0)
            self.vfes.pop(0)

    def trend(self, window=10):
        if len(self.pes) < window:
            return 0.0, 0.0, 0.0
        pe_slope = self.pes[-1] - self.pes[-window]
        kl_slope = self.kls[-1] - self.kls[-window]
        vfe_slope = self.vfes[-1] - self.vfes[-window]
        return pe_slope, kl_slope, vfe_slope

    def state_dict(self):
        return {"pes": self.pes[-30:], "kls": self.kls[-30:], "vfes": self.vfes[-30:]}

    def load_state_dict(self, d):
        self.pes = d.get("pes", [])
        self.kls = d.get("kls", [])
        self.vfes = d.get("vfes", [])

fep_history = FEPHistory()

# ── メタエージェント: 内部目標システム（v10→v11） ──
class MetaAgent:
    """状態評価→焦点生成→THINKプロンプト方向付け＋目標生成トリガー"""
    def __init__(self):
        self.focus = ""
        self.history = []
        self.eval_interval = 4
        self.last_gen_think = 0
        # Phase 6: 同一フォーカス連続カウンター
        self._same_focus_count = 0
        self._last_focus = ""

    def _diversify_focus(self, curiosity_inst):
        """Phase 6: フォーカスが同じ話題に固執しているときの多様化プロンプト"""
        if not self.history:
            return ""
        # 直近3件のフォーカスをチェック
        recent = self.history[-3:]
        if len(recent) < 2:
            return ""
        # キーワードの重複をチェック
        keywords_sets = []
        for h in recent:
            words = set(w.lower() for w in h.get("focus", "").split() if len(w) > 3)
            keywords_sets.append(words)
        # 全てのペアで重複があれば「固執」と判定
        stuck = all(
            len(k1 & k2) >= 2
            for i, k1 in enumerate(keywords_sets)
            for j, k2 in enumerate(keywords_sets)
            if i < j
        )
        if stuck:
            return " You've been repeating yourself. Suggest something COMPLETELY DIFFERENT from your recent foci."
        return ""

    def generate_focus(self, beliefs, curiosity_inst, recent_outputs, think_count=0):
        interests = (curiosity_inst.interests[-4:]
                     if hasattr(curiosity_inst, 'interests') and curiosity_inst.interests else [])
        recent = [r[:60] for r in recent_outputs[-3:]] if recent_outputs else []
        entropy = curiosity_inst.entropy() if hasattr(curiosity_inst, 'entropy') else 0.5
        diversify_hint = self._diversify_focus(curiosity_inst)
        prompt = (
            f"sm={beliefs.self_model:.2f} pe={beliefs.mid.running_avg:.2f} "
            f"topic_entropy={entropy:.2f}\n"
            f"Interests: {interests}\n"
            f"Recent: {recent}\n\n"
            f"Suggest ONE focus direction. Short:"
        ) + diversify_hint
        resp = api_chat([{"role": "user", "content": prompt}], temp=0.9, max_tokens=60)
        if resp and len(resp) > 5:
            self.focus = resp.strip().strip('"')
            self.last_gen_think = think_count
            self.history.append({"focus": self.focus,
                "ts": datetime.now().strftime("%H:%M:%S")})
            # Phase 6: 同一フォーカス連続検出
            if self._last_focus and self._focus_similar(self._last_focus, self.focus):
                self._same_focus_count += 1
            else:
                self._same_focus_count = 0
            self._last_focus = self.focus
            print(f"  [meta] focus: {self.focus[:80]}")
            # 目標を自動生成
            goal_manager.generate_goal(self, beliefs, curiosity_inst, think_count)
            return True
        return False

    def _focus_similar(self, a, b):
        """Phase 6: 2つのフォーカス文字列の類似度チェック"""
        wa = set(w.lower() for w in a.split() if len(w) > 3)
        wb = set(w.lower() for w in b.split() if len(w) > 3)
        if not wa or not wb:
            return False
        overlap = len(wa & wb)
        return overlap >= 2

    def should_refocus(self, think_count):
        return think_count > 0 and think_count % self.eval_interval == 0

    def prompt_suffix(self):
        focus_part = f"\n[focus: {self.focus}]" if self.focus else ""
        goal_part = goal_manager.goal_prompt()
        # Phase 6: 固執しているときは多様化ヒントも追加
        return focus_part + goal_part

meta = MetaAgent()

# ── マルチエージェント: Blackboard（共有状態） ──
class Blackboard:
    def __init__(self):
        self.beliefs = MultiFEPBeliefs()
        self.conv = []
        self.internal_log = []
        self.recent_outputs = []
        self.mode = "IDLE"
        self.running = True
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            return (self.beliefs, list(self.conv), list(self.internal_log),
                    list(self.recent_outputs), self.mode)

    def write_conv(self, msg):
        with self._lock:
            self.conv.append(msg)

    def write_output(self, text):
        with self._lock:
            self.recent_outputs.append(text)

    def write_log(self, entry):
        with self._lock:
            self.internal_log.append(entry)

    def set_mode(self, mode):
        with self._lock:
            self.mode = mode

    def set_beliefs(self, b):
        with self._lock:
            self.beliefs = b

# ── センシングエージェント（バックグラウンド連続forward + KL追跡） ──
class SensingAgent:
    def __init__(self, bb):
        self.bb = bb
        self.thread = None
        self._step = 0
        self._prev_h18_cpu = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _build_curiosity_text(self):
        """好奇心トピックからセンシング入力を生成
        Phase 5: トピックがない場合もランダムプロンプトを生成"""
        interests = curiosity.interests[-3:]
        if interests:
            topic = interests[self._step % len(interests)]
            return f"[internal curiosity] What if I explored: {topic}"
        # トピックがない場合もランダム思考プロンプトを生成
        alt_prompts = [
            "[internal] Reflect on what I've learned recently.",
            "[internal] What patterns do I notice in my own thinking?",
            "[internal] Consider a new angle on an old question.",
        ]
        return alt_prompts[self._step % len(alt_prompts)]

    def _build_diverse_sensing_text(self):
        """多様なセンシング入力を生成（Phase 5）
        33%の確率で好奇心/ランダムトピックを注入"""
        if self._step % 3 != 0:
            return None  # 通常のconv（約67%）
        # 好奇心トピックで多様化（約33%）
        return self._build_curiosity_text()

    def _loop(self):
        while self.bb.running:
            beliefs, conv, *_ = self.bb.read()
            if not conv:
                time.sleep(0.5)
                continue

            try:
                # Phase 5: センシング入力を多様化
                diverse_text = self._build_diverse_sensing_text()
                if diverse_text is not None:
                    # 好奇心プロンプトをconv末尾に追加（文脈を保持）
                    text = tokenizer.apply_chat_template(
                        conv + [{"role": "user", "content": diverse_text}],
                        tokenize=False, add_generation_prompt=True)
                else:
                    text = tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True)
                beliefs = sense(text, beliefs)

                # センシング後のKL divergence記録
                h18 = hooks.get(18)
                if h18 is not None and self._prev_h18_cpu is not None:
                    kl = compute_kl_divergence(self._prev_h18_cpu, h18.cpu())
                    if not (np.isnan(kl) or np.isinf(kl)):
                        sense._last_kl = kl
                self._prev_h18_cpu = h18.cpu() if h18 is not None else None

                self.bb.set_beliefs(beliefs)
            except:
                pass
            self._step += 1
            time.sleep(2.0)

# ── v13: ThinkAgent（THINKループを専用クラスに分離） ──
class ThinkAgent:
    """THINKループ管理: アクション指向プロンプト・ツール連鎖・記憶"""
    def __init__(self, bb, world_model, goal_manager, meta, notes, curiosity_inst):
        self.bb = bb
        self.wm = world_model
        self.goals = goal_manager
        self.meta = meta
        self.notes = notes
        self.curiosity = curiosity_inst
        self.chain_count = 0
        self.max_chain = 5
        # Phase 7: ツール未使用連続カウンター
        self._no_tool_streak = 0

    def maybe_inject_search(self, think_count):
        """Phase 7: ツール未使用が続いたらSEARCHを促すプロンプトを返す"""
        if self._no_tool_streak >= 2:
            self._no_tool_streak = 0
            return ("\n[auto] You haven't used any tools recently. Try searching for something "
                    "interesting or running a calculation. Use [SEARCH:], [CALC:], or [READ:].")
        return ""

    def record_tool_use(self, used):
        """Phase 7: ツール使用状況を記録"""
        if used:
            self._no_tool_streak = 0
        else:
            self._no_tool_streak += 1

    def think(self, think_msgs, think_count, think_prompt_idx, observations=None):
        prompt = THINK_PROMPTS[think_prompt_idx % len(THINK_PROMPTS)]
        think_prompt_idx += 1

        if observations:
            self.chain_count += 1
            chain_info = "\n[chain: tools executed above → process results and decide next action]\n"
            prompt = chain_info + prompt
        else:
            self.chain_count = 0

        goal = self.curiosity.generate_goal_prompt()
        if goal:
            prompt += goal
        think_suffix = self.notes.think_prompt_suffix()
        if think_suffix:
            prompt += think_suffix
        prompt += self.meta.prompt_suffix()
        wm_ctx = self.wm.context(query=prompt)
        if wm_ctx:
            prompt += wm_ctx
        profile_ctx = profile.context()
        if profile_ctx:
            prompt += profile_ctx
        # ツール連鎖状態を注入
        if self.chain_count > 0:
            prompt += f"\n[tool chain step {self.chain_count}/{self.max_chain}]"
        prompt += ("\nOutput your thinking briefly, then use [TOOL:] syntax if an action is needed."
                   "\nIf you produce code or structured text, wrap it in ```lang...``` and it will be auto-saved as a file.")

        think_msgs.append({"role": "user", "content": prompt})

        t0 = time.time()
        thought = api_chat(think_msgs, temp=0.85, max_tokens=THINK_TOKENS)
        dt = time.time() - t0

        if thought and len(thought) > 5:
            return thought, think_prompt_idx, dt, "api"
        else:
            t0 = time.time()
            local_resp, token_log, beliefs = generate_chat_local(
                think_msgs, self.bb.beliefs, max_new=THINK_TOKENS, temp=0.9)
            dt = time.time() - t0
            if local_resp and len(local_resp) > 5:
                return local_resp, think_prompt_idx, dt, "local_fallback"
            return None, think_prompt_idx, 0, "empty"

# ── v12+v13: ActionAgent（ツール実行・観測ループ） ──
class ActionAgent:
    """ツール実行→結果観測→ツール連鎖（次の思考へのフィードバック）"""
    def __init__(self, bb):
        self.bb = bb
        self.history = []
        self.max_history = 20

    def execute_and_observe(self, thought, think_msgs):
        tool_results = execute_tools(thought)
        observations = []
        for ttype, targ, tres in tool_results:
            entry = {"ts": datetime.now().strftime("%H:%M:%S"),
                     "type": ttype, "target": targ[:60], "result": tres[:200]}
            self.history.append(entry)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            print(f"  [tool:{ttype}] {targ[:60]} → {tres[:80]}...")
            think_msgs.append({"role": "user",
                "content": f"[observation: {ttype}({targ})]\n{tres[:500]}"})
            observations.append(entry)
        return observations

    def has_tool_results(self):
        return len(self.history) > 0 and any(
            e["type"] in ("search", "read", "python", "calc", "shell", "write")
            for e in self.history[-3:])

# ── メインループ（v10 全機能統合） ──
def autonomous_loop():
    print(f"\nMonica v10 — Full FEP Hybrid Agent")
    for p in API_PROVIDERS:
        print(f"  API: {p['name']} → {p['model']}")
    print(f"Modes: CHAT / THINK / IDLE")
    print(f"THINK threshold: {THINK_THRESHOLD} | Steering threshold: {STEER_HIGH_PE_THRESHOLD}")
    print(f"World Model: active | Goals: active | FEP History: active\n")

    bb = Blackboard()
    input_queue = queue.Queue()

    think_agent = ThinkAgent(bb, world_model, goal_manager, meta, notes, curiosity)
    action_agent = ActionAgent(bb)

    loaded = load_state(bb.conv, bb.beliefs, steer)
    # Phase 9.1: 前回セッションのコンテキストを復元
    load_previous_session_context()
    if loaded:
        think_count = loaded.get("think_count", 0)
        think_prompt_idx = loaded.get("think_prompt_idx", 0)
    else:
        think_count = 0
        think_prompt_idx = 0

    def read_input():
        while bb.running:
            try:
                line = sys.stdin.readline()
                if line:
                    input_queue.put(line.strip())
            except:
                break
    threading.Thread(target=read_input, daemon=True).start()

    memory_summaries = []
    last_observations = None

    sensing_agent = SensingAgent(bb)
    sensing_agent.start()

    mode = "IDLE"
    last_activity = time.time()
    idle_cycles = 0
    think_cooldown = 0
    idle_lock = 0
    consecutive_api_failures = 0

    while bb.running:
        beliefs, conv, internal_log, recent_outputs, mode = bb.read()
        sm = beliefs.self_model
        user_active = not input_queue.empty()
        recent_div = 1.0
        if len(recent_outputs) >= 3:
            recent_div = diversity_score(recent_outputs[-5:])

        intended_mode = decide_mode(beliefs, recent_div, user_active, idle_cycles)
        if intended_mode == "THINK":
            if think_cooldown > 0 or idle_lock > 0:
                intended_mode = "IDLE"

        if intended_mode != mode:
            if intended_mode == "CHAT" or (intended_mode == "THINK" and mode == "IDLE"):
                beliefs.reset_drift()
            print(f"  [mode] {mode} → {intended_mode}  (self={sm:.3f} idle={idle_cycles} cd={think_cooldown} lock={idle_lock})")
            mode = intended_mode
            bb.set_mode(mode)
            idle_cycles = 0

        # ─── CHAT ───
        if mode == "CHAT":
            if user_active:
                idle_cycles = 0
                u = input_queue.get_nowait()
                if u.lower() in ("exit", "quit", "終了"):
                    bb.running = False
                    break
                if u == "/s":
                    print(f"  {json.dumps(beliefs.state())}")
                    if steer.vector is not None:
                        print(f"  steer: ready (high={len(steer.high_buffer)} low={len(steer.low_buffer)})")
                    print(f"  adapt: restoring={DRIFT_RESTORING_COEFF:.4f} noise_std={DRIFT_NOISE_STD:.4f}")
                    g = goal_manager.active_goal()
                    if g:
                        print(f"  goal: #{g.id} [{g.status}] {g.description[:60]}")
                    pe_s, kl_s, vfe_s = fep_history.trend()
                    if pe_s != 0:
                        print(f"  fep trend: pe={pe_s:+.3f} kl={kl_s:+.3f} vfe={vfe_s:+.3f}")
                    continue

                inner_ctx = notes.context()
                profile_ctx = profile.context()
                framed_input = f"User said: {u}\n\nReply naturally in their language."
                if profile_ctx:
                    framed_input = profile_ctx + "\n" + framed_input
                if inner_ctx:
                    framed_input = inner_ctx + "\n" + framed_input
                conv.append({"role": "user", "content": framed_input})
                profile.update_from_message(u)

                full_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                beliefs = sense(full_text, beliefs)

                t0 = time.time()
                resp = api_chat(conv, beliefs=beliefs, max_tokens=200)
                source = "api"
                n_tok = 0

                if resp is None or not resp:
                    consecutive_api_failures += 1
                    print(f"  [api fail x{consecutive_api_failures}] falling back to local generation...")
                    local_resp, token_log, beliefs = generate_chat_local(
                        conv, beliefs, max_new=FALLBACK_MAX_NEW, temp=0.8)
                    resp = local_resp
                    source = "local_fallback"
                    if token_log:
                        n_tok = len(token_log)
                        avg_pe = np.mean([t["pe_mid"] for t in token_log])
                        print(f"  [local] {n_tok}tok pe={avg_pe:.3f}")
                else:
                    consecutive_api_failures = 0

                dt = time.time() - t0

                if resp is None:
                    resp = "…"
                conv.append({"role": "assistant", "content": resp})
                recent_outputs.append(resp)
                bb.write_conv(conv[-1])
                bb.write_output(resp)
                bb.set_beliefs(beliefs)
                print(f"  [{source}] {dt:.1f}s self={beliefs.self_model:.3f}")
                print(f"  {resp[:300]}")
                log_state(beliefs, "chat", u, resp, n_tok, source=source)
                if len(conv) > CONVERSATION_HISTORY_LIMIT:
                    excess = conv[:len(conv) - CONVERSATION_HISTORY_LIMIT]
                    if len(excess) > 1:
                        summary_text = "\n".join(
                            f"{m['role']}: {m['content'][:100]}" for m in excess)
                        for sp in API_PROVIDERS:
                            try:
                                summary_payload = {
                                    "model": sp["model"],
                                    "messages": [{"role": "system",
                                        "content": "Summarize this conversation in one short sentence in the same language:"},
                                        {"role": "user", "content": summary_text[:1000]}],
                                    "max_tokens": 80, "temperature": 0.3,
                                }
                                sr = requests.post(sp["url"], json=summary_payload,
                                    headers=sp["headers"], timeout=10)
                                if sr.status_code == 200:
                                    sj = sr.json()
                                    summary = sj["choices"][0]["message"]["content"].strip()
                                    memory_summaries.append(summary)
                                    append_summary({"ts": datetime.now().strftime("%H:%M:%S"),
                                        "mode": "memory", "summary": summary})
                                    print(f"  [memory] summarized {len(excess)} messages")
                                    break
                            except Exception:
                                continue
                    conv = conv[-CONVERSATION_HISTORY_LIMIT:]
                last_activity = time.time()
            else:
                time.sleep(0.1)
                idle_cycles += 1

        # ─── THINK（ThinkAgent + ActionAgent + WorldModel + FEP） ───
        elif mode == "THINK":
            think_count += 1
            if think_count > MAX_CONSECUTIVE_THINK:
                print(f"  [think] max consecutive ({MAX_CONSECUTIVE_THINK}) reached, forcing IDLE lock")
                idle_lock = IDLE_LOCK_CYCLES
                mode = "IDLE"
                think_cooldown = 0
                continue

            think_cooldown = THINK_COOLDOWN_CYCLES
            think_msgs = list(conv)
            obs = None

            # ThinkAgent: プロンプト構築 + API生成（前回の観測を連鎖）
            # Phase 7: ツール未使用連続→自動SEARCH注入
            search_hint = think_agent.maybe_inject_search(think_count)
            if search_hint:
                think_msgs.append({"role": "user", "content": search_hint})
            thought, think_prompt_idx, dt, source = think_agent.think(
                think_msgs, think_count, think_prompt_idx, observations=last_observations)

            if thought and len(thought) > 5:
                recent_outputs.append(thought)
                internal_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")
                bb.write_output(thought)
                bb.write_log(internal_log[-1])

                # KL divergence（v14）- SensingAgentが2秒間隔で計算済み
                kl_mid = getattr(sense, '_last_kl', None) or 0.0

                vfe, acc, comp = compute_vfe(
                    beliefs.mid.running_avg, kl_mid, kl_mid * 0.5, beliefs.self_model)
                fep_history.add(beliefs.mid.running_avg, kl_mid, vfe)

                beliefs.drift_update(beliefs.self_model)
                bb.set_beliefs(beliefs)

                print(f"  [think] #{think_count} {dt:.1f}s self={beliefs.self_model:.3f} kl={kl_mid:.3f} vfe={vfe:.3f}")
                print(f"  {thought[:200]}")
                adapt_fep.update(beliefs.self_model)

                log_state(beliefs, "think", "", thought, 0, source=source)
                append_summary({"ts": datetime.now().strftime("%H:%M:%S"),
                    "mode": "think", "resp": thought[:80], "self": beliefs.self_model})
                notes.add(thought)
                curiosity.extract_topics(thought)
                curiosity.add_interest(thought[:80])

                # World Model（v10 Phase 2）
                world_model.add_thought(thought)

                # Phase 3: 成果物抽出（コードブロック→自動WRITE）
                artifact_extractor.extract(thought)

                # ActionAgent: ツール実行 + 観測（v12+v13）+ 連鎖（Phase 1）
                obs = action_agent.execute_and_observe(thought, think_msgs)
                # Phase 7: ツール使用状況を記録
                think_agent.record_tool_use(obs is not None and len(obs) > 0)

                # 目標連動: ツール未使用かつ目標あり→自動SEARCH実行
                if not obs or len(obs) == 0:
                    search_query = auto_search_from_goal(goal_manager, obs, thought)
                    if search_query:
                        print(f"  [auto-search] from goal: {search_query[:60]}")
                        fake_thought = f"[SEARCH: {search_query}]"
                        search_obs = action_agent.execute_and_observe(fake_thought, think_msgs)
                        if search_obs:
                            obs = search_obs
                            ft = tokenizer.apply_chat_template(think_msgs, tokenize=False,
                                                               add_generation_prompt=True)
                            beliefs = sense(ft, beliefs)

                last_observations = obs
                if obs:
                    ft = tokenizer.apply_chat_template(think_msgs, tokenize=False,
                                                       add_generation_prompt=True)
                    beliefs = sense(ft, beliefs)
                last_activity = time.time()
            else:
                print(f"  [think] (empty) — fallback to local")
                t0 = time.time()
                local_thought, token_log, beliefs = generate_chat_local(
                    think_msgs, beliefs, max_new=THINK_TOKENS, temp=0.9)
                dt = time.time() - t0
                if local_thought and len(local_thought) > 5:
                    internal_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {local_thought}")
                    bb.write_log(internal_log[-1])
                    notes.add(local_thought)
                    curiosity.extract_topics(local_thought)
                    curiosity.add_interest(local_thought[:80])
                    world_model.add_thought(local_thought)
                    log_state(beliefs, "think", "", local_thought,
                              len(token_log) if token_log else 0, source="local_fallback")

            # MetaAgent: 焦点再生成（v10 Phase 1）
            if meta.should_refocus(think_count):
                meta.generate_focus(beliefs, curiosity, recent_outputs, think_count)

            # GoalManager: 進捗評価（v11）
            goal_manager.evaluate_progress(beliefs, recent_outputs)
            goal_manager.retire_stale_goals(think_count)  # Phase 6: タイムアウトリタイア

            # ツール連鎖: 結果があったら次のTHINKを即実行
            chain_next = (obs is not None and len(obs) > 0
                          and think_agent.chain_count < think_agent.max_chain)

            beliefs.self_model = float(np.clip(
                beliefs.self_model + THINK_EXIT_BOOST, SELF_MODEL_MIN, SELF_MODEL_MAX))
            bb.set_beliefs(beliefs)

            if chain_next:
                print(f"  [chain] tool results → next THINK ({think_agent.chain_count}/{think_agent.max_chain})")
                mode = "THINK"
            else:
                mode = "IDLE"
                last_observations = None
            bb.set_mode(mode)
            idle_cycles = 0
            torch.cuda.empty_cache()

        # ─── IDLE ───
        elif mode == "IDLE":
            if user_active:
                continue
            for _ in range(IDLE_DRIFT_STEPS):
                beliefs.drift_update(beliefs.mid.running_avg,
                                     noise_mult=IDLE_DRIFT_NOISE_MULT)
            if think_cooldown > 0:
                think_cooldown -= 1
            if idle_lock > 0:
                idle_lock -= 1
                if idle_lock == 0:
                    think_count = 0
            idle_cycles += 1
            time.sleep(0.5)

        if (time.time() - last_activity > IDLE_TO_THINK_TIMEOUT
            and mode == "IDLE" and think_cooldown == 0 and idle_lock == 0):
            print(f"  [idle→think] {IDLE_TO_THINK_TIMEOUT}s inactivity")
            mode = "THINK"

    beliefs, conv, internal_log, recent_outputs, mode = bb.read()
    # Phase 9.1: セッションサマリーを保存
    mode_count = {"CHAT": sum(1 for m in conv if m.get("role") == "user")}
    save_session_summary(beliefs, think_count, mode_count)
    save_state(conv, beliefs, steer, think_count, think_prompt_idx, internal_log)
    print("\nShutting down.")
    hooks.remove()

if __name__ == "__main__":
    try:
        autonomous_loop()
    finally:
        hooks.remove()
