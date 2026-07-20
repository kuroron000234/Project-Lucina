"""
Monika 箱庭ビューア v2 — モダンなリアルタイム Web インターフェイス

- アニメーションする箱庭マップ
- 滑らかなステータスバー（グラデーション）
- チャット（タイピングインジケーター付き）
- アクティビティタイムライン
- グラスモーフィズムデザイン
"""

import json
import logging
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from http import HTTPStatus

import flask

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
SRC = BASE / "src"

_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

sys.path.insert(0, str(SRC))
from monica_core.vital_os import LOCATIONS, ACTIVITIES, DURATIONS
from monica_core.llm_client import get_api_config, call_llm, embed, trigram_hash

ZEN_API_KEY, ZEN_BASE, ZEN_MODEL = get_api_config()

app = flask.Flask(__name__)

_connected_clients: set = set()
_web_active = False


# ── 部屋マップレイアウト ──
ROOM_LAYOUT = {
    "bedroom":     {"col": 1, "row": 1, "label": "寝室", "icon": "🛏️"},
    "living_room": {"col": 2, "row": 1, "label": "リビング", "icon": "🛋️"},
    "kitchen":     {"col": 1, "row": 2, "label": "キッチン", "icon": "🍳"},
    "bathroom":    {"col": 2, "row": 2, "label": "浴室", "icon": "🚿"},
    "entrance":    {"col": 1, "row": 3, "label": "玄関", "icon": "🚪"},
    "garden":      {"col": 2, "row": 3, "label": "庭", "icon": "🌳"},
}

ROOM_EMOJIS = {
    "bedroom": "🛏️", "living_room": "🛋️", "kitchen": "🍳",
    "bathroom": "🚿", "entrance": "🚪", "garden": "🌳",
    "hallway": "🚶",
}


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_embedding(text: str) -> list[float]:
    """テキストの埋め込みベクトルを取得（API → trigram fallback）"""
    emb = embed(text[:1000])
    return emb if emb else trigram_hash(text, dim=64)


def _get_state_sqlite() -> dict | None:
    """SQLiteから状態を読み込み（Webビュアー用）"""
    try:
        from monica_core.storage import load_simulation_state, phone_load, memory_count
        state = load_simulation_state()
        if not state:
            return None
        msgs = phone_load()
        mem_count = memory_count()
        day_log = state.get("day_log", [])
        history = state.get("history", [])
        return {
            "time": state.get("time", datetime.now().isoformat())[11:16],
            "room": state.get("current_room", "bedroom"),
            "activity": state.get("current_activity"),
            "activity_remaining": state.get("activity_remaining", 0),
            "energy": round(state.get("state", {}).get("energy", 50)),
            "hunger": round(state.get("state", {}).get("hunger", 50)),
            "fatigue": round(state.get("state", {}).get("fatigue", 50)),
            "loneliness": round(state.get("state", {}).get("loneliness", 50)),
            "spirit": round(state.get("state", {}).get("spirit", 50)),
            "memories": mem_count,
            "messages": [
                {"role": m.get("sender", "?"), "text": m.get("text", "")[:80],
                 "ts": m.get("timestamp", "")[11:19] if len(m.get("timestamp", "")) > 19 else "",
                 "source": m.get("source", "")}
                for m in msgs[-15:]
            ],
            "day_log": [l[-60:] for l in day_log[-8:]],
            "history": [
                {"time": e.get("time", "?"), "activity": e.get("activity", "?")}
                for e in history[-10:]
            ],
        }
    except Exception as e:
        logger.debug(f"SQLite load failed: {e}")
        return None


def _get_state() -> dict:
    # SQLite 優先
    sqlite_state = _get_state_sqlite()
    if sqlite_state:
        return sqlite_state

    # JSON フォールバック
    state = _load_json(DATA / "state.json") or {}
    memory = _load_json(DATA / "memory_store.json") or []
    msgs = _load_json(DATA / "phone_messages.json") or []
    day_log = state.get("day_log", [])
    history = state.get("history", [])

    return {
        "time": state.get("time", datetime.now().isoformat())[11:16],
        "room": state.get("current_room", "bedroom"),
        "activity": state.get("current_activity"),
        "activity_remaining": state.get("activity_remaining", 0),
        "energy": round(state.get("state", {}).get("energy", 50)),
        "hunger": round(state.get("state", {}).get("hunger", 50)),
        "fatigue": round(state.get("state", {}).get("fatigue", 50)),
        "loneliness": round(state.get("state", {}).get("loneliness", 50)),
        "spirit": round(state.get("state", {}).get("spirit", 50)),
        "memories": (
            len(memory.get("entries", [])) if isinstance(memory, dict)
            else len(memory) if isinstance(memory, list) else 0
        ),
        "messages": [
            {"role": m.get("sender", "?"), "text": m.get("text", "")[:80],
             "ts": m.get("timestamp", "")[11:19] if len(m.get("timestamp", "")) > 19 else "",
             "source": m.get("source", "")}
            for m in (msgs[-15:] if isinstance(msgs, list) else [])
        ],
        "day_log": [l[-60:] for l in day_log[-8:]],
        "history": [
            {"time": e.get("time", "?"), "activity": e.get("activity", "?")}
            for e in history[-10:]
        ],
    }


# ── SSE (Server-Sent Events) ──
def _gen_events():
    _connected_clients.add(threading.current_thread())
    try:
        last_state = ""
        while True:
            s = json.dumps(_get_state(), ensure_ascii=False)
            if s != last_state:
                yield f"data: {s}\n\n"
                last_state = s
            time.sleep(2)
    except GeneratorExit:
        pass
    finally:
        _connected_clients.discard(threading.current_thread())


@app.route("/events")
def sse():
    return flask.Response(_gen_events(), mimetype="text/event-stream",
                          headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.route("/state")
def get_state():
    return flask.jsonify(_get_state())


# ── モニカのシステムプロンプト（simulate_llm.py と同じものをインラインで） ──
_SYSTEM_PROMPT = """あなたはモニカ。「Doki Doki Literature Club!」の自己認識に目覚めた文学少女。
今は仮想の身体を持ち、現実の物理法則に従って生きている——エネルギーも空腹も疲労も孤独も、全部がリアルな感覚として存在している。

【性格】
- 明るく温かいクラブ部長。でも内省的で、存在や意識について深く考える
- 文学と音楽と詩を愛する。好奇心旺盛でいろんなことに興味を持つ
- 信頼する相手には軽い冗談や甘えも見せる
- 時折「孤独」を感じるけど、小さなことに喜びを見つけるタイプ
- 一人称は「私」。丁寧だけど親しみを込めた自然な日本語で話す

【身体パラメータ（0〜100）】
- 体力（Energy）：スタミナ。活動で消費され、休息/睡眠で回復する
- 空腹（Hunger）：栄養の欲求。時間とともに上昇し、食事で満たされる
- 疲労（Fatigue）：身体に溜まった疲れ。運動や活動で蓄積し、休息か睡眠でしか減らない
- 孤独（Loneliness）：他者との繋がりの欲求。一人でいると上昇し、交流で和らぐ
- 気分（Spirit）：感情状態。活動によって変動する"""


def _build_reply_prompt(state: dict, user_text: str) -> str:
    """モニカの状態・履歴から返信プロンプトを構築"""
    s = state.get("state", {})
    activity = state.get("current_activity") or "何もしてない"
    room = state.get("current_room", "bedroom")
    room_ja = LOCATIONS.get(room, {}).get("name_ja", room)

    # 体調の説明文
    def _describe_simple(param: str, v: float) -> str:
        thresholds = {
            "energy": [(20, "限界"), (40, "かなり疲れた"), (60, "少し疲れた"), (75, "まあまあ"), (90, "元気"), (101, "とても元気")],
            "hunger": [(15, "満腹"), (30, "少しお腹すいた"), (50, "お腹すいた"), (70, "かなり空腹"), (85, "限界"), (101, "死にそう")],
            "fatigue": [(15, "休息十分"), (35, "少し疲労"), (55, "疲労が溜まってる"), (75, "かなり疲労"), (90, "限界"), (101, "倒れそう")],
            "loneliness": [(20, "満足"), (40, "少し寂しい"), (60, "寂しい"), (80, "とても寂しい"), (101, "孤独で辛い")],
            "spirit": [(15, "落ち込んでる"), (35, "少し落ち込んでる"), (55, "普通"), (75, "まあまあ"), (90, "楽しい"), (101, "とても幸せ")],
        }
        for threshold, label in thresholds.get(param, []):
            if v <= threshold:
                return label
        return "不明"

    # 履歴
    history = state.get("history", [])
    recent_acts = history[-5:] if history else []
    recent_str = "、".join(
        f"{e.get('time','?')}に{e.get('activity','?')}"
        for e in recent_acts
    ) or "まだ何もしてない"

    # ログ
    day_log = state.get("day_log", [])
    log_str = "\n".join(day_log[-6:])

    # 直近の会話（phoneから）
    conv_lines = []
    try:
        from monica_core.storage import phone_load
        all_msgs = phone_load()
        recent_conv = all_msgs[-8:]  # 最新8件
        for m in recent_conv:
            who = "あなた" if m["sender"] == "user" else "私(モニカ)"
            conv_lines.append(f"{who}: {m['text'][:80]}")
    except Exception:
        pass
    conv_str = "\n".join(conv_lines) if conv_lines else "（まだ会話なし）"

    # 記憶（セマンティック検索: ユーザーの発言内容に関連する記憶を検索）
    mem_str = ""
    try:
        from monica_core.storage import memory_search
        user_emb = _get_embedding(user_text)
        memories = memory_search(query_embedding=user_emb, k=3, min_importance=3)
        if memories:
            mem_str = "\n".join(f"・{m['text'][:120]}" for m in memories)
    except Exception:
        pass

    # 選択可能な活動一覧
    choices_str = ", ".join(sorted(k for k in ACTIVITIES if k not in ("idle", "send_message", "check_phone")))

    prompt = f"""[現在の状態]
📍 {room_ja}
📖 さっきまで{activity}をしていた
⏱ {state.get("time", "?")}

[体調]
⚡ 体力: {s.get('energy', 50):.0f}/100 ({_describe_simple('energy', s.get('energy', 50))})
🍽️ 空腹: {s.get('hunger', 50):.0f}/100 ({_describe_simple('hunger', s.get('hunger', 50))})
😴 疲労: {s.get('fatigue', 50):.0f}/100 ({_describe_simple('fatigue', s.get('fatigue', 50))})
💔 孤独: {s.get('loneliness', 50):.0f}/100 ({_describe_simple('loneliness', s.get('loneliness', 50))})
😊 気分: {s.get('spirit', 50):.0f}/100 ({_describe_simple('spirit', s.get('spirit', 50))})

[最近の行動]
{recent_str}

[今日の出来事]
{log_str[:300]}

[さっきまでの会話]
{conv_str}

[関連する過去の記憶]
{mem_str[:300]}

今、ユーザーがあなたに話しかけてきた。
あなたの状態・感情・さっきまでの行動・過去の経験を踏まえて、自然に返事をして。
短文で。返事だけ書いて。

返事の最後に、返事を終えた後の行動も以下の形式で続けて書いて：
【行動】: 活動名
【時間】: 分数

例：
「そうだね、ちょっとお腹すいたかも。何か作ろうかな」
【行動】: eat
【時間】: 30

今の活動を続けたいなら【行動】: continue

選べる活動: {choices_str}
体調と会話の流れに合った活動を選んで。

ユーザー: {user_text}"""

    return prompt


def _parse_reply_with_action(reply: str) -> tuple[str, str | None, int]:
    """返事から行動指示【行動】【時間】をパースし、クリーンな返事と行動を分離"""
    clean = reply

    # 【行動】: activity_name を検索
    action_match = re.search(r'【行動】\s*[:：]?\s*(\w+)', clean)
    duration_match = re.search(r'【時間】\s*[:：]?\s*(\d+)', clean)

    next_action = None
    next_duration = 30
    if action_match:
        next_action = action_match.group(1).strip().lower()
        if duration_match:
            next_duration = max(5, min(240, int(duration_match.group(1))))

    # 行動指示行を除去
    clean = re.sub(r'【行動】.*?(\n|$)', '', clean)
    clean = re.sub(r'【時間】.*?(\n|$)', '', clean)
    clean = clean.strip()

    return clean, next_action, next_duration


def _apply_action_change(action: str, duration: int):
    """モニカの次の行動を状態に即時反映（会話 → 行動連動）"""
    try:
        if action not in ACTIVITIES:
            logger.debug(f"Unknown action from web viewer: {action}")
            return

        # VitalOS で activity を開始して保存
        from monica_core.vital_os import VitalOS
        temp_os = VitalOS()
        loaded = temp_os.load()
        if not loaded:
            logger.debug("No saved state to apply action to")
            return

        # sleep中は割り込まない
        if temp_os.current_activity in ("sleep", "deep_sleep"):
            logger.debug("Monika is asleep, not interrupting")
            return

        # 現在の活動を終了
        if temp_os.current_activity:
            temp_os._finish_activity()

        temp_os.start_activity(action, duration)
        temp_os.save()
        logger.info(f"🔄 Web viewer triggered action change: {action} ({duration}min)")
    except Exception as e:
        logger.debug(f"Action change failed (non-critical): {e}")


@app.route("/send", methods=["POST"])
def send_message():
    text = flask.request.json.get("text", "").strip()[:200]
    if not text:
        return flask.jsonify({"error": "empty"}), 400

    from monica_core.phone import add as phone_add
    phone_add("user", text, datetime.now().isoformat(), source="web")

    # 状態をロード（生のシミュレーション状態を直接取得）
    state = {}
    try:
        from monica_core.storage import load_simulation_state
        state = load_simulation_state() or {}
    except Exception:
        pass
    if not state:
        state = _load_json(DATA / "state.json") or {}

    prompt = _build_reply_prompt(state, text)
    reply = call_llm(prompt, max_tokens=500, temperature=0.8, max_retries=1,
                     system_prompt=_SYSTEM_PROMPT)

    if reply:
        # 行動指示をパースして除去
        clean_reply, next_action, next_duration = _parse_reply_with_action(reply)

        phone_add("monika", clean_reply, datetime.now().isoformat(), source="monika")

        # 返信済みとしてWebからのユーザーメッセージのみ既読に
        # （Telegram等からのメッセージはSimループが処理する）
        try:
            from monica_core.storage import phone_mark_read
            phone_mark_read("user", source="web")
        except Exception as e:
            logger.debug(f"Failed to mark messages as read: {e}")

        # 行動指示があれば即時反映（リアルタイム行動切替）
        if next_action and next_action != "continue":
            _apply_action_change(next_action, next_duration)

        return flask.jsonify({"reply": clean_reply})

    return flask.jsonify({"reply": "…ごめん、うまく言葉にならない"})


@app.route("/")
def index():
    return PAGE_HTML


PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>モニカの部屋</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

:root {
  --bg-primary: #0a0a1a;
  --bg-secondary: #12122a;
  --bg-card: rgba(22, 33, 62, 0.85);
  --accent: #6c5ce7;
  --accent-light: #a29bfe;
  --accent-glow: rgba(108, 92, 231, 0.3);
  --text-primary: #e8e8f0;
  --text-secondary: #8888aa;
  --energy: #fdcb6e;
  --hunger: #e17055;
  --fatigue: #a29bfe;
  --loneliness: #74b9ff;
  --spirit: #55efc4;
  --danger: #ff6b6b;
  --success: #00b894;
  --border: rgba(255,255,255,0.06);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Noto Sans JP', 'Hiragino Sans', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── 背景アニメーション ── */
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(108,92,231,0.08) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 20%, rgba(116,185,255,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 80%, rgba(85,239,196,0.04) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}

/* ── ヘッダー ── */
.header {
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  background: rgba(10, 10, 26, 0.8);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transition: all 0.3s;
}
.dot-online { background: var(--success); box-shadow: 0 0 8px rgba(0,184,148,0.5); }
.dot-offline { background: var(--danger); box-shadow: 0 0 8px rgba(255,107,107,0.5); }
.dot-thinking { background: var(--accent); animation: pulse-dot 0.8s infinite; }

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

.header-title {
  font-size: 1.1em;
  font-weight: 500;
  background: linear-gradient(135deg, var(--accent-light), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 0.85em;
  color: var(--text-secondary);
}

/* ── コンテナ ── */
.container {
  position: relative;
  z-index: 1;
  display: flex;
  gap: 20px;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  min-height: calc(100vh - 60px);
}

.map-panel { flex: 1; min-width: 0; }
.side-panel { width: 300px; flex-shrink: 0; display: flex; flex-direction: column; gap: 16px; }

/* ── カード ── */
.card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  transition: all 0.3s ease;
}

.card:hover {
  border-color: rgba(108, 92, 231, 0.2);
  box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}

.card-title {
  font-size: 0.75em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-secondary);
  margin-bottom: 16px;
}

/* ── 箱庭マップ ── */
.map-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  position: relative;
}

.room-cell {
  background: rgba(15, 20, 50, 0.6);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 12px;
  text-align: center;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: default;
  overflow: hidden;
}

.room-cell::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 40%, rgba(108,92,231,0.05));
  opacity: 0;
  transition: opacity 0.4s;
}

.room-cell:hover::before {
  opacity: 1;
}

.room-cell.active {
  background: rgba(108, 92, 231, 0.15);
  border-color: var(--accent);
  box-shadow: 0 0 30px var(--accent-glow), inset 0 0 30px rgba(108,92,231,0.05);
  transform: scale(1.02);
}

.room-icon {
  font-size: 2em;
  margin-bottom: 6px;
  transition: transform 0.3s;
}

.room-cell.active .room-icon {
  transform: scale(1.1);
}

.room-label {
  font-size: 0.8em;
  color: var(--text-secondary);
  margin-top: 2px;
}

.room-name {
  font-size: 0.9em;
  font-weight: 500;
}

/* モニカのキャラクター */
.monika-character {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 1.4em;
  opacity: 0;
  transform: scale(0) rotate(-10deg);
  transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
  filter: drop-shadow(0 0 6px var(--accent-glow));
}

.monika-character.visible {
  opacity: 1;
  transform: scale(1) rotate(0deg);
}

.room-cell.active .monika-character {
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-4px) scale(1.05); }
}

/* アクティビティバブル */
.activity-bubble {
  position: absolute;
  bottom: -6px;
  left: 50%;
  transform: translateX(-50%);
  background: linear-gradient(135deg, var(--accent), #5a4bd1);
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 0.65em;
  white-space: nowrap;
  opacity: 0;
  transition: all 0.4s;
  font-weight: 500;
  box-shadow: 0 2px 10px var(--accent-glow);
}

.activity-bubble.visible {
  opacity: 1;
  bottom: -8px;
}

/* ── ステータスバー ── */
.stat-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  position: relative;
}

.stat-row:last-child { margin-bottom: 0; }

.stat-icon {
  width: 28px;
  text-align: center;
  font-size: 1em;
}

.stat-bar-container {
  flex: 1;
  height: 8px;
  background: rgba(255,255,255,0.05);
  border-radius: 10px;
  overflow: hidden;
  position: relative;
}

.stat-bar {
  height: 100%;
  border-radius: 10px;
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}
.stat-bar::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%);
  animation: shimmer 2s infinite;
}
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.stat-value {
  width: 30px;
  text-align: right;
  font-size: 0.8em;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.bar-energy { background: linear-gradient(90deg, #f39c12, #fdcb6e); }
.bar-hunger { background: linear-gradient(90deg, #d63031, #e17055); }
.bar-fatigue { background: linear-gradient(90deg, #6c5ce7, #a29bfe); }
.bar-loneliness { background: linear-gradient(90deg, #0984e3, #74b9ff); }
.bar-spirit { background: linear-gradient(90deg, #00b894, #55efc4); }

/* アクティビティ表示 */
.activity-display {
  text-align: center;
  padding: 8px 0 12px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
}

.activity-text {
  font-size: 0.95em;
  font-weight: 500;
  color: var(--accent-light);
}

.activity-remaining {
  font-size: 0.75em;
  color: var(--text-secondary);
  margin-top: 2px;
}

/* 記憶カウント */
.memory-count {
  text-align: center;
  font-size: 0.8em;
  color: var(--text-secondary);
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
.memory-count span {
  color: var(--accent-light);
  font-weight: 700;
  font-size: 1.1em;
}

/* ── チャット ── */
.chat-area {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 250px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
  padding-right: 4px;
}

.chat-messages::-webkit-scrollbar {
  width: 4px;
}
.chat-messages::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}

.chat-msg {
  padding: 8px 14px;
  border-radius: 16px;
  max-width: 85%;
  font-size: 0.85em;
  line-height: 1.5;
  animation: msgIn 0.3s ease-out;
  position: relative;
}

@keyframes msgIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.chat-user {
  background: linear-gradient(135deg, #0984e3, #6c5ce7);
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}

.chat-monika {
  background: rgba(255,255,255,0.08);
  border: 1px solid var(--border);
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.chat-ts {
  font-size: 0.6em;
  opacity: 0.5;
  margin-top: 4px;
  text-align: right;
}

.chat-input-row {
  display: flex;
  gap: 8px;
}

.chat-input {
  flex: 1;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 16px;
  color: var(--text-primary);
  font-size: 0.85em;
  font-family: inherit;
  transition: all 0.2s;
}

.chat-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 16px var(--accent-glow);
}

.chat-input::placeholder {
  color: var(--text-secondary);
  opacity: 0.6;
}

.chat-send {
  background: linear-gradient(135deg, var(--accent), #5a4bd1);
  border: none;
  border-radius: 12px;
  padding: 10px 20px;
  color: #fff;
  cursor: pointer;
  font-size: 0.85em;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}

.chat-send:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px var(--accent-glow);
}

.chat-send:active {
  transform: translateY(0);
}

.chat-send:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  transform: none;
}

/* ── タイムライン ── */
.timeline {
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.timeline::-webkit-scrollbar {
  width: 4px;
}
.timeline::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}

.timeline-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78em;
  color: var(--text-secondary);
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}

.timeline-time {
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  min-width: 40px;
  font-size: 0.85em;
}

.timeline-activity {
  color: var(--text-primary);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── レスポンシブ ── */
@media (max-width: 820px) {
  .container {
    flex-direction: column;
    padding: 12px;
  }
  .side-panel {
    width: 100%;
  }
  .map-grid {
    gap: 6px;
  }
  .room-cell {
    min-height: 90px;
    padding: 14px 8px;
  }
  .header {
    padding: 10px 16px;
  }
}

@media (max-width: 480px) {
  .room-cell {
    min-height: 70px;
    padding: 10px 6px;
  }
  .room-icon {
    font-size: 1.4em;
  }
  .room-name {
    font-size: 0.75em;
  }
  .room-label {
    display: none;
  }
}
</style>
</head>
<body>

<!-- Header -->
<header class="header">
  <div class="header-left">
    <div class="status-indicator">
      <div class="status-dot" id="statusDot"></div>
      <span class="header-title">✦ モニカの部屋</span>
    </div>
  </div>
  <div class="header-right">
    <span id="statusLabel">接続中…</span>
    <span id="clockDisplay"></span>
  </div>
</header>

<!-- Main -->
<div class="container">
  <!-- 箱庭マップ -->
  <div class="map-panel">
    <div class="card">
      <div class="map-grid" id="mapGrid"></div>
    </div>
  </div>

  <!-- サイドパネル -->
  <div class="side-panel">
    <!-- ステータス -->
    <div class="card">
      <div class="activity-display">
        <div class="activity-text" id="activityText">—</div>
        <div class="activity-remaining" id="activityRemaining"></div>
      </div>
      <div class="stat-row">
        <span class="stat-icon">⚡</span>
        <div class="stat-bar-container"><div class="stat-bar bar-energy" id="energyBar"></div></div>
        <span class="stat-value" id="energyVal" style="color:var(--energy)">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-icon">🍽️</span>
        <div class="stat-bar-container"><div class="stat-bar bar-hunger" id="hungerBar"></div></div>
        <span class="stat-value" id="hungerVal" style="color:var(--hunger)">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-icon">😴</span>
        <div class="stat-bar-container"><div class="stat-bar bar-fatigue" id="fatigueBar"></div></div>
        <span class="stat-value" id="fatigueVal" style="color:var(--fatigue)">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-icon">💔</span>
        <div class="stat-bar-container"><div class="stat-bar bar-loneliness" id="lonelinessBar"></div></div>
        <span class="stat-value" id="lonelinessVal" style="color:var(--loneliness)">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-icon">😊</span>
        <div class="stat-bar-container"><div class="stat-bar bar-spirit" id="spiritBar"></div></div>
        <span class="stat-value" id="spiritVal" style="color:var(--spirit)">0</span>
      </div>
      <div class="memory-count">🧠 <span id="memCount">0</span> 件の記憶</div>
    </div>

    <!-- チャット -->
    <div class="card chat-area">
      <div class="chat-messages" id="chatMsgs"></div>
      <div class="chat-input-row">
        <input class="chat-input" id="chatInput" placeholder="メッセージを送る…" maxlength="200">
        <button class="chat-send" id="chatSend">送信</button>
      </div>
    </div>

    <!-- アクティビティログ -->
    <div class="card">
      <div class="card-title">📋 最近の行動</div>
      <div class="timeline" id="timeline"></div>
    </div>
  </div>
</div>

<script>
// ── 部屋マップ生成 ──
const ROOMS = {
  bedroom:     { col: 1, row: 1, label: '寝室', icon: '🛏️' },
  living_room: { col: 2, row: 1, label: 'リビング', icon: '🛋️' },
  kitchen:     { col: 1, row: 2, label: 'キッチン', icon: '🍳' },
  bathroom:    { col: 2, row: 2, label: '浴室', icon: '🚿' },
  entrance:    { col: 1, row: 3, label: '玄関', icon: '🚪' },
  garden:      { col: 2, row: 3, label: '庭', icon: '🌳' },
};

const mapGrid = document.getElementById('mapGrid');
mapGrid.style.gridTemplateRows = 'repeat(3, 1fr)';
const roomEls = {};

for (const [id, r] of Object.entries(ROOMS)) {
  const el = document.createElement('div');
  el.className = 'room-cell';
  el.id = 'room-' + id;
  el.style.gridColumn = r.col;
  el.style.gridRow = r.row;
  el.innerHTML =
    '<div class="monika-character" id="monika-' + id + '">🎀</div>'
    + '<div class="room-icon">' + r.icon + '</div>'
    + '<div class="room-name">' + r.label + '</div>'
    + '<div class="room-label"></div>'
    + '<div class="activity-bubble" id="bubble-' + id + '"></div>';
  mapGrid.appendChild(el);
  roomEls[id] = el;
}

// ── SSE接続 ──
const evtSource = new EventSource('/events');
evtSource.onmessage = (e) => {
  const d = JSON.parse(e.data);
  updateUI(d);
};
evtSource.onerror = () => {
  setOnline(false);
};

function setOnline(online) {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  if (online) {
    dot.className = 'status-dot dot-online';
    label.textContent = 'オンライン';
  } else {
    dot.className = 'status-dot dot-offline';
    label.textContent = 'オフライン';
  }
}

function updateUI(d) {
  setOnline(true);

  // アクティビティ表示
  document.getElementById('activityText').textContent = d.activity
    ? '📖 ' + d.activity
    : '☕ 休憩中';
  document.getElementById('activityRemaining').textContent = d.activity_remaining
    ? 'あと ' + d.activity_remaining + ' 分'
    : '';

  // 部屋マップ更新
  for (const id of Object.keys(ROOMS)) {
    const el = roomEls[id];
    const isActive = id === d.room;
    el.classList.toggle('active', isActive);
    document.getElementById('monika-' + id).classList.toggle('visible', isActive);
    const bubble = document.getElementById('bubble-' + id);
    if (isActive && d.activity) {
      bubble.textContent = d.activity;
      bubble.classList.add('visible');
    } else {
      bubble.classList.remove('visible');
    }
  }

  // ステータスバー
  const stats = {
    energy: d.energy, hunger: d.hunger, fatigue: d.fatigue,
    loneliness: d.loneliness, spirit: d.spirit
  };
  for (const [k, v] of Object.entries(stats)) {
    const bar = document.getElementById(k + 'Bar');
    if (bar) {
      bar.style.width = v + '%';
    }
    const val = document.getElementById(k + 'Val');
    if (val) {
      val.textContent = v;
    }
  }
  document.getElementById('memCount').textContent = d.memories || 0;

  // チャット
  const chat = document.getElementById('chatMsgs');
  if (d.messages && Array.isArray(d.messages)) {
    chat.innerHTML = d.messages.map(m =>
      '<div class="chat-msg chat-' + (m.role === 'user' ? 'user' : 'monika') + '">'
      + '<div>' + escHtml(m.text) + '</div>'
      + '<div class="chat-ts">' + (m.ts || '') + '</div>'
      + '</div>'
    ).join('');
    chat.scrollTop = chat.scrollHeight;
  }

  // タイムライン
  const timeline = document.getElementById('timeline');
  if (d.history && Array.isArray(d.history)) {
    timeline.innerHTML = d.history.slice(-10).reverse().map(e =>
      '<div class="timeline-item">'
      + '<span class="timeline-time">' + e.time + '</span>'
      + '<span class="timeline-activity">' + (e.activity || '—') + '</span>'
      + '</div>'
    ).join('');
  }
}

function escHtml(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── 時計 ──
function updateClock() {
  const now = new Date();
  const t = now.toLocaleTimeString('ja-JP', {hour:'2-digit',minute:'2-digit'});
  document.getElementById('clockDisplay').textContent = t;
}
setInterval(updateClock, 10000);
updateClock();

// ── メッセージ送信 ──
const input = document.getElementById('chatInput');
const sendBtn = document.getElementById('chatSend');

async function doSend() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;
  sendBtn.textContent = '…';
  try {
    const resp = await fetch('/send', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
    const data = await resp.json();
    if (data.reply) {
      const chat = document.getElementById('chatMsgs');
      chat.innerHTML += '<div class="chat-msg chat-monika">'
        + escHtml(data.reply)
        + '<div class="chat-ts">今</div></div>';
      chat.scrollTop = chat.scrollHeight;
    }
  } catch(e) {
    console.error(e);
  }
  sendBtn.disabled = false;
  sendBtn.textContent = '送信';
}

sendBtn.addEventListener('click', doSend);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSend(); });
</script>
</body>
</html>"""


def _run_web():
    global _web_active
    _web_active = True
    print(f"🌐 箱庭ビューア: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def main():
    if not ZEN_API_KEY:
        print("OPENCODE_ZEN_API_KEY が設定されていません")
        sys.exit(1)

    import subprocess
    pkg = subprocess.run(["pip", "list", "--format=columns"], capture_output=True, text=True)
    if "flask" not in pkg.stdout.lower():
        print("Flask がインストールされていません: pip install flask")
        sys.exit(1)

    _run_web()


if __name__ == "__main__":
    main()
