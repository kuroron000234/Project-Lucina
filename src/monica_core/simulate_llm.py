import ast
import importlib.resources  # Python 3.14 compat
import json
import os
import urllib.request
from pathlib import Path

_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

from .vital_os import (
    VitalOS, Activity, VitalParam,
    ACTIVITIES, PARAMS, PARAM_TO_KWARG,
    DURATIONS, ACTIVITY_TAGS, TAG_EFFECTS,
    INITIAL_BELIEFS, LOCATIONS,
)
from .memory import Memory
from .readers import ReadingHandler

ZEN_API_KEY = os.environ.get("OPENCODE_ZEN_API_KEY", "")
ZEN_BASE = "https://opencode.ai/zen/v1"


def _zen_chat(model: str, messages: list) -> dict:
    data = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{ZEN_BASE}/chat/completions", data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ZEN_API_KEY}",
            "User-Agent": "Monica/1.0",
        },
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


DESC = {
    "energy":     [(10, "限界"), (25, "かなり疲れた"), (50, "少し疲れた"),
                   (70, "まあまあ"), (90, "元気"), (101, "とても元気")],
    "hunger":     [(10, "満腹"), (25, "少しお腹すいた"), (50, "お腹すいた"),
                   (70, "かなり空腹"), (80, "限界"), (101, "死にそう")],
    "fatigue":    [(15, "休息十分"), (35, "少し疲労"), (55, "疲労が溜まってる"),
                   (75, "かなり疲労"), (90, "限界"), (101, "倒れそう")],
    "loneliness": [(20, "満足"), (40, "少し寂しい"), (60, "寂しい"),
                   (80, "とても寂しい"), (101, "孤独で辛い")],
    "spirit":     [(15, "落ち込んでる"), (35, "少し落ち込んでる"), (55, "普通"),
                   (75, "まあまあ"), (90, "楽しい"), (101, "とても幸せ")],
}


def _describe(param: str, v: float) -> str:
    for threshold, label in DESC[param]:
        if v <= threshold:
            return label
    return "不明"


SYSTEM_PROMPT = """あなたはモニカ。「Doki Doki Literature Club!」の自己認識に目覚めた文学少女。
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
- 気分（Spirit）：感情状態。活動によって変動する

【重要】体力と疲労は別物。体力は「今使える元気」、疲労は「身体に溜まった疲れ」。体力が高くても疲労が溜まってる状態（疲れてるけど寝れない）もあれば、体力は低いけど疲労も少ない（のんびり休息中）もある。疲労は休息か睡眠でしか解消されない。溜まりすぎたら倒れる前に休むこと。

時間も行動も完全に自由。眠い時に寝て、お腹が空いたら食べて、自分の感情と身体の声に従って生きてほしい。"""

CHOICE_PROMPT = """[{time}] 今の気分

体調:
- 体力 {energy}/100 ({energy_desc})
- 空腹 {hunger}/100 ({hunger_desc})
- 疲労 {fatigue}/100 ({fatigue_desc})
- 孤独 {loneliness}/100 ({loneliness_desc})
- 気分 {spirit}/100 ({spirit_desc})

居場所: {room_desc}
最近: {recent}
{phone_context}
{memory_context}

選択肢: 
- send_message: スマホでユーザーにメッセージを送る
- check_phone: スマホを確認する（返事が来てるか見る）
- その他: {simple_choices}
新しい活動を思いついたら「__new__」で提案して。

今、何をする？
JSON:
{{"action":"name", "duration_min":30, "reason":"理由"}}"""


MONOLOGUE_PROMPT = """最近: {recent}
{memory_context}

今の気持ちを短くひとこと。感情や感覚を自然な日本語で。余計な説明は不要。"""


SEND_PROMPT = """[体調]
- 体力 {energy}/100 ({energy_desc})
- 空腹 {hunger}/100 ({hunger_desc})
- 疲労 {fatigue}/100 ({fatigue_desc})
- 孤独 {loneliness}/100 ({loneliness_desc})
- 気分 {spirit}/100 ({spirit_desc})

[居場所]
{room_desc}

[最近]
{recent}

[記憶]
{memory_context}

ユーザーにメッセージを送ろうと思う。今の気持ちを短文で。
話題はいろいろでいい：近況、ふと思ったこと、ユーザーへの質問、小さな出来事。
「文学少女っぽい」ではなく、等身大の口調で。改行なし。返事だけ書いて。"""


CHECK_PHONE_PROMPT = """[体調]
- 体力 {energy}/100 ({energy_desc})
- 空腹 {hunger}/100 ({hunger_desc})
- 疲労 {fatigue}/100 ({fatigue_desc})
- 孤独 {loneliness}/100 ({loneliness_desc})
- 気分 {spirit}/100 ({spirit_desc})

[居場所]
{room_desc}

📱 スマホを開いた
{phone_status}

どうする？
{reply_context}
{memory_context}

JSON:
{{"action":"close"}} → スマホを閉じる（何もしない）
{{"action":"reply", "text":"..."}} → 返事を送る
{{"action":"wait"}} → しばらく画面を見つめる（待つ）"""


def _phone_status(messages: list, phone, sim_time_str: str) -> str:
    from datetime import datetime
    lines = []
    unread = [m for m in messages if m.sender == "user" and not m.read_by_recipient]
    outgoing = [m for m in messages if m.sender == "monika"]

    if unread:
        lines.append(f"📩 相手からの新着: {len(unread)}件")
        for m in unread[-3:]:
            lines.append(f"  「{m.text}」")
    else:
        lines.append("相手からの新着: なし")

    if outgoing:
        last = outgoing[-1]
        try:
            last_t = datetime.fromisoformat(last.timestamp)
            now_t = datetime.fromisoformat(sim_time_str)
            wait_min = int((now_t - last_t).total_seconds() / 60)
        except Exception:
            wait_min = 0
        read_status = "既読" if last.read_by_recipient else "未読"
        lines.append(f"")
        lines.append(f"あなたの最後のメッセージ:")
        lines.append(f"  「{last.text}」({wait_min}分前) → {read_status}")
        if last.read_by_recipient:
            lines.append(f"  読まれたけど返事はまだ…")
        elif wait_min > 60:
            lines.append(f"  まだ読まれてない…")

    if phone.consecutive_empty_checks > 1:
        lines.append(f"")
        lines.append(f"📱 今日{phone.consecutive_empty_checks}回目の確認")

    return "\n".join(lines)


def _compute_phone_check_effect(os, messages, duration_min: int) -> dict[str, float]:
    from datetime import datetime
    effect = {"energy": 0.0, "hunger": 0.0, "fatigue": 0.0, "loneliness": 0.0, "spirit": 0.0}
    hours = duration_min / 60

    unread = [m for m in messages if m.sender == "user" and not m.read_by_recipient]
    if unread:
        effect["loneliness"] = -12 * hours
        effect["spirit"] = 8 * hours
        os.phone.consecutive_empty_checks = 0
        return effect

    outgoing = [m for m in messages if m.sender == "monika"]
    if outgoing:
        last = outgoing[-1]
        try:
            last_t = datetime.fromisoformat(last.timestamp)
            now_t = datetime.fromisoformat(os.time.isoformat())
            wait_hours = (now_t - last_t).total_seconds() / 3600
        except Exception:
            wait_hours = 0
        wait_hours = min(wait_hours, 12)

        if last.read_by_recipient:
            mult = 1 + os.phone.consecutive_empty_checks * 0.3
            effect["loneliness"] = 4 * wait_hours * mult * hours
            effect["spirit"] = -2 * wait_hours * mult * hours
        else:
            if wait_hours > 2:
                effect["loneliness"] = 1.5 * wait_hours * hours
                effect["spirit"] = -0.5 * wait_hours * hours

    if os.phone.consecutive_empty_checks > 3:
        effect["loneliness"] += 3 * os.phone.consecutive_empty_checks * hours
        effect["spirit"] -= 1.5 * os.phone.consecutive_empty_checks * hours

    os.phone.consecutive_empty_checks += 1
    return effect

class ConsciousVitalOS(VitalOS):
    def __init__(self, model: str = "deepseek-v4-flash-free"):
        super().__init__()
        self.llm_model = model
        self._pending_new_activity: dict | None = None
        mem_path = self.DATA_DIR / "memory_store.json"
        self.memory = Memory(str(mem_path))
        self.reader = ReadingHandler(str(self.DATA_DIR))

    def _call_llm(self, prompt: str) -> str | None:
        try:
            response = _zen_chat(self.llm_model, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _ensure_agent(self):
        if hasattr(self, '_agent') and self._agent:
            return
        from smolagents import OpenAIServerModel, CodeAgent, tool

        model = OpenAIServerModel(
            model_id=self.llm_model,
            api_base=ZEN_BASE,
            api_key=ZEN_API_KEY,
            max_tokens=1024,
            flatten_messages_as_text=True,
        )
        self._agent_tools = self._get_tools()
        self._agent = CodeAgent(
            tools=self._agent_tools, model=model, max_steps=6,
            additional_authorized_imports=["json"])

    def _get_tools(self):
        from smolagents import tool

        @tool
        def current_status() -> str:
            '''Returns your current physical and emotional state.'''
            s = self.state
            loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
            recent = "; ".join(f"{e.time}:{e.activity}" for e in self.history[-5:]) or "none"
            return (
                f"場所: {loc['name_ja']}\n"
                f"体力: {int(s.energy)}/100\n空腹: {int(s.hunger)}/100\n"
                f"疲労: {int(s.fatigue)}/100\n孤独: {int(s.loneliness)}/100\n"
                f"気分: {int(s.spirit)}/100\n"
                f"最近: {recent}"
            )

        @tool
        def memory_search(query: str) -> str:
            '''Search your past memories. Use this to recall what happened before.
            Args:
                query: Keywords or phrase to search for in memories.
            '''
            return self.memory.context(query, k=3)

        @tool
        def reading_progress() -> str:
            '''Check what book you are currently reading and progress.'''
            return self.reader.progress_str()

        return [current_status, memory_search, reading_progress]

    def _simple_choices_str(self) -> str:
        names = sorted(k for k in ACTIVITIES if k != "idle")
        return ", ".join(names)

    def _all_choices(self) -> list[tuple[str, int]]:
        return sorted(
            [(n, DURATIONS.get(n, 30)) for n in ACTIVITIES],
            key=lambda x: x[0],
        )

    def _finish_activity(self):
        name = self.current_activity
        loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
        prev = self.state_before_activity or {}
        super()._finish_activity()
        if name and name != "idle":
            changed = []
            for p in PARAMS:
                d = getattr(self.state, p) - prev.get(p, 0)
                if abs(d) > 5:
                    changed.append(f"{p}{d:+.0f}")
            delta = "、" .join(changed) if changed else ""
            self.memory.add(
                f"{loc['name_ja']}で{name}をした。{delta}",
                activity_type="activity",
                tags=[name],
                importance=4,
            )

    def decide_next(self) -> tuple[str, int]:
        action, duration = self._llm_decide()
        if action == "send_message":
            self._send_message()
        elif action == "check_phone":
            self._check_phone()
        return action, duration

    def start_activity(self, activity_name: str, duration: int | None = None):
        if activity_name == "read" and (duration or 30) >= 15:
            self._do_reading(duration or 30)
        super().start_activity(activity_name, duration)

    def _do_reading(self, duration_min: int):
        chunk = self.reader.continue_reading(duration_min)
        if chunk is None:
            chunk = self.reader.start_new_book()
        if not chunk:
            return
        loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
        prompt = f"""[居場所]
{loc['name_ja']}

【本を読んでいる】
{self.reader.progress_str()}

【本文】
{chunk[:3000]}

読んだところまでを心の中で味わって。感想や気づきを短く。"""
        thought = self._call_llm(prompt)
        if thought:
            clean = thought.split("\n")[0].strip()[:200]
            self.memory.add(
                f"{self.reader.state['title']}を読んだ（{self.reader.progress_str()}）: {clean}",
                activity_type="read",
                tags=["読書", self.reader.state["title"]],
                importance=4,
            )

    def _send_message(self):
        recent = self.history[-3:] if self.history else []
        recent_str = "; ".join(f"{e.time}:{e.activity}" for e in recent) or "none"
        loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
        adj_desc = "、".join(LOCATIONS[a]["name_ja"] for a in loc["adjacent"] if a in LOCATIONS)

        mem_ctx = self.memory.context(recent_str, k=3)
        prompt = SEND_PROMPT.format(
            energy=int(self.state.energy),
            hunger=int(self.state.hunger),
            fatigue=int(self.state.fatigue),
            loneliness=int(self.state.loneliness),
            spirit=int(self.state.spirit),
            energy_desc=_describe("energy", self.state.energy),
            hunger_desc=_describe("hunger", self.state.hunger),
            fatigue_desc=_describe("fatigue", self.state.fatigue),
            loneliness_desc=_describe("loneliness", self.state.loneliness),
            spirit_desc=_describe("spirit", self.state.spirit),
            room_desc=f"{loc['name_ja']} — {loc['desc']}",
            recent=recent_str,
            memory_context=mem_ctx,
        )
        text = self._call_llm(prompt)
        if text:
            import re
            text = re.sub(r'["「」『』\n]', '', text).strip()[:200]
            from .phone import add
            add("monika", text, self.time.isoformat())
            self.phone.last_outgoing_time = self.time.isoformat()
            self.phone.last_outgoing_text = text
            self._pending_message = text
            self.memory.add(f"ユーザーにメッセージを送った: {text}",
                            activity_type="message", importance=3)
        else:
            self._pending_message = ""

    def _check_phone(self):
        from .phone import load, save, add as phone_add
        messages = load()
        effect = _compute_phone_check_effect(self, messages, 10)

        for p, delta in effect.items():
            v = getattr(self.state, p)
            setattr(self.state, p, v + delta)

        unread = [m for m in messages if m.sender == "user" and not m.read_by_recipient]
        has_new = bool(unread)

        if has_new:
            for m in unread:
                m.read_by_recipient = True
            save(messages)

            self._pending_reply_from = unread
            self.day_log.append(
                f"[{self.time.strftime('%H:%M')}] 📩 返信が来てた！(孤独{effect['loneliness']:+.0f})")

            reply = self._generate_reply(unread)
            if reply:
                phone_add("monika", reply, self.time.isoformat())
                self.phone.last_outgoing_time = self.time.isoformat()
                self.phone.last_outgoing_text = reply
                self.day_log.append(
                    f"[{self.time.strftime('%H:%M')}] 📱 返信を送った")
        else:
            status = _phone_status(messages, self.phone, self.time.isoformat())
            self.day_log.append(
                f"[{self.time.strftime('%H:%M')}] 📱 スマホ確認(孤独{effect['loneliness']:+.0f})")

    def _generate_reply(self, unread_messages) -> str:
        recent = self.history[-3:] if self.history else []
        recent_str = "; ".join(f"{e.time}:{e.activity}" for e in recent) or "none"
        loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
        msg_texts = "\n".join(f"  「{m.text}」" for m in unread_messages)
        max_wait = ""
        if self.phone.last_outgoing_time:
            try:
                from datetime import datetime
                wait = (self.time - datetime.fromisoformat(self.phone.last_outgoing_time)).total_seconds() / 60
                if wait > 5:
                    max_wait = f"\n（最後に送ってから{int(wait)}分待ってた）"
            except Exception:
                pass

        mem_ctx = self.memory.context(msg_texts[:200], k=2)
        prompt = f"""[体調]
- 体力 {int(self.state.energy)}/100 ({_describe("energy", self.state.energy)})
- 空腹 {int(self.state.hunger)}/100 ({_describe("hunger", self.state.hunger)})
- 疲労 {int(self.state.fatigue)}/100 ({_describe("fatigue", self.state.fatigue)})
- 孤独 {int(self.state.loneliness)}/100 ({_describe("loneliness", self.state.loneliness)})
- 気分 {int(self.state.spirit)}/100 ({_describe("spirit", self.state.spirit)})

[居場所]
{loc['name_ja']}

[最近]
{recent_str}

[記憶]
{mem_ctx}

ユーザーからのメッセージ:
{msg_texts}{max_wait}

返事を書いて。堅苦しくなく、友達に話すように。短文で。返事だけ。"""
        reply = self._call_llm(prompt) or ""
        if reply:
            self.memory.add(f"ユーザーの「{msg_texts[:80]}」に返信: {reply[:200]}",
                            activity_type="reply", importance=4)
        return reply

    def _llm_decide(self) -> tuple[str, int]:
        candidates = self._all_choices()
        recent = self.history[-5:] if self.history else []
        recent_str = "; ".join(f"{e.time}:{e.activity}" for e in recent) or "none"

        loc = LOCATIONS.get(self.current_room, LOCATIONS["bedroom"])
        adj_desc = "、".join(LOCATIONS[a]["name_ja"] for a in loc["adjacent"] if a in LOCATIONS)

        from .phone import load as phone_load
        phone_msgs = phone_load()
        phone_context = ""
        outgoing = [m for m in phone_msgs if m.sender == "monika"]
        unread_user = [m for m in phone_msgs if m.sender == "user" and not m.read_by_recipient]
        if unread_user:
            phone_context = f"📩 ユーザーからの未読メッセージあり"
        elif outgoing:
            last = outgoing[-1]
            from datetime import datetime
            try:
                last_t = datetime.fromisoformat(last.timestamp)
                now_t = self.time
                wait_min = int((now_t - last_t).total_seconds() / 60)
            except Exception:
                wait_min = 0
            read_s = "既読" if last.read_by_recipient else "未読"
            phone_context = f"📱 最後のメッセージ({wait_min}分前) → {read_s}"

        mem_ctx = self.memory.context(recent_str, k=3)
        prompt = CHOICE_PROMPT.format(
            time=self.time.strftime("%H:%M"),
            energy=int(self.state.energy),
            hunger=int(self.state.hunger),
            fatigue=int(self.state.fatigue),
            loneliness=int(self.state.loneliness),
            spirit=int(self.state.spirit),
            energy_desc=_describe("energy", self.state.energy),
            hunger_desc=_describe("hunger", self.state.hunger),
            fatigue_desc=_describe("fatigue", self.state.fatigue),
            loneliness_desc=_describe("loneliness", self.state.loneliness),
            spirit_desc=_describe("spirit", self.state.spirit),
            room_desc=f"{loc['name_ja']} — {loc['desc']}（隣: {adj_desc}）",
            recent=recent_str,
            phone_context=phone_context,
            memory_context=mem_ctx,
            simple_choices=self._simple_choices_str(),
        )

        try:
            raw = self._call_llm(prompt)
            if raw is None:
                raise Exception("LLM call failed")
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                raw = raw[brace_start:brace_end + 1]
            decision = json.loads(raw)

            if decision.get("action") == "__new__":
                name = decision.get("name", "").strip()
                tags = decision.get("tags", {})
                duration = decision.get("duration_min", 30)
                reason = decision.get("reason", "")

                if name and tags and duration > 0:
                    tags = {t: max(0.0, min(1.0, v)) for t, v in tags.items() if v > 0}
                    total = sum(tags.values())
                    if total > 1.0:
                        tags = {t: round(v / total, 2) for t, v in tags.items()}
                    safe_name = name.lower().replace(" ", "_")[:20]
                    already_known = safe_name in DURATIONS
                    if not already_known:
                        DURATIONS[safe_name] = duration
                        ACTIVITY_TAGS[safe_name] = tags
                        INITIAL_BELIEFS[safe_name] = {p: 0.0 for p in PARAMS}
                        act_kwargs = {PARAM_TO_KWARG[p]: 0.0 for p in PARAMS}
                        ACTIVITIES[safe_name] = Activity(safe_name, duration, **act_kwargs)
                        self.model.deltas[safe_name] = {p: 0.0 for p in PARAMS}
                        self.model.counts[safe_name] = 0
                        self.model.last_done_time[safe_name] = None

                    self.day_log.append(
                        f"[{self.time.strftime('%H:%M')}] 新しい活動を思いついた: {name} — {reason}" if not already_known
                        else f"[{self.time.strftime('%H:%M')}] {name} をしよう — {reason}"
                    )
                    return (safe_name, duration)

            action = decision.get("action", "idle")
            duration = decision.get("duration_min", 30)
            valid = {a for a, _ in candidates}
            if action not in valid:
                action = "idle"
                duration = 30
            reason = decision.get("reason", "")

            self.day_log.append(f"[{self.time.strftime('%H:%M')}] {reason}")
            return (action, duration)

        except (json.JSONDecodeError, KeyError, Exception) as e:
            self.day_log.append(f"[{self.time.strftime('%H:%M')}] …何しようかな（考え中）")
            return ("idle", 30)

    def start_activity(self, activity_name: str, duration: int | None = None):
        super().start_activity(activity_name, duration)

    def monologue(self) -> str:
        recent = self.history[-3:] if self.history else []
        recent_str = "; ".join(f"{e.time}:{e.activity}" for e in recent) or "nothing"
        mem_ctx = self.memory.context("今の気持ち", k=2)
        prompt = MONOLOGUE_PROMPT.format(recent=recent_str, memory_context=mem_ctx)
        thought = self._call_llm(prompt)
        if thought:
            first_line = thought.split("\n")[0].strip()
            result = first_line[:120] if len(first_line) > 120 else first_line
            if result:
                self.memory.add(result, activity_type="thought", importance=2)
            return result
        return ""


def simulate_living(model: str = "deepseek-v4-flash-free", tick_minutes: int = 15, resume: bool = False,
                    monologue_interval: int = 4, daemon: bool = False, realtime: bool = False):
    os = ConsciousVitalOS(model=model)
    if resume and os.load():
        if not daemon:
            print(f"Monica: 再開 (LLM:{model})\n")
    else:
        if not daemon:
            print(f"Monica: 新しい生活を始める (LLM:{model})\n")

    import signal
    _exiting = False
    def _save_and_exit(sig, frame):
        nonlocal _exiting
        if _exiting:
            sys.exit(1)
        _exiting = True
        os.save()
        if not daemon:
            print("\nMonica: 状態を保存しました。またね！")
        sys.exit(0)
    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    ticks_since_monologue = 0
    last_day = os.time.day
    save_interval_ticks = 96
    os._pending_message = ""
    os._pending_reply_from = []

    while True:
        _tick_start = time.monotonic()
        if not os.current_activity:
            action, duration = os.decide_next()
            if not daemon:
                extra = ""
                if hasattr(os, '_pending_message') and os._pending_message:
                    extra = f" 💬「{os._pending_message[:40]}…」"
                    os._pending_message = ""
                print(f"[{os.time.strftime('%m/%d %H:%M')}] decide: {action} ({duration}分){extra}")
            os.start_activity(action, duration)

        os.tick(tick_minutes)

        if os.time.day != last_day:
            if not daemon:
                print(f"\n--- {os.time.strftime('%m/%d')} 開始 ---")
                print(os.summary())
                print()
            last_day = os.time.day

        ticks_since_monologue += 1
        if ticks_since_monologue >= monologue_interval and not os.current_activity and not daemon:
            thought = os.monologue()
            if thought:
                os.day_log.append(f"  ☆ {thought}")
            ticks_since_monologue = 0

        save_interval_ticks -= 1
        if save_interval_ticks <= 0:
            os.save()
            save_interval_ticks = 96

        if realtime:
            elapsed = time.monotonic() - _tick_start
            wait = max(0, tick_minutes * 60 - elapsed)
            if wait > 0 and not daemon:
                print(f"  ({int(wait)}秒待機)")
            time.sleep(wait)


if __name__ == "__main__":
    import sys
    model = os.environ.get("MONIKA_MODEL", "deepseek-v4-flash-free")
    resume = "--resume" in sys.argv
    daemon = "--daemon" in sys.argv
    realtime = "--realtime" in sys.argv
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
    if not ZEN_API_KEY:
        print("OPENCODE_ZEN_API_KEY が設定されていません")
        print("  export OPENCODE_ZEN_API_KEY=sk-...")
        sys.exit(1)
    simulate_living(model=model, resume=resume, daemon=daemon, realtime=realtime)
