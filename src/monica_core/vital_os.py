import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import math
from datetime import datetime, timedelta
from pathlib import Path


class VitalParam(Enum):
    ENERGY = "energy"
    HUNGER = "hunger"
    LONELINESS = "loneliness"
    SPIRIT = "spirit"


PARAMS = [p.name.lower() for p in VitalParam]

PARAM_TO_KWARG = {
    "energy": "energy_delta", "hunger": "hunger_delta",
    "loneliness": "loneliness_delta", "spirit": "spirit_delta",
}

# ── 世界層（True Physics）──
# タグ1単位あたりのE/H/L/Sへの1時間あたり効果
# 設計指針: 8h睡眠後の起床時Eが~70程度、3食で空腹が持続可能な範囲
TAG_EFFECTS = {
    "exertion":  {"energy": -10, "hunger": +2,  "loneliness":  0, "spirit": +3},
    "social":    {"energy":  -2, "hunger":  0,  "loneliness":-18, "spirit": +6},
    "mental":    {"energy":  -5, "hunger":  0,  "loneliness":  0, "spirit": +3},
    "outdoor":   {"energy":   0, "hunger":  0,  "loneliness": -6, "spirit": +4},
    "play":      {"energy":  -3, "hunger":  0,  "loneliness": -2, "spirit":+10},
    "rest":      {"energy":  +9, "hunger": +1,  "loneliness":  0, "spirit": -1},
    "nourish":   {"energy":  +4, "hunger":-40,  "loneliness":  0, "spirit": +2},
    "boredom":   {"energy":  -1, "hunger": +1,  "loneliness": +1, "spirit": -1},
}

ACTIVITY_TAGS = {
    "sleep":        {"rest": 0.9, "boredom": 0.3},
    "deep_sleep":   {"rest": 1.0, "boredom": 0.4},
    "eat":          {"nourish": 1.0, "rest": 0.1},
    "piano":        {"play": 0.8, "mental": 0.3, "exertion": 0.1},
    "piano_long":   {"play": 0.9, "mental": 0.4, "exertion": 0.2},
    "read":         {"mental": 0.7, "play": 0.2, "boredom": 0.3},
    "idle":         {"boredom": 0.8, "rest": 0.1},
    "rest":         {"rest": 0.7, "boredom": 0.3},
    "stretch":      {"exertion": 0.4, "rest": 0.3},
    "write_diary":  {"mental": 0.5, "play": 0.4},
    "walk":         {"exertion": 0.3, "outdoor": 0.8, "play": 0.1},
    "talk":         {"social": 0.9, "mental": 0.2},
    "talk_to_user": {"social": 0.7, "mental": 0.3, "play": 0.1},
}


def compute_true_effect(activity_name: str, duration_minutes: int) -> dict[str, float]:
    """世界層: タグ→パラメータ効果 を計算（活動1回ぶんの総効果）"""
    tags = ACTIVITY_TAGS.get(activity_name, {})
    per_hour = {p: 0.0 for p in PARAMS}
    for tag, intensity in tags.items():
        tag_row = TAG_EFFECTS.get(tag, {p: 0 for p in PARAMS})
        for param in PARAMS:
            per_hour[param] += intensity * tag_row[param]
    hours = duration_minutes / 60
    return {p: round(v * hours, 2) for p, v in per_hour.items()}


# エージェントの事前信念（初期モデル、世界層とはズレている）
INITIAL_BELIEFS = {
    "sleep": {"energy": 30, "hunger": 5, "loneliness": 2, "spirit": -3},
    "deep_sleep": {"energy": 40, "hunger": 5, "loneliness": 1, "spirit": 0},
    "eat": {"energy": 5, "hunger": -40, "loneliness": 0, "spirit": 2},
    "piano": {"energy": -8, "hunger": 2, "loneliness": -2, "spirit": 12},
    "piano_long": {"energy": -15, "hunger": 3, "loneliness": -3, "spirit": 20},
    "read": {"energy": -5, "hunger": 1, "loneliness": -1, "spirit": 8},
    "idle": {"energy": -3, "hunger": 1, "loneliness": 5, "spirit": -3},
    "rest": {"energy": 10, "hunger": 0, "loneliness": 4, "spirit": -2},
    "stretch": {"energy": 5, "hunger": 0, "loneliness": 0, "spirit": 3},
    "write_diary": {"energy": -3, "hunger": 0, "loneliness": -5, "spirit": 5},
    "walk": {"energy": -5, "hunger": 3, "loneliness": -5, "spirit": 10},
    "talk": {"energy": -2, "hunger": 0, "loneliness": -15, "spirit": 5},
    "talk_to_user": {"energy": -3, "hunger": 0, "loneliness": -14, "spirit": 6},
}

DURATIONS = {"sleep": 60, "deep_sleep": 60, "eat": 30, "rest": 30,
             "stretch": 15, "talk": 15, "walk": 30, "piano_long": 60,
             "piano": 30, "read": 60, "idle": 30, "write_diary": 30,
             "talk_to_user": 10}


@dataclass
class Activity:
    name: str
    duration_minutes: int
    energy_delta: float = 0
    hunger_delta: float = 0
    loneliness_delta: float = 0
    spirit_delta: float = 0
    is_rest: bool = False


ACTIVITIES = {}
for name, dur in DURATIONS.items():
    kwargs = {PARAM_TO_KWARG[p]: INITIAL_BELIEFS[name][p] for p in PARAMS}
    ACTIVITIES[name] = Activity(name, dur, **kwargs)
ACTIVITIES["rest"].is_rest = True

SETPOINTS = {
    VitalParam.ENERGY: 65,
    VitalParam.HUNGER: 25,
    VitalParam.LONELINESS: 25,
    VitalParam.SPIRIT: 65,
}

DRIFT_RATES = {
    VitalParam.ENERGY: -4,
    VitalParam.HUNGER: 2,
    VitalParam.LONELINESS: 0.5,
    VitalParam.SPIRIT: -1.0,
}


@dataclass
class VitalState:
    energy: float = 55
    hunger: float = 45
    loneliness: float = 40
    spirit: float = 50

    def as_dict(self):
        return {p.name.lower(): round(getattr(self, p.name.lower()), 1) for p in VitalParam}

    def as_float_dict(self):
        return {p.name.lower(): getattr(self, p.name.lower()) for p in VitalParam}

    def clone(self):
        return VitalState(**self.as_float_dict())

    def clamp(self):
        for p in VitalParam:
            v = getattr(self, p.name.lower())
            setattr(self, p.name.lower(), max(0.0, min(100.0, v)))


@dataclass
class Event:
    time: str
    activity: str
    state_before: dict
    state_after: dict


EXPLORATION_EPSILON = 0.15


class WorldModel:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.deltas: dict[str, dict[str, float]] = {
            name: dict(beliefs) for name, beliefs in INITIAL_BELIEFS.items()
        }
        self.counts: dict[str, int] = {name: 0 for name in INITIAL_BELIEFS}
        self.last_done_time: dict[str, Optional[datetime]] = {name: None for name in INITIAL_BELIEFS}

    def mark_done(self, activity: str, current_time: datetime):
        self.counts[activity] = self.counts.get(activity, 0) + 1
        self.last_done_time[activity] = current_time

    def hours_since_done(self, activity: str, current_time: datetime) -> float:
        last = self.last_done_time.get(activity)
        if last is None:
            return 999.0
        return (current_time - last).total_seconds() / 3600

    def predict_state(self, activity: str, state: VitalState) -> VitalState:
        pred = state.clone()
        effect = compute_true_effect(activity, DURATIONS.get(activity, 30))
        for param in PARAMS:
            v = getattr(pred, param) + effect[param]
            setattr(pred, param, v)
        pred.clamp()
        return pred

    def predict_own_model(self, activity: str, state: VitalState,
                          duration_minutes: int = 30) -> VitalState:
        """エージェント自身の信念に基づく予測（効用計算用、時間単位に正規化）"""
        hours = duration_minutes / 60
        pred = state.clone()
        deltas = self.deltas.get(activity, {p: 0 for p in PARAMS})
        for param in PARAMS:
            v = getattr(pred, param) + deltas[param] * hours
            setattr(pred, param, v)
        pred.clamp()
        return pred

    def update(self, activity: str, total_effect: dict[str, float],
               clamped_params: set[str] = set()):
        old = self.deltas.get(activity, {p: 0 for p in PARAMS})
        for param in PARAMS:
            if param in clamped_params:
                continue
            actual = total_effect.get(param, 0)
            error = abs(actual - old.get(param, 0))
            if error > 0.5:
                learned = self.alpha * actual + (1 - self.alpha) * old.get(param, 0)
            else:
                learned = old.get(param, 0)
            learned = max(-60, min(60, learned))
            self.deltas[activity][param] = round(learned, 2)

    def satiation(self, activity: str, current_time: datetime) -> float:
        h = self.hours_since_done(activity, current_time)
        if h > 24:
            return 0.0
        return 8.0 * math.exp(-h / 3.0)

    def novelty_bonus(self, activity: str, current_time: datetime) -> float:
        h = self.hours_since_done(activity, current_time)
        if self.counts.get(activity, 0) == 0:
            return 12.0
        return 6.0 * math.exp(-h / 2.0)

    def utility(self, activity: str, current_state: VitalState, current_time: datetime,
                duration_minutes: int = 30) -> float:
        pred = self.predict_own_model(activity, current_state, duration_minutes)
        deviation = sum(abs(getattr(pred, p) - SETPOINTS[VitalParam[p.upper()]]) for p in PARAMS)
        return -deviation - self.satiation(activity, current_time) + self.novelty_bonus(activity, current_time)


class VitalOS:
    def __init__(self):
        self.state = VitalState()
        self.model = WorldModel()
        self.time = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
        self.history: list[Event] = []
        self.current_activity: Optional[str] = None
        self.activity_remaining: int = 0
        self.day_log: list[str] = []
        self.state_before_activity: Optional[dict] = None
        self.activity_start_time: Optional[datetime] = None
        self.activity_total_duration: int = 0

    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    
    def _state_path(self) -> Path:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return self.DATA_DIR / "state.json"

    def save(self):
        path = self._state_path()
        data = {
            "state": self.state.as_float_dict(),
            "time": self.time.isoformat(),
            "current_activity": self.current_activity,
            "activity_remaining": self.activity_remaining,
            "activity_total_duration": self.activity_total_duration,
            "history": [
                {"time": e.time, "activity": e.activity,
                 "state_before": e.state_before, "state_after": e.state_after}
                for e in self.history
            ],
            "model_deltas": {name: dict(d) for name, d in self.model.deltas.items()},
            "model_counts": dict(self.model.counts),
            "last_done": {
                name: t.isoformat() if t else None
                for name, t in self.model.last_done_time.items()
            },
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> bool:
        path = self._state_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data["state"].items():
                setattr(self.state, k, v)
            self.time = datetime.fromisoformat(data["time"])
            self.current_activity = data["current_activity"]
            self.activity_remaining = data["activity_remaining"]
            self.activity_total_duration = data["activity_total_duration"]
            self.history = [
                Event(e["time"], e["activity"], e["state_before"], e["state_after"])
                for e in data["history"]
            ]
            self.model.deltas.update(data["model_deltas"])
            self.model.counts.update(
                {k: int(v) for k, v in data["model_counts"].items()}
            )
            for name, t_str in data.get("last_done", {}).items():
                self.model.last_done_time[name] = (
                    datetime.fromisoformat(t_str) if t_str else None
                )
            return True
        except Exception:
            return False

    def tick(self, minutes: int = 5):
        self.time += timedelta(minutes=minutes)
        self._apply_drift(minutes)
        if self.current_activity:
            was_remaining = self.activity_remaining
            self.activity_remaining -= minutes
            effective = minutes if self.activity_remaining >= 0 else was_remaining
            self._apply_activity_effects(self.current_activity, effective)
            if self.activity_remaining <= 0:
                self._finish_activity()
        self.state.clamp()

    def _apply_drift(self, minutes: int):
        hours = minutes / 60
        for p in VitalParam:
            v = getattr(self.state, p.name.lower())
            setattr(self.state, p.name.lower(), v + DRIFT_RATES[p] * hours)

    def _apply_activity_effects(self, activity_name: str, minutes: int):
        if self.activity_total_duration <= 0:
            return
        total_effect = compute_true_effect(activity_name, self.activity_total_duration)
        fraction = minutes / self.activity_total_duration
        for param in PARAMS:
            v = getattr(self.state, param)
            setattr(self.state, param, v + total_effect[param] * fraction)

    def start_activity(self, activity_name: str, duration: Optional[int] = None):
        act = ACTIVITIES.get(activity_name)
        if not act:
            return
        dur = duration or act.duration_minutes
        self.state_before_activity = self.state.as_float_dict()
        self.activity_start_time = self.time
        self.activity_total_duration = dur
        self.activity_remaining = dur
        self.current_activity = activity_name
        self._apply_activity_effects(activity_name, 0)
        self.history.append(Event(
            time=self.time.strftime("%H:%M"),
            activity=activity_name,
            state_before={k: round(v, 1) for k, v in self.state_before_activity.items()},
            state_after={k: round(v, 1) for k, v in self.state.as_float_dict().items()},
        ))
        act_name = act.name
        self.day_log.append(f"[{self.time.strftime('%H:%M')}] {act_name} を始めた")

    def _finish_activity(self):
        act = ACTIVITIES.get(self.current_activity)
        if act:
            self.day_log.append(f"[{self.time.strftime('%H:%M')}] {act.name} を終えた")
            self.model.mark_done(self.current_activity, self.time)
            if self.state_before_activity and self.activity_start_time:
                before = VitalState(**self.state_before_activity)
                elapsed_hours = (self.time - self.activity_start_time).total_seconds() / 3600
                if elapsed_hours > 0:
                    observed = {p: getattr(self.state, p) - getattr(before, p) for p in PARAMS}
                    drift_effect = {p: DRIFT_RATES[VitalParam[p.upper()]] * elapsed_hours for p in PARAMS}
                    total_effect = {p: observed[p] - drift_effect[p] for p in PARAMS}
                    clamped_params = {
                        p for p in PARAMS
                        if getattr(self.state, p) <= 0 or getattr(self.state, p) >= 100
                    }
                    per_hour = {p: total_effect[p] / elapsed_hours for p in PARAMS}
                    self.model.update(self.current_activity, per_hour,
                                      clamped_params=clamped_params)

        self.current_activity = None
        self.activity_remaining = 0
        self.state_before_activity = None
        self.activity_start_time = None

    def _critical_needs(self) -> Optional[tuple[str, int]]:
        s = self.state
        if s.hunger > 80:
            return ("eat", 30)
        if s.energy < 15:
            return ("sleep", 120)
        if s.energy < 25:
            return ("rest", 60)
        if s.loneliness > 80:
            return ("walk", 30)
        if s.spirit < 15:
            return ("piano", 30)
        return None

    def _mandatory_slot(self, hour: int) -> Optional[tuple[str, int]]:
        if hour >= 23 or hour < 6:
            return ("sleep", 480 if hour >= 23 else 360)
        if 6 <= hour < 7:
            return ("stretch", 15)
        if 12 <= hour < 13:
            return ("eat", 30)
        if 13 <= hour < 14:
            return ("rest", 30)
        if 17 <= hour < 18:
            return ("eat", 30)
        if 21 <= hour < 23:
            return ("write_diary", 30)
        return None

    def free_time_choices(self) -> list[tuple[str, int]]:
        s = self.state
        choices = [("read", 60), ("write_diary", 30), ("walk", 30), ("idle", 30)]
        if s.energy > 20:
            choices.append(("stretch", 15))
        if s.energy > 30:
            choices.append(("piano", 30))
        if s.energy > 45:
            choices.append(("piano_long", 60))
        if s.loneliness > 40 or self.model.counts.get("talk", 0) == 0:
            choices.append(("talk", 15))
        if s.energy > 15 and self.model.counts.get("piano", 0) == 0:
            choices.append(("piano", 30))
        choices.append(("talk_to_user", 10))
        return choices

    def decide_next(self) -> tuple[str, int]:
        hour = self.time.hour
        if hour >= 23 or hour < 6:
            return ("sleep", 480 if hour >= 23 else 360)
        mandatory = self._mandatory_slot(hour)
        if mandatory:
            return mandatory
        critical = self._critical_needs()
        if critical:
            return critical
        candidates = self.free_time_choices()
        if random.random() < EXPLORATION_EPSILON:
            pick = random.choice(candidates)
            self.day_log.append(f"   → 冒険: {pick[0]}を試してみよう")
            return pick
        best = max(candidates, key=lambda c: self.model.utility(c[0], self.state, self.time, c[1]))
        return (best[0], best[1])

    def deviation(self) -> dict[VitalParam, float]:
        return {p: abs(getattr(self.state, p.name.lower()) - SETPOINTS[p])
                for p in VitalParam}

    def summary(self) -> str:
        t = self.time.strftime("%H:%M")
        s = self.state.as_dict()
        dev = self.deviation()
        parts = [f"[{t}] E:{s['energy']:.0f} H:{s['hunger']:.0f} L:{s['loneliness']:.0f} S:{s['spirit']:.0f}"]
        if self.current_activity:
            parts.append(f" [{self.current_activity}あと{self.activity_remaining}分]")
        parts.append(f" | 乖離: {max(dev.values()):.0f}")
        return "".join(parts)

    def model_report(self) -> str:
        lines = ["--- エージェントの学習済みモデル (1時間あたりの効果) ---"]
        for name, deltas in sorted(self.model.deltas.items()):
            n = self.model.counts[name]
            dur = DURATIONS.get(name, 30)
            true_eff = compute_true_effect(name, dur)
            true_hour = {p: true_eff[p] / (dur / 60) for p in PARAMS}
            d = " ".join(f"{p}:{deltas[p]:+.1f}" for p in PARAMS)
            t = " ".join(f"T:{true_hour[p]:+.1f}" for p in PARAMS)
            lines.append(f"  {name:15s} x{n:2d}  信念:{d}  真値:{t}")
        return "\n".join(lines)
