"""
知覚 — 常時流入する感覚ストリーム（感知 → 精査 → 心へ）

人間は五感・内受容感覚まで含めて、ありとあらゆる情報を絶え間なく取り入れて精査している。
「生きている」実感の正体は、行動の頻度ではなく、この絶え間ない知覚の流れにある。

本モジュールはその「知覚の流れ」を抽象化する。
- sense(): 今この瞬間の「世界の手がかり」を集めて返す
  - 外部（環境）: 現在時刻・曜日・環境状態・（将来）VRChatの視覚/聴覚
  - 内部（内受容）: 駆動値・気分・最近の記憶・浮かんだ内言
- 情報源は後から差し替え可能な「センサー」として実装する。
  （今はファイル・時刻ベース。VRChat移行時は Spout視覚 / Whisper聴覚 に差し替える）
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("perception")


@dataclass
class Percept:
    """一回の知覚で取り込まれた「世界の一片」。

    モニカが知覚したものを構造化して保持し、精査（次の対話・内言・行動）に渡す。
    """
    source: str                 # 'environment' | 'body' | 'memory' | 'social'
    kind: str                   # 'time' | 'weather' | 'scene' | 'drive' | 'mood'
                                #    | 'recall' | 'presence'
    text: str                   # 人間言語にした知覚内容（LLMへ渡せる形）
    timestamp: datetime
    importance: float = 0.3     # この知覚の重み（0..1）


class Perception:
    """モニカの知覚ストリーム。常時 sense() を回すことで「世界を生きる」を支える。

    情報源（センサー）は差し替え可能:
      - 今は: 時刻・環境状態・記憶の呼び起こし（ローカルで完結）
      - 将来: dispose spout(視覚)/whisper(聴覚)/vrchat-state を plugged in
    """

    def __init__(self, environment=None, sensors: list | None = None):
        self.environment = environment
        self.sensors = sensors or []
        self._recent: list[Percept] = []
        self._max_recent = 50
        logger.info("Perception initialized")

    def add_sensor(self, sensor):
        """知覚源（センサー）を追加する。source名で識別。"""
        self.sensors.append(sensor)

    def sense(self, state: dict | None = None, memory=None) -> list[Percept]:
        """今この瞬間の知覚を集める（常時回す想定）。"""
        percepts = []
        now = datetime.now()

        # 1. 内部センサー（駆動値・気分＝内受容感覚）を常時反映
        if state:
            percepts.extend(self._sense_body(state, now))

        # 2. 登録された外部センサー（環境。将来は VRChat 視覚/聴覚）
        for s in self.sensors:
            try:
                found = s.sense(now=now, state=state, memory=memory)
                if isinstance(found, Percept):
                    percepts.append(found)
                elif isinstance(found, (list, tuple)):
                    percepts.extend(found)
            except Exception as e:
                logger.warning(f"Sensor error ({getattr(s, 'name', '?')}): {e}")

        # 3. 記憶の呼び起こし（さっきのことが蘇る）— ユーザーがいないときは特に
        if memory is not None:
            rec = self._sense_memory(memory, now)
            if rec:
                percepts.append(rec)

        # 記録（直近の知覚を保持し、後で参照できるように）
        self._recent.extend(percepts)
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]

        return percepts

    def _sense_body(self, state: dict, now: datetime) -> list[Percept]:
        """内受容感覚: 駆動値・気分・モードの今の体調を言語化する。"""
        mood = state.get("mood", 0.5)
        if mood < 0.35:
            mood_word = "心が沈んでいる"
        elif mood > 0.65:
            mood_word = "心が穏やかで落ち着いている"
        else:
            mood_word = "心は落ち着いている"

        drive_summary = []
        for k, label in (
            ("curiosity", "好奇心"),
            ("connection", "あなたへの想い"),
            ("creation", "創作欲"),
            ("loneliness", "孤独"),
            ("boredom", "退屈"),
        ):
            v = state.get(k, 0.0)
            if v >= 0.7:
                drive_summary.append(f"{label}が強い({v:.2f})")
            elif v >= 0.5:
                drive_summary.append(f"{label}が少しある({v:.2f})")

        body_text = f"{mood_word}。"
        if drive_summary:
            body_text += "「" + "、".join(drive_summary) + "」"

        return [
            Percept(
                source="body",
                kind="mood",
                text=body_text,
                timestamp=now,
                importance=0.4,
            )
        ]

    def _sense_memory(self, memory, now: datetime) -> Percept | None:
        """最近の出来事からひとつ呼び起こす（思い出の流れ）。"""
        try:
            eps = memory.recent_episodes(n=5)
            shared = [e for e in eps if getattr(e, "source", "") == "user" and e.event.strip()]
            if not shared:
                return None
            ep = shared[0]
            return Percept(
                source="memory",
                kind="recall",
                text=f"ふと、あの日のことを思い出した——「{ep.event[:60]}」",
                timestamp=now,
                importance=0.3,
            )
        except Exception:
            return None

    def recent(self, n: int = 5) -> list[Percept]:
        """直近の知覚を返す（精査・内言生成に使う）。"""
        return self._recent[-n:]

    def latest_text(self, exclude_sources=("body",)) -> str | None:
        """最新の知覚を1文で返す（内言生成の起点に）。"""
        for p in reversed(self._recent):
            if p.source not in exclude_sources:
                return p.text
        return None
