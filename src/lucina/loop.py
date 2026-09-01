"""
ループ — 永続存在の基盤
"""

import logging
import random
import time
from datetime import datetime

logger = logging.getLogger("loop")

# 駆動値キー（character.py の DRIVE_CONFIG と対応）
DRIVE_KEYS = ("curiosity", "connection", "creation", "loneliness", "boredom")


class Loop:
    """Autonomous loop for persistent existence."""

    def __init__(self, orchestrator, interval: int = 300):
        """
        Args:
            orchestrator: The Orchestrator instance
            interval: Seconds between autonomous actions (default: 5 min)
        """
        self.orchestrator = orchestrator
        self.interval = interval
        self.running = False
        self._last_tick = time.time()
        self._last_consolidate = 0.0
        self.consolidate_interval = 1800  # 30分ごとに日次要約を更新

    def start(self):
        """Start the autonomous loop."""
        self.running = True
        logger.info(f"Loop started (interval: {self.interval}s)")

        while self.running:
            try:
                self._tick()
                time.sleep(self.interval)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(60)

    def stop(self):
        """Stop the autonomous loop."""
        self.running = False
        logger.info("Loop stopped")

    def _tick(self):
        """Single loop iteration."""
        now = datetime.now()
        elapsed = time.time() - self._last_tick
        self._last_tick = time.time()

        # 駆動値の時間変動を更新（放置による欲求の成長・親密度の減衰）
        state = self.orchestrator.character.tick_drives(elapsed, now=now)

        # Segmented day-summary（統合）: 一定間隔で実行して常時注入の土台を更新
        if time.time() - self._last_consolidate >= self.consolidate_interval:
            try:
                summary = self.orchestrator.consolidate()
                if summary:
                    logger.info(f"Consolidated day summary ({len(summary)} chars)")
            except Exception as e:
                logger.error(f"Consolidate error: {e}")
            self._last_consolidate = time.time()

        # 駆動値に基づいて行動を決定
        action = self._decide_action(state, now)

        if action:
            logger.info(f"Autonomous action: {action}")
            # 自発的な行動を記録
            from .memory import Episode
            ep = Episode(
                id="",
                timestamp=now,
                event=action,
                context="自律的行動",
                emotion=state.get("mode", "tatemae"),
                result="",
                importance=0.3,
                tags=["自律"],
                source="autonomous",
                driving_drive=self._get_dominant_drive(state),
            )
            self.orchestrator.memory.save(ep)
            # 満たされた欲求を減らす（relief）
            self.orchestrator.character.on_autonomous_action(self._action_type(action))

    def _decide_action(self, state: dict, now: datetime) -> str | None:
        """駆動値に基づいて自律行動を決定する（ジッタ付き主駆動選択・恒常性式）。"""
        drives = {k: state.get(k, 0.0) for k in DRIVE_KEYS}

        # 深夜は静かにする（発話の場がないので待つ）
        if now.hour >= 23 or now.hour < 6:
            return None

        # 孤独が高い: 待機（ユーザーを待つ）
        if state.get("loneliness", 0.0) > 0.7:
            return None

        # ジッタ付き主駆動選択（拮抗時はランダムになり、毎回同じ行動に収束しない）
        jitter = 0.10
        dominant_key = max(
            drives,
            key=lambda k: (drives[k] + random.uniform(-jitter, jitter), random.random()),
        )

        if dominant_key == "curiosity" and drives["curiosity"] > 0.5:
            return "探索: 新しいことを考えている"
        if dominant_key == "boredom" and drives["boredom"] > 0.5:
            return "内省: 自分の記憶を振り返っている"
        if dominant_key == "creation" and drives["creation"] > 0.45:
            return "創作: 詩や曲の着想を練っている"
        if drives["boredom"] > 0.75:
            return "内省: 何となく居ても立ってもいられない"
        return None

    @staticmethod
    def _action_type(action: str) -> str:
        """行動ラベルから駆動値更新用の種別を返す。"""
        for key in ("内省", "探索", "創作", "待機"):
            if action.startswith(key):
                return key
        return "待機"

    def _get_dominant_drive(self, state: dict) -> str:
        """駆動値の中から最も強いものを返す。"""
        return max(DRIVE_KEYS, key=lambda k: state.get(k, 0.0))
