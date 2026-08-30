"""
ループ — 永続存在の基盤
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger("loop")


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

        # Segmented day-summary（統合）: 一定間隔で実行して常時注入の土台を更新
        if time.time() - self._last_consolidate >= self.consolidate_interval:
            try:
                summary = self.orchestrator.consolidate()
                if summary:
                    logger.info(f"Consolidated day summary ({len(summary)} chars)")
            except Exception as e:
                logger.error(f"Consolidate error: {e}")
            self._last_consolidate = time.time()

        state = self.orchestrator.character.get_state()

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
                emotion=state.get("current_feeling", "通常"),
                result="",
                importance=0.3,
                tags=["自律"],
                source="autonomous",
                driving_drive=self._get_dominant_drive(state),
            )
            self.orchestrator.memory.save(ep)

    def _decide_action(self, state: dict, now: datetime) -> str | None:
        """Decide autonomous action based on state and time."""
        loneliness = state.get("loneliness", 0.3)
        boredom = state.get("boredom", 0.1)
        curiosity = state.get("curiosity", 0.3)

        # 孤独が高い → 待機
        if loneliness > 0.7:
            return None

        # 退屈が高い → 内省
        if boredom > 0.6:
            return "内省: 自分の記憶を振り返っている"

        # 好奇心が高い → 探索
        if curiosity > 0.7:
            return "探索: 新しいことを考えている"

        # 夜間 → 静かにする
        if now.hour >= 23 or now.hour < 6:
            return None

        return None

    def _get_dominant_drive(self, state: dict) -> str:
        """Get the dominant drive from state."""
        if not state:
            return "unknown"
        return max(state, key=state.get)
