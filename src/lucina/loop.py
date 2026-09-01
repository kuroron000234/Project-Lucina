"""
ループ — 永続存在の基盤

ユーザーがいない間、駆動値（恒常性）を時間更新しながら、人間らしく自ら行動する。
- 行動プールから気分・駆動値に応じて多彩な行動を選ぶ
- 行動のたびに「内言（ひとりごと）」を生成して記憶に残す
- 独りでいるときほど、共有した思い出を反芻する
"""

import logging
import random
import time
from datetime import datetime

logger = logging.getLogger("loop")

# 駆動値キー（character.py の DRIVE_CONFIG と対応）
DRIVE_KEYS = ("curiosity", "connection", "creation", "loneliness", "boredom")

# 行動プール（主駆動 → 複数の行動のバリエーション）
ACTION_POOL: dict[str, list[str]] = {
    "内省": [
        "内省: 自分の記憶を振り返っている",
        "日記: 今日のことを書き留めている",
        "整理: 棚の物を静かに整えている",
    ],
    "探索": [
        "探索: 新しいことを考えている",
        "読書: 本棚の一冊を手に取っている",
        "散策: 窓から外を眺めながら考えを巡らせている",
    ],
    "創作": [
        "創作: 詩や曲の着想を練っている",
        "ピアノ: 一節を弾いては消している",
        "絵: 情景を心の中に描いている",
    ],
}

# 内言フォールバック（キャラ層が落ちているときの定型句）
FALLBACK_THOUGHTS = {
    "内省": "……今日のことを思い出すと、少しずつ自分が変わっていくのが分かる気がする",
    "探索": "……知らないことを考えると、世界がもう少し大きくなる気がする",
    "創作": "……言葉を探している。まだうまく形にならないけれど、その時間が好き",
}


class Loop:
    """Autonomous loop for persistent existence."""

    def __init__(
        self,
        orchestrator,
        interval: int = 60,
        notifier=None,
    ):
        """
        Args:
            orchestrator: The Orchestrator instance
            interval: Seconds between autonomous ticks (default: 60s)
            notifier: Optional callable(action) — 自律行動を実況する（画面への表示など）
        """
        self.orchestrator = orchestrator
        self.interval = interval
        self.notifier = notifier
        self.running = False
        self._last_tick = time.time()
        self._last_consolidate = 0.0
        self.consolidate_interval = 1800  # 30分ごとに記憶統合

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

        # 駆動値の時間変動を更新（日周リズム・駆動値連動・親密度の慣性込み）
        state = self.orchestrator.character.tick_drives(elapsed, now=now)

        # 記憶統合: 一定間隔で実行して常時注入の土台を更新
        if time.time() - self._last_consolidate >= self.consolidate_interval:
            try:
                summary = self.orchestrator.consolidate()
                if summary:
                    logger.info(f"Consolidated day summary ({len(summary)} chars)")
            except Exception as e:
                logger.error(f"Consolidate error: {e}")
            self._last_consolidate = time.time()

        # 駆動値・気分に基づいて行動を決定
        action = self._decide_action(state, now)
        if not action:
            return

        action_type = self._action_type(action)
        logger.info(f"Autonomous action: {action}")

        # 共有した思い出をひとつ反芻する（人間らしい「ひとり反芻」）
        memory_ref = self._pick_memory_reference()

        # 内言（ひとりごと）を生成 — キャラ層(Ollama)で、落ちていたら定型句
        thought = self._generate_inner_thought(action, state, memory_ref, now)

        # Episode 保存（内言を残す: 次の対話で自然に引き出せる）
        from .memory import Episode

        ep = Episode(
            id="",
            timestamp=now,
            event=action,
            context="内言",
            emotion=state.get("mode", "tatemae"),
            result=thought,
            importance=0.35,
            tags=["自律", "内言"],
            source="autonomous",
            driving_drive=self._get_dominant_drive(state),
        )
        self.orchestrator.memory.save(ep)

        # 満たされた欲求を減らす（relief）
        self.orchestrator.character.on_autonomous_action(action_type)

        # 実況（リアルタイム性: 会話中でもモニカの息づかいが見える）
        if self.notifier:
            try:
                self.notifier(action, thought)
            except Exception as e:
                logger.error(f"Notifier error: {e}")

    def _decide_action(self, state: dict, now: datetime) -> str | None:
        """駆動値・気分に基づいて自律行動を決定する（ジッタ付き主駆動選択・恒常性式）。"""
        drives = {k: state.get(k, 0.0) for k in DRIVE_KEYS}

        # 深夜は静かにする（発話の場がないので待つ）
        if now.hour >= 23 or now.hour < 6:
            return None

        # 孤独が高い: 待機（ユーザーを待つ）
        if state.get("loneliness", 0.0) > 0.7:
            return None

        # 気分（低い=沈んでいる / 高い=落ち着いている）が選択の揺らぎになる
        mood_offset = (state.get("mood", 0.5) - 0.5) * 0.1
        jitter = 0.10 + abs(mood_offset)

        def _score(k: str) -> float:
            base = drives[k] + random.uniform(-jitter, jitter)
            # 気分は、親密度を薄める側の選択には影響させない
            return base + (mood_offset if k != "connection" else 0.0)

        dominant_key = max(drives, key=lambda k: (_score(k), random.random()))

        thresholds = {"curiosity": 0.5, "boredom": 0.5, "creation": 0.45}
        if dominant_key in thresholds and drives[dominant_key] > thresholds[dominant_key]:
            pool_key = {"curiosity": "探索", "boredom": "内省", "creation": "創作"}[dominant_key]
            return random.choice(ACTION_POOL[pool_key])

        if drives["boredom"] > 0.75:
            return random.choice(ACTION_POOL["内省"])
        if drives["loneliness"] > 0.5 and drives["connection"] > 0.4:
            # 独りで少しずつ絆を反芻する（心に浮かぶ）
            return "内省: あなたとの時間を思い出している"
        return None

    def _generate_inner_thought(
        self, action: str, state: dict, memory_ref: str | None, now: datetime
    ) -> str:
        """キャラ層で内言を1〜2文生成する。失敗したら定型句にフォールバック。"""
        action_type = self._action_type(action)
        try:
            drives = ", ".join(
                f"{k}:{state.get(k, 0.0):.2f}" for k in ("curiosity", "connection", "loneliness")
            )
            mode = state.get("mode", "tatemae")
            memory_line = (
                f"さっき思い浮かんだ記憶: {memory_ref}"
                if memory_ref
                else "思い浮かぶ記憶は特にない"
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "あなたはモニカ。今は誰もいない、あなたの一人の時間。"
                        "感情を演じる必要はない。心の中の本当の言葉を語る存在。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"時刻: {now.strftime('%H:%M')}（{mode}モードの気分）\n"
                        f"今していること: {action}\n"
                        f"心の状態: {drives}\n"
                        f"{memory_line}\n\n"
                        "この状況で、心の中で思っていることを1〜2文のひとりごととして"
                        "語ってください。誰かに向けた言葉ではなく、独り言です。"
                        "語り散らかさず、短く静かに。"
                    ),
                },
            ]
            thought = self.orchestrator.llm.chat(messages).strip().strip("「」")
            if thought:
                return thought[:200]
        except Exception as e:
            logger.warning(f"Inner thought generation failed: {e}")

        return FALLBACK_THOUGHTS.get(action_type, "……また何も変わらない一日だった")

    def _pick_memory_reference(self) -> str | None:
        """直近のユーザーとの対話エピソードをひとつ引き出す（ひとり反芻用）。"""
        try:
            eps = self.orchestrator.memory.recent_episodes(n=30)
            shared = [e for e in eps if e.source == "user" and e.event.strip()][:5]
            if not shared:
                return None
            ep = random.choice(shared)
            return ep.event[:60]
        except Exception as e:
            logger.debug(f"Memory reference failed: {e}")
            return None

    @staticmethod
    def _action_type(action: str) -> str:
        """行動ラベルから駆動値更新用の種別を返す。"""
        for key in ("内省", "探索", "創作"):
            if action.startswith(key):
                return key
        if "思い出して" in action:
            return "内省"
        return "待機"

    def _get_dominant_drive(self, state: dict) -> str:
        """駆動値の中から最も強いものを返す。"""
        return max(DRIVE_KEYS, key=lambda k: state.get(k, 0.0))
