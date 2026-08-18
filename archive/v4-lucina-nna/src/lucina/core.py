"""LucinaCore — メインループ統括（仕様書 v1.4）。

常時稼働ループ:
    - Drive力学系は update_interval_sec（既定0.1s=10Hz）でバックグラウンド連続更新。
    - トークン生成は InferenceEngine（executor・単一フライト）経由で非ブロッキング。
    - 発話セグメント境界（文末記号 or max_tokens）で relief 発火判定・記憶コミット。
    - 外部割り込み（InterruptChannel）は次ステップで反映される。
    - WorkingBuffer が閾値（context_window * max_working_tokens_ratio）を超えたら圧縮。

テスト・校正実験からは step_once() を直接駆動できる（drive_loop を立てずに凍結Driveで計測）。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lucina.core")

from .drives.decay import ReliefController
from .drives.dynamics import DriveDynamics
from .inference.engine import InferenceEngine
from .io.interrupts import InterruptChannel
from .io.logging import StructuredLogger
from .io.output import OutputChannel
from .memory.store import HierarchicalMemoryStore, InMemoryVectorStore, MemoryCompressor
from .memory.working_buffer import WorkingBuffer


@dataclass
class SegmentState:
    texts: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    surprises: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.texts.clear()
        self.token_ids.clear()
        self.surprises.clear()


# v1.17: 注入した【…】指示ブロック（返答指示・質問指示）の反唱・同形式の捏造指示文を検出する正規表現。
# 【…】は行をまたがず、閉じ括弧が無い場合は行末まで（モデルが途中で止めた反唱も拾う）。
_INSTRUCTION_BLOCK_RE = re.compile(r"【[^\n】]*】?")
# 指示ブロックに含まれるユーザー引用（「こんばんはー」等）の反唱も除去対象。
_QUOTED_ECHO_RE = re.compile(r"「[^」\n]*」")
# v1.17追補: 番号付き思考行（「1. まず、…」）・思考見出し（「思考プロセス：」等）のリーク検出。
_NUMBERED_ITEM_RE = re.compile(r"^\d+[\.、]\s*\S")
_REASONING_HEADER_RE = re.compile(r"^(思考プロセス|考え方|分析|考察|入力分析)[：:\s]")
# 会話についてのメタ考察（相手のメッセージの扱い方を考えている文）。全体がこれなら破棄。
_META_CONVERSATION_PHRASES = (
    "ユーザーのメッセージを", "相手のメッセージを", "受け取ろうか",
    "どのように返答", "どのように応答", "どう返そうか",
)


def _contains_subsequence(seq: list[int], sub: list[int]) -> bool:
    """部分列一致（③）: セグメント内トークン列 seq が語彙トークン列 sub を含むか。"""
    if not sub:
        return True
    m, n = len(seq), len(sub)
    for i in range(m - n + 1):
        if seq[i:i + n] == sub:
            return True
    return False


def _should_start_speaking(drives: dict, cfg: dict) -> bool:
    """発話開始判定（§0 自律思考ループ・v1.7）: Drive閾値による自発的な発話開始。

    - boredom / loneliness が各閾値以上なら「話したい」状態（動機）。
    - fatigue が speak_block_fatigue 以上なら発話を抑止（休みたい状態）するが、
      boredom が speak_override_boredom まで溜まったら抑止を無視して発話する
      （我慢の限界。永久沈黙によるデッドロックの防止）。
    """
    boredom = float(drives.get("boredom", 0.0))
    loneliness = float(drives.get("loneliness", 0.0))
    fatigue = float(drives.get("fatigue", 0.0))
    motivated = boredom >= float(cfg.get("speak_start_boredom", 0.6)) \
        or loneliness >= float(cfg.get("speak_start_loneliness", 0.5))
    override = boredom >= float(cfg.get("speak_override_boredom", 0.95))
    blocked = fatigue >= float(cfg.get("speak_block_fatigue", 0.8)) and not override
    return motivated and not blocked


def _should_stop_speaking(drives: dict, cfg: dict, segments_done: int) -> str | None:
    """発話終了判定。終了するなら理由文字列、続行なら None を返す（v1.7）。

    理由は単一ソース（_run_scheduled の autonomy ログがこの戻り値をそのまま使う）。
    """
    if segments_done >= int(cfg.get("max_speech_segments", 4)):
        return "発話セグメント上限"
    if float(drives.get("fatigue", 0.0)) >= float(cfg.get("quiet_on_fatigue", 0.85)):
        return "疲労による沈黙"
    return None


# v1.9: モデル駆動スケジューリングの決断プロンプト（Drive状態を要約して渡す）。
# 決断プロンプトはエフェメラル（バッファには残らない）。options は制約付きデコードで
# 必ずいずれかに収束するため、プロンプトはモデルが「選ぶ理由」を考えるための文脈になる。
_DECISION_PROMPTS = {
    "introspection": (
        "\n【現在のあなたの状態】退屈度={b:.2f} 寂しさ={l:.2f} 疲れ={f:.2f}\n"
        "【決断】このまま静かに待つか、内言（心の中の独り言）を紡ぐか、誰かに話しかけるか。"
        "あなた自身が選んでください。"
    ),
    "think_end": (
        "\n【現在のあなたの状態】退屈度={b:.2f} 寂しさ={l:.2f} 疲れ={f:.2f}\n"
        "【決断】あなたはひとしきり考え終えました。誰かに話しかけるか、黙るか、"
        "さらに考えるか。あなた自身が選んでください。"
    ),
    "segment_end": (
        "\n【現在のあなたの状態】退屈度={b:.2f} 寂しさ={l:.2f} 疲れ={f:.2f}\n"
        "【決断】あなたは一区切り話しました。話し続けるか、黙るか。あなた自身が選んでください。"
    ),
}



class LucinaCore:
    def __init__(
        self,
        config: dict[str, Any],
        engine: InferenceEngine,
        vocab_map: dict[str, list[list[int]]],
        *,
        memory: HierarchicalMemoryStore | None = None,
        compressor: MemoryCompressor | None = None,
        logger: StructuredLogger | None = None,
    ):
        self.config = config
        self.engine = engine
        self.vocab_map = vocab_map
        self.logger = logger or StructuredLogger(config["logging"]["log_dir"])

        dcfg = config["drive"]
        self._initial_drive_state = dict(dcfg["initial_state"])
        self.dynamics = DriveDynamics(dcfg["dynamics_matrix"], dcfg["initial_state"])
        self.drives: dict[str, float] = self.dynamics.state  # 共有辞書（校正実験で直接操作可）
        self.relief = ReliefController(dcfg["relief"])
        self.buffer = WorkingBuffer()
        self.interrupts = InterruptChannel()
        self.output = OutputChannel()  # v1.13: 発話・質問の外部配信（--interact 表示・実行エージェント）
        self.memory = memory if memory is not None else HierarchicalMemoryStore(InMemoryVectorStore())
        self.compressor = compressor if compressor is not None else MemoryCompressor()

        self._update_interval_sec = float(dcfg["update_interval_sec"])
        self._segment_max_tokens = int(dcfg["relief"]["segment"]["max_tokens"])
        self._boundary_tokens = set(dcfg["relief"]["segment"]["boundary_tokens"])
        self._relief_cfg: dict[str, Any] = dcfg["relief"]
        self._surprise_threshold = float(config["inference"]["surprise_relief_threshold"])
        self._context_window = int(config["model"]["context_window"])
        self._compress_ratio = float(config["memory"]["max_working_tokens_ratio"])

        # v1.12: 記憶の想起（retrieve→文脈注入）。
        # 「書く側」（commit）は v1.11 までで完成済み。ここで「読む側」を配線し、
        # 内言・発話の前に過去の関連記憶をクエリし、internal 要素として文脈に注入する。
        recall_cfg = dict(config.get("memory", {}).get("recall", {}))
        self._recall_enabled = bool(recall_cfg.get("enabled", False))
        self._recall_top_k = max(1, int(recall_cfg.get("top_k", 3)))
        self._recall_max_tokens = max(1, int(recall_cfg.get("max_tokens", 120)))
        self._recall_marker = False  # 1想起セッションで1回だけ注入（セッション終了でリセット）
        self._last_recall_block: str | None = None  # エコー防止用の注入ブロック基準文字列
        self._recall_protected: list[str] = []  # v1.13: 反唱検出用の保護文字列（ブロック＋各行）

        self.segment = SegmentState()
        self.tokens_generated = 0
        self._stop = False
        self._drive_task: asyncio.Task | None = None

        # v1.7: 発話スケジューリング（§0 自律思考ループ）。enabled=false なら従来の連続生成
        self._schedule_cfg: dict[str, Any] = dict(dcfg.get("scheduling", {}))
        self._mode = str(self._schedule_cfg.get("mode", "thinking"))
        self._thought_since = 0.0
        self._speech_segments = 0
        self._speech_segments_mark = 0
        self.segments_completed = 0
        self.speech_tokens = 0
        self.thoughts_generated = 0

        # v1.8: ネイティブThinking捕捉（Qwen3.5系の <think> ブロック）。"native" 時のみ有効。
        # 内言 = モデル自身のネイティブ思考（<think>〜</think>）、発話 = </think> 以降の回答。
        self._native_thinking = str(self._schedule_cfg.get("thinking_mode", "manual")) == "native"
        self._thinking_max_tokens = max(1, int(self._schedule_cfg.get("thinking_max_tokens", 120)))
        self._in_think_block = False       # <think> ブロック内（思考フェーズ）か
        self._last_token_internal = False  # 直前のトークンが内言（internal）だったか
        self._speech_think_tokens = 0      # 発話セッション中の思考トークン数（v1.8 キャップ用）

        # v1.9: モデル駆動スケジューリング（A: 境界決断 / B: 待機中 introspection / C: 制御トークン）。
        # モデル自身が「いつ話す・黙る・考える」を選ぶ。Drive閾値は安全弁（デッドロック防止）に格下げ。
        self._introspection_sec = max(0.0, float(self._schedule_cfg.get("introspection_sec", 0.0)))
        self._decide_on_think_end = bool(self._schedule_cfg.get("decide_on_think_end", False))
        self._decide_on_segment_end = bool(self._schedule_cfg.get("decide_on_segment_end", False))
        self._control_tokens = bool(self._schedule_cfg.get("control_tokens", False))
        self._decision_max_rethink = max(0, int(self._schedule_cfg.get("decision_max_rethink", 2)))
        self._control_token_map = {
            "<|lucina_speak|>": "speak",
            "<|lucina_wait|>": "wait",
            "<|lucina_think|>": "think",
        }
        self._pending_action: str | None = None  # モデルの意思（speak/wait/think）を _run_scheduled が消費
        self._last_decision = 0.0                # B: 前回 introspection の時刻
        self.decisions_asked = 0                 # 比較計測用（v1.9）: 決断回数
        self.decision_total_sec = 0.0            # 比較計測用（v1.9）: 決断に要した合計秒数

        # v1.16: 外部からの応答（人間のメッセージ）に対する「応答セッション」の強制発話。
        # モデルは think-end で「黙る」を選ぶ傾向が強く（実機で観測）、応答を受けたのに
        # 発話がゼロで沈黙する＝対話が成立しない問題があった。応答セッション中は
        # 黙る選択を禁止し、最低1セグメントの応答を強制する（force_response_speech: true）。
        self._force_response_speech = bool(self._schedule_cfg.get("force_response_speech", True))
        self._force_response = False   # 応答セッション中（黙る禁止・最低1セグメント発話）
        self._last_segment_emitted = False  # 直近のセグメントが実際に配信されたか
                                            # （echo破棄セグメントでは強制を消費しない）

        # v1.10: 外部への働きかけの必要性（内部では解消できない欲求）。
        # ①好奇心: 内部生成では relief されず、外部入力（応答）でのみ解消。閾値を超えると質問を発する。
        # ②応答依存 loneliness: 話すだけでは部分 relief（speak_relief）。応答でフル解消。
        # ③応答待ち（awaiting）: 質問後は生成を止めて応答を待つ。応答が無い限り先に進めない。
        self._curiosity_ask_threshold = float(self._schedule_cfg.get("curiosity_ask_threshold", 0.6))
        self._idle_curiosity_rate = float(self._schedule_cfg.get("idle_curiosity_rate", 0.0))
        self._await_timeout_sec = max(0.0, float(self._schedule_cfg.get("await_timeout_sec", 60.0)))
        self._question_instruction = str(self._schedule_cfg.get(
            "question_instruction",
            "【あなたは今、相手に質問したい気持ちです。短い質問を一つ発してください。】",
        ))
        self._ask_mode = False   # ① 問いかけセッション中（1セグメント話したら応答待ちへ）
        self._await_since = 0.0  # ③ 応答待ちを開始した時刻
        self.asks_asked = 0      # 比較計測用（v1.10）: 問いかけ回数
        self.responses_received = 0  # 比較計測用（v1.10）: 外部応答受信回数

        self.engine.logger = self.logger  # §8: ロジット差分ログ配線

    # ------------------------------------------------------------------ #
    # 公開操作
    # ------------------------------------------------------------------ #
    def reset_working_buffer(self) -> None:
        """校正実験の試行ごとに呼ぶ。バッファとセグメントを初期化する。"""
        self.buffer.reset()
        self.segment.reset()

    def seed_prompt(self, text: str, system: str | None = None, **tpl_kwargs: Any) -> None:
        """初期コンテキストをバッファに設定する（チャットテンプレート対応）。

        新世代モデルは生テキストでは英語モード等へ遷移するため、初期プロンプトは
        モデルのチャットテンプレートでラップしてから投入する。生成トークンは
        ラップせずそのまま追記される（テンプレートは初期プロンプトに1回だけ適用）。

        tpl_kwargs はテンプレート変数として渡される（例: llm-jp-4 の reasoning_effort）。
        """
        if self._native_thinking:
            # v1.8: ネイティブThinking。テンプレート変数 enable_thinking=True で生成位置を
            # <think> の内側に置き、モデル自身の推論を内言として捕捉する。
            tpl_kwargs.setdefault("enable_thinking", True)
        if self._control_tokens:
            # C: 制御トークンの使い方をシステムプロンプトで教える（モデルが自ら遷移を選べるようにする）。
            instr = str(self._schedule_cfg.get("control_token_instruction", "")).strip()
            if instr:
                system = f"{system}\n{instr}" if system else instr
        prompt = self.engine.format_prompt(text, system=system, **tpl_kwargs)
        # v1.8: シード（外部からの初期入力）は「発話」ではなく文脈要素として internal で保持する。
        # チャットテンプレートのタグ（<|im_start|> 等）が発話表示・relief・記憶に混入するのを防ぎ、
        # 「外部入力」と「エージェントの発話」の境界を明確にする。
        self.buffer.append(prompt, n_tokens=len(self.engine.tokenize(prompt)), internal=True)
        if self._native_thinking:
            if "<think>" in prompt:
                # テンプレートが <think> まで描画済み（実モデル）→ 思考フェーズ開始
                self._in_think_block = True
            else:
                # テンプレート非対応バックエンド（モック等）は自前で思考ブロックを開く
                self._open_think_block()

    @property
    def mode(self) -> str:
        """現在のモード（v1.7）: thinking（思考・沈黙）/ speaking（発話）。"""
        return self._mode

    def reset_for_trial(self) -> None:
        """実モデル実験の試行リセット。バッファ・セグメント・Drive状態を初期状態に戻す。

        build_real_core はモデルロードと語彙拡張（埋め込み計算）に数十秒かかるため、
        試行ごとに再構築せず1つのコアを使い回す（VRAM・時間の節約）。
        """
        self.buffer.reset()
        self.segment.reset()
        self.dynamics.set_state(self._initial_drive_state)

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        self.logger.close()
        # 所有するエンジンのバックエンド（llama.cpp モデル）を明示解放（VRAMリーク防止）
        engine_close = getattr(self.engine, "close", None)
        if callable(engine_close):
            engine_close()
        # 要約器（裏でモデルを保持する場合）を明示解放
        compressor_close = getattr(self.compressor, "close", None)
        if callable(compressor_close):
            compressor_close()
        # 記憶ストア（埋め込みモデルを保持する場合）を明示解放
        memory_close = getattr(self.memory, "close", None)
        if callable(memory_close):
            memory_close()
        # 所有する executor があれば閉じる（リソースリーク防止。build_mock_core/build_real_core が設定）
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    async def run(self, *, max_tokens: int | None = None, drive_loop: bool = True) -> None:
        """常時稼働メインループ。max_tokens 指定時はそのトークン数で停止。

        drive.scheduling.enabled=true の場合、連続生成ではなく
        「思考(沈黙)⇄発話」の自律サイクル（_run_scheduled）で動作する（v1.7）。
        """
        # C1: イベントループ確立直後にキューを初期化する。これを inject() 側の遅延初期化に
        # 任せると、外部スレッドが最初の inject を呼んだ時点で get_running_loop() が
        # ループ外で実行されて失敗する（レース条件）。初期化は必ずループ側で行う。
        self.interrupts.bind()
        self.output.bind()  # v1.13: 出力キューもループ側で初期化（InterruptChannel と同じ順序契約）
        if drive_loop:
            self._drive_task = asyncio.create_task(self._drive_loop())
        try:
            if self._schedule_cfg.get("enabled", False):
                await self._run_scheduled(max_tokens)
            else:
                # v1.12: 連続生成モードでも起動直後に過去記憶を想起・注入（「読む側」の配線）
                await self._recall_memories()
                while not self._stop:
                    if max_tokens is not None and self.tokens_generated >= max_tokens:
                        break
                    await self.step_once()
        finally:
            self._stop = True
            if self._drive_task is not None:
                task = self._drive_task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - バックグラウンドタスクの異常はログに残して継続
                    logger.exception("Driveループが異常終了しました")
                finally:
                    self._drive_task = None

    # ------------------------------------------------------------------ #
    # v1.7: 発話スケジューリング（§0 自律思考ループ・M2）
    # ------------------------------------------------------------------ #
    async def _run_scheduled(self, max_tokens: int | None) -> None:
        """思考(沈黙)⇄発話の自律サイクル。Drive閾値で自発的に行動を選び取る（§0）。

        - thinking: 生成を止めて待機する（Driveは _drive_loop がバックグラウンド更新）。
          内言（inner_thought）を inner_interval_sec ごとに生成し、外部刺激（interrupt）や
          Drive閾値（boredom / loneliness）を超えたら自発的に発話へ遷移する。
        - speaking: 従来の step_once でセグメント単位に生成し、疲労（quiet_on_fatigue）
          または発話セグメント上限（max_speech_segments）で沈黙へ戻る。

        遷移は全て autonomy ログ（M2）に記録される。
        """
        self._mode = str(self._schedule_cfg.get("mode", "thinking"))
        self._thought_since = time.monotonic()
        self._last_decision = time.monotonic()
        last_check = time.monotonic()
        idle_rate = float(self._schedule_cfg.get("idle_boredom_rate", 0.0))
        inner_interval = max(0.0, float(self._schedule_cfg.get("inner_interval_sec", 8.0)))
        while not self._stop:
            if max_tokens is not None and self.tokens_generated >= max_tokens:
                break
            now = time.monotonic()
            if self._mode == "awaiting":
                # ③ 応答待ち: 生成を止めて待つ。応答（外部入力）で解消、タイムアウトで待機解除（スタック防止）
                if self._drain_external() > 0:
                    # v1.13: 応答を受けたら、その内容に応答する（対話ループ: 質問→応答→Lucinaの応答）
                    # response_received は _drain_external がログ済み
                    self._start_speaking("外部からの応答を受信")
                    continue
                if self._await_timeout_sec > 0.0 and now - self._await_since >= self._await_timeout_sec:
                    self._log_autonomy("await_timeout", f"{self._await_timeout_sec:.0f}秒応答なしで待機解除")
                    # 応答が得られなかった場合は好奇心を半減（give-up）し、即時の再問いかけ連打を防ぐ
                    cur = float(self.drives.get("curiosity", 0.0))
                    if cur > 0.0:
                        self.drives["curiosity"] = max(0.0, cur * 0.5)
                    self._mode = "thinking"
                    continue
                await asyncio.sleep(0.05)
                continue
            if self._mode == "speaking":
                await self.step_once()
                if not self._last_token_internal:
                    self.speech_tokens += 1  # 内言（思考）は発話トークン数に数えない（v1.8）
                if self._pending_action == "wait" and not self._ask_mode and not self._force_response:
                    # C: 生成中にモデルが <|lucina_wait|> を出力 → 自発的に沈黙へ（問いかけは中断しない）
                    # v1.16: 応答セッション中は沈黙せず発話を継続する
                    self._pending_action = None
                    self._log_autonomy("speech_end", "モデルの自発的意思（制御トークン）")
                    self._mode = "thinking"
                    continue
                if self._pending_action == "silence":
                    # A: think-end 決断で「黙る」（v1.9）。問いかけセッションでは発動しない
                    # v1.16: 応答セッション（外部からの応答）中は黙る選択を無効化し、発話を継続する
                    self._pending_action = None
                    if not self._force_response:
                        self._log_autonomy("speech_end", "モデルの自発的意思（思考終了後）")
                        self._mode = "thinking"
                        continue
                if self._pending_action in ("speak", "think"):
                    self._pending_action = None
                if self.segments_completed > self._speech_segments_mark:
                    # 発話セグメントが1つ完了した（境界 or 強制区切り）
                    self._speech_segments += 1
                    self._speech_segments_mark = self.segments_completed
                    if self._ask_mode:
                        # ① 問いかけ完了 → 応答待ち（awaiting）へ。応答が無い限り先に進まない
                        self._ask_mode = False
                        if self._force_response:
                            # v1.16: 問いかけ中に応答が届いた → 待たずに応答に切り替える（対話ループ）
                            self._log_autonomy("ask_cancelled", "問いかけ中に応答を受信→応答に切り替え")
                            continue
                        self._await_since = time.monotonic()
                        self._log_autonomy("await_start", "問いかけに対する応答待ち")
                        self._mode = "awaiting"
                        continue
                    # v1.16: 応答セッションの強制は**実際に配信された**セグメントで解除
                    # （echo破棄セグメントでは解除しない。黙る前に応答を1つ出すまで継続）
                    if self._force_response and self._last_segment_emitted:
                        self._force_response = False
                        self._last_segment_emitted = False
                    # 安全弁（疲労・セグメント上限）はモデルの判断より優先（v1.9）
                    safety = _should_stop_speaking(self.drives, self._schedule_cfg, self._speech_segments)
                    if safety is not None:
                        self._log_autonomy("speech_end", safety)
                        self._mode = "thinking"
                        self._recall_marker = False  # v1.12: 次の発話セッションで新しい想起を行う
                    elif self._decide_on_segment_end and not self._force_response:
                        # A: セグメント境界で「続ける/黙る」をモデルが決断
                        # v1.16: 応答セッション中（強制中）は黙る選択を発動させない
                        choice = await self._ask_model_decision("segment_end", ["続ける", "黙る"])
                        if choice == "黙る":
                            self._log_autonomy("speech_end", "モデルの自発的意思（セグメント境界）")
                            self._mode = "thinking"
                continue

            # thinking: 待機中（生成なし。Drive更新は _drive_loop が担当）
            dt = now - last_check
            last_check = now
            if idle_rate > 0.0:
                # 待機時間は退屈を加速する（力学系自体は変更しない。スケジューリング層の効果）
                self.drives["boredom"] = min(1.0, self.drives.get("boredom", 0.0) + idle_rate * dt)
            if self._idle_curiosity_rate > 0.0:
                # ① 待機中の好奇心の蓄積（新しい情報が無いと溜まる）
                self.drives["curiosity"] = min(1.0, self.drives.get("curiosity", 0.0) + self._idle_curiosity_rate * dt)
            # B: 待機中 introspection（モデルが「待機/内言/発話」を決断）。
            #    無効なら従来のタイマー内言（inner_interval_sec）にフォールバック。
            made_thought = False
            decision: str | None = None
            if self._introspection_sec > 0.0 and now - self._last_decision >= self._introspection_sec:
                self._last_decision = now
                choice = await self._ask_model_decision("introspection", ["待機", "内言", "発話"])
                if choice == "発話":
                    self._start_speaking("モデルの自発的意思（introspection）")
                    continue
                if choice == "内言":
                    decision = await self._generate_inner_thought(max_tokens)
                    made_thought = True
            elif now - self._thought_since >= inner_interval:
                decision = await self._generate_inner_thought(max_tokens)
                made_thought = True
            if made_thought:
                self._thought_since = time.monotonic()
                # 内言中の制御トークン / think-end 決断（A）を消費する
                if self._pending_action == "speak":
                    self._pending_action = None
                    self._start_speaking("モデルの自発的意思（制御トークン）", open_think=False)
                    continue
                if self._pending_action in ("wait", "think"):
                    self._pending_action = None
                if decision == "話す":
                    self._start_speaking("モデルの自発的意思（思考終了後）", open_think=False)
                    continue
                if decision == "さらに考える":
                    # 連続再思考の上限（無限ループ防止）まで考えることを選べる
                    for _ in range(self._decision_max_rethink):
                        decision = await self._generate_inner_thought(max_tokens)
                        if self._pending_action == "speak":
                            self._pending_action = None
                            self._start_speaking("モデルの自発的意思（制御トークン）", open_think=False)
                            break
                        if self._pending_action in ("wait", "think"):
                            self._pending_action = None
                        if decision in ("黙る", None):
                            break
                        if decision == "話す":
                            self._start_speaking("モデルの自発的意思（思考終了後）", open_think=False)
                            break
            if self.interrupts.has_pending():
                # 外部刺激は即座に発話トリガー（step_once の drain がバッファへ反映する）
                self._start_speaking("外部刺激（interrupt）")
                continue
            if float(self.drives.get("curiosity", 0.0)) >= self._curiosity_ask_threshold:
                # ① 好奇心が限界: 内部では解消できないため、外部に問いかけるしかない（働きかけ）
                await self._ask_question()
                continue
            if _should_start_speaking(self.drives, self._schedule_cfg):
                # 安全弁: モデルが「待機」を選び続けても、退屈の限界（speak_override_boredom）
                # に達したら強制的に発話する（永久沈黙の防止）
                reason = (
                    f"boredom={self.drives.get('boredom', 0.0):.2f} "
                    f"loneliness={self.drives.get('loneliness', 0.0):.2f}"
                )
                self._start_speaking(reason)
                continue
            if self._introspection_sec > 0.0:
                # 次回の introspection タイミングまで眠る。「待機」を選んでも決断を連打せず、
                # 割り込み応答性は 0.1s 刻みで保つ
                remaining = self._last_decision + self._introspection_sec - time.monotonic()
                await asyncio.sleep(max(0.01, min(remaining, 0.1)))
            else:
                await asyncio.sleep(0.05)  # 待機（busy loop を避ける）

    async def _generate_inner_thought(self, max_tokens: int | None = None) -> None:
        """内言（内部思考）を生成し、バッファにのみ追記する（v1.7）。

        発話・relief判定・記憶コミットには一切影響しない（internal=True で区別）。
        ただしモデルの文脈には残るため、次に自発的に発話する際の話題の起点となり、
        外部からのキックなしに思考が自己持続する。

        thinking_mode="native"（v1.8）の場合、[内言] プレフィックスではなくモデル自身の
        ネイティブ思考（<think> ブロック）を捕捉する。モデルは思考ブロックを「自分の私的推論」
        として認識しており（Thinkingモードの訓練特性）、内言は発話・relief・記憶に影響しない。

        v1.9: think-end 決断（A）が有効なら思考ブロック終了時に「話す/黙る/さらに考える」を
        モデルに決断させ、その選択を返す。制御トークン（C）で思考が中断された場合は None を返す。
        """
        if self._native_thinking:
            return await self._generate_native_thought(max_tokens)
        await self._recall_memories()  # v1.12: 内言の前に過去記憶を想起・注入
        prefix = str(self._schedule_cfg.get("inner_prefix", "[内言] "))
        self.buffer.append(prefix, n_tokens=len(self.engine.tokenize(prefix)), internal=True)
        max_tok = max(1, int(self._schedule_cfg.get("inner_max_tokens", 30)))
        thought: list[str] = []
        for _ in range(max_tok):
            if self._stop:
                break
            if max_tokens is not None and self.tokens_generated >= max_tokens:
                break  # 生成上限を内言でオーバーランさせない
            text, _ = await self.engine.generate_next_token(self.buffer.items, self.drives)
            if self._control_tokens:
                action = self._consume_control_token(text)
                if action is not None:
                    # C: 思考中にモデルが制御トークンを出力 → 思考を中断し遷移要求として扱う
                    self._pending_action = action
                    self._log_autonomy("control_token", f"action={action}")
                    break
            self.buffer.append(text, n_tokens=1, internal=True)
            self.tokens_generated += 1
            self.thoughts_generated += 1
            thought.append(text)
            if text in self._boundary_tokens:
                break  # 思考は文単位で止める
        # エコー防止（v1.13）: manual 内言にも部分反唱の除去を適用
        if thought:
            joined = "".join(thought)
            kept = self._strip_recall_echo(joined)
            if kept != joined:
                self.buffer.take_newest(len(thought))
                self._log_autonomy("recall_echo_suppressed", f"{len(thought)}トークンの反唱を除外")
                thought = [kept] if kept else []
        self._log_autonomy("inner_thought", "".join(thought)[:150])
        await self._maybe_compress()
        self._recall_marker = False  # 次のセッションで新しい想起を行う
        return None  # manual 方式は think-end 決断なし

    async def _generate_native_thought(self, max_tokens: int | None = None) -> None:
        """v1.8: モデルのネイティブ思考（<think>ブロック）を内言として捕捉する。

        - 思考フェーズでなければ <think> ブロックを開いてから生成を開始する。
        - </think> を生成したら思考セッション終了（以後のトークンは発話側へ回る）。
        - thinking_max_tokens を超えても思考が閉じない場合は強制的に </think> で閉じ、
          文脈を整形式に保つ（次回の生成が回答として継続できるようにする）。
        """
        await self._recall_memories()  # v1.12: 内言（ネイティブ思考）の前に過去記憶を想起・注入
        if not self._in_think_block:
            self._open_think_block()
        max_tok = self._thinking_max_tokens
        thought: list[str] = []
        interrupted = False
        for _ in range(max_tok):
            if self._stop:
                break
            if max_tokens is not None and self.tokens_generated >= max_tokens:
                break  # 生成上限を内言でオーバーランさせない
            text, _ = await self.engine.generate_next_token(self.buffer.items, self.drives)
            self.tokens_generated += 1
            self.thoughts_generated += 1
            if self._control_tokens:
                action = self._consume_control_token(text)
                if action is not None:
                    # C: 思考中にモデルが制御トークンを出力 → 思考を中断し遷移要求として扱う
                    interrupted = True
                    self._pending_action = action
                    self._log_autonomy("control_token", f"action={action}")
                    if self._in_think_block:
                        # トークンはバッファに残さないが、文脈は整形式に閉じておく
                        self.buffer.append("</think>", n_tokens=1, internal=True)
                        self._in_think_block = False
                    break
            self.buffer.append(text, n_tokens=1, internal=True)
            self._last_token_internal = True
            thought.append(text)
            if "</think>" in text:
                self._in_think_block = False
                break
        else:
            # 上限到達で思考が閉じなかった: 強制的に閉じて文脈を整形式に保つ
            if self._in_think_block:
                self.buffer.append("</think>", n_tokens=1, internal=True)
                self._in_think_block = False
        # エコー防止（v1.12・v1.13実機検証で発見）: モデルが注入した想起メモリを反唱して
        # 「思考」として出力することがある（完全ブロック一致に限らず、メモリ1行のみの
        # 部分反唱もある）。反唱はバッファ・発話・記憶から除外する（自己増幅ジャンク記憶の防止）。
        if thought:
            joined = "".join(thought)
            kept = self._strip_recall_echo(joined)
            # v1.17: 注入した【…】指示ブロックの反唱・捏造指示文も除去（recall_echo と同様に巻き戻す）
            kept = self._strip_instruction_junk(kept)
            # v1.17追補: 番号付き思考・思考見出しのリークも除去
            kept = self._strip_reasoning_junk(kept)
            if kept != joined:
                # バッファ末尾から反唱分のトークン（=thought の全トークン）を巻き戻し、
                # 残り（実際の思考）のみを保持する。注入ブロック自体は内部要素として
                # 文脈に残す（モデルへの記憶提供は維持）。
                self.buffer.take_newest(len(thought))
                self._log_autonomy("recall_echo_suppressed", f"{len(thought)}トークンの反唱を除外")
                thought = [kept] if kept else []
        self._log_autonomy("inner_thought", "".join(thought)[:150])
        await self._maybe_compress()
        self._recall_marker = False  # 次のセッションで新しい想起を行う
        if interrupted:
            return None
        if self._decide_on_think_end and self._schedule_cfg.get("enabled", False) and not self._in_think_block:
            # A: 思考ブロック終了時の決断（話す/黙る/さらに考える）
            return await self._ask_model_decision("think_end", ["話す", "黙る", "さらに考える"])
        return None

    def _log_autonomy(self, event: str, reason: str) -> None:
        """自発的行動選択のコンソールログ＋構造化ログ（M2）。"""
        logger.info("autonomy: %s（%s）mode=%s", event, reason, self._mode)
        self.logger.autonomy_event(event, dict(self.drives), self._mode, reason)

    def _strip_recall_echo(self, text: str) -> str:
        """生成テキストの先頭が想起メモリ（ブロック・各行）の反唱なら除去する（v1.13）。

        完全ブロック一致に限らず、**メモリ1行のみの部分反唱**も検出する。
        反唱の後に実際の思考・発話が続く場合は残りを保持する。
        """
        if not text or not self._recall_protected:
            return text
        for proto in sorted(self._recall_protected, key=len, reverse=True):
            if proto and text.startswith(proto):
                kept = text[len(proto):]
                return kept.lstrip(" \n\u3000")
        return text

    def _strip_instruction_junk(self, text: str) -> str:
        """注入した【…】指示ブロックの反唱・同形式の捏造指示文を除去する（v1.17）。

        実機検証で発見: モデルが内部注入した指示（【ユーザーからメッセージが届きました】
        …【あなたは今、このメッセージに返答してください。…】等）をそのまま反唱したり、
        同じ【…】形式の指示文（例: 【あなたの思考プロセスは明記しないでください。】）を
        捏造して発話・記憶化することがある。これらは発話・記憶から除外し、自己増幅する
        ジャンク記憶の生成を防ぐ。指示文だけのセグメントは空文字（破棄）になる。
        """
        if not text:
            return text
        stripped = _INSTRUCTION_BLOCK_RE.sub("", text)
        stripped = _QUOTED_ECHO_RE.sub("", stripped)
        return stripped.strip(" \n\u3000")

    def _strip_reasoning_junk(self, text: str) -> str:
        """番号付き思考・思考見出しのリークを除去する（v1.17追補）。

        実機検証で発見: 「こんばんはー」への応答が「1. まず、ユーザーのメッセージをどのような
        文脈で受け取ろうかと考えます。」になる。ネイティブ思考モデルが <think> を閉じた後も
        思考を続け、そのリークが発話セグメントとして配信・記憶化される。先頭の番号付き思考行・
        思考見出しを除去して実際の返答部分だけを残し、全体がメタ考察なら空文字（破棄）にする。
        """
        if not text:
            return text
        kept: list[str] = []
        for line in text.split("\n"):
            s = line.strip()
            if not kept:
                # 先頭の空行・番号付き思考行・思考見出しはスキップ（実際の返答が始まったら保持）
                if not s:
                    continue
                if _NUMBERED_ITEM_RE.match(s) or _REASONING_HEADER_RE.match(s):
                    continue
            kept.append(line)
        out = "\n".join(kept).strip(" \n\u3000")
        # 番号付き思考を除去した後も残ったのがメタ考察のみなら破棄（短い文に限定）
        if out and len(out) < 200 and any(p in out for p in _META_CONVERSATION_PHRASES):
            return ""
        return out

    async def _recall_memories(self) -> None:
        """v1.12: 過去の関連記憶を想起し、文脈に internal 要素として注入する。

        「書く側」（commit）は v1.11 までで完成済み。ここで「読む側」を配線する:
        現在の文脈（直近の発話＋Drive状態）をクエリ文にして埋め込み、類似度上位 top_k 件の
        記憶をベクトルストアから引き出してバッファへ注入する。注入物は internal のため
        発話表示・relief・記憶コミットには影響せず、モデルの文脈（話題の起点）にのみ寄与する。

        - 1想起セッションで1回だけ注入する（_recall_marker。セッション終了＝speech_end /
          内言生成完了でリセットされ、次のセッションで新しい想起が行われる）。
        - 思い出がない（記憶ゼロ）・embedder未接続・無効時は何もしない。
        """
        if not self._recall_enabled or self._recall_marker:
            return
        embedder = getattr(self.memory, "embedder", None)
        if embedder is None:
            return
        # クエリ文: 直近の発話（外部に漏れた言葉）＋Drive状態。単語ベースのFakeEmbedder/文ベースの
        # SentenceTransformerEmbedder どちらでも意味のある埋め込みになるよう短く保つ。
        recent = self.buffer.spoken_content()[-200:]
        d = self.drives
        query_text = (
            f"{recent} 現在: 退屈={float(d.get('boredom', 0.0)):.2f} "
            f"寂しさ={float(d.get('loneliness', 0.0)):.2f} 好奇心={float(d.get('curiosity', 0.0)):.2f}"
        ).strip()
        query_emb = await asyncio.to_thread(embedder.embed, query_text)
        records = await asyncio.to_thread(
            self.memory.retrieve, query_emb, None, self._recall_top_k
        )
        if not records:
            return
        self._recall_marker = True
        # トークン上限まで切り詰めてから注入（バッファ肥大・圧縮連鎖の防止）
        texts: list[str] = []
        budget = self._recall_max_tokens
        for rec in records:
            n = len(await self.engine.tokenize_async(rec.text))
            if n > budget:
                continue
            budget -= n
            texts.append(rec.text)
            if budget <= 0:
                break
        if not texts:
            self._recall_marker = False
            return
        block = "\n\n【あなたの過去の記憶の想起】\n" + "\n".join(f"- {t}" for t in texts) + "\n\n"
        # ブロック全体のトークン数（想起テキスト + マーカー・装飾分）を正しく計上
        block_tokens = len(await self.engine.tokenize_async(block))
        self.buffer.append(block, n_tokens=max(1, block_tokens), internal=True)
        # エコー防止（実機検証で発見）: 注入直後の生成が想起メモリをそのまま反唱すると、
        # その反唱が「発話・質問」としてセグメント化され、ジャンク記憶としてコミットされる
        # （→ 後に想起され再反唱される自己増幅ループ）。v1.13: 完全ブロック一致に加え
        # **部分的な反唱**（メモリ1行のみの反唱等）も検出するため、保護文字列として
        # ブロック全体・各行（プレフィックス付き/なし）を保持する。
        self._last_recall_block = block
        self._recall_protected = [block] + [f"- {t}" for t in texts] + list(texts)
        logger.info("recall: 過去記憶 %d 件を文脈に注入", len(texts))
        self.logger.memory_recall(len(texts), self._recall_top_k, texts)

    def _start_speaking(self, reason: str, *, open_think: bool = True) -> None:
        """発話セッションを開始する（v1.9）。モデル駆動・Drive閾値・割り込みの共通エントリ。

        open_think=True（既定）なら「考えてから話す」（v1.8）。モデルが think-end 決断や
        制御トークンで「話す」を選んだ直後は open_think=False にする（思考を挟まず直ちに
        発話へ。再オープンすると発話冒頭が思考として消費されてしまう）。
        """
        self._log_autonomy("speech_start", reason)
        self._speech_segments = 0
        self._speech_segments_mark = self.segments_completed
        self._last_segment_emitted = False  # v1.16: 新セッションでは未配信から開始
        self._mode = "speaking"
        if open_think and self._native_thinking and not self._in_think_block:
            self._open_think_block()  # v1.8: 「考えてから話す」

    async def _ask_model_decision(self, purpose: str, options: list[str]) -> str:
        """モデル自身に行動を決断させる（v1.9・制約付きデコード）。

        決断プロンプトはエフェメラル（バッファには残さない）。Drive状態の要約を渡し、
        モデルが「今の気分」を踏まえて選べるようにする。選択は必ず options のいずれかに
        収束する（engine.generate_decision の制約付きデコード）。
        """
        d = self.drives
        prompt = _DECISION_PROMPTS[purpose].format(
            b=float(d.get("boredom", 0.0)),
            l=float(d.get("loneliness", 0.0)),
            f=float(d.get("fatigue", 0.0)),
        )
        context = list(self.buffer.items) + [prompt]
        t0 = time.monotonic()
        choice = await self.engine.generate_decision(context, options)
        dt = time.monotonic() - t0
        self.decisions_asked += 1
        self.decision_total_sec += dt
        self._log_autonomy("decision", f"{purpose}→{choice}")
        return choice

    def _consume_control_token(self, text: str) -> str | None:
        """生成トークンに制御トークン（C）が含まれていればアクション名（speak/wait/think）を返す。"""
        for token, action in self._control_token_map.items():
            if token in text:
                return action
        return None

    def _drain_external(self) -> int:
        """外部入力（割り込み・応答）を吸い上げ、バッファに反映し、外部依存の relief を発火する（v1.10）。

        - ①好奇心: 外部情報は唯一の解消手段。内部生成では決して解消されない。
        - ②loneliness: 応答が返る＝聞いてもらえた、でフル解消。
        外部イベントはセグメント単位と違い即時反映が必要（応答直後に再問いかけしない）ため、
        pending キューではなく Drive 値へ直接適用する（_relieve_drive）。戻り値は受信メッセージ数。
        """
        msgs = self.interrupts.drain()
        for msg in msgs:
            # v1.16: ユーザーからのメッセージには明示的に返答するよう指示する（internal）。
            # 従来は [interrupt] 接頭辞だけだったため、瞑想的なシステムプロンプトの下では
            # モデルが「黙る」を選び、対話が成立しなかった。
            # v1.17追補: 「思考の過程・分析・前置きは書かず、いきなり返答の言葉だけを」と明示。
            # 9Bのネイティブ思考モデルは <think> を閉じた後も番号付き思考を続け、それが
            # そのまま応答になる問題があった（実機検証）。返答の形式を限定して抑止する。
            self.buffer.append(
                f"【ユーザーからメッセージが届きました】「{msg}」\n"
                "【このメッセージに直接返答してください。思考の過程・分析・前置きは一切書かず、"
                "いきなり相手への返答の言葉だけを、日本語で短く自然に書いてください。】",
                n_tokens=32,
                internal=True,
            )
        if msgs:
            self.responses_received += 1
            self._relieve_drive("curiosity", self.relief.per_action_of("curiosity"))
            self._relieve_drive("loneliness", self.relief.per_action_of("loneliness"))
            # v1.16: 応答を受けたら現在のセッションを「応答セッション」にする（黙る禁止・
            # 最低1セグメントの発話を強制。対話が成立しない問題への対応）
            if self._force_response_speech:
                self._force_response = True
                self._last_segment_emitted = False
            self._log_autonomy("response_received", f"msg={msgs[0][:60]}")
        return len(msgs)

    def _relieve_drive(self, drive: str, amount: float) -> None:
        """Drive値を直接減らす（v1.10・外部イベント駆動の即時 relief）。"""
        amount = max(0.0, float(amount))
        if amount <= 0.0:
            return
        self.drives[drive] = max(0.0, float(self.drives.get(drive, 0.0)) - amount)

    async def _ask_question(self) -> None:
        """① 好奇心駆動の問いかけ（外部への働きかけ）。

        好奇心は内部生成では解消できないため、外部に質問を発し、応答待ち（awaiting）へ
        遷移する。質問の意図は質問指示文（internal）を文脈に残すことでモデルに伝える。
        1セグメント話し終えたら _run_scheduled が awaiting へ移す。
        """
        self.asks_asked += 1
        self._log_autonomy("ask_start", f"curiosity={self.drives.get('curiosity', 0.0):.2f}")
        await self._recall_memories()  # v1.12: 問いかけ（外部への働きかけ）の前に過去記憶を想起・注入
        self.buffer.append(
            self._question_instruction,
            n_tokens=len(self.engine.tokenize(self._question_instruction)),
            internal=True,
        )
        self._ask_mode = True
        self._speech_segments = 0
        self._speech_segments_mark = self.segments_completed
        self._mode = "speaking"
        if self._native_thinking and not self._in_think_block:
            self._open_think_block()  # 考えてから問いかける

    def _open_think_block(self) -> None:
        """v1.8: <think> ブロックを開いて思考フェーズに入る（ネイティブThinking）。

        開きタグはバッファに internal 要素として残す（モデルの文脈には必要。
        発話表示・relief・記憶からは除外される）。
        """
        tag = "<think>\n"
        self.buffer.append(tag, n_tokens=len(self.engine.tokenize(tag)), internal=True)
        self._in_think_block = True
        self._speech_think_tokens = 0

    async def step_once(self) -> tuple[str, float]:
        """生成ループ1反復（トークン生成→反映→割り込み→セグメント境界→圧縮チェック）。

        thinking_mode="native"（v1.8）の場合、<think> ブロック内のトークンは内言
        （internal）として扱い、発話・relief・記憶に一切影響させない。</think> を生成したら
        思考フェーズ終了で、以降のトークンは発話（回答）として通常のセグメント追跡に入る。
        """
        text, surprise = await self.engine.generate_next_token(self.buffer.items, self.drives)
        self.tokens_generated += 1

        if self._control_tokens:
            action = self._consume_control_token(text)
            if action is not None:
                # C: 制御トークンはバッファ・発話・セグメントに一切残さず、遷移要求として扱う
                self._pending_action = action
                self._log_autonomy("control_token", f"action={action}")
                await self._maybe_compress()
                return text, surprise

        if self._native_thinking:
            if self._in_think_block:
                # 思考フェーズ: 内言としてのみ扱う
                if "</think>" in text:
                    self._in_think_block = False
                    self._speech_think_tokens = 0
                    self.buffer.append(text, n_tokens=1, internal=True)
                    self.thoughts_generated += 1
                    self._last_token_internal = True
                    if (self._decide_on_think_end and self._schedule_cfg.get("enabled", False)
                            and not self._ask_mode and not self._force_response):
                        # A: 「考えてから話す」直後、話し続けるか黙るかをモデルが決断
                        #    （問いかけセッションは強制された外部行動のため決断で中断しない）
                        # v1.16: 応答セッション中（強制中）も黙る決断を発動させない
                        choice = await self._ask_model_decision("segment_end", ["続ける", "黙る"])
                        if choice == "黙る":
                            self._pending_action = "silence"
                    await self._maybe_compress()
                    return text, surprise
                # v1.8: 発話セッション中の思考にも上限を適用。超えたら強制クローズして回答へ
                # （Qwen3.5 は思考が長引きやすく、無制限だと回答に到達しない）
                self._speech_think_tokens += 1
                if self._speech_think_tokens >= self._thinking_max_tokens:
                    self._in_think_block = False
                    self._speech_think_tokens = 0
                    self.buffer.append(text, n_tokens=1, internal=True)
                    self.buffer.append("</think>", n_tokens=1, internal=True)
                    self.thoughts_generated += 2
                    self._last_token_internal = True
                    await self._maybe_compress()
                    return text, surprise
                self.buffer.append(text, n_tokens=1, internal=True)
                self.thoughts_generated += 1
                self._last_token_internal = True
                await self._maybe_compress()
                return text, surprise
            if "<think>" in text:
                # 回答中にモデルが再思考を開いた（自己修正等）→ 内言として扱う
                self._in_think_block = True
                self._speech_think_tokens = 0
                self.buffer.append(text, n_tokens=1, internal=True)
                self.thoughts_generated += 1
                self._last_token_internal = True
                await self._maybe_compress()
                return text, surprise
            if "</think>" in text:
                # 強制クローズ後の重複出力など: 境界タグは発話に漏らさない（v1.8）
                self.buffer.append(text, n_tokens=1, internal=True)
                self._last_token_internal = True
                await self._maybe_compress()
                return text, surprise
        self._last_token_internal = False
        self.buffer.append(text, n_tokens=1)

        # セグメント追跡（relief判定・記憶用にトークンIDとサプライズを保持）
        self.segment.texts.append(text)
        self.segment.token_ids.extend(await self.engine.tokenize_async(text))
        self.segment.surprises.append(surprise)

        # 外部入力（応答・割り込み）を反映し、外部依存の relief（好奇心・lonelinessフル）を発火
        self._drain_external()

        if self._is_boundary(text) or len(self.segment.texts) >= self._segment_max_tokens:
            await self._finalize_segment()

        await self._maybe_compress()
        return text, surprise

    # ------------------------------------------------------------------ #
    # 内部処理
    # ------------------------------------------------------------------ #
    async def _drive_loop(self) -> None:
        interval = self._update_interval_sec
        last = time.monotonic()
        while not self._stop:
            now = time.monotonic()
            dt = now - last
            last = now
            relief_delta = self.relief.step(dt, self.drives)
            self.dynamics.step(dt, relief_delta)  # self.drives を in-place 更新
            self.logger.drive_step(self.drives, now)
            await asyncio.sleep(interval)

    def _is_boundary(self, token_text: str) -> bool:
        return token_text in self._boundary_tokens

    def _evaluate_relief(self, avg_surprise: float) -> None:
        """①: セグメント単位で発火判定。条件を満たすDriveの per_action を1回分キューする。"""
        rel = self._relief_cfg
        if rel.get("boredom", {}).get("enabled", True) and avg_surprise >= self._surprise_threshold:
            self.relief.apply_per_action("boredom")
        if rel.get("fatigue", {}).get("enabled", True) and avg_surprise < self._surprise_threshold:
            self.relief.apply_per_action("fatigue")
        lon_cfg = rel.get("loneliness", {})
        if lon_cfg.get("enabled", True) and self._segment_matches_vocab("loneliness"):
            if "speak_relief" in lon_cfg:
                # v1.10 ②: 話すだけでは部分 relief。フル解消は応答（外部入力）が返ってから
                self.relief.apply_amount("loneliness", float(lon_cfg["speak_relief"]))
            else:
                self.relief.apply_per_action("loneliness")  # 旧仕様（後方互換）

    def _segment_matches_vocab(self, drive: str) -> bool:
        """③: セグメント内トークン列が対象Driveの語彙トークン列と部分列一致するか。"""
        for seq in self.vocab_map.get(drive, []):
            if _contains_subsequence(self.segment.token_ids, list(seq)):
                return True
        return False

    async def _finalize_segment(self) -> None:
        text = "".join(self.segment.texts).strip()
        if not text:
            self.segment.reset()
            return
        self.segments_completed += 1  # v1.7: 発話スケジューリングのセグメント完了検知用
        # v1.13: 想起メモリの反唱を除去（完全一致に限らず部分反唱も）。セグメント全体が
        # 反唱ならコミット・配信せず破棄する（ジャンク記憶化とジャンク質問ルーティングの防止）。
        stripped = self._strip_recall_echo(text)
        if not stripped.strip():
            self.segment.reset()
            return
        # v1.17: 注入した【…】指示ブロックの反唱・同形式の捏造指示文も除去。
        # v1.17追補: 番号付き思考・思考見出しのリークも除去（「1. まず、…」等）。
        # 指示文・思考リークのみのセグメント（例: 「【あなたの思考プロセスは明記しないでください。」
        # 「1. まず、ユーザーのメッセージを…」）はコミット・配信せず破棄する（ジャンク記憶の
        # 自己増幅ループの防止）。
        stripped = self._strip_instruction_junk(stripped)
        stripped = self._strip_reasoning_junk(stripped)
        if not stripped.strip():
            self._log_autonomy("junk_suppressed", "指示文・思考リークのみのセグメントを破棄")
            self.segment.reset()
            return
        text = stripped
        avg_surprise = (
            sum(self.segment.surprises) / len(self.segment.surprises) if self.segment.surprises else 0.0
        )
        self._evaluate_relief(avg_surprise)
        # B3: サプライズを記憶の重要度として使用。分類結果（EPISODIC/SEMANTIC/EMOTIONAL/…）をログ
        record = await self.memory.commit(text, dict(self.drives), importance=avg_surprise)
        self.logger.memory_commit(record.kind.value, record.importance, text)
        # v1.13: 発話・質問を外部へ配信（--interact のリアルタイム表示・実行エージェントのルーティング）
        self.output.emit("question" if self._ask_mode else "speech", text)
        # v1.16: 実際に配信されたセグメントを記録（応答セッションの強制解除判定に使用）
        self._last_segment_emitted = True
        self.segment.reset()

    async def _maybe_compress(self) -> None:
        if not self.buffer.is_over_threshold(self._context_window, self._compress_ratio):
            return
        result = await self.compressor.compress(self.buffer, self._context_window, self._compress_ratio)
        if result is not None:
            removed_text, removed_tokens, summary = result
            self.logger.compression_event(removed_tokens=removed_tokens, summary=summary[:200])
