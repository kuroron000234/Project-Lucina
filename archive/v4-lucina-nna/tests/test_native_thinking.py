"""ネイティブThinking捕捉のテスト（仕様書 v1.4 §5.6・v1.8）。

thinking_mode="native" で:
- <think>ブロック内のトークン（モデルのネイティブ思考）は内言（internal）として扱われ、
  発話表示・relief・記憶に影響しない。
- </think> を境に以降のトークンは発話（回答）としてセグメント追跡に入る。
- 思考が thinking_max_tokens を超えても閉じない場合は強制クローズされる。

ScriptedBackend が指定テキスト列を順に「生成」するため、実際のモデルなしで
ルーティングを決定論的に検証できる。
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lucina.core import LucinaCore  # noqa: F401
from lucina.inference.engine import InferenceEngine
from lucina.io.logging import StructuredLogger
from lucina.memory.store import HierarchicalMemoryStore, InMemoryVectorStore, MemoryCompressor
from lucina.testing import FakeSummarizer, MockTokenizer, make_test_config


class ScriptedBackend:
    """指定したテキスト列を順に生成する決定的バックエンド（ルーティング検証用）。

    next_token_logits は該当テキストに対応する ID に巨大なピークを立てるため、
    greedy サンプリング（make_test_config(sampling="greedy")）で必ずそのトークンが選ばれる。
    """

    def __init__(self, texts: list[str], tokenizer: MockTokenizer) -> None:
        self._texts = list(texts)
        self._tok = tokenizer
        self._step = 0
        self._ids: dict[int, str] = {}

    def _alloc(self, text: str) -> int:
        tid = self._tok.vocab_size() - 1 - len(self._ids)
        self._ids[tid] = text
        return tid

    def next_token_logits(self, context_text: str) -> np.ndarray:
        text = self._texts[min(self._step, len(self._texts) - 1)]
        self._step += 1
        tid = self._alloc(text)
        n = self._tok.vocab_size()
        logits = np.full(n, -100.0)
        logits[tid] = 100.0
        return logits

    def decode(self, token_id: int) -> str:
        tid = int(token_id)
        if tid in self._ids:
            return self._ids[tid]
        return self._tok.decode(tid)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def vocab_size(self) -> int:
        return self._tok.vocab_size()


def _build_native_core(tmp_path, texts: list[str], **sched_overrides) -> LucinaCore:
    cfg = make_test_config(sampling="greedy", log_dir=str(tmp_path))
    sched = cfg["drive"]["scheduling"]
    sched["enabled"] = True
    sched["thinking_mode"] = "native"
    sched["thinking_max_tokens"] = 3
    for key, value in sched_overrides.items():
        sched[key] = value
    tokenizer = MockTokenizer()
    backend = ScriptedBackend(texts, tokenizer)
    executor = ThreadPoolExecutor(max_workers=2)
    engine = InferenceEngine("mock", executor, backend=backend, vocab_map={}, config=cfg)
    logger = StructuredLogger(str(tmp_path))
    memory = HierarchicalMemoryStore(InMemoryVectorStore(), embedder=None)
    compressor = MemoryCompressor(FakeSummarizer())
    core = LucinaCore(cfg, engine, {}, memory=memory, compressor=compressor, logger=logger)
    core._executor = executor  # noqa: SLF001 - close() 用の参照
    return core


def _events(tmp_path) -> list[str]:
    return [json.loads(line)["event"] for line in (tmp_path / "autonomy.jsonl").read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# seed_prompt: ネイティブThinkingの開始位置
# --------------------------------------------------------------------------- #
async def test_seed_prompt_opens_think_block(tmp_path) -> None:
    """シード投入後に思考フェーズが開かれ、<think> タグは発話表示から除外される。"""
    core = _build_native_core(tmp_path, [])
    try:
        core.seed_prompt("起動")
        assert core._in_think_block  # noqa: SLF001
        assert "<think>" in core.buffer.content()      # モデルの文脈には残る
        # シード（外部入力）は internal: 発話表示には一切出ない（v1.8）
        assert core.buffer.spoken_content() == ""
        assert "<think>" not in core.buffer.spoken_content()  # noqa: SLF001
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# step_once: 思考→回答のルーティング
# --------------------------------------------------------------------------- #
async def test_step_once_routes_think_then_speech(tmp_path) -> None:
    """<think>内は内言（internal）、</think>以降は発話としてセグメント追跡に入る。"""
    core = _build_native_core(
        tmp_path, ["考え", "中", "</think>", "寂しい", "です", "。"]
    )
    try:
        core.seed_prompt("起動")
        results = []
        for _ in range(6):
            results.append(await core.step_once())
        assert [t for t, _ in results] == ["考え", "中", "</think>", "寂しい", "です", "。"]

        # 思考は internal（発話・セグメント・relief に入らない）
        assert core.thoughts_generated == 3          # 考え・中・</think>
        assert not core._in_think_block  # noqa: SLF001
        spoken = core.buffer.spoken_content()
        assert "考え" not in spoken and "<think>" not in spoken and "</think>" not in spoken
        assert spoken.endswith("寂しいです。")
        # 回答セグメントは境界（。）で最終化済み
        assert core.segments_completed == 1
        assert core.segment.texts == []
        # relief は回答セグメント（低サプライズ→fatigue休息）のみ発火。思考は関与しない
        assert "fatigue" in core.relief.pending
        assert "boredom" not in core.relief.pending
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# 内言セッション（_generate_inner_thought・native）
# --------------------------------------------------------------------------- #
async def test_native_inner_thought_captures_model_thinking(tmp_path) -> None:
    """内言セッションがモデルのネイティブ思考を捕捉し、発話・relief・記憶に影響しない。"""
    core = _build_native_core(
        tmp_path, ["静寂", "の", "中", "</think>"], thinking_max_tokens=10
    )
    try:
        core.seed_prompt("起動")
        await core._generate_inner_thought()  # noqa: SLF001
        assert not core._in_think_block  # noqa: SLF001
        assert core.buffer.content().endswith("</think>")  # 思考ブロックが閉じている
        assert core.buffer.spoken_content() == ""          # 内言は発話に影響しない
        assert core.segment.texts == []
        assert core.relief.pending == {}
        assert "inner_thought" in _events(tmp_path)        # autonomy ログに記録される
    finally:
        core.close()


async def test_speech_phase_thinking_capped(tmp_path) -> None:
    """発話セッション中の思考も thinking_max_tokens で打ち切られ、回答に到達する（v1.8）。"""
    core = _build_native_core(tmp_path, ["思", "考", "中", "で", "す", "。"], thinking_max_tokens=3)
    try:
        core.seed_prompt("起動")
        core._open_think_block()  # noqa: SLF001 - 発話セッション開始
        for _ in range(6):
            await core.step_once()
        # 思考3トークン（思・考・中）でキャップ → </think> 強制クローズ → 回答へ
        assert core.thoughts_generated == 4   # 思・考・中・</think>(強制)
        assert not core._in_think_block  # noqa: SLF001
        assert "中</think>" in core.buffer.content()   # 強制クローズ済み
        spoken = core.buffer.spoken_content()
        assert "思" not in spoken and "<think>" not in spoken
        assert spoken.endswith("です。")                  # 回答は発話として残る
        assert core.segments_completed == 1
    finally:
        core.close()


async def test_forced_close_on_cap(tmp_path) -> None:
    """思考が thinking_max_tokens を超えても閉じない場合、強制クローズで文脈を整形式に保つ。"""
    core = _build_native_core(tmp_path, ["長い", "思考", "続く", "まだ", "続く"])
    try:
        core.seed_prompt("起動")
        await core._generate_inner_thought()  # noqa: SLF001
        assert core.thoughts_generated == 3   # 上限3トークンで停止
        assert not core._in_think_block  # noqa: SLF001
        assert core.buffer.content().endswith("</think>")  # 強制クローズ済み
        assert core.buffer.spoken_content() == ""
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# 発話セッション: 「考えてから話す」
# --------------------------------------------------------------------------- #
async def test_stray_think_end_never_leaks_to_speech(tmp_path) -> None:
    """強制クローズ後の重複 </think> 出力は発話（spoken）に漏れない（v1.8）。"""
    core = _build_native_core(
        tmp_path, ["思", "</think>", "</think>", "答", "え", "。"], thinking_max_tokens=2
    )
    try:
        core.seed_prompt("起動")
        core._open_think_block()  # noqa: SLF001
        for _ in range(6):
            await core.step_once()
        spoken = core.buffer.spoken_content()
        assert "<think>" not in spoken and "</think>" not in spoken
        assert spoken == "答え。"
    finally:
        core.close()


async def test_speech_session_thinks_then_speaks(tmp_path) -> None:
    """speech_start で思考ブロックが開かれ、思考は内言・回答は発話になる。"""
    core = _build_native_core(
        tmp_path, ["思", "考", "</think>", "答", "え", "。"],
        mode="thinking",
        speak_start_boredom=0.0,      # 直ちに自発的に発話開始
        speak_start_loneliness=1.0,
        speak_block_fatigue=1.0,
        quiet_on_fatigue=1.0,
        max_speech_segments=1,
        inner_interval_sec=1000.0,    # 内言セッションは挟まない
    )
    try:
        core.seed_prompt("起動")
        await core.run(max_tokens=6, drive_loop=False)
        # 発話セッション内の思考は内言として扱われ、発話には回答のみが残る
        spoken = core.buffer.spoken_content()
        assert "思" not in spoken and "<think>" not in spoken
        assert spoken.endswith("答え。")
        assert core.thoughts_generated == 3   # 思・考・</think>
        assert core.speech_tokens == 3        # 答・え・。
        assert core.segments_completed == 1
        assert "speech_start" in _events(tmp_path)
        assert "speech_end" in _events(tmp_path)
    finally:
        core.close()
