"""モデル駆動スケジューリングのテスト（仕様書 v1.4 §5.6・v1.9）。

v1.9 で「モデル自身がいつ話す・黙る・考えるかを選ぶ」3方式を追加した:
- A: 境界決断（decide_on_think_end / decide_on_segment_end）— 思考ブロック終了時・
  発話セグメント境界でモデルが「話す/黙る/さらに考える」「続ける/黙る」を決断。
- B: 待機中 introspection（introspection_sec）— 待機中も数秒ごとにモデルが
  「待機/内言/発話」を決断し、自発的に話し始める瞬間を選べる。
- C: 制御トークン（control_tokens）— 生成中にモデルが <|lucina_speak|> 等を出力し、
  ランタイムがそれを解釈して遷移する。

共通基盤: engine.generate_decision の制約付きデコード（選択肢のトークン列で
prefix-trie を組み、ロジットをマスクして必ず選択肢の1つに収束させる）。

Drive閾値は安全弁として残り、モデルが「待機」を選び続けても退屈の限界
（speak_override_boredom）で強制発話する（デッドロック防止）。
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
    """指定したテキスト列を順に生成する決定的バックエンド（ルーティング検証用）。"""

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


class TokenSequenceBackend(ScriptedBackend):
    """指定したトークンID列を順に吐くバックエンド（制約付きデコード検証用）。"""

    def __init__(self, token_ids: list[int], tokenizer: MockTokenizer) -> None:
        super().__init__([], tokenizer)
        self._token_ids = list(token_ids)
        self._step = 0

    def next_token_logits(self, context_text: str) -> np.ndarray:
        tid = self._token_ids[min(self._step, len(self._token_ids) - 1)]
        self._step += 1
        n = self._tok.vocab_size()
        logits = np.full(n, -100.0)
        logits[tid] = 100.0
        return logits


def _build_core(tmp_path, texts: list[str], **sched_overrides) -> LucinaCore:
    cfg = make_test_config(sampling="greedy", log_dir=str(tmp_path))
    sched = cfg["drive"]["scheduling"]
    sched["enabled"] = True
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


def _inject_decisions(core: LucinaCore, script: list[str]) -> None:
    """engine.generate_decision を決定的なスクリプトに差し替える（ルーティング検証用）。

    スクリプトが尽きた後は最後の選択を繰り返す（ScriptedBackend と同じ「末尾繰り返し」
    で、後続の決断が意図しない選択肢に変わらないようにする）。
    """
    queue = list(script)

    async def fake_decision(context: list[str], options: list[str]) -> str:
        if queue:
            return queue.pop(0)
        return script[-1] if script else options[0]

    core.engine.generate_decision = fake_decision  # type: ignore[method-assign]


def _events(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "autonomy.jsonl").read_text(encoding="utf-8").splitlines()]


def _event_names(tmp_path) -> list[str]:
    return [d["event"] for d in _events(tmp_path)]


# --------------------------------------------------------------------------- #
# 共通基盤: 制約付きデコード（engine.generate_decision）
# --------------------------------------------------------------------------- #
async def test_generate_decision_picks_exact_option(tmp_path) -> None:
    """バックエンドが該当オプションのトークン列を吐けば、そのオプションに必ず収束する。"""
    tok = MockTokenizer()
    ids = tok.encode("待機")
    backend = TokenSequenceBackend(ids, tok)
    executor = ThreadPoolExecutor(max_workers=1)
    engine = InferenceEngine("mock", executor, backend=backend, vocab_map={}, config=make_test_config(log_dir=str(tmp_path)))
    try:
        choice = await engine.generate_decision(["文脈"], ["待機", "内言", "発話"])
        assert choice == "待機"
    finally:
        engine.close()
        executor.shutdown(wait=False)


async def test_generate_decision_never_leaves_options(tmp_path) -> None:
    """バックエンドが選択肢外のトークンを吐こうとしても、選択肢のいずれかに収束する。"""
    tok = MockTokenizer()
    garbage = TokenSequenceBackend([tok.vocab_size() - 5], tok)
    executor = ThreadPoolExecutor(max_workers=1)
    engine = InferenceEngine("mock", executor, backend=garbage, vocab_map={}, config=make_test_config(log_dir=str(tmp_path)))
    try:
        choice = await engine.generate_decision(["文脈"], ["待機", "内言", "発話"])
        assert choice in ("待機", "内言", "発話")
    finally:
        engine.close()
        executor.shutdown(wait=False)


# --------------------------------------------------------------------------- #
# B: 待機中 introspection（モデルが「待機/内言/発話」を決断）
# --------------------------------------------------------------------------- #
async def test_introspection_model_chooses_speak_without_drive_threshold(tmp_path) -> None:
    """Drive閾値が発話しない設定でも、モデルが「発話」を選べば自発的に話し始める。"""
    core = _build_core(
        tmp_path, ["</think>", "答", "え", "。"],
        mode="thinking",
        introspection_sec=0.01,
        inner_interval_sec=1000.0,     # タイマー内言は挟まない
        speak_start_boredom=1.0,       # Drive閾値では決して発話しない
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,    # 安全弁も発動させない
        speak_block_fatigue=1.0,
        quiet_on_fatigue=1.0,
        thinking_mode="native",
        thinking_max_tokens=3,
    )
    _inject_decisions(core, ["発話"])
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=10, drive_loop=False))
        await asyncio.sleep(0.5)  # introspection→発話→発話セッション の遷移が進むのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path)
        starts = [e for e in events if e["event"] == "speech_start"]
        assert starts, "モデルの決断で発話が開始されていない"
        assert "introspection" in starts[0]["reason"]
        assert core.speech_tokens >= 3
        assert "答え" in core.buffer.spoken_content()
    finally:
        core.close()


async def test_introspection_model_chooses_think(tmp_path) -> None:
    """モデルが「内言」を選べば、タイマーではなく意思で内言セッションが走る。"""
    core = _build_core(
        tmp_path, ["考", "え", "。"],
        mode="thinking",
        introspection_sec=0.01,
        inner_interval_sec=1000.0,     # タイマー内言は無効
        speak_start_boredom=1.0,
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        thinking_mode="manual",
        inner_max_tokens=6,
    )
    _inject_decisions(core, ["内言", "待機"])
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=10, drive_loop=False))
        await asyncio.sleep(0.5)  # introspection→内言→待機 の遷移が進むのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path)
        decisions = [e for e in events if e["event"] == "decision"]
        assert any("introspection→内言" in e["reason"] for e in decisions)
        assert any(e["event"] == "inner_thought" for e in events)
        # 「待機」を選んだ後は Drive 閾値も発話しないため、発話は起きない
        assert not any(e["event"] == "speech_start" for e in events)
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# A: 境界決断（think_end / segment_end）
# --------------------------------------------------------------------------- #
async def test_think_end_model_chooses_silence(tmp_path) -> None:
    """思考ブロック終了時にモデルが「黙る」を選べば、発話せず沈黙を続ける。"""
    core = _build_core(
        tmp_path, ["考", "え", "</think>"],
        mode="thinking",
        introspection_sec=0.0,
        inner_interval_sec=0.01,
        decide_on_think_end=True,
        speak_start_boredom=1.0,
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        thinking_mode="native",
        thinking_max_tokens=5,
    )
    _inject_decisions(core, ["黙る"])
    try:
        core.seed_prompt("起動")
        await core.run(max_tokens=10, drive_loop=False)
        events = _events(tmp_path)
        assert any("think_end→黙る" in e["reason"] for e in events if e["event"] == "decision")
        assert not any(e["event"] == "speech_start" for e in events)
        assert core.speech_tokens == 0
        assert core.buffer.spoken_content() == ""
    finally:
        core.close()


async def test_think_end_model_chooses_speak(tmp_path) -> None:
    """思考ブロック終了時にモデルが「話す」を選べば、そのまま自発的に発話する。"""
    core = _build_core(
        tmp_path, ["考", "え", "</think>", "答", "え", "。"],
        mode="thinking",
        introspection_sec=0.0,
        inner_interval_sec=0.01,
        decide_on_think_end=True,
        speak_start_boredom=1.0,
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        quiet_on_fatigue=1.0,
        max_speech_segments=10,
        thinking_mode="native",
        thinking_max_tokens=5,
    )
    _inject_decisions(core, ["話す"])
    try:
        core.seed_prompt("起動")
        await core.run(max_tokens=12, drive_loop=False)
        events = _events(tmp_path)
        assert any("think_end→話す" in e["reason"] for e in events if e["event"] == "decision")
        starts = [e for e in events if e["event"] == "speech_start"]
        assert starts and "思考終了後" in starts[0]["reason"]
        assert "答え" in core.buffer.spoken_content()
    finally:
        core.close()


async def test_segment_end_model_chooses_silence(tmp_path) -> None:
    """発話セグメント境界でモデルが「黙る」を選べば、そこで話すのをやめる。"""
    core = _build_core(
        tmp_path, ["こ", "ん", "にち", "は", "。"],
        mode="speaking",               # 発話から開始
        introspection_sec=0.0,
        inner_interval_sec=1000.0,
        decide_on_segment_end=True,
        speak_start_boredom=1.0,       # 沈黙後は再発話しない
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        quiet_on_fatigue=1.0,
        max_speech_segments=10,
        thinking_mode="manual",
    )
    _inject_decisions(core, ["黙る"])
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=20, drive_loop=False))
        await asyncio.sleep(0.5)  # 発話→セグメント境界→黙る の遷移が進むのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path)
        ends = [e for e in events if e["event"] == "speech_end"]
        assert ends and "セグメント境界" in ends[0]["reason"]
        assert any("segment_end→黙る" in e["reason"] for e in events if e["event"] == "decision")
        # 1セグメントだけ話して黙った（モデルの意思で打ち切られた）
        assert core.segments_completed == 1
        assert core.buffer.spoken_content() == "こんにちは。"
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# C: 制御トークン（<|lucina_speak|> / <|lucina_wait|> / <|lucina_think|>）
# --------------------------------------------------------------------------- #
async def test_control_token_wait_stops_speech(tmp_path) -> None:
    """発話中にモデルが <|lucina_wait|> を出力すると、自発的に沈黙する。トークンは発話に漏れない。"""
    core = _build_core(
        tmp_path, ["こ", "ん", "にち", "は", "。", "<|lucina_wait|>"],
        mode="speaking",
        introspection_sec=0.0,
        inner_interval_sec=1000.0,
        control_tokens=True,
        speak_start_boredom=1.0,
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        quiet_on_fatigue=1.0,
        max_speech_segments=10,
        thinking_mode="manual",
    )
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=20, drive_loop=False))
        await asyncio.sleep(0.5)  # 発話→制御トークン→沈黙 の遷移が進むのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path)
        assert any(e["event"] == "control_token" and e["reason"] == "action=wait" for e in events)
        ends = [e for e in events if e["event"] == "speech_end"]
        assert ends and "制御トークン" in ends[0]["reason"]
        assert "<|lucina_wait|>" not in core.buffer.spoken_content()
        assert core.buffer.spoken_content() == "こんにちは。"
    finally:
        core.close()


async def test_control_token_speak_in_inner_thought_starts_speech(tmp_path) -> None:
    """待機中の内言セッションでモデルが <|lucina_speak|> を出力すると、自発的に発話する。"""
    core = _build_core(
        tmp_path, ["<|lucina_speak|>"],
        mode="thinking",
        introspection_sec=0.0,
        inner_interval_sec=0.01,       # タイマーで内言セッションを回す
        control_tokens=True,
        speak_start_boredom=1.0,       # Drive閾値では発話しない
        speak_start_loneliness=1.0,
        speak_override_boredom=1.0,
        thinking_mode="manual",
        inner_max_tokens=10,
    )
    try:
        core.seed_prompt("起動")
        await core.run(max_tokens=10, drive_loop=False)
        events = _events(tmp_path)
        assert any(e["event"] == "control_token" and e["reason"] == "action=speak" for e in events)
        starts = [e for e in events if e["event"] == "speech_start"]
        assert starts and "制御トークン" in starts[0]["reason"]
        assert "<|lucina_speak|>" not in core.buffer.spoken_content()
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# 安全弁: モデルが「待機」を選び続けても退屈の限界で強制発話
# --------------------------------------------------------------------------- #
async def test_drive_override_rescues_from_model_silence(tmp_path) -> None:
    """モデルが何度も「待機」を選んでも、退屈の限界（安全弁）で発話が強制される。"""
    core = _build_core(
        tmp_path, ["答", "え", "。"],
        mode="thinking",
        introspection_sec=0.05,
        inner_interval_sec=1000.0,
        idle_boredom_rate=0.5,         # 待機中に退屈が溜まる（安全弁が発動するまで）
        speak_start_boredom=0.5,       # 退屈が溜まったら話したくなる（動機）
        speak_start_loneliness=1.0,
        speak_override_boredom=0.5,    # 安全弁: 退屈の限界
        speak_block_fatigue=1.0,
        quiet_on_fatigue=1.0,
        thinking_mode="manual",
    )
    _inject_decisions(core, ["待機"])
    try:
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=10, drive_loop=False))
        await asyncio.sleep(3.0)  # 待機決断を重ねた後に退屈が限界へ達するのを待つ
        core.stop()
        await run_task
        events = _events(tmp_path)
        decisions = [e for e in events if e["event"] == "decision"]
        assert any("introspection→待機" in e["reason"] for e in decisions), "モデルは待機を選んでいる"
        starts = [e for e in events if e["event"] == "speech_start"]
        assert starts, "モデルが待機を選び続けても安全弁で発話すべき"
        assert "boredom=" in starts[0]["reason"]
    finally:
        core.close()
