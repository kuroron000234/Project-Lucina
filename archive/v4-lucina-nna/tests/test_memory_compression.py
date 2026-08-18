"""メモリ圧縮・EMOTIONAL強制タグのテスト（仕様書 v1.4 §6タスク4・6）。"""

from __future__ import annotations

import asyncio

import pytest

from lucina.memory.schema import MemoryKind
from lucina.memory.store import HierarchicalMemoryStore, InMemoryVectorStore, MemoryCompressor
from lucina.memory.working_buffer import WorkingBuffer
from lucina.testing import FakeSummarizer, build_mock_core, make_test_config


def _fill_buffer(buffer: WorkingBuffer, n_items: int, tokens_per_item: int = 5) -> None:
    for i in range(n_items):
        buffer.append(f"内容{i}", tokens_per_item)


def test_compression_triggers_at_threshold() -> None:
    """タスク4 DoD: コンテキスト上限に近づいたら例外なく圧縮がトリガーされる。"""
    buf = WorkingBuffer()
    _fill_buffer(buf, 80, tokens_per_item=5)  # 400 tokens
    assert buf.is_over_threshold(context_window=256, ratio=0.75)  # 閾値 192
    compressor = MemoryCompressor(FakeSummarizer())
    result = asyncio.run(compressor.compress(buf, context_window=256, ratio=0.75))
    assert result is not None
    removed_text, removed_tokens, summary = result
    assert removed_tokens > 0
    assert buf.token_count <= 192
    assert buf.content().startswith("[要約]")  # 要約が先頭に挿入される


def test_compression_without_summarizer_no_overflow() -> None:
    """要約器未設定でもオーバーフローしない（警告ログ付きで破棄）。"""
    buf = WorkingBuffer()
    _fill_buffer(buf, 100, tokens_per_item=10)  # 1000 tokens
    compressor = MemoryCompressor(None)
    asyncio.run(compressor.compress(buf, context_window=256, ratio=0.75))
    assert buf.token_count <= 192


def test_no_compression_under_threshold() -> None:
    buf = WorkingBuffer()
    _fill_buffer(buf, 10, tokens_per_item=5)  # 50 tokens < 192
    compressor = MemoryCompressor(FakeSummarizer())
    assert asyncio.run(compressor.compress(buf, 256, 0.75)) is None
    assert buf.token_count == 50


def test_emotional_forced_on_large_drive_delta() -> None:
    """タスク6 DoD: Drive変化量 ±0.3 以上で強制EMOTIONALタグが付与される。"""

    async def go() -> list[MemoryKind]:
        store = HierarchicalMemoryStore(InMemoryVectorStore())
        r1 = await store.commit("初回", {"boredom": 0.1})
        r2 = await store.commit("大きな変化", {"boredom": 0.5})   # delta 0.4 >= 0.3
        r3 = await store.commit("小さな変化", {"boredom": 0.55})  # delta 0.05
        return [r1.kind, r2.kind, r3.kind]

    kinds = asyncio.run(go())
    assert kinds[0] == MemoryKind.SEMANTIC
    assert kinds[1] == MemoryKind.EMOTIONAL
    assert kinds[2] == MemoryKind.SEMANTIC


def test_retrieve_by_kind_filter() -> None:
    async def go() -> None:
        store = HierarchicalMemoryStore(InMemoryVectorStore())
        await store.commit("知識1", {"boredom": 0.1})
        r = await store.commit("感情的な体験", {"boredom": 0.9})  # delta 0.8 → EMOTIONAL
        assert r.kind == MemoryKind.EMOTIONAL

    asyncio.run(go())


def test_core_compression_via_steps(tmp_path) -> None:
    """コア統合: 小さいコンテキストで step を回すと圧縮イベントが構造化ログに残る。"""
    config = make_test_config(context_window=32, log_dir=str(tmp_path))
    core = build_mock_core(config, log_dir=str(tmp_path))

    async def go() -> None:
        for _ in range(300):
            await core.step_once()

    asyncio.run(go())
    core.close()
    comp_log = tmp_path / "compression.jsonl"
    assert comp_log.exists()
    lines = comp_log.read_text(encoding="utf-8").splitlines()
    assert any("removed_tokens" in line for line in lines)
