"""記憶の想起（v1.12: retrieve→文脈注入）のテスト。

- 内言（manual / native）の前に過去の関連記憶が internal 要素として文脈に注入される
- 1想起セッションで1回だけ注入される（マーカー重複防止）
- 無効（enabled=false）・embedder未接続時は何もしない
- 想起は発話表示（spoken_content）に影響しない（internal）
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from lucina.core import LucinaCore
from lucina.memory.store import HierarchicalMemoryStore, InMemoryVectorStore, MemoryCompressor
from lucina.testing import build_mock_core, make_test_config


def _cfg_with_recall(*, enabled: bool = True, top_k: int = 3, max_tokens: int = 120) -> dict:
    return make_test_config(
        **{
            "memory.recall.enabled": enabled,
            "memory.recall.top_k": top_k,
            "memory.recall.max_tokens": max_tokens,
        }
    )


@pytest.mark.asyncio
async def test_recall_injects_memories_into_buffer(tmp_path) -> None:
    """想起が internal 要素としてバッファに注入され、発話表示には混ざらない。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    # 過去記憶を1件コミット（「寂しい」= loneliness 軸の記憶）
    await core.memory.commit("寂しい夜でした。", {"boredom": 0.5}, importance=0.5)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001

    assert core.buffer.contains("【あなたの過去の記憶の想起】")
    assert core.buffer.contains("寂しい夜でした")
    # internal 要素のみで構成されるため、発話（spoken）表示には出ない
    assert "寂しい夜でした" not in core.buffer.spoken_content()
    core.close()


@pytest.mark.asyncio
async def test_recall_once_per_session(tmp_path) -> None:
    """1想起セッションで1回だけ注入される（マーカーによる重複防止）。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("孤独を感じる日々。", {"loneliness": 0.7}, importance=0.6)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001
    await core._recall_memories()  # noqa: SLF001 - 2回目はマーカーでスキップ

    assert core.buffer.content().count("【あなたの過去の記憶の想起】") == 1
    core.close()


@pytest.mark.asyncio
async def test_recall_disabled_no_injection(tmp_path) -> None:
    """enabled=false なら注入されない。"""
    cfg = _cfg_with_recall(enabled=False)
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("楽しい思い出。", {"boredom": 0.2}, importance=0.5)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001

    assert not core.buffer.contains("【あなたの過去の記憶の想起】")
    core.close()


@pytest.mark.asyncio
async def test_recall_without_embedder_skipped(tmp_path) -> None:
    """embedder 未接続のストアなら何もしない（実装が落ちない）。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path, with_embedder=False)
    core.memory = HierarchicalMemoryStore(InMemoryVectorStore())  # embedder なし
    core.compressor = MemoryCompressor()

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001

    assert not core.buffer.contains("【あなたの過去の記憶の想起】")
    core.close()


@pytest.mark.asyncio
async def test_recall_before_inner_thought_manual(tmp_path) -> None:
    """manual 内言の生成前に想起が注入される（バッファに残る）。"""
    cfg = make_test_config(
        **{
            "memory.recall.enabled": True,
            "memory.recall.top_k": 3,
            "drive.scheduling.enabled": True,
            "drive.scheduling.thinking_mode": "manual",
            "drive.scheduling.inner_interval_sec": 0.0,
            "drive.scheduling.inner_max_tokens": 10,
        }
    )
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("月が綺麗な夜でした。", {"boredom": 0.3}, importance=0.5)

    core.seed_prompt("起動")
    task = asyncio.create_task(core.run(max_tokens=5, drive_loop=False))
    await asyncio.sleep(0.05)
    core.stop()
    await task

    # 内言が生成され、想起ブロックが文脈に残っている
    assert core.buffer.contains("【あなたの過去の記憶の想起】")
    core.close()


@pytest.mark.asyncio
async def test_recall_relevant_memory_ranked_first(tmp_path) -> None:
    """クエリ（現在の文脈）に最も関連する記憶が上位で返る（類似度ランキング）。"""
    cfg = _cfg_with_recall(top_k=3)
    core = build_mock_core(cfg, log_dir=tmp_path)
    # 関連記憶（loneliness 軸）と無関連記憶（fatigue 軸）をコミット
    await core.memory.commit("孤独で誰かに会いたい。", {"loneliness": 0.8}, importance=0.8)
    await core.memory.commit("よく眠った一日だった。", {"fatigue": 0.6}, importance=0.6)

    core.seed_prompt("起動")
    # 現在の文脈を loneliness 寄りにする
    core.buffer.append("寂しい 会いたい", n_tokens=3)
    await core._recall_memories()  # noqa: SLF001

    content = core.buffer.content()
    recall_block = content[content.index("【あなたの過去の記憶の想起】"):]
    assert recall_block.index("孤独で誰かに会いたい") < recall_block.index("よく眠った一日だった")
    core.close()


@pytest.mark.asyncio
async def test_recall_echo_suppressed(tmp_path) -> None:
    """エコー抑制: 反唱テキストを末尾から巻き戻しても、注入ブロック本体は文脈に残る。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("過去の思い出。", {"boredom": 0.4}, importance=0.5)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001
    block = core._last_recall_block  # noqa: SLF001
    assert block is not None
    n_before = len(core.buffer.items)

    # モデルが想起ブロックをそのまま反唱した状況を再現（思考トークンとして末尾に追記）
    core.buffer.append(block, n_tokens=10, internal=True)
    core.buffer.take_newest(1)  # 反唱トークンを巻き戻し

    # 反唱は除去され、注入ブロック（internal の記憶提供）は文脈に残っている
    assert len(core.buffer.items) == n_before
    assert core.buffer.contains("【あなたの過去の記憶の想起】")
    assert core.buffer.contains("過去の思い出")
    core.close()


@pytest.mark.asyncio
async def test_strip_partial_recall_echo(tmp_path) -> None:
    """v1.13: メモリ1行のみの部分反唱も除去される（完全ブロック一致に限らない）。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("新しいことを学びたいという気持ちが強い。", {"boredom": 0.4}, importance=0.5)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001

    # 完全ブロック反唱
    assert core._strip_recall_echo(core._last_recall_block) == ""  # noqa: SLF001
    # 部分反唱（メモリ1行のみ・プレフィックス付き）
    partial = "- 新しいことを学びたいという気持ちが強い。だから質問したいです。"
    assert core._strip_recall_echo(partial) == "だから質問したいです。"
    # 部分反唱（プレフィックスなし・実思考が続く）
    partial2 = "新しいことを学びたいという気持ちが強い。何を調べようか。"
    assert core._strip_recall_echo(partial2) == "何を調べようか。"
    # 反唱でない通常テキストは無変更
    assert core._strip_recall_echo("今日の天気はどうですか") == "今日の天気はどうですか"
    core.close()


@pytest.mark.asyncio
async def test_resets_between_sessions(tmp_path) -> None:
    """speech_end（セッション終了）でマーカーがリセットされ、次のセッションで再想起できる。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    await core.memory.commit("過去の記憶その一。", {"boredom": 0.4}, importance=0.5)
    await core.memory.commit("過去の記憶その二。", {"boredom": 0.4}, importance=0.5)

    core.seed_prompt("起動")
    await core._recall_memories()  # noqa: SLF001
    assert core.buffer.content().count("【あなたの過去の記憶の想起】") == 1

    # セッション終了相当（speech_end と同じリセット）
    core._recall_marker = False  # noqa: SLF001
    await core._recall_memories()  # noqa: SLF001

    assert core.buffer.content().count("【あなたの過去の記憶の想起】") == 2
    core.close()


@pytest.mark.asyncio
async def test_strip_instruction_junk(tmp_path) -> None:
    """v1.17: 注入した【…】指示ブロックの反唱・捏造指示文が発話テキストから除去される。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)

    # モデルが捏造した指示文（実機で発生: 「こんばんはー」への応答がこれになった）
    assert core._strip_instruction_junk("【あなたの思考プロセスは明記しないでください。") == ""  # noqa: SLF001
    # 注入した返答指示そのものの反唱
    echo = (
        "【ユーザーからメッセージが届きました】「こんばんはー」\n"
        "【あなたは今、このメッセージに返答してください。日本語で、短く、自然に。】"
    )
    assert core._strip_instruction_junk(echo) == ""  # noqa: SLF001
    # 文頭の指示反唱の後に実際の発話が続く場合は発話だけ残る
    mixed = "【あなたは今、このメッセージに返答してください。】こんばんは。今日はいい天気ですね。"
    assert core._strip_instruction_junk(mixed) == "こんばんは。今日はいい天気ですね。"  # noqa: SLF001
    # 通常の発話は無変更
    normal = "こんばんは。今日はいい天気ですね。"
    assert core._strip_instruction_junk(normal) == normal  # noqa: SLF001
    core.close()


@pytest.mark.asyncio
async def test_instruction_junk_not_committed_nor_emitted(tmp_path) -> None:
    """v1.17: 指示文のみのセグメントは記憶コミットも発話配信もされない（ジャンク記憶の自己増幅防止）。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    n0 = len(core.memory._vector_store)  # noqa: SLF001

    core.seed_prompt("起動")
    # モデルが捏造した指示文がセグメントとして完成した状況を再現
    core.segment.texts = ["【あなたの思考プロセスは明記しないでください。"]
    core.segment.surprises = [0.8]
    await core._finalize_segment()  # noqa: SLF001

    # 記憶にはコミットされない・発話として配信されない
    assert len(core.memory._vector_store) == n0  # noqa: SLF001
    emitted = core.output.drain()
    assert all(kind != "speech" for kind, _ in emitted)
    core.close()


@pytest.mark.asyncio
async def test_response_instruction_echo_not_committed(tmp_path) -> None:
    """v1.17: 注入した返答指示をモデルが反唱しても、ジャンク記憶にならない（実機バグの回帰防止）。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    n0 = len(core.memory._vector_store)  # noqa: SLF001

    core.seed_prompt("起動")
    echo = (
        "【ユーザーからメッセージが届きました】「こんばんはー」\n"
        "【あなたは今、このメッセージに返答してください。日本語で、短く、自然に。】"
    )
    core.segment.texts = [echo]
    core.segment.surprises = [0.9]
    await core._finalize_segment()  # noqa: SLF001

    assert len(core.memory._vector_store) == n0  # noqa: SLF001
    emitted = core.output.drain()
    assert all(kind != "speech" for kind, _ in emitted)
    core.close()


@pytest.mark.asyncio
async def test_strip_reasoning_junk(tmp_path) -> None:
    """v1.17追補: 番号付き思考・思考見出しのリークが発話テキストから除去される。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)

    # 実機で発生した「こんばんはー」への応答（番号付き思考リーク）
    junk = "1. まず、ユーザーのメッセージをどのような文脈で受け取ろうかと考えます。"
    assert core._strip_reasoning_junk(junk) == ""  # noqa: SLF001
    # 思考リークの後に実際の返答が続く場合は返答だけ残す
    mixed = "1. まず、ユーザーのメッセージをどのような文脈で受け取ろうかと考えます。\n\nこんばんは！今日はどうでしたか？"
    assert core._strip_reasoning_junk(mixed) == "こんばんは！今日はどうでしたか？"  # noqa: SLF001
    # 思考見出し形式
    assert core._strip_reasoning_junk("思考プロセス：\n1. 考えます。") == ""  # noqa: SLF001
    # 番号なしのメタ考察（会話について考えている文）も破棄
    assert core._strip_reasoning_junk("ユーザーのメッセージをどう扱うか、まず考えます。") == ""  # noqa: SLF001
    # 通常の発話は無変更
    normal = "こんばんは。今日はいい夜ですね。"
    assert core._strip_reasoning_junk(normal) == normal  # noqa: SLF001
    core.close()


@pytest.mark.asyncio
async def test_reasoning_junk_not_committed_nor_emitted(tmp_path) -> None:
    """v1.17追補: 番号付き思考リークのみのセグメントは記憶コミットも発話配信もされない。"""
    cfg = _cfg_with_recall()
    core = build_mock_core(cfg, log_dir=tmp_path)
    n0 = len(core.memory._vector_store)  # noqa: SLF001

    core.seed_prompt("起動")
    core.segment.texts = ["1. まず、ユーザーのメッセージをどのような文脈で受け取ろうかと考えます。"]
    core.segment.surprises = [0.8]
    await core._finalize_segment()  # noqa: SLF001

    assert len(core.memory._vector_store) == n0  # noqa: SLF001
    emitted = core.output.drain()
    assert all(kind != "speech" for kind, _ in emitted)
    core.close()
