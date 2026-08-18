"""外部への働きかけの必要性（v1.10）のテスト。

v1.9 までの Lucina は「内部の欲求（Drive）を内部の出力で解消できる」ため、外部に
働きかける必要性が構造的に発生しなかった。v1.10 では「内部では解消できない欲求」を導入する:

- ①好奇心（curiosity）: 時間で自然上昇。relief は**外部入力（応答）でのみ**発火。
  閾値を超えるとモデルが外部に質問を発する（働きかけ）→ 応答待ち（awaiting）へ。
- ②応答依存 loneliness: 話すだけでは**部分 relief**（speak_relief）。応答が返って初めて
  フル解消 → 無視されると寂しさが蓄積し、届く言葉が必然化する。
- ③応答待ち（awaiting）: 質問後は生成を止めて応答を待つ。応答で解除・タイムアウトで待機解除。
"""

from __future__ import annotations

import asyncio
import json

from lucina.core import LucinaCore
from lucina.testing import build_mock_core, make_test_config


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _names(path) -> list[str]:
    return [e["event"] for e in _events(path)]


# --------------------------------------------------------------------------- #
# ① 好奇心: 内部生成では解消されない
# --------------------------------------------------------------------------- #
async def test_curiosity_not_relieved_by_speech(tmp_path) -> None:
    """発話（内部生成）では好奇心は解消されない。relief 源は外部入力のみ。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.curiosity_ask_threshold": 1.0,  # 問いかけはさせない
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.5
        # 発話セグメントを1つ完了させる（relief 判定が走る）
        core.segment.texts = ["こんにちは"]
        core.segment.token_ids = [1, 2]
        core.segment.surprises = [0.5]
        await core._finalize_segment()  # noqa: SLF001
        assert core.drives["curiosity"] == 0.5, "発話で好奇心は解消されてはいけない"
    finally:
        core.close()


async def test_curiosity_accumulates_over_time(tmp_path) -> None:
    """新しい情報が無い環境では好奇心が時間とともに蓄積する。"""
    cfg = make_test_config(log_dir=str(tmp_path))
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.05
        core.dynamics.step(60.0, {})  # 60秒の経過（外部情報なし）
        assert core.drives["curiosity"] > 0.05, "好奇心は時間で蓄積すべき"
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# ① + ③: 好奇心閾値 → 問いかけ → 応答待ち
# --------------------------------------------------------------------------- #
async def test_curiosity_ask_enters_awaiting(tmp_path) -> None:
    """好奇心が閾値を超えると質問を発し、応答待ち（awaiting）に入る。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.introspection_sec": 0.0,
        "drive.scheduling.inner_interval_sec": 1000.0,
        "drive.scheduling.speak_start_boredom": 1.0,   # Drive発話はさせない（好奇心が唯一の動機）
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_override_boredom": 1.0,
        "drive.scheduling.curiosity_ask_threshold": 0.5,
        "drive.scheduling.await_timeout_sec": 100.0,
        "drive.scheduling.thinking_mode": "manual",
        "drive.relief.segment.max_tokens": 4,          # モックは文末記号を出しにくいため強制区切り
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.9
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=50, drive_loop=False))
        await asyncio.sleep(0.5)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "ask_start" in names
        assert "await_start" in names
        assert core.mode == "awaiting"
        assert core.buffer.spoken_content() != "", "質問は発話として外部へ出ているべき"
    finally:
        core.close()


async def test_ask_survives_think_end_decision(tmp_path) -> None:
    """問いかけは強制された外部行動のため、think-end 決断（A）で中断されない（v1.10バグ修正）。

    修正前: _ask_question が思考ブロックを開き、decide_on_think_end=true でモデルが
    「黙る」を選ぶと質問が発話される前に abort され、await_start に到達しなかった。
    """
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.introspection_sec": 0.0,
        "drive.scheduling.inner_interval_sec": 1000.0,
        "drive.scheduling.speak_start_boredom": 1.0,
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_override_boredom": 1.0,
        "drive.scheduling.curiosity_ask_threshold": 0.5,
        "drive.scheduling.await_timeout_sec": 100.0,
        "drive.scheduling.decide_on_think_end": True,   # これが有効でも中断されないこと
        "drive.scheduling.decide_on_segment_end": True,
        "drive.scheduling.thinking_mode": "native",
        "drive.scheduling.thinking_max_tokens": 3,
        "drive.relief.segment.max_tokens": 4,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.9
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=50, drive_loop=False))
        await asyncio.sleep(0.5)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "ask_start" in names
        assert "await_start" in names, "問いかけが think-end 決断で中断されてはいけない"
        assert core.buffer.spoken_content() != "", "質問は発話として外部へ出ているべき"
    finally:
        core.close()


async def test_awaiting_response_relieves_curiosity(tmp_path) -> None:
    """応答待ち中に外部応答が届くと、好奇心・寂しさがフル解消され待機解除される。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.introspection_sec": 0.0,
        "drive.scheduling.inner_interval_sec": 1000.0,
        "drive.scheduling.speak_start_boredom": 1.0,
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_override_boredom": 1.0,
        "drive.scheduling.curiosity_ask_threshold": 0.5,
        "drive.scheduling.await_timeout_sec": 100.0,
        "drive.scheduling.thinking_mode": "manual",
        "drive.relief.segment.max_tokens": 4,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.9
        core.seed_prompt("起動")
        # drive_loop=True: relief が 0.1s で消費され、好奇心が閾値未満に下がって再問いかけしないことを確認する
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=True))
        await asyncio.sleep(0.3)
        core.interrupts.inject("あなたの質問に答えます。")
        await asyncio.sleep(0.5)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "response_received" in names
        assert core.drives["curiosity"] < 0.5, "外部応答で好奇心が閾値未満に relief されるべき"
        assert core.mode == "thinking", "応答後は待機解除され、再問いかけしない"
        assert names.count("await_start") == 1, "好奇心が解消されたので2回目の問いかけは起きない"
    finally:
        core.close()


async def test_awaiting_timeout_releases_with_partial_relief(tmp_path) -> None:
    """応答が無いままタイムアウトすると待機解除され、好奇心は半分だけ解消（give-up）される。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.introspection_sec": 0.0,
        "drive.scheduling.inner_interval_sec": 1000.0,
        "drive.scheduling.speak_start_boredom": 1.0,
        "drive.scheduling.speak_start_loneliness": 1.0,
        "drive.scheduling.speak_override_boredom": 1.0,
        "drive.scheduling.curiosity_ask_threshold": 0.5,
        "drive.scheduling.await_timeout_sec": 0.3,
        "drive.scheduling.thinking_mode": "manual",
        "drive.relief.segment.max_tokens": 4,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["curiosity"] = 0.9
        core.seed_prompt("起動")
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=True))
        await asyncio.sleep(1.2)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "await_timeout" in names
        # タイムアウトで好奇心は半減（0.9→0.45）→ 閾値0.5未満で即時再問いかけしない
        assert core.drives["curiosity"] <= 0.5
        assert core.mode == "thinking"
        assert names.count("await_start") == 1
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# ② 応答依存 loneliness: 話すだけでは部分 relief
# --------------------------------------------------------------------------- #
async def test_speaking_gives_partial_loneliness_relief(tmp_path) -> None:
    """loneliness語彙を話しても部分 relief（speak_relief）のみ。フル解消は応答待ち。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.relief.loneliness.speak_relief": 0.2,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.drives["loneliness"] = 0.8
        core.segment.texts = ["寂しい"]
        core.segment.token_ids = core.engine.tokenize("寂しい")  # 語彙マップのトークン列と一致
        core.segment.surprises = [0.3]
        await core._finalize_segment()  # noqa: SLF001
        assert core.relief.pending.get("loneliness", 0.0) == 0.2, "話すだけでは部分 relief のみ"
        assert core.relief.pending.get("loneliness", 0.0) < 0.6, "フル relief は応答待ち"
    finally:
        core.close()


async def test_loneliness_needs_response_for_full_relief(tmp_path) -> None:
    """話す（部分0.2）→ 応答（フル0.6）の順で解消。応答が無い環境では寂しさが残り続ける。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.relief.loneliness.speak_relief": 0.2,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        # 話す（部分 relief）
        core.drives["loneliness"] = 0.5
        core.segment.texts = ["寂しい"]
        core.segment.token_ids = core.engine.tokenize("寂しい")
        core.segment.surprises = [0.3]
        await core._finalize_segment()  # noqa: SLF001
        core.dynamics.step(0.0, core.relief.step(0.0, core.drives))  # relief を消費
        assert core.drives["loneliness"] == 0.3, "話すだけでは 0.5-0.2=0.3 が残る（解消されない）"

        # 応答（外部入力）でフル relief（bind してから inject する: ループ未起動のため）
        core.interrupts.bind()
        core.interrupts.inject("私も寂しかった。")
        await asyncio.sleep(0.05)  # call_soon_threadsafe のキュー反映を待つ
        n = core._drain_external()  # noqa: SLF001
        assert n == 1
        assert core.drives["loneliness"] == 0.0, "応答が返って初めて完全に解消される（外部reliefは即時適用）"
    finally:
        core.close()


# --------------------------------------------------------------------------- #
# v1.16: 応答セッションの強制発話（対話が成立しない問題への対応）
# --------------------------------------------------------------------------- #
def _run_speaking_with_force(tmp_path, *, force: bool, ask_mode: bool = False) -> LucinaCore:
    """speaking セッションを force_response 設定付きで構築し、core を返す（後処理は呼び出し側）。"""
    cfg = make_test_config(log_dir=str(tmp_path), **{
        "drive.scheduling.enabled": True,
        "drive.scheduling.mode": "speaking",  # _run_scheduled が起動時にモードを config 値で上書きするため
        "drive.scheduling.thinking_mode": "manual",
        "drive.relief.segment.max_tokens": 3,   # モックは文末記号を出しにくいため強制区切り
        "drive.scheduling.force_response_speech": force,
        "drive.scheduling.decide_on_segment_end": False,
        "drive.scheduling.await_timeout_sec": 0.3,
    })
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    core.seed_prompt("起動")
    core._ask_mode = ask_mode
    if force:
        # 外部からの応答を受けた状態（_drain_external が立てるのと同値）を模擬
        core._force_response = True
        core._last_segment_emitted = False
    core._start_speaking("テストセッション")
    return core


async def test_response_injects_reply_instruction(tmp_path) -> None:
    """v1.16: 応答受信時に「返答せよ」という明示指示（internal）がバッファへ注入され、
    force_response が有効になる（モデルに応答を促す・黙る選択を抑止）。"""
    cfg = make_test_config(log_dir=str(tmp_path))
    core = build_mock_core(cfg, token_delay_ms=0.0, log_dir=str(tmp_path))
    try:
        core.interrupts.bind()
        core.interrupts.inject("こんにちは")
        await asyncio.sleep(0.05)
        n = core._drain_external()  # noqa: SLF001
        assert n == 1
        content = core.buffer.content()
        assert "【ユーザーからメッセージが届きました】" in content, "返答指示がバッファに注入される"
        assert "返答してください" in content
        assert core._force_response is True  # noqa: SLF001
        assert core._last_segment_emitted is False  # noqa: SLF001
    finally:
        core.close()


async def test_response_during_ask_cancels_await_and_speaks(tmp_path) -> None:
    """v1.16: 問いかけ中に応答が届いたら応答待ち（awaiting）に入らず、応答に切り替える。

    実機で「/send のメッセージが受信（response_received）されるのに await_start に入り、
    Lucina が応答しない」問題の再現と修正確認。
    """
    core = _run_speaking_with_force(tmp_path, force=True, ask_mode=True)
    try:
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=False))
        await asyncio.sleep(0.4)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "ask_cancelled" in names, "応答が届いた問いかけは応答待ちではなく応答に切り替わる"
        assert "await_start" not in names, "応答があるのに待ちに入ってはいけない"
        assert core.segments_completed >= 1, "応答セグメントが生成されている"
    finally:
        core.close()


async def test_response_during_ask_waits_without_force(tmp_path) -> None:
    """v1.16 対照: force 無効なら従来どおり問いかけは応答待ち（await_start）に入る。"""
    core = _run_speaking_with_force(tmp_path, force=False, ask_mode=True)
    try:
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=False))
        await asyncio.sleep(0.4)
        core.stop()
        await run_task
        names = _names(tmp_path / "autonomy.jsonl")
        assert "await_start" in names
        assert "ask_cancelled" not in names
    finally:
        core.close()


async def test_force_response_suppresses_silence_decision(tmp_path) -> None:
    """v1.16: 応答セッション中はモデルの「黙る」決断（think-end）を無効化し、応答を続ける。

    実機で「応答を受信 → think-end で黙る → 発話ゼロで沈黙」になる問題の修正確認。
    """
    core = _run_speaking_with_force(tmp_path, force=True)
    try:
        core._pending_action = "silence"  # モデルが think-end で「黙る」を選んだ状態を模擬
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=False))
        await asyncio.sleep(0.4)
        core.stop()
        await run_task
        assert core.segments_completed >= 1, "黙る決断でも応答セグメントが生成される（強制発話）"
        assert core.buffer.spoken_content() != "", "応答が発話として出ている"
        # 応答セグメントの完了後に安全弁（セグメント上限）で終了する speech_end は正常
        assert core.mode == "thinking", "応答セッションが正しく終了している"
    finally:
        core.close()


async def test_without_force_silence_halts(tmp_path) -> None:
    """v1.16 対照: force 無効なら「黙る」決断で即沈黙し、応答セグメントは生成されない。"""
    core = _run_speaking_with_force(tmp_path, force=False)
    try:
        core._pending_action = "silence"
        run_task = asyncio.create_task(core.run(max_tokens=200, drive_loop=False))
        await asyncio.sleep(0.4)
        core.stop()
        await run_task
        assert core.segments_completed == 0, "黙る決断で沈黙する（従来挙動・完了セグメントなし）"
    finally:
        core.close()
