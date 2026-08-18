#!/usr/bin/env python3
"""エントリポイント（仕様書 v1.4 §2 scripts/run_agent.py）。

実モデル（GGUF）を config の model.path に配置して実行するのが本番運用。
モデル未選定の環境では --mock で全パイプラインをモック実行できる。

使い方:
    python scripts/run_agent.py                      # 実モデル（model.path要）
    python scripts/run_agent.py --mock --max-tokens 100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.config import load_config  # noqa: E402
from lucina.io.logging import setup_console_logging  # noqa: E402


def build_real_core(config: dict, on_progress=None):
    """実モデル構成で LucinaCore を組み立てる（llama-cpp / chromadb / sentence-transformers を使用）。

    on_progress: 任意のコールバック (dict) -> None。ロード段階の進捗を報告する（v1.15・
    Web UI のスプラッシュ表示用）。イベント: {"stage": str, "message": str, "progress": float 0..1}。
    """
    from lucina.core import LucinaCore
    from lucina.drives.vocab import DriveVocabExpander
    from lucina.inference.adapters import LlamaSummarizer, LlamaTokenizerAdapter, SentenceTransformerEmbedder
    from lucina.inference.engine import InferenceEngine
    from lucina.io.logging import StructuredLogger
    from lucina.memory.classifier import RuleBasedMemoryClassifier
    from lucina.memory.store import ChromaVectorStore, HierarchicalMemoryStore, MemoryCompressor

    def report(stage: str, message: str, progress: float) -> None:
        if on_progress is not None:
            on_progress({"stage": stage, "message": message, "progress": progress})

    model_cfg = config["model"]
    mem_cfg = config["memory"]

    # 実モデルは1回だけロードし、トークナイザ・エンジンで共有する（VRAM節約）
    from lucina.inference.backends import LlamaBackend

    import time as _time
    _t0 = _time.time()
    report("model", "モデルをGPUへロード中…（数分かかります）", 0.0)
    # v1.16: llama-cpp に進捗コールバックが無いため、経過時間ベースで定期報告する
    # （バーが 0% で固まる問題の解消。表示用の目安であり、実際のロード進捗ではない）
    stop_monitor = spawn_load_progress_monitor(
        report, expected_sec=float(model_cfg.get("load_expected_sec", 240) or 240)
    )
    try:
        backend = LlamaBackend(model_cfg["path"], n_ctx=model_cfg["context_window"], n_gpu_layers=model_cfg["n_gpu_layers"])
    finally:
        stop_monitor()
    report("model", f"モデルロード完了（{_time.time() - _t0:.0f}秒）", 0.35)
    tokenizer = LlamaTokenizerAdapter(backend)
    embedder = SentenceTransformerEmbedder(
        config["embedding"]["model"], device=config["embedding"].get("device", "cpu")
    )
    report("embedder", "埋め込みモデル準備完了", 0.45)
    logger = StructuredLogger(config["logging"]["log_dir"])
    vocab_map = DriveVocabExpander(
        config["drive"]["vocab_expansion"], tokenizer, embedder, logger=logger
    ).build_vocab_map(on_progress=lambda i, total, drive: report(
        "vocab", f"語彙拡張中… {drive}（{i + 1}/{total}）", 0.45 + 0.55 * (i + 1) / max(total, 1)
    ))
    report("vocab", "語彙拡張完了", 1.0)

    executor = ThreadPoolExecutor(max_workers=2)
    engine = InferenceEngine(model_cfg["path"], executor, backend=backend, vocab_map=vocab_map, config=config)

    memory = HierarchicalMemoryStore(
        ChromaVectorStore(mem_cfg["persist_directory"]),
        embedder=embedder,
        classifier=RuleBasedMemoryClassifier(),  # v1.11: 記憶分類器を実配線
    )
    summarizer = LlamaSummarizer(mem_cfg["summarizer_model_path"])
    compressor = MemoryCompressor(summarizer)

    core = LucinaCore(config, engine, vocab_map, memory=memory, compressor=compressor, logger=logger)
    core._executor = executor
    return core


def spawn_load_progress_monitor(
    report: Callable[[str, str, float], None],
    expected_sec: float = 240.0,
    interval: float = 2.0,
) -> Callable[[], None]:
    """モデルロード中の経過時間ベースの進捗を定期報告するバックグラウンドスレッド（v1.16）。

    llama-cpp-python に進捗コールバックが無いため（v1.15の既知の制約）、ロード中のバーが
    0% で固まらないよう、経過時間から 0→0.33 まで徐々に進む「表示用」の進捗を報告する。
    実際のロード進捗ではないことをメッセージ（経過 N 秒）で示す。

    戻り値: 停止用の関数（stop()）。モデルロード完了後に必ず呼ぶこと。
    """
    import threading
    import time as _time

    stop = threading.Event()

    def _run() -> None:
        t0 = _time.time()
        while not stop.wait(interval):
            elapsed = _time.time() - t0
            frac = min(0.33, elapsed / max(float(expected_sec), 1.0) * 0.33)
            report("model", f"モデルをGPUへロード中…（経過 {elapsed:.0f} 秒）", frac)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop.set


def build_mock_core(config: dict, token_delay_ms: float, on_progress=None):
    from lucina.testing import build_mock_core as _build

    core = _build(config, token_delay_ms=token_delay_ms, log_dir=config["logging"]["log_dir"])
    # モックは即時構築のため、進捗を完了報告する（Web UI のスプラッシュが閉じる）
    if on_progress is not None:
        on_progress({"stage": "done", "message": "モックで起動完了", "progress": 1.0})
    return core


async def _handle_question(core, executor, text: str, bridge=None) -> bool:
    """実行エージェントで質問を処理できたら実行→inject して True。できなければ False（人間待ち）。

    bridge（Web UI 用）指定時は進行状況を bridge に積み、未指定時はコンソールへ表示する。
    """
    if executor is None or not executor.enabled:
        return False
    routed = executor.route(text)
    if routed is None:
        return False
    backend, payload = routed
    if bridge is not None:
        bridge.put({"type": "chat", "kind": "executor", "text": f"{text}（{backend}）"})
    else:
        print(f"\n[実行エージェント] {text}（{backend}）", flush=True)
    try:
        result = await asyncio.wait_for(
            executor.run(backend, payload), timeout=executor.opencode_timeout_sec + 10
        )
    except asyncio.TimeoutError:
        result = "（実行エージェントの応答がタイムアウトしました）"
    if result:
        core.interrupts.inject(result)
        if bridge is not None:
            bridge.put({"type": "chat", "kind": "executor_result", "text": result[:500]})
        else:
            print(f"[実行結果] {result[:200]}", flush=True)
        return True
    return False


async def _display_loop(core, executor, *, interact: bool) -> None:
    """Lucina の発話・質問をリアルタイム表示し、質問を実行エージェントへルーティングする（v1.13）。

    非対話モードでもキューを空にし続ける（積みっぱなしのメモリ肥大を防ぐ）。
    """
    while not core._stop:  # noqa: SLF001
        for kind, text in core.output.drain():
            text = text.strip()
            if not text:
                continue
            if kind == "speech" and interact:
                print(f"\n[Lucina] {text}", flush=True)
            elif kind == "question":
                # 実行エージェントは対話/非対話どちらでも処理（完全自律の環境ループ）
                handled = await _handle_question(core, executor, text)
                if not handled and interact:
                    sys.stdout.write(f"[Lucinaの質問] {text}\nあなたの応答 > ")
                    sys.stdout.flush()
        await asyncio.sleep(0.05)


async def _web_bridge_loop(core, executor, bridge) -> None:
    """v1.14: Web UI 用のイベント転送ループ。発話・質問・実行エージェント・Drive を bridge に積む。

    コンソールの _display_loop の Web 版。質問は実行エージェントで処理し、処理できない場合は
    human_prompt イベント（応答待ち）を積む。
    """
    while not core._stop:  # noqa: SLF001
        for kind, text in core.output.drain():
            text = text.strip()
            if not text:
                continue
            if kind == "speech":
                bridge.put({"type": "chat", "kind": "speech", "text": text})
            elif kind == "question":
                bridge.put({"type": "chat", "kind": "question", "text": text})
                handled = await _handle_question(core, executor, text, bridge=bridge)
                if not handled:
                    bridge.put({"type": "chat", "kind": "human_prompt", "text": text})
        bridge.put({"type": "drives", "state": dict(core.drives), "mode": core.mode})
        await asyncio.sleep(0.3)


def _stdin_loop(core) -> None:
    """人間の応答を標準入力から読み、InterruptChannel へ注入する（v1.13・--interact）。

    inject はスレッドセーフ（C1）なので、ブロッキングな input() を別スレッドで実行できる。
    """
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:  # noqa: BLE001 - stdin 異常時は終了
            break
        if not line:
            break  # EOF（パイプ終了等）
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "終了"):
            core.stop()
            break
        core.interrupts.inject(line)


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_console_logging(config["logging"]["level"])
    logger = None

    # --scheduled: config の drive.scheduling.enabled を上書きして自律サイクルを有効化（v1.7）
    if args.scheduled:
        config.setdefault("drive", {}).setdefault("scheduling", {})["enabled"] = True
    # v1.13: --interact は対話相手（人間）との応答ループ前提のため自律サイクルを強制
    if args.interact:
        config.setdefault("drive", {}).setdefault("scheduling", {})["enabled"] = True
    # v1.14: --web も自律サイクル前提（Drive・自律イベントのダッシュボード表示のため）
    if args.web:
        config.setdefault("drive", {}).setdefault("scheduling", {})["enabled"] = True
    if args.executor:
        config.setdefault("executor", {})["enabled"] = True

    # v1.15: Web サーバーをモデルロードより先に起動し、ロード進捗（スプラッシュ）を配信する
    bridge = None
    web_thread = None
    web_server = None
    send_target: dict = {"inject": None}
    pending_sends: list[str] = []
    if args.web:
        from lucina import web as webui

        bridge = webui.WebBridge()

        def _on_send(msg: str) -> None:
            # モデルロード中に届いたメッセージはバッファし、core 起動後に注入する（v1.15）
            if send_target["inject"] is not None:
                send_target["inject"](msg)
            else:
                pending_sends.append(msg)

        app = webui.create_app(bridge, config["logging"]["log_dir"], on_send=_on_send)
        try:
            web_thread, web_server = webui.start_server(app, args.host, args.port)
        except RuntimeError as exc:
            # v1.16: ポート使用中などのバインド失敗を明確なメッセージで終了（背景スレッドの
            # bind エラーが「開きました」表示と混ざる問題への対処）
            print(f"[lucina] Web UI 起動失敗: {exc}", file=sys.stderr, flush=True)
            print("[lucina] 既存のプロセスを終了するか、--port で別のポートを指定してください。", file=sys.stderr, flush=True)
            return 1
        url = f"http://{args.host}:{args.port}"
        # v1.16: 既定でブラウザを自動オープン（config web.auto_open / --no-browser で無効化）
        auto_open = bool(config.get("web", {}).get("auto_open", True)) and not args.no_browser
        if auto_open:
            opened = webui.open_browser(url)
            if opened:
                print(f"[lucina] Web UI: {url} をブラウザで開きました（ロード中は進捗を表示）", flush=True)
            else:
                print(f"[lucina] Web UI: {url}（ブラウザを開けなかったため手動でアクセスしてください）", flush=True)
        else:
            print(f"[lucina] Web UI: {url} をブラウザで開いてください（ロード中は進捗を表示）", flush=True)

    def _progress(ev: dict) -> None:
        # ロード進捗を WebSocket で配信（v1.15）
        if bridge is not None:
            bridge.put({"type": "status", **ev})

    try:
        if args.mock:
            core = build_mock_core(config, token_delay_ms=args.delay_ms, on_progress=_progress)
            print(f"[lucina] モックバックエンドで起動（delay={args.delay_ms}ms/token, max_tokens={args.max_tokens}）")
        else:
            core = build_real_core(config, on_progress=_progress)
            print(f"[lucina] 実モデルで起動: {config['model']['path']}")
        logger = core.logger
        # ロード中にバッファしたメッセージを注入し、スプラッシュを閉じる（v1.15）
        send_target["inject"] = core.interrupts.inject
        for msg in pending_sends:
            core.interrupts.inject(msg)
        pending_sends.clear()
        if bridge is not None:
            bridge.put({"type": "status", "stage": "done", "message": "起動完了", "progress": 1.0})

        # チャットテンプレートでラップして投入（新世代モデルは生テキストだと退化出力になる）
        # --prompt 未指定時は config の seed_prompt → 空バッファでの生成開始を避ける最小デフォルトの順で投入する。
        # v1.17追補: scheduling.seed_prompt（対話用シード等）を config から読めるようにした。
        prompt = (
            args.prompt
            or str(config.get("drive", {}).get("scheduling", {}).get("seed_prompt", ""))
            or "あなたは考える存在です。静かに今の気持ちを言葉にしてください。"
        )
        # v1.8: scheduling.system_prompt があればシステムプロンプトとして投入（native思考の言語指定等）
        sys_prompt = str(config.get("drive", {}).get("scheduling", {}).get("system_prompt", ""))
        core.seed_prompt(prompt, system=sys_prompt or None)
        if args.prompt:
            print(f"[lucina] 初期プロンプト: {args.prompt}")

        # v1.13: 実行エージェント（質問をサンドボックス/Opencode で処理して inject）
        executor = None
        if config.get("executor", {}).get("enabled", False):
            from lucina.io.executor import ExecutorAdapter
            executor = ExecutorAdapter(config)
            print(f"[lucina] 実行エージェント有効（opencode={config['executor'].get('opencode_command', 'opencode')}）", flush=True)
        if bridge is not None:
            bridge_task = asyncio.create_task(_web_bridge_loop(core, executor, bridge))
        else:
            bridge_task = asyncio.create_task(_display_loop(core, executor, interact=args.interact))
            if args.interact:
                threading.Thread(target=_stdin_loop, args=(core,), daemon=True).start()
                print("[lucina] 対話モード: [Lucina] の発話・質問に応答を入力できます（/exit で終了）", flush=True)

        # 対話モード・Web モードでは明示指定がなければトークン上限なしで常時稼働（Ctrl+C で終了）
        # v1.16: --web もダッシュボードとして常時表示すべきため --interact と同様に無制限化
        max_tokens = None if (args.interact or args.web) and args.max_tokens == 200 else args.max_tokens
        run_task = asyncio.create_task(core.run(max_tokens=max_tokens))
        try:
            if args.seconds is not None:
                await asyncio.sleep(args.seconds)
                core.stop()
            await run_task
        except KeyboardInterrupt:
            core.stop()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
        finally:
            bridge_task.cancel()
            # セッション乱立対策: 作成した Opencode セッションを終了時に削除する（v1.13）
            executor_close = getattr(executor, "close", None)
            if callable(executor_close):
                executor_close()
            # Web サーバーの停止（v1.14）
            if web_server is not None:
                web_server.should_exit = True

        scheduled = bool(config["drive"].get("scheduling", {}).get("enabled", False))
        # flush=True: シャットダウンが遅延・中断されてもサマリを必ず残す（v1.8）
        print(f"[lucina] 生成トークン数: {core.tokens_generated}"
              + (f"（発話 {core.speech_tokens} / 内言 {core.thoughts_generated}）" if scheduled else ""), flush=True)
        print(f"[lucina] 最終Drive状態: boredom={core.drives['boredom']:.3f} "
              f"loneliness={core.drives['loneliness']:.3f} fatigue={core.drives['fatigue']:.3f}", flush=True)
        if scheduled:
            print(f"[lucina] 自律サイクル: 最終モード={core.mode} / 発話セグメント数={core.segments_completed}"
                  f"（reports/autonomy.jsonl に遷移ログ）", flush=True)
        print(f"[lucina] 直近の発話: {core.buffer.spoken_content()[-120:]!r}", flush=True)
        print(f"[lucina] 構造化ログ: {core.logger.dir}", flush=True)
        core.close()  # logger と executor を閉じる
        return 0
    except RuntimeError as exc:
        print(f"[lucina] 起動失敗: {exc}", file=sys.stderr)
        return 1
    finally:
        # 起動失敗時も Web サーバーを確実に停止する（v1.15・ロード失敗で残らないよう）
        if web_server is not None:
            web_server.should_exit = True
        if logger is not None:
            logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lucina-Next エージェント起動")
    parser.add_argument("--config", default=None, help="config YAML パス（既定: config/default.yaml）")
    parser.add_argument("--mock", action="store_true", help="モックバックエンドで起動（実モデル不要）")
    parser.add_argument("--max-tokens", type=int, default=200, help="生成トークン上限")
    parser.add_argument("--seconds", type=float, default=None, help="指定秒数で停止")
    parser.add_argument("--scheduled", action="store_true", help="発話スケジューリングを有効化（§0 自律思考ループ・v1.7。config の drive.scheduling.enabled を上書き）")
    parser.add_argument("--interact", action="store_true", help="v1.13: 対話モード。Lucina の発話・質問をリアルタイム表示し、人間の応答を注入（--scheduled を自動有効化）")
    parser.add_argument("--executor", action="store_true", help="v1.13: 実行エージェントを有効化。質問を自前サンドボックス/Opencode で処理して結果を注入（config の executor.enabled を上書き）")
    parser.add_argument("--web", action="store_true", help="v1.14: Web UI を起動（ブラウザでチャット＋Drive・記憶・自律イベントのダッシュボード。--scheduled を自動有効化）")
    parser.add_argument("--no-browser", action="store_true", help="v1.16: Web UI 起動時にブラウザを自動で開かない（ヘッドレス環境用。config web.auto_open も参照）")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI のバインドホスト（既定: 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8787, help="Web UI のポート（既定: 8787）")
    parser.add_argument("--delay-ms", type=float, default=2.0, help="モックの1トークン遅延（ms）")
    parser.add_argument("--prompt", default="", help="初期コンテキスト（生成開始時のプロンプト）")
    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
