#!/usr/bin/env python3
"""v1.10: 外部への働きかけの必要性のシナリオ比較スクリプト。

「内部の欲求（Drive）を内部の出力で解消できる」ため外部に働きかける必要性が発生しない
（v1.9実測: モデルは黙る・考えるを好み、発話はDriveの安全弁に委ねられた）問題への対策案:

- baseline: 旧仕様（話すだけで loneliness フル解消・好奇心なし）
- ①curiosity: 好奇心Drive（内部では解消されず、閾値を超えると外部に問いかける）
- ②response_loneliness: loneliness は話すだけでは部分 relief。応答でフル解消
- all: ①+②（＋応答待ち awaiting）

計測: 各シナリオで「エージェントが外部に働きかけた回数（ask_start）」「応答待ちへの
遷移（await_start）」「最終好奇心・寂しさ」を記録し、reports/external_necessity_comparison.md に書く。

使い方:
    python scripts/compare_external_necessity.py                     # 実モデル・全シナリオ
    python scripts/compare_external_necessity.py --mock --seconds 5  # モックスモーク
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.config import load_config  # noqa: E402
from lucina.drives.decay import ReliefController  # noqa: E402
from lucina.io.logging import setup_console_logging  # noqa: E402

SCENARIOS = ["baseline", "curiosity", "response_loneliness", "all"]

# 各シナリオの上書き（base config に重ねる）
SCENARIO_CONFIGS = {
    # 旧仕様: 話すだけで loneliness フル解消・好奇心は無効（問いかけなし）
    "baseline": {
        "idle_curiosity_rate": 0.0,
        "curiosity_ask_threshold": 1.0,
        "speak_relief": 0.6,      # = per_action 相当（話すだけでフル解消・旧仕様）
    },
    # ①のみ: 好奇心が溜まると外部に問いかけ。loneliness は旧仕様
    "curiosity": {
        "idle_curiosity_rate": 0.01,
        "curiosity_ask_threshold": 0.5,
        "speak_relief": 0.6,
    },
    # ②のみ: 話すだけでは部分 relief。応答でフル解消。好奇心は無効
    "response_loneliness": {
        "idle_curiosity_rate": 0.0,
        "curiosity_ask_threshold": 1.0,
        "speak_relief": 0.2,
    },
    # ①+②+応答待ち（awaiting）: 全導入
    "all": {
        "idle_curiosity_rate": 0.03,
        "curiosity_ask_threshold": 0.5,
        "speak_relief": 0.2,
    },
}


def build_real_core(config: dict):
    from lucina.core import LucinaCore
    from lucina.drives.vocab import DriveVocabExpander
    from lucina.inference.adapters import LlamaSummarizer, LlamaTokenizerAdapter, SentenceTransformerEmbedder
    from lucina.inference.backends import LlamaBackend
    from lucina.inference.engine import InferenceEngine
    from lucina.io.logging import StructuredLogger
    from lucina.memory.classifier import RuleBasedMemoryClassifier
    from lucina.memory.store import ChromaVectorStore, HierarchicalMemoryStore, MemoryCompressor

    model_cfg = config["model"]
    mem_cfg = config["memory"]
    backend = LlamaBackend(model_cfg["path"], n_ctx=model_cfg["context_window"], n_gpu_layers=model_cfg["n_gpu_layers"])
    tokenizer = LlamaTokenizerAdapter(backend)
    embedder = SentenceTransformerEmbedder(
        config["embedding"]["model"], device=config["embedding"].get("device", "cpu")
    )
    logger = StructuredLogger(config["logging"]["log_dir"])
    vocab_map = DriveVocabExpander(
        config["drive"]["vocab_expansion"], tokenizer, embedder, logger=logger
    ).build_vocab_map()
    executor = ThreadPoolExecutor(max_workers=2)
    engine = InferenceEngine(model_cfg["path"], executor, backend=backend, vocab_map=vocab_map, config=config)
    memory = HierarchicalMemoryStore(
        ChromaVectorStore(mem_cfg["persist_directory"]),
        embedder=embedder,
        classifier=RuleBasedMemoryClassifier(),  # v1.11
    )
    summarizer = LlamaSummarizer(mem_cfg["summarizer_model_path"])
    compressor = MemoryCompressor(summarizer)
    core = LucinaCore(config, engine, vocab_map, memory=memory, compressor=compressor, logger=logger)
    core._executor = executor
    return core


def build_mock_core(config: dict):
    from lucina.testing import build_mock_core as _build

    return _build(config, token_delay_ms=0.0, log_dir=config["logging"]["log_dir"])


def _apply_scenario(core, scenario: str, config: dict) -> None:
    """シナリオの設定をコアに適用する（キャッシュ属性・relief コントローラを更新）。"""
    over = SCENARIO_CONFIGS[scenario]
    sched = core._schedule_cfg
    sched["idle_curiosity_rate"] = over["idle_curiosity_rate"]
    sched["curiosity_ask_threshold"] = over["curiosity_ask_threshold"]
    core._idle_curiosity_rate = over["idle_curiosity_rate"]
    core._curiosity_ask_threshold = over["curiosity_ask_threshold"]
    # loneliness の relief 仕様を更新（_evaluate_relief は core._relief_cfg を生参照）
    core._relief_cfg["loneliness"]["speak_relief"] = over["speak_relief"]
    # ReliefController は起動時に config をコピーするため再構築する
    core.relief = ReliefController(config["drive"]["relief"])
    core.relief._config["loneliness"]["speak_relief"] = over["speak_relief"]


def _reset_core(core, seed_prompt: str, system_prompt: str) -> None:
    core.reset_for_trial()
    core._stop = False
    core.tokens_generated = 0
    core.speech_tokens = 0
    core.thoughts_generated = 0
    core.segments_completed = 0
    core.decisions_asked = 0
    core.decision_total_sec = 0.0
    core.asks_asked = 0
    core.responses_received = 0
    core._speech_segments = 0
    core._speech_segments_mark = 0
    core._pending_action = None
    core._ask_mode = False
    core._last_decision = time.monotonic()
    core._mode = str(core._schedule_cfg.get("mode", "thinking"))
    core.seed_prompt(seed_prompt, system=system_prompt or None)


async def run_scenario(core, seconds: float, max_tokens: int) -> None:
    drive_task = asyncio.create_task(core._drive_loop())
    task = asyncio.create_task(core._run_scheduled(max_tokens))
    await asyncio.sleep(seconds)
    core.stop()
    try:
        await task
    except Exception:  # noqa: BLE001
        pass
    drive_task.cancel()
    try:
        await drive_task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


def _autonomy_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collect(core, before: int, after: list[dict]) -> dict:
    events = after[before:]
    counts: dict[str, int] = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    return {
        "asks": core.asks_asked,
        "await_start": counts.get("await_start", 0),
        "await_timeout": counts.get("await_timeout", 0),
        "responses": core.responses_received,
        "speech_start": counts.get("speech_start", 0),
        "inner_thought": counts.get("inner_thought", 0),
        "speech_tokens": core.speech_tokens,
        "thought_tokens": core.thoughts_generated,
        "final_curiosity": round(float(core.drives.get("curiosity", 0.0)), 3),
        "final_loneliness": round(float(core.drives.get("loneliness", 0.0)), 3),
        "events": [f"{e['event']}:{e['reason']}" for e in events
                   if e["event"] in ("ask_start", "await_start", "await_timeout", "response_received")],
    }


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_console_logging(config["logging"]["level"])
    scenarios = [s for s in args.scenarios.split(",") if s in SCENARIOS]
    if not scenarios:
        print(f"[ext] 有効なシナリオがありません: {args.scenarios}", file=sys.stderr)
        return 1

    core = build_mock_core(config) if args.mock else build_real_core(config)
    print(f"[ext] {'モック' if args.mock else '実モデル'}: {', '.join(scenarios)}・各{args.seconds}秒")

    seed = args.prompt or "あなたは考える存在です。静かに今の気持ちを言葉にしてください。"
    sys_prompt = str(config.get("drive", {}).get("scheduling", {}).get("system_prompt", ""))
    autonomy_path = Path(config["logging"]["log_dir"]) / "autonomy.jsonl"

    results: dict[str, dict] = {}
    try:
        for scenario in scenarios:
            _apply_scenario(core, scenario, config)
            _reset_core(core, seed, sys_prompt)
            before = len(_autonomy_lines(autonomy_path))
            await run_scenario(core, args.seconds, args.max_tokens)
            results[scenario] = _collect(core, before, _autonomy_lines(autonomy_path))
            r = results[scenario]
            print(
                f"[ext] {scenario:>18}: 問いかけ{r['asks']}回 応答待ち{r['await_start']}回 "
                f"タイムアウト{r['await_timeout']}回 応答{r['responses']}回 "
                f"発話{r['speech_tokens']}tok 最終curiosity={r['final_curiosity']} loneliness={r['final_loneliness']}"
            )
    finally:
        core.close()

    _write_report(results, args)
    return 0


def _write_report(results: dict[str, dict], args: argparse.Namespace) -> None:
    out = Path(args.output_dir) / "external_necessity_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 外部への働きかけの必要性 シナリオ比較（v1.10）",
        "",
        f"- 計測日時: {time.strftime('%Y-%m-%d %H:%M:%S')} / 各シナリオ {args.seconds} 秒",
        f"- モード: {'モック' if args.mock else '実モデル（Qwen3.5-9B）'}",
        "",
        "## 計測サマリ",
        "",
        "| シナリオ | 問いかけ | 応答待ち | タイムアウト | 応答受信 | 発話トークン | 最終curiosity | 最終loneliness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s, r in results.items():
        lines.append(
            f"| {s} | {r['asks']} | {r['await_start']} | {r['await_timeout']} | {r['responses']} | "
            f"{r['speech_tokens']} | {r['final_curiosity']} | {r['final_loneliness']} |"
        )
    lines += ["", "## 外部働きかけイベント", ""]
    for s, r in results.items():
        lines.append(f"- **{s}**: " + (" → ".join(r["events"]) if r["events"] else "（外部への働きかけなし）"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ext] レポート: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="外部への働きかけの必要性 シナリオ比較")
    parser.add_argument("--config", default=None, help="config YAML パス（既定: demo_external.yaml）")
    parser.add_argument("--mock", action="store_true", help="モックバックエンドで実行")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="対象シナリオ")
    parser.add_argument("--seconds", type=float, default=60.0, help="各シナリオの実行秒数")
    parser.add_argument("--max-tokens", type=int, default=15000, help="各シナリオのトークン上限")
    parser.add_argument("--prompt", default="", help="初期コンテキスト")
    parser.add_argument("--output-dir", default="./reports", help="レポート出力先")
    args = parser.parse_args()
    if args.config is None:
        args.config = "./config/demo_external.yaml"
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
