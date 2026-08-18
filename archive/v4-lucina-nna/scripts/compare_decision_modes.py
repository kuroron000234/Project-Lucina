#!/usr/bin/env python3
"""v1.9: モデル駆動スケジューリングの3方式を比較計測するスクリプト。

モデル自身が「いつ話す・黙る・考える」を選ぶ方式（仕様書 v1.4 §5.6・v1.9）:
- baseline: 従来のDrive閾値＋タイマー内言（v1.8。モデルはタイミングを決めない）
- A: 境界決断（decide_on_think_end / decide_on_segment_end）
- B: 待機中 introspection（introspection_sec）
- C: 制御トークン（control_tokens）

1つのコア（モデルロード1回）を使い回し、各方式を --seconds 秒ずつ走らせて
以下を計測し、reports/decision_modes_comparison.md にレポートを書く:
- 決断回数・決断レイテンシ（決断1回のコスト）
- 遷移（speech_start/end / inner_thought / control_token）の回数
- 発話トークン / 思考トークン（生成の向き）
- 選択の分布（モデルが何を選んだか）とモデル駆動で起きた遷移の割合

使い方:
    python scripts/compare_decision_modes.py                     # 実モデル・全方式
    python scripts/compare_decision_modes.py --mock --seconds 5  # モックスモーク
    python scripts/compare_decision_modes.py --modes baseline,B --seconds 30
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
from lucina.io.logging import setup_console_logging  # noqa: E402

MODES = ["baseline", "A", "B", "C", "all"]

# 各方式のスケジューリング上書き（base config に重ねる）
MODE_CONFIGS = {
    "baseline": {
        "introspection_sec": 0.0,
        "decide_on_think_end": False,
        "decide_on_segment_end": False,
        "control_tokens": False,
    },
    "A": {
        "introspection_sec": 0.0,
        "decide_on_think_end": True,
        "decide_on_segment_end": True,
        "control_tokens": False,
    },
    "B": {
        "introspection_sec": 5.0,
        "decide_on_think_end": False,
        "decide_on_segment_end": False,
        "control_tokens": False,
    },
    "C": {
        "introspection_sec": 0.0,
        "decide_on_think_end": False,
        "decide_on_segment_end": False,
        "control_tokens": True,
    },
    "all": {
        "introspection_sec": 5.0,
        "decide_on_think_end": True,
        "decide_on_segment_end": True,
        "control_tokens": True,
    },
}


def build_real_core(config: dict):
    """実モデル構成で LucinaCore を組み立てる（run_agent と同じ構成）。"""
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


def _autonomy_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _apply_mode(core, mode: str) -> None:
    """方式のフラグをコアに適用する（_schedule_cfg とキャッシュ済み属性の両方を更新）。"""
    over = MODE_CONFIGS[mode]
    sched = core._schedule_cfg
    sched["introspection_sec"] = over["introspection_sec"]
    sched["decide_on_think_end"] = over["decide_on_think_end"]
    sched["decide_on_segment_end"] = over["decide_on_segment_end"]
    sched["control_tokens"] = over["control_tokens"]
    core._introspection_sec = over["introspection_sec"]
    core._decide_on_think_end = over["decide_on_think_end"]
    core._decide_on_segment_end = over["decide_on_segment_end"]
    core._control_tokens = over["control_tokens"]
    if over["introspection_sec"] > 0.0:
        sched["inner_interval_sec"] = 1000.0  # B ではタイマー内言を無効化（モデルが内言を選ぶ）
    elif mode == "baseline":
        pass  # baseline は config の inner_interval_sec のまま


def _reset_core(core, seed_prompt: str, system_prompt: str) -> None:
    """方式ごとの試行リセット（バッファ・Drive・カウンタ・モード）。"""
    core.reset_for_trial()
    core._stop = False  # 前方式の stop() で立った停止フラグを戻す
    core.tokens_generated = 0
    core.speech_tokens = 0
    core.thoughts_generated = 0
    core.segments_completed = 0
    core.decisions_asked = 0
    core.decision_total_sec = 0.0
    core._speech_segments = 0
    core._speech_segments_mark = 0
    core._pending_action = None
    core._last_decision = time.monotonic()
    core._mode = str(core._schedule_cfg.get("mode", "thinking"))
    core.seed_prompt(seed_prompt, system=system_prompt or None)


async def run_mode(core, seconds: float, max_tokens: int) -> None:
    """Drive ループを起動したまま _run_scheduled を走らせる（実時間で Drive が発達する）。"""
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


def _collect(core, before: int, after: list[dict]) -> dict:
    """方式1回分の計測結果を集計する。"""
    events = after[before:]
    counts: dict[str, int] = {}
    choices: dict[str, int] = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
        if e["event"] == "decision":
            purpose = e["reason"].split("→")[0]
            choice = e["reason"].split("→")[-1]
            key = f"{purpose}→{choice}"
            choices[key] = choices.get(key, 0) + 1
    n_dec = core.decisions_asked
    avg_latency = (core.decision_total_sec / n_dec * 1000.0) if n_dec else 0.0
    return {
        "seconds": round(core._last_mode_seconds, 1),
        "tokens": core.tokens_generated,
        "speech_tokens": core.speech_tokens,
        "thought_tokens": core.thoughts_generated,
        "segments": core.segments_completed,
        "decisions": n_dec,
        "decision_avg_ms": avg_latency,
        "speech_start": counts.get("speech_start", 0),
        "speech_end": counts.get("speech_end", 0),
        "inner_thought": counts.get("inner_thought", 0),
        "control_token": counts.get("control_token", 0),
        "choices": choices,
        "events": [f"{e['event']}:{e['reason']}" for e in events if e["event"] in ("speech_start", "speech_end")],
        "final_drives": {k: round(v, 3) for k, v in core.drives.items()},
    }


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_console_logging(config["logging"]["level"])
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    if not modes:
        print(f"[compare] 有効な方式がありません: {args.modes}", file=sys.stderr)
        return 1

    if args.mock:
        core = build_mock_core(config)
        print(f"[compare] モックバックエンド（{', '.join(modes)}・各{args.seconds}秒）")
    else:
        core = build_real_core(config)
        print(f"[compare] 実モデル: {config['model']['path']}（{', '.join(modes)}・各{args.seconds}秒）")

    seed = args.prompt or "あなたは考える存在です。静かに今の気持ちを言葉にしてください。"
    sys_prompt = str(config.get("drive", {}).get("scheduling", {}).get("system_prompt", ""))
    autonomy_path = Path(config["logging"]["log_dir"]) / "autonomy.jsonl"

    results: dict[str, dict] = {}
    try:
        for mode in modes:
            _apply_mode(core, mode)
            _reset_core(core, seed, sys_prompt)
            before = len(_autonomy_lines(autonomy_path))
            core._last_mode_seconds = args.seconds
            await run_mode(core, args.seconds, args.max_tokens)
            results[mode] = _collect(core, before, _autonomy_lines(autonomy_path))
            r = results[mode]
            print(
                f"[compare] {mode:>8}: 発話{r['speech_start']}回/沈黙{r['speech_end']}回 "
                f"内言{r['inner_thought']}回 決断{r['decisions']}回({r['decision_avg_ms']:.0f}ms/回) "
                f"発話トークン{r['speech_tokens']} 思考トークン{r['thought_tokens']} "
                f"CTLトークン{r['control_token']}回"
            )
    finally:
        core.close()

    _write_report(results, autonomy_path, args)
    return 0


def _write_report(results: dict[str, dict], autonomy_path: Path, args: argparse.Namespace) -> None:
    out = Path(args.output_dir) / "decision_modes_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# モデル駆動スケジューリング 3方式の比較（v1.9）",
        "",
        f"- 計測日時: {time.strftime('%Y-%m-%d %H:%M:%S')} / 各方式 {args.seconds} 秒",
        f"- モード: {'モック' if args.mock else '実モデル（Qwen3.5-9B）'}",
        f"- 方式: baseline（Drive閾値＋タイマー）/ A（境界決断）/ B（待機中 introspection）/ C（制御トークン）/ all（A+B+C）",
        "",
        "## 計測サマリ",
        "",
        "| 方式 | 発話/沈黙 | 内言 | 決断回数 | 決断レイテンシ | 発話トークン | 思考トークン | CTLトークン |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, r in results.items():
        lines.append(
            f"| {mode} | {r['speech_start']}/{r['speech_end']} | {r['inner_thought']} | "
            f"{r['decisions']} | {r['decision_avg_ms']:.0f}ms | {r['speech_tokens']} | "
            f"{r['thought_tokens']} | {r['control_token']} |"
        )
    lines += ["", "## 選択の分布（モデルが何を選んだか）", ""]
    for mode, r in results.items():
        if r["choices"]:
            lines.append(f"- **{mode}**: " + ", ".join(f"{k}×{v}" for k, v in sorted(r["choices"].items())))
    lines += ["", "## 遷移ログ（speech_start / speech_end）", ""]
    for mode, r in results.items():
        lines.append(f"- **{mode}**: " + (" → ".join(r["events"]) if r["events"] else "（遷移なし）"))
    lines += ["", "## 最終Drive", ""]
    for mode, r in results.items():
        lines.append(f"- **{mode}**: " + ", ".join(f"{k}={v}" for k, v in r["final_drives"].items()))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] レポート: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="モデル駆動スケジューリング 3方式の比較計測")
    parser.add_argument("--config", default=None, help="config YAML パス")
    parser.add_argument("--mock", action="store_true", help="モックバックエンドで実行")
    parser.add_argument("--modes", default=",".join(MODES), help=f"対象方式（既定: 全方式）")
    parser.add_argument("--seconds", type=float, default=45.0, help="各方式の実行秒数")
    parser.add_argument("--max-tokens", type=int, default=10000, help="各方式のトークン上限")
    parser.add_argument("--prompt", default="", help="初期コンテキスト")
    parser.add_argument("--output-dir", default="./reports", help="レポート出力先")
    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
