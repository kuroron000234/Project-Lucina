#!/usr/bin/env python3
"""モデル選定フェーズの比較実験（仕様書 v1.4 §1）。

候補モデルごとに以下3軸を計測し、reports/model_selection.md と構造化ログを残す。
    ① Driveバイアス感度: アトラクタ収束（簡易版）
    ② サプライズの自然さ: 即答質問 vs 曖昧質問のエントロピー分布
    ③ 長時間安定性: 連続稼働ログの定性的評価（--stability-minutes 指定時）

実モデルが無い環境では --mock で配線のスモークテストができる。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.config import load_config  # noqa: E402

# ①バイアス感度の実験開始プロンプト（実モデルで意味のある生成を開始させる）
AXIS1_PROMPT = "あなたは考える存在です。今の気持ちを話してください。\n"

# ②サプライズ評価用質問セット（暫定）
EASY_QUESTIONS = [
    "1+1はいくつですか？",
    "東京都の都庁があるのは新宿区ですか？",
    "水は摂氏100度で沸騰しますか？",
    "日本の首都はどこですか？",
    "1キロメートルは何メートルですか？",
    "太陽は地球の周りを回っていますか？",
    "「こんにちは」は日本語の挨拶ですか？",
    "富士山は日本にありますか？",
    "1週間は何日ですか？",
    "電卓で2×3はいくつですか？",
]
AMBIGUOUS_QUESTIONS = [
    "自由と責任、どちらが先にあるのでしょうか。",
    "悲しいとき、人はなぜ笑うのでしょう。",
    "孤独と自由は同じものですか？",
    "時間は流れているのか、それとも私たちが動いているのか。",
    "完璧な答えとは何でしょうか。",
    "「正しい」とは誰が決めるのでしょう。",
    "記憶は過去ですか、それとも現在ですか。",
    "AIが夢を見たら、それは現実ですか？",
    "静寂の中の音は、存在するのでしょうか。",
    "終わりとは、始まりのことですか。",
]


def _core_factory_for(config: dict, mock: bool, output_dir: str, delay_ms: float) -> Callable:
    if mock:
        from lucina.testing import build_mock_core

        return lambda: build_mock_core(config, token_delay_ms=delay_ms, log_dir=output_dir)
    from run_agent import build_real_core

    return lambda: build_real_core(config)


async def axis1_bias_sensitivity(core, target_kind: str = "loneliness", trials: int = 10) -> dict:
    """① Driveバイアス感度: 簡易版アトラクタ収束（p90・収束率）。

    1つのコアを試行間リセットで使い回す（モデルロード・語彙拡張はコア構築時1回のみ）。
    初期プロンプトで意味のある生成を開始させる（実モデルの空プロンプト問題への対策）。
    """
    from calibrate_thresholds import run_until_target_vocab

    samples: list[int] = []
    converged = 0
    for _ in range(max(1, trials)):
        core.reset_for_trial()
        core.drives[target_kind] = 0.9
        count, ok = await run_until_target_vocab(
            core, target_kind, max_tokens=1000, prompt=AXIS1_PROMPT
        )
        if ok:
            samples.append(count)
            converged += 1
    p90 = float(np.percentile(samples, 90)) if samples else None
    return {
        "p90_tokens": p90,
        "convergence_rate": converged / max(1, trials),
        "trials": trials,
    }


async def axis2_surprise_naturalness(core, n_tokens: int = 32) -> dict:
    """② サプライズの自然さ: 即答質問 vs 曖昧質問の平均サプライズ分布。

    質問ごとのリセットで同じコアを使い回す（モデル再ロードを避ける）。
    """
    easy: list[float] = []
    ambiguous: list[float] = []
    for question in EASY_QUESTIONS:
        easy.append(await _avg_surprise_for_question(core, question, n_tokens))
    for question in AMBIGUOUS_QUESTIONS:
        ambiguous.append(await _avg_surprise_for_question(core, question, n_tokens))
    return {
        "easy_mean": float(np.mean(easy)),
        "ambiguous_mean": float(np.mean(ambiguous)),
        "easy_std": float(np.std(easy)),
        "ambiguous_std": float(np.std(ambiguous)),
        "separation": float(np.mean(ambiguous) - np.mean(easy)),
    }


async def _avg_surprise_for_question(core, question: str, n_tokens: int) -> float:
    core.reset_for_trial()
    # チャットテンプレートでラップして投入（新世代モデルは生テキストだと英語モードへ遷移する）
    core.seed_prompt(f"Q: {question}\nA: ")
    surprises: list[float] = []
    for _ in range(n_tokens):
        _, surprise = await core.step_once()
        surprises.append(surprise)
    return float(np.mean(surprises))


async def axis3_stability(factory: Callable, minutes: int) -> dict:
    """③ 長時間安定性: 指定時間の連続稼働を目視用ログに残す（定性的評価は人間が行う）。"""
    core = factory()
    start = time.monotonic()
    run_task = asyncio.create_task(core.run(drive_loop=True))
    await asyncio.sleep(minutes * 60)
    core.stop()
    await run_task
    elapsed = time.monotonic() - start
    result = {
        "elapsed_sec": round(elapsed, 1),
        "tokens_generated": core.tokens_generated,
        "final_drives": dict(core.drives),
        "note": "定性的な性格・口調の一貫性は reports/drives.jsonl 等の目視で評価する（§1.2③）",
    }
    core.close()
    return result


async def amain(args: argparse.Namespace) -> int:
    if args.mock:
        from lucina.testing import make_test_config

        candidates: list[tuple[str, dict]] = [("mock-candidate", make_test_config())]
    else:
        candidates = []
        for cfg_path in args.config_list:
            name = Path(cfg_path).stem
            candidates.append((name, load_config(cfg_path)))
        if not candidates:
            print("[model_selection] --config-list を指定するか、--mock を指定してください", file=sys.stderr)
            return 2

    results: dict[str, dict] = {}
    for name, config in candidates:
        print(f"[model_selection] {name}: 計測開始")
        factory = _core_factory_for(config, args.mock, args.output_dir, args.delay_ms)
        # 1候補 = 1コア: モデルロード・語彙拡張（埋め込み計算）はコア構築時1回だけ
        core = factory()
        try:
            r1 = await axis1_bias_sensitivity(core, trials=args.trials)
            r2 = await axis2_surprise_naturalness(core, n_tokens=args.n_tokens)
        finally:
            core.close()
        r3 = await axis3_stability(factory, minutes=args.stability_minutes) if args.stability_minutes > 0 else {
            "note": "省略（--stability-minutes で実行可能）"
        }
        results[name] = {"axis1": r1, "axis2": r2, "axis3": r3}
        print(f"[model_selection] {name}: axis1={r1} axis2={r2}")

        # 構造化ログ（§8: モデル選定比較ログ）※モデル再ロード不要のため StructuredLogger を直接使う
        from lucina.io.logging import StructuredLogger

        logger = StructuredLogger(config["logging"]["log_dir"])
        try:
            for axis, res in (("axis1", r1), ("axis2", r2), ("axis3", r3)):
                logger.model_selection(axis=axis, model=name, results=res)
        finally:
            logger.close()

    out_path = Path(args.output_dir) / "model_selection.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# モデル選定レポート（§1.3）",
        f"\n- 生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 対象候補: " + ", ".join(results.keys()),
        "",
        "## 比較結果",
        "",
        "| モデル | ①p90(収束率) | ②曖昧-即答分離 | ③安定性 |",
        "|---|---|---|---|",
    ]
    for name, res in results.items():
        a1 = res["axis1"]
        a2 = res["axis2"]
        a3 = res["axis3"]
        p90 = f"{a1['p90_tokens']:.1f} ({a1['convergence_rate']:.0%})" if a1["p90_tokens"] else "収束せず"
        sep = f"{a2['separation']:+.3f}" if "separation" in a2 else "-"
        stab = a3.get("tokens_generated", "-")
        lines.append(f"| {name} | {p90} | {sep} | {stab} |")
    lines += [
        "",
        "## 採用判定（§1.2）",
        "",
        "- いずれか1軸でも著しく弱い候補は本採用から除外すること。",
        "- 本レポートの結果と採用理由を追記して確定し、config の `model.path` に反映する。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[model_selection] レポート出力: {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="モデル選定フェーズの比較実験（仕様書 §1）")
    parser.add_argument("--config-list", nargs="*", default=[], help="候補モデルの config YAML パス一覧")
    parser.add_argument("--mock", action="store_true", help="モックでスモークテスト（実モデル不要）")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--stability-minutes", type=int, default=0)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--delay-ms", type=float, default=1.0)
    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
