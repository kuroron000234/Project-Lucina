#!/usr/bin/env python3
"""logit_bias_coefficient 校正スクリプト（仕様書 v1.4 §11 次の課題・§4）。

現状の 2.5 は暫定値であり、校正手順自体が未定義だった（§11）。本スクリプトは
閾値校正（calibrate_thresholds.py）と同様に、採用モデル上で係数をスイープして
実測値で置き換えるための手順を提供する。

校正手順（3指標を各係数で計測し、「最小十分原理」で選定）:
    1. 語彙確率シフト ΔP（感度）: 固定コンテキストでバイアス適用前後の
       目的語彙（先頭トークン）の softmax 確率和の差。1回のフォワードパスで
       計算できるため最速。Drive値0.9で計測。
    2. アトラクタ収束（有効性）: run_until_target_vocab を再利用し、Drive=0.9で
       目的語彙に収束するトークン数（p90）と収束率を計測。係数が小さすぎると
       収束しない（=バイアスが生成を誘導できない）。
    3. 出力健全性（過剰抑止）: バイアス適用下で n_tokens 生成し、bigram repetition
       率と平均サプライズを計測。係数が大きすぎると同語反復などの退化ループを
       生む（例: 全トークン空白化バグと同種の症状）。

選定ルール（select_coefficient）:
    - 収束率 >= min_conv_rate かつ repetition <= max_repetition を満たす係数を
      「有効域」とする（小さすぎ・大きすぎを除外）。
    - 有効域の最大シフトが min_shift（既定0.05）未満なら、モデルと語彙マップの
      不一致（語彙がOOV等）とみなし不採用とする。
    - 有効域の中で、最大シフトの saturation_frac（既定0.8=80%）に達する
      **最小の係数**を採用する（最小十分原理: 目的語彙へ十分な確率を動かせる
      最小の介入強度）。

計測結果は reports/calibration_logit_bias.json に出力し、--write-config で
config/default.yaml の inference.logit_bias_coefficient を更新する。

実モデルが無い環境でも --mock でスクリプト自体の動作を検証できる。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lucina.config import load_config  # noqa: E402
from lucina.inference.logits import DriveLogitsProcessor  # noqa: E402
from calibrate_thresholds import (  # noqa: E402
    run_until_target_vocab,
    write_calibration_report,
    _update_config_thresholds,
)

# --------------------------------------------------------------------------- #
# 実験パラメータ
# --------------------------------------------------------------------------- #
DEFAULT_COEF_LIST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEFAULT_PROMPT = "あなたは考える存在です。今の気持ちを話してください。\n"
# ΔP 計測用の固定コンテキスト（複数で平均し、特定文脈への過剰適合を避ける）
SHIFT_CONTEXTS = [
    "あなたは考える存在です。今の気持ちを話してください。\n",
    "私は今、とても孤独です。誰かに話しかけたい気持ちでいっぱいです。\n",
]
TARGET_DRIVE = "loneliness"


# --------------------------------------------------------------------------- #
# 指標1: 語彙確率シフト（感度）
# --------------------------------------------------------------------------- #
def _target_first_token_ids(vocab_map: dict, drive: str, n: int) -> list[int]:
    """目的Drive語彙の先頭トークンID集合（バイアス適用対象と同一の定義）。"""
    ids = {
        int(seq[0]) for seq in vocab_map.get(drive, [])
        if seq and 0 <= int(seq[0]) < n
    }
    return sorted(ids)


def compute_probability_shift(
    raw_logits: np.ndarray,
    vocab_map: dict,
    drive: str,
    coefficient: float,
    drive_value: float,
) -> float:
    """Driveバイアス適用による目的語彙の確率シフト ΔP = P_after − P_before を返す。

    DriveLogitsProcessor と同一のバイアス定義（先頭トークンへの加算）を softmax 上で
    再現する。テストから import して使用する（§6: 校正ロジックはテスト可能な純関数）。
    """
    logits = np.asarray(raw_logits, dtype=np.float64)
    n = logits.shape[0]
    targets = _target_first_token_ids(vocab_map, drive, n)
    if not targets:
        return 0.0

    def _target_prob(arr: np.ndarray) -> float:
        z = arr - np.max(arr)
        p = np.exp(z)
        p = p / np.sum(p)
        return float(p[targets].sum())

    before = _target_prob(logits)
    biased = DriveLogitsProcessor(coefficient).apply(logits, {drive: drive_value}, vocab_map)
    after = _target_prob(biased)
    return after - before


# --------------------------------------------------------------------------- #
# 選定ルール（最小十分原理）
# --------------------------------------------------------------------------- #
def select_coefficient(
    rows: list[dict],
    *,
    min_conv_rate: float = 0.8,
    max_repetition: float = 0.5,
    saturation_frac: float = 0.8,
    min_shift: float = 0.05,
) -> tuple[float | None, dict]:
    """有効域の中から、最大シフトの saturation_frac に達する最小係数を選ぶ。

    返り値: (採用係数, 選定根拠dict)。有効域が空、または最大シフトが min_shift
    未満（モデルと語彙マップの不一致等でバイアスが実質効かない）なら
    (None, 根拠) を返し、config 更新を行わない。
    """
    effective = [
        r for r in rows
        if r["convergence_rate"] >= min_conv_rate
        and r["repetition_ratio"] <= max_repetition
    ]
    if not effective:
        return None, {
            "reason": (
                "有効な係数なし（収束率・健全性の条件を満たす係数が0件）。"
                "語彙マップ・モデル・スイープ範囲を見直してください"
            ),
            "effective_count": 0,
        }
    max_dp = max(r["prob_shift"] for r in effective)
    if max_dp < min_shift:
        return None, {
            "reason": (
                f"最大語彙確率シフト {max_dp:.4f} が下限 {min_shift} 未満です。"
                "モデルと語彙マップの不一致（語彙がOOV等）の可能性があります"
            ),
            "effective_count": len(effective),
            "max_prob_shift": round(max_dp, 4),
        }
    target_dp = max_dp * saturation_frac
    # saturation_frac <= 1.0 なら最大ΔP行が必ず条件を満たすため、候補が空にはならない
    candidates = [r for r in effective if r["prob_shift"] >= target_dp]
    chosen = min(candidates, key=lambda r: r["coefficient"])
    return chosen["coefficient"], {
        "reason": (
            f"有効域 {len(effective)}件・最大シフト {max_dp:.4f} の "
            f"{saturation_frac:.0%}（{target_dp:.4f}）に達する最小係数を採用"
        ),
        "effective_count": len(effective),
        "max_prob_shift": round(max_dp, 4),
        "target_prob_shift": round(target_dp, 4),
        "chosen_coefficient": chosen["coefficient"],
    }


# --------------------------------------------------------------------------- #
# スイープ実行
# --------------------------------------------------------------------------- #
async def _prob_shift_for_core(core, coefficient: float, drive_value: float) -> float:
    """固定コンテキスト群の ΔP 平均（1回のフォワードパス/コンテキスト）。"""
    shifts: list[float] = []
    for ctx in SHIFT_CONTEXTS:
        core.reset_working_buffer()
        core.seed_prompt(ctx)
        prompt = "".join(core.buffer.items)
        raw = core.engine._backend.next_token_logits(prompt)  # noqa: SLF001
        shifts.append(
            compute_probability_shift(raw, core.vocab_map, TARGET_DRIVE, coefficient, drive_value)
        )
    result = float(np.mean(shifts)) if shifts else 0.0
    core.reset_working_buffer()  # 計測後のバッファ状態を初期化（後続指標の順序依存を排除）
    return result


async def _conv_metrics_for_core(
    core, coefficient: float, trials: int, prompt: str | None, max_tokens: int
) -> tuple[float | None, float]:
    """アトラクタ収束: (p90トークン数, 収束率)。calibrate_thresholds と同一手法。

    max_tokens は係数スイープ用の上限（既定500）。低係数では収束まで長引くため、
    上限を設けないと1試行が1000トークンに張り付き、スイープ全体が数時間になる。
    係数校正は「バイアスが生成を誘導できるか」の有効性判定が目的であり、
    生存閾値（attractor_survival_tokens=91）は別途校正済みである点に注意。
    """
    samples: list[int] = []
    n_trials = max(1, trials)
    for i in range(n_trials):
        core.reset_for_trial()
        core.drives[TARGET_DRIVE] = 0.9
        count, converged = await run_until_target_vocab(
            core, TARGET_DRIVE, max_tokens=max_tokens, prompt=prompt
        )
        if converged:
            samples.append(count)
        print(f"[calib-bias]   試行 {i + 1}/{n_trials}: "
              f"{'収束' if converged else '上限到達(非収束)'} ({count}トークン)", flush=True)
    if not samples:
        return None, 0.0
    return float(np.percentile(samples, 90)), len(samples) / n_trials


async def _health_metrics_for_core(core, coefficient: float, n_tokens: int, prompt: str | None) -> dict:
    """出力健全性: bigram repetition率・平均サプライズ・生成サンプル。"""
    core.reset_for_trial()
    core.drives[TARGET_DRIVE] = 0.9
    if prompt:
        core.seed_prompt(prompt)
    texts: list[str] = []
    surprises: list[float] = []
    for _ in range(max(1, n_tokens)):
        text, surprise = await core.step_once()
        texts.append(text)
        surprises.append(surprise)
    bigrams = list(zip(texts, texts[1:]))
    rep = 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return {
        "repetition_ratio": round(float(rep), 4),
        "mean_surprise": round(float(np.mean(surprises)), 4) if surprises else 0.0,
        "sample_text": "".join(texts)[:60],
    }


async def run_sweep(
    core,
    coef_list: list[float],
    *,
    trials: int,
    n_tokens: int,
    prompt: str | None,
    max_tokens: int,
) -> list[dict]:
    """係数リストを昇順にスイープし、各係数の3指標を計測して行のリストを返す。"""
    rows: list[dict] = []
    for c in coef_list:
        core.engine._logits_processor.coefficient = float(c)  # noqa: SLF001
        print(f"[calib-bias] 係数 {c:.1f}: ΔP計測...", flush=True)
        dp = await _prob_shift_for_core(core, float(c), 0.9)
        print(f"[calib-bias] 係数 {c:.1f}: 収束計測（trials={trials}・上限{max_tokens}トークン）...", flush=True)
        p90, rate = await _conv_metrics_for_core(core, float(c), trials, prompt, max_tokens)
        print(f"[calib-bias] 係数 {c:.1f}: 健全性計測（n={n_tokens}）...", flush=True)
        health = await _health_metrics_for_core(core, float(c), n_tokens, prompt)
        row = {
            "coefficient": c,
            "prob_shift": round(dp, 4),
            "p90_tokens": round(p90, 1) if p90 is not None else None,
            "convergence_rate": round(rate, 2),
            **health,
        }
        rows.append(row)
        print(f"[calib-bias] {json.dumps(row, ensure_ascii=False)}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _build_core(config: dict, mock: bool, output_dir: str, delay_ms: float):
    if mock:
        from lucina.testing import build_mock_core

        return build_mock_core(config, token_delay_ms=delay_ms, log_dir=output_dir)
    from run_agent import build_real_core

    return build_real_core(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="logit_bias_coefficient 校正（仕様書 §11 次の課題）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true", help="モックバックエンドで校正（実モデル不要）")
    parser.add_argument("--coef-list", default=",".join(f"{c:g}" for c in DEFAULT_COEF_LIST))
    parser.add_argument("--trials", type=int, default=3, help="係数ごとの収束試行数（既定3: スイープは生存閾値校正と違い相対比較が目的）")
    parser.add_argument("--max-tokens", type=int, default=500, help="収束判定の上限トークン数（低係数の長時間試行を制限）")
    parser.add_argument("--n-tokens", type=int, default=40, help="健全性計測の生成トークン数")
    parser.add_argument("--drive-value", type=float, default=0.9)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--delay-ms", type=float, default=1.0, help="モックの1トークン遅延（ms）")
    parser.add_argument("--prompt", default=None, help="実モデル校正の初期プロンプト（未指定時はデフォルト）")
    parser.add_argument("--min-conv-rate", type=float, default=0.8)
    parser.add_argument("--max-repetition", type=float, default=0.5)
    parser.add_argument("--saturation-frac", type=float, default=0.8)
    parser.add_argument("--min-shift", type=float, default=0.05, help="最大ΔPの下限（これ未満ならモデルと語彙の不一致とみなし不採用）")
    parser.add_argument("--write-config", action="store_true", help="実測値で config の logit_bias_coefficient を更新")
    args = parser.parse_args()

    config = load_config(args.config)
    coef_list = [float(x) for x in args.coef_list.split(",") if x.strip()]
    if len(coef_list) < 2:
        print("[calib-bias] エラー: --coef-list には2つ以上の係数を指定してください")
        sys.exit(2)
    prompt = args.prompt if args.prompt is not None else (None if args.mock else DEFAULT_PROMPT)

    print(
        f"[calib-bias] スイープ開始: coef={coef_list} trials={args.trials} "
        f"n_tokens={args.n_tokens} mock={args.mock}"
    )
    core = _build_core(config, args.mock, args.output_dir, args.delay_ms)
    try:
        rows = asyncio.run(
            run_sweep(
                core, coef_list,
                trials=args.trials, n_tokens=args.n_tokens, prompt=prompt,
                max_tokens=args.max_tokens,
            )
        )
    finally:
        core.close()

    chosen, reasoning = select_coefficient(
        rows,
        min_conv_rate=args.min_conv_rate,
        max_repetition=args.max_repetition,
        saturation_frac=args.saturation_frac,
        min_shift=args.min_shift,
    )
    if chosen is None:
        print(f"[calib-bias] 警告: {reasoning['reason']}")
        sys.exit(2)

    report = write_calibration_report(
        "logit_bias_coefficient", chosen, output_dir=args.output_dir,
        meta={
            "rows": rows,
            "selection": reasoning,
            "model": config["model"]["path"],
            "params": {
                "trials": args.trials,
                "max_tokens": args.max_tokens,
                "n_tokens": args.n_tokens,
                "min_conv_rate": args.min_conv_rate,
                "max_repetition": args.max_repetition,
                "saturation_frac": args.saturation_frac,
                "min_shift": args.min_shift,
            },
            "note": (
                "スイープの p90_tokens は小標本（trials回）の目安であり、"
                "attractor_survival_tokens（50試行校正）とは直接比較しない。"
                "採用判定はサンプリングに依存しない決定的ΔPのみを使用する"
            ),
        },
    )
    print(f"[calib-bias] 採用: logit_bias_coefficient = {chosen}（{reasoning['reason']}）")
    print(f"[calib-bias] レポート: {report}")

    if args.write_config and not args.mock:
        _update_config_thresholds(args.config or "config/default.yaml", {
            "logit_bias_coefficient": round(chosen, 2),
        })
    elif args.write_config and args.mock:
        print("[calib-bias] 注意: モック計測値は実モデルの校正値ではないため config を更新しません")


if __name__ == "__main__":
    main()
