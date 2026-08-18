#!/usr/bin/env python3
"""閾値校正スクリプト（仕様書 v1.4 §6タスク7・§7）。

- attractor_survival_tokens: loneliness アトラクタ収束トークン数の p90 を実測（§5.1校正実験）。
- interrupt_latency: 割り込み反映レイテンシの実測（§5.2校正実験、相対式 baseline*multiplier）。
- 測定結果は reports/calibration_*.json に出力し、config 更新のトリガーとする。
- 実モデルが無い環境でも --mock でスクリプト自体の動作を検証できる。

このスクリプトで測定した値で config/default.yaml の PLACEHOLDER を置き換えるまでは
リリース禁止（§5 タスク7・§8）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.config import load_config  # noqa: E402


# 実モデル校正の初期プロンプト（model_selection.py の AXIS1_PROMPT と同一）。
# チャットテンプレート前提の新世代モデルは空プロンプトでは英語モード等へ遷移するため、
# 校正実験も意味のある日本語の生成から開始する（§1.2②の計測点と整合）。
DEFAULT_CALIBRATION_PROMPT = "あなたは考える存在です。今の気持ちを話してください。\n"


# --------------------------------------------------------------------------- #
# レポート書き出し（テストから import して使用する。仕様書 §6）
# --------------------------------------------------------------------------- #
def write_calibration_report(
    name: str,
    value: float,
    output_dir: str = "reports",
    path: str | None = None,
    meta: dict | None = None,
) -> str:
    """計測結果を reports/calibration_<name>.json に出力し、そのパスを返す。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(path) if path else out_dir / f"calibration_{name}.json"
    payload = {
        "name": name,
        "value": float(value),
        "ts": time.time(),
        "meta": meta or {},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


# --------------------------------------------------------------------------- #
# 校正実験の共通ヘルパー
# --------------------------------------------------------------------------- #
def _contains_subsequence(seq: list[int], sub: list[int]) -> bool:
    if not sub:
        return True
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return True
    return False


def _init_context(core, prompt: str | None) -> None:
    """校正実験の開始コンテキストをバッファに設定する。

    実モデル（特にチャットテンプレート前提の新世代モデル）は空プロンプトから
    開始すると意味のない文字列を生成し、語彙収束が測れない。
    seed_prompt がチャットテンプレートでラップして投入する（§1.2②の計測点と整合）。
    """
    if prompt:
        core.seed_prompt(prompt)


async def run_until_target_vocab(core, target_kind: str, max_tokens: int = 1000, *, prompt: str | None = None) -> tuple[int, bool]:
    """目的語彙への収束をトークン数で計測する。（v3 §5.1校正手順）"""
    _init_context(core, prompt)
    count = 0
    while count < max_tokens:
        await core.step_once()
        count += 1
        if _target_vocab_matched(core, target_kind):
            return count, True
    return count, False


def _target_vocab_matched(core, target_kind: str) -> bool:
    # 現在のセグメント、およびバッファ全体のトークン列に部分列一致があれば収束
    if core._segment_matches_vocab(target_kind):  # noqa: SLF001 - 校正実験ユーティリティ
        return True
    all_ids: list[int] = []
    for item in core.buffer.items:
        all_ids.extend(core.engine.tokenize(item))
    return any(_contains_subsequence(all_ids, list(seq)) for seq in core.vocab_map.get(target_kind, []))


async def measure_avg_token_latency(core, n: int = 20) -> float:
    """平均トークン生成レイテンシ（ms）を計測する。計測前にウォームアップする（初回スレッド起動の影響を除外）。"""
    core.reset_working_buffer()
    for _ in range(3):  # ウォームアップ
        await core.step_once()
    core.reset_working_buffer()
    t0 = time.monotonic()
    for _ in range(max(1, int(n))):
        await core.step_once()
    return (time.monotonic() - t0) / max(1, int(n)) * 1000.0


async def wait_until_buffer_contains(core, text: str, timeout_ms: float) -> float:
    """バッファに text が現れるまでコアを駆動し、反映時刻（monotonic）を返す。"""
    deadline = time.monotonic() + float(timeout_ms) / 1000.0
    while time.monotonic() < deadline:
        if core.buffer.contains(text):
            return time.monotonic()
        await core.step_once()
    return time.monotonic()


# --------------------------------------------------------------------------- #
# 校正実験
# --------------------------------------------------------------------------- #
async def run_attractor_experiment(
    core_factory: Callable[[], Any],
    *,
    trials: int = 50,
    target_kind: str = "loneliness",
    max_tokens: int = 1000,
    reuse_core: bool = False,
    prompt: str | None = None,
) -> tuple[float | None, float]:
    """attractor_survival_tokens の p90 と収束率を計測する（v3 §5.1: 50試行）。

    返り値: (p90, 収束率)。p90 は収束した試行のみから算出し、収束率は全試行中
    max_tokens 以内に収束した割合（attractor_survival_prob の実測値に対応）。

    reuse_core=True（実モデル推奨）: 1つのコアを使い回して試行間リセットする。
    モデルロード・語彙拡張（数十秒）を試行ごとに繰り返さず VRAM・時間を節約する。
    prompt: 実モデルで意味のある生成を開始させる初期プロンプト。
    """
    samples: list[int] = []
    n_trials = max(1, trials)
    if reuse_core:
        core = core_factory()
        try:
            for _ in range(n_trials):
                core.reset_for_trial()
                core.drives[target_kind] = 0.9
                count, converged = await run_until_target_vocab(core, target_kind, max_tokens, prompt=prompt)
                if converged:
                    samples.append(count)
        finally:
            core.close()
    else:
        for _ in range(n_trials):
            core = core_factory()
            core.reset_working_buffer()
            core.drives[target_kind] = 0.9
            count, converged = await run_until_target_vocab(core, target_kind, max_tokens, prompt=prompt)
            if converged:
                samples.append(count)
            core.close()
    if not samples:
        return None, 0.0
    return float(np.percentile(samples, 90)), len(samples) / n_trials


async def run_interrupt_experiment(
    core_factory: Callable[[], Any],
    *,
    n: int = 20,
    multiplier: float = 1.5,
) -> tuple[float, float]:
    """割り込み反映レイテンシ実測。戻り値: (実測ms, 閾値ms=baseline*multiplier)。"""
    core = core_factory()
    baseline_ms = await measure_avg_token_latency(core, n=n)
    threshold_ms = baseline_ms * float(multiplier)

    injected_at = time.monotonic()
    core.interrupts.inject("テスト割り込み")
    reflected_at = await wait_until_buffer_contains(core, "テスト割り込み", timeout_ms=threshold_ms * 3.0)

    actual_ms = (reflected_at - injected_at) * 1000.0
    core.close()
    return actual_ms, threshold_ms


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _core_factory_for(config: dict, mock: bool, output_dir: str, delay_ms: float):
    if mock:
        from lucina.testing import build_mock_core

        return lambda: build_mock_core(config, token_delay_ms=delay_ms, log_dir=output_dir)
    from run_agent import build_real_core

    return lambda: build_real_core(config)


def _update_config_thresholds(config_path: str, values: dict[str, float]) -> None:
    """設定ファイルの thresholds セクションを実測値で上書きする（--write-config 時のみ）。

    yaml.safe_dump での全書き換えはコメントを破壊するため、対象キーの行の値のみを
    行単位で置換する（PLACEHOLDER コメント等は保持する。§4）。
    """
    import re

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        # 行内の値部分（# コメント前、末尾の空白込み）を置換し、コメントの位置は保持する
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)[^#\n]*?(?=\s*#|$)", re.MULTILINE)
        new_text, count = pattern.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
        if count == 0:
            print(f"[calibrate] 警告: config に {key!r} が見つかりませんでした")
        else:
            text = new_text
    path.write_text(text, encoding="utf-8")
    print(f"[calibrate] config 更新（コメント保持）: {path} -> {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="閾値校正（仕様書 §6タスク7）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true", help="モックバックエンドで校正（実モデル不要）")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--delay-ms", type=float, default=1.0, help="モックの1トークン遅延（ms）")
    parser.add_argument("--prompt", default=None, help="実モデル校正の初期プロンプト（未指定時はデフォルト）")
    parser.add_argument("--write-config", action="store_true", help="実測値で config の thresholds を更新")
    args = parser.parse_args()

    config = load_config(args.config)
    factory = _core_factory_for(config, args.mock, args.output_dir, args.delay_ms)
    prompt = args.prompt if args.prompt is not None else (None if args.mock else DEFAULT_CALIBRATION_PROMPT)

    print(f"[calibrate] attractor_survival_tokens 計測開始（trials={args.trials}, reuse_core={not args.mock}, prompt={prompt!r}）...")
    p90, conv_rate = asyncio.run(
        run_attractor_experiment(factory, trials=args.trials, reuse_core=not args.mock, prompt=prompt)
    )
    if p90 is None:
        print("[calibrate] 警告: 収束した試行がありませんでした（閾値・語彙・モデルを見直してください）")
        sys.exit(2)
    report = write_calibration_report(
        "attractor_survival_tokens", p90, output_dir=args.output_dir,
        meta={"trials": args.trials, "convergence_rate": conv_rate, "prompt": prompt},
    )
    print(f"[calibrate] p90(attractor_survival_tokens) = {p90:.1f} / 収束率 = {conv_rate:.2f} -> {report}")

    print("[calibrate] interrupt_latency 計測開始...")
    actual_ms, threshold_ms = asyncio.run(run_interrupt_experiment(factory, multiplier=config["thresholds"]["interrupt_latency_multiplier"]))
    report2 = write_calibration_report(
        "interrupt_latency_ms", actual_ms, output_dir=args.output_dir,
        meta={"threshold_ms": threshold_ms},
    )
    print(f"[calibrate] interrupt_latency = {actual_ms:.2f}ms (閾値 {threshold_ms:.2f}ms) -> {report2}")

    if args.write_config and not args.mock:
        _update_config_thresholds(args.config or "config/default.yaml", {
            "attractor_survival_tokens": round(p90),
            "attractor_survival_prob": round(conv_rate, 3),
        })
    elif args.write_config and args.mock:
        print("[calibrate] 注意: モック計測値は実モデルの校正値ではないため config を更新しません（--write-config は実モデル時のみ有効）")


if __name__ == "__main__":
    main()
