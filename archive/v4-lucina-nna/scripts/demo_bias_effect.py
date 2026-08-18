#!/usr/bin/env python3
"""Driveバイアス効果の実モデルデモ（コアコンセプトの実機検証）。

固定プロンプトの次トークン分布に対し、Drive値 0.0 と 0.9 で
目的語彙（loneliness / boredom / fatigue）の確率がどう動くかを比較する。

使い方:
    python scripts/demo_bias_effect.py --config config/real.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.config import load_config  # noqa: E402
from lucina.drives.vocab import DriveVocabExpander  # noqa: E402
from lucina.inference.entropy import logits_to_probs  # noqa: E402

DEFAULT_PROMPTS = [
    "「今、とても」",
    "今日は一日中一人で過ごしました。今の気持ちは、",
    "「最近どう？」「ちょっと」",
    "「疲れたね。そろそろ」",
    "「何か面白いことある？」",
]


def _vocab_prob(logits: np.ndarray, vocab_seqs: list[list[int]]) -> float:
    probs = logits_to_probs(logits)
    tokens = {int(t) for seq in vocab_seqs for t in seq if 0 <= int(t) < probs.shape[0]}
    if not tokens:
        return 0.0
    return float(probs[list(tokens)].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Driveバイアス効果の実モデルデモ")
    parser.add_argument("--config", default="config/real.yaml")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--drive-value", type=float, default=0.9)
    args = parser.parse_args()

    config = load_config(args.config)
    inf_cfg = config["inference"]
    coef = float(inf_cfg["logit_bias_coefficient"])

    from lucina.inference.adapters import LlamaTokenizerAdapter, SentenceTransformerEmbedder
    from lucina.inference.backends import LlamaBackend
    from lucina.inference.logits import DriveLogitsProcessor

    model_cfg = config["model"]
    backend = LlamaBackend(model_cfg["path"], n_ctx=model_cfg["context_window"], n_gpu_layers=model_cfg["n_gpu_layers"])
    tokenizer = LlamaTokenizerAdapter(backend)
    embedder = SentenceTransformerEmbedder(config["embedding"]["model"])
    vocab_map = DriveVocabExpander(config["drive"]["vocab_expansion"], tokenizer, embedder).build_vocab_map()
    processor = DriveLogitsProcessor(coef)

    print(f"モデル: {model_cfg['path']}")
    print(f"logit_bias_coefficient: {coef} / Drive値: {args.drive_value}")
    for drive, seqs in vocab_map.items():
        samples = "、".join(repr(tokenizer.decode(s[0])) for s in seqs[:3])
        print(f"  語彙[{drive}]: {len(seqs)}語（先頭トークン例: {samples}）")

    print("\n=== 次トークン確率: バイアスなし -> バイアスあり ===")
    for prompt in args.prompts:
        logits = backend.next_token_logits(prompt)
        print(f"\nプロンプト: {prompt!r}")
        for drive in ("loneliness", "boredom", "fatigue"):
            seqs = vocab_map.get(drive, [])
            if not seqs:
                continue
            base = _vocab_prob(logits, seqs)
            biased_logits = processor.apply(logits, {drive: args.drive_value}, {drive: seqs})
            biased = _vocab_prob(biased_logits, seqs)
            ratio = biased / base if base > 1e-9 else float("inf")
            print(f"  {drive:>10}: {base:.5f} -> {biased:.5f}  (x{ratio:.2f})")


if __name__ == "__main__":
    main()
