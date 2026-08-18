#!/usr/bin/env python3
"""一時デバッグ: LlamaBackend の next_token_logits がプロンプトに依存するか確認する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lucina.inference.backends import LlamaBackend

model = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

b = LlamaBackend(model, n_ctx=8192, n_gpu_layers=-1)
prompts = ["「今、とても」", "今日は一人です", "The quick brown fox", "1 + 1 = 2 です。"]

prev = None
for p in prompts:
    ids = b.encode(p)
    l = b.next_token_logits(p)
    top = np.argsort(l)[-5:][::-1]
    dec = [b.decode(int(t)) for t in top]
    same = "SAME" if prev is not None and np.array_equal(prev, l) else "diff"
    print(f"{same} | {p!r}: n_tokens={len(ids)} | top5={list(zip(top, dec))}")
    prev = l
