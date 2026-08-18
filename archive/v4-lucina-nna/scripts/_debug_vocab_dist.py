#!/usr/bin/env python3
"""一時デバッグ: Qwen3.5-9B の語彙サイズと日本語トークンの分布を確認する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))

from lucina.inference.backends import LlamaBackend

b = LlamaBackend("models/qwen3.5-9b-q4_k_m.gguf", n_ctx=8192, n_gpu_layers=-1)
n = b.vocab_size()
print(f"vocab_size: {n}", flush=True)

import re
jp_re = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

# サンプリングで日本語トークンの分布を確認（全スキャンは遅いので10000件サンプリング）
import random
random.seed(42)
ids = random.sample(range(n), min(20000, n))
jp_count = 0
jp_samples = []
for tid in ids:
    w = b.decode(tid).strip()
    if jp_re.search(w):
        jp_count += 1
        if len(jp_samples) < 15:
            jp_samples.append((tid, w))

print(f"サンプル20000中の日本語トークン: {jp_count} ({jp_count/len(ids)*100:.1f}%)", flush=True)
print("日本語トークン例:", jp_samples[:10], flush=True)
b.close()
