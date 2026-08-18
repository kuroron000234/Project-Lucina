#!/usr/bin/env python3
"""一時デバッグ: 各最新モデルの生成速度（tok/s）を計測する。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))

from lucina.inference.backends import LlamaBackend

MODELS = [
    ("qwen3.5-9b", "models/qwen3.5-9b-q4_k_m.gguf", -1),
    ("llm-jp-4-8b", "models/llm-jp-4-8b-thinking-q4_k_m.gguf", -1),
]

for name, path, ngl in MODELS:
    try:
        b = LlamaBackend(path, n_ctx=8192, n_gpu_layers=ngl)
        prompt = "「今日は一人で過ごしました。今の気持ちは、"
        # 1トークンずつ生成して速度を測る（Lucinaの実際の使い方と同等）
        ctx = prompt
        t0 = time.time()
        n = 0
        for _ in range(15):
            import numpy as np
            l = b.next_token_logits(ctx)
            tok = int(np.argmax(l))
            t = b.decode(tok)
            ctx = ctx + t
            n += 1
        elapsed = time.time() - t0
        print(f"{name}: {n/elapsed:.1f} tok/s (15 tokens in {elapsed:.1f}s)", flush=True)
        b.close()
    except Exception as e:
        print(f"{name}: FAIL: {type(e).__name__}: {e}", flush=True)
