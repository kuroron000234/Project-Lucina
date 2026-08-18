#!/usr/bin/env python3
"""一時デバッグ: シード語彙と日本語候補トークンの埋め込み類似度を確認する。"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))

from lucina.config import load_config
from lucina.inference.backends import LlamaBackend
from lucina.inference.adapters import LlamaTokenizerAdapter, SentenceTransformerEmbedder

config = load_config("config/candidate_qwen3.yaml")
backend = LlamaBackend(config["model"]["path"], n_ctx=8192, n_gpu_layers=-1)
tokenizer = LlamaTokenizerAdapter(backend)
embedder = SentenceTransformerEmbedder(config["embedding"]["model"], device="cpu")

seeds = ["寂しい", "孤独", "一人", "会いたい", "誰か", "独り言"]
cands = ["寂", "会", "一緒に", "話", "誰", "独り", "仲間", "友達", "孤独感", "一人きり"]

seed_embs = np.stack([embedder.embed(w) for w in seeds])
cand_embs = np.stack([embedder.embed(w) for w in cands])
sims = cand_embs @ seed_embs.T
max_sim = sims.max(axis=1)
for w, ms in zip(cands, max_sim):
    print(f"候補'{w}': max_sim={ms:.3f}")

# シード同士の類似度
print("--- シード同士 ---")
for i, a in enumerate(seeds):
    for j, b in enumerate(seeds):
        if i < j:
            s = float(seed_embs[i] @ seed_embs[j])
            print(f"  {a} vs {b}: {s:.3f}")
backend.close()
