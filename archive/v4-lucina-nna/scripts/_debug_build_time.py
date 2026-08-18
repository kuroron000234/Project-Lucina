#!/usr/bin/env python3
"""一時デバッグ: Qwen3.5-9B の build_real_core 構築時間を段階別に計測する。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))
sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "scripts"))

from lucina.config import load_config

config = load_config("config/candidate_qwen3.yaml")

from lucina.inference.backends import LlamaBackend

t0 = time.time()
backend = LlamaBackend(config["model"]["path"], n_ctx=config["model"]["context_window"], n_gpu_layers=config["model"]["n_gpu_layers"])
print(f"[1] LlamaBackend load: {time.time()-t0:.1f}s", flush=True)

from lucina.inference.adapters import LlamaTokenizerAdapter, SentenceTransformerEmbedder

t0 = time.time()
tokenizer = LlamaTokenizerAdapter(backend)
print(f"[2] tokenizer: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
embedder = SentenceTransformerEmbedder(config["embedding"]["model"], device="cpu")
print(f"[3] embedder load: {time.time()-t0:.1f}s", flush=True)

from lucina.drives.vocab import DriveVocabExpander

t0 = time.time()
vocab_map = DriveVocabExpander(config["drive"]["vocab_expansion"], tokenizer, embedder).build_vocab_map()
print(f"[4] vocab_expansion: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
backend.close()
print(f"[5] close: {time.time()-t0:.1f}s", flush=True)
