#!/usr/bin/env python3
"""一時デバッグ: BOS除去修正後の tokenizer.encode と部分列一致を確認する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))

from lucina.inference.adapters import LlamaTokenizerAdapter
from lucina.inference.backends import LlamaBackend

MODELS = [
    ("qwen", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("gemma", "models/gemma-2-2b-it-q4_k_m.gguf"),
    ("elyza", "models/llama-3-elyza-jp-8b-q4_k_m.gguf"),
]

def contains(seq, sub):
    if not sub:
        return True
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return True
    return False

for name, path in MODELS:
    b = LlamaBackend(path, n_ctx=8192, n_gpu_layers=-1)
    tok = LlamaTokenizerAdapter(b)
    probe = tok.encode("寂しい")
    vocab_seq = tok.encode("寂しい")
    gen_seq = [101749, 124127, 224, 102800, 15973]  # BOSなしの生成トークン列イメージ
    # 実際に「寂しい」のBOSなしトークン列が、BOS付きエンコード列から除去されているか
    print(f"{name}: encode(寂しい)={probe} BOS={tok._bos_id}")
    print(f"  部分列一致(生成列vs語彙列)={'YES' if contains(gen_seq, vocab_seq) else 'NO'}")
    b.close()
