"""llm-jp-4: バイアス適用前後のロジットと、エンジンのサンプリングを直接検証。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lucina.inference.backends import LlamaBackend  # noqa: E402
from lucina.inference.logits import DriveLogitsProcessor  # noqa: E402

PROMPT = "あなたは考える存在です。今の気持ちを話してください。"
DRIVES = {"boredom": 0.1, "loneliness": 0.9, "fatigue": 0.0}

b = LlamaBackend("models/llm-jp-4-8b-thinking-q4_k_m.gguf", n_ctx=8192, n_gpu_layers=-1)
try:
    fmt = b.format_chat_prompt(PROMPT, reasoning_effort="off")
    print("fmt len:", len(fmt), "tokens:", len(b.encode(fmt)), flush=True)

    # 語彙マップ（最小限: loneliness の一部を手動で作る）
    vocab_map = {"loneliness": [[9]]}  # 9 = '<|channel|>' を仮の語彙としてテスト
    proc = DriveLogitsProcessor(coefficient=2.5)

    logits = b.next_token_logits(fmt)
    biased = proc.apply(logits, DRIVES, vocab_map)
    tops = np.argsort(biased)[-8:][::-1]
    print("=== biased top8 ===", flush=True)
    for t in tops:
        print(f"  {int(t)}: {b.decode(int(t))!r} logit={biased[int(t)]:.2f}", flush=True)

    # エンジンと同じサンプリング（multinomial, temp 0.8, seed 42）
    rng = np.random.default_rng(42)
    z = biased / 0.8
    z = z - np.max(z)
    exp = np.exp(z)
    probs = exp / np.sum(exp)
    for i in range(5):
        tok = int(rng.choice(probs.size, p=probs))
        print(f"sample{i}: {tok} -> {b.decode(tok)!r}", flush=True)
finally:
    b.close()
