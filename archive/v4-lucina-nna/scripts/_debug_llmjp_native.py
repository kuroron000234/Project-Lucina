"""llm-jp-4-8b-thinking のネイティブ応答と生ロジットの確認。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.inference.backends import LlamaBackend  # noqa: E402

PROMPT = "あなたは考える存在です。今の気持ちを話してください。"

b = LlamaBackend("models/llm-jp-4-8b-thinking-q4_k_m.gguf", n_ctx=8192, n_gpu_layers=-1)
try:
    # 1) ネイティブ create_chat_completion（チャットテンプレート自動適用）
    print("=== native create_chat_completion ===", flush=True)
    res = b._llm.create_chat_completion(  # noqa: SLF001
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=60,
        temperature=0.8,
    )
    print("response:", repr(res["choices"][0]["message"]["content"]), flush=True)
    if "reasoning" in res["choices"][0]["message"]:
        print("reasoning:", repr(res["choices"][0]["message"]["reasoning"])[:200], flush=True)

    # 2) 生ロジット: フォーマット済みプロンプトの次トークン分布
    print("=== raw logits top5 ===", flush=True)
    fmt = b.format_chat_prompt(PROMPT, reasoning_effort="off")
    print("formatted:", repr(fmt[-120:]), flush=True)
    logits = b.next_token_logits(fmt)
    import numpy as np

    tops = np.argsort(logits)[-8:][::-1]
    for t in tops:
        print(f"  {int(t)}: {b.decode(int(t))!r} logit={logits[int(t)]:.2f}", flush=True)
finally:
    b.close()
