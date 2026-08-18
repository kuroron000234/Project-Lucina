"""llm-jp-4 / gemma-4 のチャットテンプレートの思考制御機構を確認する。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.inference.backends import LlamaBackend  # noqa: E402

for name, path in [
    ("llm-jp-4", "models/llm-jp-4-8b-thinking-q4_k_m.gguf"),
    ("gemma-4", "models/gemma-4-12b-it-q4_k_m.gguf"),
]:
    b = LlamaBackend(path, n_ctx=8192, n_gpu_layers=-1)
    try:
        meta = getattr(b._llm, "metadata", None) or {}  # noqa: SLF001
        tpl = meta.get("tokenizer.chat_template", "")
        print(f"===== {name}: template len={len(tpl)} =====")
        # thinking 関連の行を抽出
        for i, line in enumerate(tpl.splitlines()):
            low = line.lower()
            if "think" in low or "reasoning" in low or "message" in low:
                print(f"  {i}: {line.strip()[:160]}")
        print()
    finally:
        b.close()
