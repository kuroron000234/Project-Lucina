"""チャットテンプレート検証スクリプト。Qwen3.5-9B で metadata / formatter を確認する。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucina.inference.backends import LlamaBackend  # noqa: E402

b = LlamaBackend("models/qwen3.5-9b-q4_k_m.gguf", n_ctx=8192, n_gpu_layers=-1)
llm = b._llm  # noqa: SLF001
try:
    meta = getattr(llm, "metadata", None) or {}
    keys = [k for k in meta.keys() if "chat" in k or "template" in k]
    print("metadata chat/template keys:", keys)
    tpl = meta.get("tokenizer.chat_template")
    print("has template:", bool(tpl))
    if tpl:
        print("template head:", tpl[:120].replace("\n", "\\n"))
        try:
            print("token_eos():", repr(llm.token_eos()))
            print("token_bos():", repr(llm.token_bos()))
        except Exception as e:  # noqa: BLE001
            print("token_eos/bos error:", type(e).__name__, e)
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        fmt = Jinja2ChatFormatter(
            template=tpl,
            eos_token=str(llm.token_eos()),
            bos_token=str(llm.token_bos()),
            add_generation_prompt=True,
        )
        resp = fmt(messages=[{"role": "user", "content": "あなたは考える存在です。今の気持ちを話してください。"}])
        print("--- formatted prompt ---")
        print(resp.prompt)
        print("--- tokens ---")
        ids = b.encode(resp.prompt)
        print("n_tokens:", len(ids), "first3:", ids[:3])
finally:
    b.close()
