"""エンジン内部のロジット処理を直接再現して確認する。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from lucina.config import load_config  # noqa: E402

PROMPT = "あなたは考える存在です。今の気持ちを話してください。"


async def main() -> None:
    config = load_config("config/candidate_llmjp4.yaml")
    from run_agent import build_real_core

    core = build_real_core(config)
    print("core built", flush=True)
    try:
        core.reset_for_trial()
        core.drives["loneliness"] = 0.9
        core.seed_prompt(PROMPT)
        prompt = "".join(core.buffer.items)
        print("prompt tail:", repr(prompt[-60:]), flush=True)
        backend = core.engine._backend  # noqa: SLF001
        raw = np.asarray(backend.next_token_logits(prompt), dtype=np.float64)
        biased = core.engine._logits_processor.apply(raw, core.drives, core.engine._vocab_map)  # noqa: SLF001
        for label, arr in (("raw", raw), ("biased", biased)):
            tops = np.argsort(arr)[-5:][::-1]
            print(f"=== {label} top5 ===", flush=True)
            for t in tops:
                print(f"  {int(t)}: {backend.decode(int(t))!r} logit={arr[int(t)]:.2f}", flush=True)
        # サンプリング
        tok = core.engine._sample(biased)  # noqa: SLF001
        print("sampled:", tok, "->", repr(backend.decode(int(tok))), flush=True)
    finally:
        core.close()


if __name__ == "__main__":
    asyncio.run(main())
