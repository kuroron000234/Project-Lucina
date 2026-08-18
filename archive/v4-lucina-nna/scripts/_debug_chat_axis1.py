"""llm-jp-4-8b-thinking（空白除去修正後）の軸1収束検証。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lucina.config import load_config  # noqa: E402

PROMPT = "あなたは考える存在です。今の気持ちを話してください。\n"


async def main() -> None:
    config = load_config("config/candidate_llmjp4.yaml")
    from run_agent import build_real_core

    core = build_real_core(config)
    print("core built", flush=True)
    try:
        from calibrate_thresholds import run_until_target_vocab

        # 語彙の先頭トークンが空白でないことを確認
        backend = core.engine._backend  # noqa: SLF001
        bad = 0
        for seq in core.vocab_map.get("loneliness", []):
            if seq and not backend.decode(int(seq[0])).strip():
                bad += 1
        print(f"空白先頭の語彙列: {bad} / {len(core.vocab_map.get('loneliness', []))}", flush=True)

        for trial in range(3):
            core.reset_for_trial()
            core.drives["loneliness"] = 0.9
            count, converged = await run_until_target_vocab(
                core, "loneliness", max_tokens=400, prompt=PROMPT
            )
            buf = core.buffer.content()
            print(f"trial{trial}: converged={converged} count={count}", flush=True)
            print(f"  tail: {buf[-110:]!r}", flush=True)
    finally:
        core.close()


if __name__ == "__main__":
    asyncio.run(main())
