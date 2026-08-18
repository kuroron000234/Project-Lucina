#!/usr/bin/env python3
"""一時デバッグ: 初期プロンプトありで収束するか確認する（語彙一致をトークン単位で観察）。"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "src"))
sys.path.insert(0, str(Path(__file__).resolve()).replace("scripts", "scripts"))

from lucina.config import load_config
from run_agent import build_real_core

config = load_config("config/candidate_qwen3.yaml")
core = build_real_core(config)

async def check():
    core.reset_for_trial()
    core.drives["loneliness"] = 0.9
    core.buffer.append("あなたは考える存在です。今の気持ちを話してください。\n", n_tokens=core.engine.tokenize("あなたは考える存在です。今の気持ちを話してください。\n").__len__())
    matched_ids = {int(s[0]) for s in core.vocab_map.get("loneliness", []) if s}
    print("loneliness 先頭トークンID集合サイズ:", len(matched_ids), flush=True)
    for _ in range(60):
        text, surprise = await core.step_once()
        for tid in core.segment.token_ids[-1:]:
            if tid in matched_ids:
                print(f"  ヒット! token={tid} text={text!r}", flush=True)
        if core._segment_matches_vocab("loneliness"):
            print("セグメント一致!", flush=True)
    print("バッファ:", core.buffer.content()[-150:], flush=True)

asyncio.run(check())
core.close()
