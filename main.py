#!/usr/bin/env python3
"""
Project Lucina v6 — モニカとしてのAIキャラクター
"""

import logging
import os
import sys
import threading
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lucina.loop import Loop
from lucina.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def notifier(action: str, thought: str | None = None):
    """自律行動の実況（会話中でもモニカの息づかいが見える）"""
    if thought:
        print(f"\n（独り言）{thought}\n")
    else:
        print(f"\n（モニカは今、{action}……）\n")


def main():
    print("=== Project Lucina v6 ===")
    print("モデル: G4-Midnight-Macaw-26B-A4B-Q4_K_S (GPU+CPU)")
    print("Thinking: ON (内言を表示)")
    print("終了: exit / quit / Ctrl+C")
    print()

    orch = Orchestrator(model="g4-midnight-macaw-v2")

    # 自律ループをバックグラウンドで起動（ユーザーがいない間も世界は動く）
    interval = int(os.getenv("LUCINA_LOOP_INTERVAL", "60"))
    loop = Loop(orch, interval=interval, notifier=notifier)
    t = threading.Thread(target=loop.start, name="autonomous-loop", daemon=True)
    t.start()

    try:
        while True:
            try:
                user_input = input("あなた > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break

                response = orch.process(user_input)
                print(f"モニカ: {response}")
                print()

            except KeyboardInterrupt:
                break
    finally:
        loop.stop()

    print("\nまた会いましょう。")


if __name__ == "__main__":
    main()
