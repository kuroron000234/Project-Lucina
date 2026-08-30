#!/usr/bin/env python3
"""
Project Lucina v6 — モニカとしてのAIキャラクター
"""

import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lucina.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    print("=== Project Lucina v6 ===")
    print("モデル: G4-Midnight-Macaw-26B-A4B-Q4_K_S (GPU+CPU)")
    print("Thinking: ON (内言を表示)")
    print("終了: exit / quit / Ctrl+C")
    print()

    orch = Orchestrator(model="g4-midnight-macaw-v2")

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

    print("\nまた会いましょう。")


if __name__ == "__main__":
    main()
