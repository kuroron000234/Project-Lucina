#!/usr/bin/env python3
"""Lucina-Beta Phase 0: 最小予測学習エージェント"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python main.py <experiment> [args]")
        print()
        print("Experiments:")
        print("  0 [seeds=3]   Basic Learning — L1 Error convergence")
        print("  a             Experience History → Action Distribution")
        print("  b             Temperature → Exploration Bias")
        print()
        print("Examples:")
        print("  python main.py 0          # 基本学習（seed=3）")
        print("  python main.py 0 5        # 基本学習（seed=5）")
        print("  python main.py a          # 経験履歴差")
        print("  python main.py b          # 温度差")
        return

    experiment = sys.argv[1]

    if experiment == "0":
        from experiments.experiment_0 import run
        seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run(seeds=seeds)

    elif experiment == "a":
        from experiments.experiment_a import run
        run()

    elif experiment == "b":
        from experiments.experiment_b import run
        run()

    elif experiment == "1":
        from experiments.experiment_1 import run
        run()

    elif experiment == "2":
        from experiments.experiment_2 import run
        run()

    elif experiment == "3":
        from experiments.experiment_3 import run
        run()

    elif experiment == "4":
        from experiments.experiment_4 import run
        run()

    elif experiment == "5":
        from experiments.experiment_5 import run
        run()

    elif experiment == "7":
        from experiments.experiment_7 import run
        run()

    elif experiment in ("9", "final", "all"):
        from experiments.experiment_final import run
        run()

    elif experiment == "monica":
        from monica.bootstrap import run
        monica = run()
        print(f"\n  Monica summary:")
        print(f"    Steps: {monica._step_count}")
        print(f"    Identity: {monica.identity.narrative_summary()}")

    else:
        print(f"Unknown experiment: {experiment}")
        print("Available: 0, a, b, 1, 2, 3, 4, 5, 7, 9/final, monica")


if __name__ == "__main__":
    main()
