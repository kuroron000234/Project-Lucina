"""REPL: 自発的エージェントループ（Phase 13+）

AutonomousREPL は Individual の薄いCLIラッパー。
全てのオーケストレーションロジックは core/individual.py に移動済み。
"""

import time

from core.individual import Individual


def _show_step(entry: dict) -> None:
    """簡易ステップ表示。cli.display.show_step が未定義のためのフォールバック。"""
    action = entry.get("action", "?")
    outcome = entry.get("outcome", "?")
    pe = entry.get("pe", 0)
    internal = entry.get("internal", {})
    istate = f" [energy={internal.get('energy','?')}]" if internal else ""
    print(f"  [{entry.get('step','?')}] {action:12s} → {outcome:10s}  PE={pe:.3f}{istate}")


class AutonomousREPL:
    """自発的エージェントループ — Individual のCLIラッパー。"""

    def __init__(self, individual: Individual):
        self.individual = individual
        self.running = False

    def step(self) -> dict:
        entry = self.individual.step()
        _show_step(entry)
        return entry

    def run(self, n_steps: int = 10, delay: float = 0.0) -> list[dict]:
        logs = []
        self.running = True
        for _ in range(n_steps):
            if not self.running:
                break
            entry = self.step()
            logs.append(entry)
            if delay:
                time.sleep(delay)
        return logs

    def stop(self) -> None:
        self.running = False

    def summary(self) -> dict:
        return {
            "steps": self.individual._step_count,
            "development": self.individual.development.summary(),
        }
