"""Persistence: 状態の永続化（Phase 14: 長期自律）

個体の状態をJSONファイルに保存・復元する。
Memory Consolidation と Idle Cycle の管理も行う。
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .memory import Memory, Episode


class Persistence:
    """個体の状態をファイルに保存・復元する。"""

    def __init__(self, save_dir: str = "saves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_individual(self, individual: Any, name: str = "individual") -> str:
        """個体の全状態を保存する。"""
        path = self.save_dir / f"{name}_{int(time.time())}.json"
        state = {
            "name": individual.name,
            "timestamp": time.time(),
            "development": individual.development.summary(),
            "self_model": individual.self_model.summary(),
            "values": individual.values.summary(),
            "identity": individual.identity.summary(),
            "memory": individual.memory.summary(),
            "metacognition": individual.metacognition.summary(),
            "total_steps": individual._step_count,
        }
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def list_saves(self, pattern: str = "individual_*.json") -> list[dict]:
        """保存された状態を一覧表示する。"""
        saves = []
        for f in sorted(self.save_dir.glob(pattern), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                saves.append({
                    "path": f.name,
                    "name": data.get("name", "?"),
                    "timestamp": data.get("timestamp", 0),
                    "steps": data.get("total_steps", 0),
                    "stage": data.get("development", {}).get("name", "?"),
                })
            except Exception:
                continue
        return saves


class MemoryConsolidator:
    """記憶圧縮と忘却の管理サイクル。"""

    def __init__(self, memory: Memory, interval: int = 50):
        self.memory = memory
        self.interval = interval
        self._cycle_count = 0

    def tick(self) -> Optional[dict]:
        """1サイクル実行。intervalに達したら圧縮を実行する。"""
        self._cycle_count += 1
        if self._cycle_count % self.interval == 0:
            return self.consolidate()
        return None

    def consolidate(self) -> dict:
        """記憶を圧縮する：

        1. 古い低PEエピソードを忘却
        2. 自伝的記憶から長期パターンを抽出
        3. 重複したセマンティック記憶をマージ
        """
        before = self.memory.summary()

        # 古い低PEエピソードを忘却（直近50以外でPE<0.1のものを削除）
        if len(self.memory.episodic) > 50:
            recent = self.memory.episodic[-50:]
            old_low_pe = [
                ep for ep in self.memory.episodic[:-50]
                if ep.pe < 0.1
            ]
            for ep in old_low_pe:
                if ep in self.memory.episodic:
                    self.memory.episodic.remove(ep)

        after = self.memory.summary()

        return {
            "before": before,
            "after": after,
            "episodes_pruned": before["episodic"] - after["episodic"],
        }

    def summary(self) -> dict:
        return {
            "cycle": self._cycle_count,
            "next_consolidation": self.interval - (self._cycle_count % self.interval),
            "memory": self.memory.summary(),
        }


class IdleCycle:
    """アイドル状態でのバックグラウンド認知処理。

    Idle中も内部状態は変動し、記憶の再活性化が行われる。
    """

    def __init__(
        self,
        memory: Memory,
        consolidate_interval: int = 10,
    ):
        self.memory = memory
        self.consolidator = MemoryConsolidator(memory, consolidate_interval)
        self.idle_steps = 0

    def step(self) -> dict:
        """1 idle cycle を実行する。"""
        self.idle_steps += 1
        result = {"idle_step": self.idle_steps}

        # 記憶圧縮
        consolidation = self.consolidator.tick()
        if consolidation:
            result["consolidation"] = consolidation

        return result

    def summary(self) -> dict:
        return {
            "idle_steps": self.idle_steps,
            "consolidator": self.consolidator.summary(),
        }
