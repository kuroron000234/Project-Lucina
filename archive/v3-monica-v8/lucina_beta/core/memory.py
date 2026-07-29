"""Memory: 3層記憶システム

Episodic Memory       — 「何が起きたか」（具体的事象）
Semantic Memory       — 「世界について何を知っているか」（汎化された知識）
Autobiographical Memory — 「自分に何が起きたか」（自己に関する記憶の連続体）

設計原則:
  - エピソードは自動的に保存される
  - 意味記憶はエピソードの反復から圧縮される
  - 自伝的記憶は自己関連性の高いエピソードから形成される
  - 忘却は自信度の低いエピソードから行われる
"""

import math
import time
from dataclasses import dataclass
from typing import Any

# 世界モデルの結果タイプ（循環インポート回避のため定数を定義）
OUTCOMES = ["food", "nothing", "danger"]


@dataclass
class Episode:
    """単一の経験エピソード。"""
    timestamp: float
    action: str
    outcome: str
    prediction: dict[str, float] | None = None
    pe: float = 0.0  # Prediction Error (surprise)
    ev: float = 0.0  # Expected Value before action
    internal_state: dict | None = None  # Snapshot of internal state
    interpretation: str = ""  # LLM interpretation (Phase 3+)
    self_relevant: bool = False  # Whether this is relevant to self-model
    importance: float = 1.0  # [0,1] how important is this memory


class Memory:
    """3層記憶システム。

    Parameters
    ----------
    capacity : int
        エピソード記憶の最大保存数（超過時は忘却）。
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity

        # 3層記憶
        self.episodic: list[Episode] = []
        self.semantic: dict[str, dict[str, float]] = {}  # "action|outcome" → count, total_pe
        self.autobiographical: list[Episode] = []

        # 統計
        self._total_episodes = 0
        self._last_consolidated_idx = 0  # 最後に圧縮したエピソードのインデックス

    # --- Storage ---

    def store(self, episode: Episode) -> None:
        """エピソードを全層に保存する。"""
        # 1. Episodic: 常に保存
        self.episodic.append(episode)
        self._total_episodes += 1

        # 容量超過 → 最も重要度の低いエピソードを忘却
        if len(self.episodic) > self.capacity:
            self._forget()

        # 2. Autobiographical: 自己関連性が高い場合のみ
        if episode.self_relevant or episode.pe > 0.5:
            self.autobiographical.append(episode)

        # 3. 意味記憶への圧縮（10件ごとに実行）
        if self._total_episodes % 10 == 0:
            self._consolidate()

    def store_from_history(self, entry: dict) -> Episode:
        """エージェントの history entry から Episode を作成して保存する。"""
        ep = Episode(
            timestamp=time.time(),
            action=entry.get("action", "?"),
            outcome=entry.get("outcome", "?"),
            prediction=entry.get("prediction"),
            pe=entry.get("pe", 0.0),
            ev=entry.get("ev", 0.0),
            internal_state=entry.get("internal"),
            interpretation=entry.get("interpretation", ""),
            self_relevant=entry.get("pe", 0) > 0.5 or entry.get("outcome") in ("danger", "food"),
            importance=max(0.1, min(1.0, entry.get("pe", 0))),
        )
        self.store(ep)
        return ep

    # --- Consolidation ---

    def _consolidate(self) -> None:
        """未圧縮のエピソードを意味記憶に圧縮する。

        _last_consolidated_idx 以降のエピソードのみを処理することで、
        同じエピソードを重複してカウントしない。
        """
        start = self._last_consolidated_idx
        end = len(self.episodic)
        if start >= end:
            return

        for i in range(start, end):
            ep = self.episodic[i]
            key = f"{ep.action}|{ep.outcome}"
            if key not in self.semantic:
                self.semantic[key] = {"count": 0, "total_pe": 0.0}
            self.semantic[key]["count"] += 1
            self.semantic[key]["total_pe"] += ep.pe

        self._last_consolidated_idx = end

    def _forget(self) -> None:
        """最も重要度の低いエピソードを忘却する。

        忘却後は _last_consolidated_idx を調整し、意味記憶との整合性を保つ。
        """
        if len(self.episodic) < 2:
            return
        # 重要度 × 経過時間で忘却スコアを計算
        now = time.time()
        scores = []
        for i, ep in enumerate(self.episodic):
            age = now - ep.timestamp if ep.timestamp > 0 else 0
            # 古い + 重要度低い = 忘れられやすい
            forget_score = (1.0 - ep.importance) * (1.0 + math.log1p(age))
            scores.append((forget_score, i))
        # 最も忘れられやすいものを削除
        scores.sort(reverse=True)
        del_idx = scores[0][1]
        self.episodic.pop(del_idx)

        # 削除位置より前の consolidated_idx は維持
        if del_idx < self._last_consolidated_idx:
            self._last_consolidated_idx -= 1

    # --- Recall ---

    def recall_episodic(self, n: int = 5, min_pe: float = 0.0) -> list[Episode]:
        """最近のエピソードを PE の高い順に取得する。"""
        relevant = [ep for ep in self.episodic if ep.pe >= min_pe]
        relevant.sort(key=lambda e: e.timestamp, reverse=True)
        return relevant[:n]

    def recall_autobiographical(self, n: int = 5) -> list[Episode]:
        """自伝的記憶を最近の順に取得する。"""
        sorted_mem = sorted(self.autobiographical, key=lambda e: e.timestamp, reverse=True)
        return sorted_mem[:n]

    def query_semantic(self, action: str, outcome: str) -> dict:
        """意味記憶から (action, outcome) の知識を取得する。"""
        key = f"{action}|{outcome}"
        return self.semantic.get(key, {"count": 0, "total_pe": 0.0})

    def action_frequency(self, action: str) -> int:
        """その行動の総経験回数を意味記憶から推定する。"""
        total = 0
        for key, data in self.semantic.items():
            if key.startswith(f"{action}|"):
                total += data["count"]
        return total

    def outcome_probs_from_memory(self, action: str) -> dict[str, float]:
        """意味記憶から (action → outcome 確率) を推定する。"""
        counts = {o: 0 for o in OUTCOMES}
        for o in OUTCOMES:
            key = f"{action}|{o}"
            if key in self.semantic:
                counts[o] = self.semantic[key]["count"]
        total = sum(counts.values())
        if total == 0:
            uniform = 1.0 / len(OUTCOMES)
            return {o: uniform for o in OUTCOMES}
        return {o: counts[o] / total for o in OUTCOMES}

    # --- State ---

    def summary(self) -> dict:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "autobiographical": len(self.autobiographical),
            "total_episodes": self._total_episodes,
        }

    def clear(self) -> None:
        """全記憶を消去する。"""
        self.episodic.clear()
        self.semantic.clear()
        self.autobiographical.clear()
        self._total_episodes = 0
        self._last_consolidated_idx = 0
