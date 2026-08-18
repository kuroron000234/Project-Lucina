"""生成ログ・監視用ロガー（仕様書 v1.4 §8）。

- Drive時系列ログ: boredom/loneliness/fatigue を毎ステップ JSON Lines で出力。
- ロジットバイアス適用前後の差分ログ: logit_bias_coefficient チューニング時・§5.4 の
  語彙の先頭トークンへのバイアス適用ログ。
- メモリ圧縮イベントログ: 発火時刻・除去トークン数・要約結果。
- 語彙拡張結果ログ: DriveVocabExpander の結果を必ず INFO 出力（運用フック）。
- モデル選定比較ログ: §1.2 の評価軸ごとの候補モデル計測結果。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np


def setup_console_logging(level: str = "INFO") -> None:
    root = logging.getLogger("lucina")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class StructuredLogger:
    """JSON Lines 形式の構造化ログを reports/ 以下に書き出す。"""

    def __init__(self, log_dir: str | Path = "./reports"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._info = logging.getLogger("lucina.io")
        self._files: dict[str, object] = {}

    def _fh(self, name: str):
        if name not in self._files:
            self._files[name] = (self.dir / f"{name}.jsonl").open("a", encoding="utf-8")
        return self._files[name]

    def _write(self, name: str, payload: dict) -> None:
        fh = self._fh(name)
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    # ---- Drive時系列（毎ステップ） ----
    def drive_step(self, state: dict[str, float], timestamp: float | None = None) -> None:
        self._write("drives", {"ts": timestamp if timestamp is not None else time.time(), "state": dict(state)})

    # ---- ロジット差分 ----
    def logit_diff(
        self,
        before: np.ndarray,
        after: np.ndarray,
        target_ids: list[int],
        coefficient: float,
    ) -> None:
        diffs = {int(t): float(after[t] - before[t]) for t in target_ids if 0 <= int(t) < before.shape[0]}
        self._write("logits", {
            "ts": time.time(),
            "coefficient": float(coefficient),
            "target_first_token_diffs": diffs,
        })

    # ---- メモリ圧縮イベント ----
    def compression_event(self, removed_tokens: int, summary: str, timestamp: float | None = None) -> None:
        self._write("compression", {
            "ts": timestamp if timestamp is not None else time.time(),
            "removed_tokens": int(removed_tokens),
            "summary": summary,
        })

    # ---- 語彙拡張結果（運用フック・必ずINFO出力） ----
    def vocab_expansion(self, vocab_map: dict[str, list[list[int]]]) -> None:
        readable = {
            drive: [list(seq) for seq in seqs] for drive, seqs in vocab_map.items()
        }
        self._info.info("Drive語彙拡張結果（目視レビュー用）: %s", json.dumps(readable, ensure_ascii=False))
        self._write("vocab_expansion", {"ts": time.time(), "vocab_map": readable})

    # ---- モデル選定比較（§1.2） ----
    def model_selection(self, axis: str, model: str, results: dict) -> None:
        self._write("model_selection", {
            "ts": time.time(),
            "axis": axis,
            "model": model,
            "results": results,
        })

    # ---- 記憶コミット（v1.11） ----
    def memory_commit(self, kind: str, importance: float, text: str, timestamp: float | None = None) -> None:
        """長期記憶へのコミット内容（分類結果・重要度）を構造化ログに残す。

        記憶層の健全性確認（どの記憶がどの種類として保存されているか）のデータソース。
        """
        self._write("memory", {
            "ts": timestamp if timestamp is not None else time.time(),
            "kind": kind,
            "importance": float(importance),
            "text": text[:200],
        })

    # ---- 記憶の想起（v1.12: retrieve→文脈注入） ----
    def memory_recall(self, count: int, top_k: int, texts: list[str], timestamp: float | None = None) -> None:
        """想起された記憶（内言・発話前の文脈注入）を構造化ログに残す。

        記憶層の「読み出し」側の健全性確認（どの記憶が現在の文脈に戻ってきているか）の
        データソース。
        """
        self._write("memory", {
            "ts": timestamp if timestamp is not None else time.time(),
            "event": "recall",
            "count": int(count),
            "top_k": int(top_k),
            "texts": [t[:100] for t in texts],
        })

    # ---- 自発的行動選択（§0 自律思考ループ・M2・v1.7） ----
    def autonomy_event(self, event: str, drives: dict[str, float], mode: str, reason: str = "") -> None:
        """Drive駆動の自発的行動（speech_start / speech_end / inner_thought）を構造化ログに残す。

        M2「Drive駆動での自発的行動選択をログで確認できる」のデータソース。
        """
        self._write("autonomy", {
            "ts": time.time(),
            "event": event,
            "mode": mode,
            "drives": dict(drives),
            "reason": reason,
        })

    def close(self) -> None:
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:  # noqa: BLE001 - close中の失敗は無視
                pass
        self._files.clear()

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
