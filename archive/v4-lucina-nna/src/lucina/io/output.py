"""OutputChannel — Lucina の発話・質問を外部へ配信する出力キュー（v1.13）。

InterruptChannel の出力版。core が発話セグメント（speech）・質問（question）を emit し、
run_agent の表示タスク（--interact）や実行エージェント（ExecutorAdapter）が drain して消費する。

- emit はイベントループ内（core の生成経路）から呼ばれる想定。
- 初期化順序は InterruptChannel と同じ契約: キュー生成は必ずイベントループのスレッドで
  行う（bind は core.run() 起動直後に呼ばれ、drain/emit 側はループ内なら自動初期化する
  フォールバックを持つ）。
- drain はループ内の別タスク（表示ループ）から呼ばれる想定。
"""

from __future__ import annotations

import asyncio


class OutputChannel:
    """発話・質問イベントの配信キュー。 (kind, text) のタプルを積む。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, str]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self) -> None:
        """イベントループ内でキューを明示的に初期化する（冪等）。"""
        if self._queue is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "OutputChannel はイベントループ内から初期化する必要があります。"
                "core.run() を起動する前に emit/drain することはできません"
            ) from exc
        self._queue = asyncio.Queue()

    def _ensure(self) -> None:
        if self._queue is None:
            self.bind()

    def emit(self, kind: str, text: str) -> None:
        """発話（speech）・質問（question）等を出力キューへ積む（ループ内からのみ）。"""
        self._ensure()
        assert self._queue is not None
        self._queue.put_nowait((kind, text))

    def drain(self) -> list[tuple[str, str]]:
        """未配信の出力イベントを全て取り出して返す（FIFO）。"""
        if self._queue is None:
            try:
                self.bind()
            except RuntimeError:
                return []
        assert self._queue is not None
        out: list[tuple[str, str]] = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out

    def __len__(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0
