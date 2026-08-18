"""InterruptChannel — 外部刺激の注入口（仕様書 v1.4 §5.5）。

契約:
    - inject はスレッドセーフである必要がある（将来Webhookやセンサー入力など複数スレッドから注入される）。
    - C1: asyncio.Queue 自体はスレッドセーフではないため、外部スレッドからの注入は
      必ず loop.call_soon_threadsafe(queue.put_nowait, item) を経由する。
      外部スレッドから素の asyncio.Queue.put() を直接呼ぶ実装は禁止。
    - 初期化順序（C1レース条件対策）: キュー生成とイベントループ捕捉（get_running_loop）は
      必ずイベントループのスレッドでのみ行う。core.run() は起動直後に bind() を呼び、
      外部スレッドが最初の inject() を行う時点でキューが確実に初期化済みであることを保証する。
      注入側（inject）が初期化の起点になると、外部スレッドが最初に呼んだ場合に
      get_running_loop() がループ外で実行されて失敗するため、初期化は必ずループ側に置く。
    - drain はイベントループ内（core のメインループ）から呼び出される想定。
      未初期化でも安全に空リストを返す（ループ内から呼ばれた場合のみ初期化するフォールバックを持つ）。
"""

from __future__ import annotations

import asyncio
import time


class InterruptChannel:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, float]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self) -> None:
        """イベントループ内でキューを明示的に初期化する（冪等）。

        イベントループが立った直後（core.run の冒頭）に呼ぶことで、外部スレッドが
        最初の inject() を行った時点でキューが確実に初期化済みになる
        （asyncio.Queue と call_soon_threadsafe は「キューがイベントループのスレッドで
        生成されている」ことが前提のため）。ループ外から呼ぶと明確なエラーを出す。
        """
        if self._queue is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "InterruptChannel はイベントループ内から初期化する必要があります。"
                "core.run() を起動する（またはループ内で bind() を呼ぶ）前に、"
                "外部スレッドから inject() することはできません"
            ) from exc
        self._queue = asyncio.Queue()

    def _ensure(self) -> None:
        """inject 側の初期化フォールバック（イベントループ内から呼ばれる場合に限る）。"""
        if self._queue is None:
            self.bind()

    def inject(self, text: str, timestamp: float | None = None) -> None:
        """任意スレッドから安全に割り込みを注入する（call_soon_threadsafe 経由・C1）。"""
        ts = timestamp if timestamp is not None else time.time()
        self._ensure()
        assert self._loop is not None and self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (text, ts))

    def drain(self) -> list[str]:
        """未処理の割り込みを全て取り出して返す（FIFO）。

        未初期化の場合、イベントループ内から呼ばれていれば初期化する
        （step_once 直駆動の実験経路でも外部スレッドの inject を受けられるようにする）。
        ループ外（未起動）から呼ばれた場合は従来通り空リストを返す。
        """
        if self._queue is None:
            try:
                self.bind()
            except RuntimeError:
                return []
        assert self._queue is not None
        out: list[str] = []
        while not self._queue.empty():
            text, _ = self._queue.get_nowait()
            out.append(text)
        return out

    def has_pending(self) -> bool:
        """未処理の割り込みが存在するか（v1.7: 発話スケジューリングの外部刺激トリガー判定用）。"""
        return self._queue is not None and not self._queue.empty()

    def __len__(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0
