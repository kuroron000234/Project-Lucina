"""Working Buffer（仕様書 v1.4 §5.3）。

- 直近の生成トークンを保持し、トークン数を管理する。
- context_window * max_working_tokens_ratio を超えたら圧縮がトリガーされる
  （圧縮自体は MemoryCompressor が行い、本クラスは要素の出し入れを担当）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BufferItem:
    text: str
    n_tokens: int = 1
    internal: bool = False  # v1.7: 内言等、モデルには見せるが「発話」としては扱わない要素


class WorkingBuffer:
    def __init__(self, items: list[BufferItem] | None = None):
        self._items: list[BufferItem] = list(items) if items else []

    # ---- 読み取り ----
    @property
    def items(self) -> list[str]:
        return [it.text for it in self._items]

    @property
    def token_count(self) -> int:
        return sum(it.n_tokens for it in self._items)

    @property
    def is_empty(self) -> bool:
        return not self._items

    def content(self) -> str:
        return "".join(it.text for it in self._items)

    def spoken_content(self) -> str:
        """発話（非内部）要素のみを連結する。内言・思考（internal=True）は除外する（v1.7）。"""
        return "".join(it.text for it in self._items if not it.internal)

    def contains(self, substring: str) -> bool:
        return substring in self.content()

    # ---- 書き込み ----
    def append(self, text: str, n_tokens: int = 1, internal: bool = False) -> None:
        self._items.append(BufferItem(text, max(0, int(n_tokens)), bool(internal)))

    def prepend(self, text: str, n_tokens: int = 1) -> None:
        self._items.insert(0, BufferItem(text, max(0, int(n_tokens))))

    def take_oldest(self, max_tokens: int) -> tuple[str, int]:
        """古い要素を先頭から最大 max_tokens 分だけ取り除き、(除去テキスト, 除去トークン数) を返す。"""
        removed: list[BufferItem] = []
        removed_tokens = 0
        while self._items and removed_tokens < max_tokens:
            item = self._items.pop(0)
            removed.append(item)
            removed_tokens += item.n_tokens
        return "".join(it.text for it in removed), removed_tokens

    def take_newest(self, n_items: int) -> tuple[str, int]:
        """末尾（直近）の要素を最大 n_items 個取り除き、(除去テキスト, 除去トークン数) を返す。

        エコー抑制（v1.12）: 生成直後の反唱トークンをバッファ末尾から巻き戻すために使用。
        """
        removed: list[BufferItem] = []
        n = max(0, int(n_items))
        for _ in range(n):
            if not self._items:
                break
            removed.append(self._items.pop())
        removed_tokens = sum(it.n_tokens for it in removed)
        return "".join(it.text for it in removed), removed_tokens

    def is_over_threshold(self, context_window: int, ratio: float) -> bool:
        return self.token_count > max(1, int(context_window * ratio))

    def reset(self) -> None:
        self._items.clear()
