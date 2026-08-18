"""記憶スキーマ（仕様書 v1.4 §5.3 B5）。

MemoryKind: EPISODIC / SEMANTIC / EMOTIONAL / PROCEDURAL
MemoryRecord: 長期記憶の1エントリ。ChromaDBへ永続化され、プロセス再起動をまたぐ。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MemoryKind(Enum):
    EPISODIC = "episodic"      # 出来事・経験
    SEMANTIC = "semantic"      # 知識・事実
    EMOTIONAL = "emotional"    # Drive変化大（abs(delta) >= 0.3）の体験
    PROCEDURAL = "procedural"  # 行動パターン（本フェーズでは使用任意）


@dataclass
class MemoryRecord:
    text: str
    kind: MemoryKind
    drive_snapshot: dict[str, float] = field(default_factory=dict)
    importance: float = 0.0
    embedding: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind.value,
            "drive_snapshot": dict(self.drive_snapshot),
            "importance": float(self.importance),
            "created_at": float(self.created_at),
        }
