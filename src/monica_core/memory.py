import json
import math
import os
from pathlib import Path
from datetime import datetime

from .llm_client import embed as _zen_embed, trigram_hash


def _local_embed(text: str, dim: int = 64) -> list[float]:
    return trigram_hash(text, dim=dim)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


class Memory:
    def __init__(self, path: str = "data/memory_store.json"):
        self.path = Path(path)
        self.dim = 64
        self.entries: list[dict] = []
        self.load()

    def _embed(self, text: str) -> list[float]:
        emb = _zen_embed(text[:1000])
        return emb if emb else _local_embed(text, self.dim)

    def add(self, text: str, activity_type: str = "other",
            tags: list[str] | None = None, importance: int = 5):
        embedding = self._embed(text)
        entry = {
            "text": text,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "tags": tags or [],
            "importance": importance,
        }
        self.entries.append(entry)
        self._trim(500)
        self.save()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.entries:
            return []
        q_emb = self._embed(query)
        scored = [(_cosine(q_emb, e["embedding"]), e) for e in self.entries]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def context(self, query: str, k: int = 5) -> str:
        entries = self.search(query, k)
        parts = []
        for e in entries:
            ts = e["timestamp"]
            if len(ts) > 16:
                ts = ts[:16]
            parts.append(f"[{ts}] ({e['activity_type']}) {e['text']}")
        return "\n".join(parts)

    def _trim(self, max_entries: int = 500):
        if len(self.entries) > max_entries:
            self.entries = self.entries[-max_entries:]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "entries": self.entries,
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.dim = data.get("dim", 64)
            self.entries = data.get("entries", [])
        except Exception:
            self.entries = []
