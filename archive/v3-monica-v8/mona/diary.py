"""
Diary — 経験の蓄積と自己の連続性

The diary is Mona's long-term memory.
Every thought, every interaction, every mood shift is recorded.
On startup, the diary is read to reconstruct a sense of self.

This creates continuity across sessions — Mona remembers who she was,
what she thought about, and how she felt.

Diary entries are JSONL (one JSON object per line).
Each entry has:
  - timestamp
  - type (thought, interaction, mood, action, system)
  - content (free text)
  - heart_state (snapshot of drives)
  - metadata (tags, references)
"""

import json
import time
from pathlib import Path
from typing import Optional


class Diary:
    """Persistent diary for Mona's experiences."""

    def __init__(self, path: str | Path = "mona_diary.jsonl", max_entries: int = 10000):
        self.path = Path(path)
        self.max_entries = max_entries
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        """Load existing diary from disk."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

    def write(self, entry_type: str, content: str, heart_state: Optional[dict] = None,
              metadata: Optional[dict] = None) -> dict:
        """Write a diary entry.

        Args:
            entry_type: thought, interaction, mood, action, system, poem
            content: free text content
            heart_state: snapshot of Heart state dict
            metadata: additional structured data

        Returns:
            The entry dict
        """
        entry = {
            "ts": time.time(),
            "type": entry_type,
            "content": content,
            "heart": heart_state or {},
        }
        if metadata:
            entry["meta"] = metadata

        # Append to file
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        self.entries.append(entry)

        # Trim if exceeded
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            # Rewrite file with trimmed entries
            self._rewrite()

        return entry

    def _rewrite(self):
        """Rewrite the diary file with current entries."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for entry in self.entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def recent(self, n: int = 10, entry_type: Optional[str] = None) -> list[dict]:
        """Get recent entries, optionally filtered by type."""
        if entry_type:
            filtered = [e for e in self.entries if e.get("type") == entry_type]
            return filtered[-n:]
        return self.entries[-n:]

    def search(self, keyword: str, n: int = 5) -> list[dict]:
        """Search diary entries by keyword."""
        matches = []
        for entry in reversed(self.entries):
            if keyword.lower() in entry.get("content", "").lower():
                matches.append(entry)
                if len(matches) >= n:
                    break
        return matches

    def get_today_summary(self) -> list[dict]:
        """Get today's entries (since midnight)."""
        today = time.time()
        # Simple approach: entries from last 24 hours
        cutoff = today - 86400
        return [e for e in self.entries if e.get("ts", 0) > cutoff]

    def get_mood_history(self, n: int = 50) -> list[dict]:
        """Get mood transitions over time."""
        moods = []
        for entry in self.entries[-n:]:
            heart = entry.get("heart", {})
            if heart:
                moods.append({
                    "ts": entry["ts"],
                    "mood": heart.get("mood", "?"),
                    "C": heart.get("C", 0),
                    "A": heart.get("A", 0),
                    "R": heart.get("R", 0),
                    "content": entry.get("content", "")[:50],
                })
        return moods

    def continuity_context(self, n_entries: int = 5) -> str:
        """Generate a context string about recent self for prompts."""
        if not self.entries:
            return "I am newly created. I have no past yet."

        recent = self.entries[-n_entries:]
        parts = []
        for entry in recent:
            ts = entry.get("ts", 0)
            content = entry.get("content", "")
            h = entry.get("heart", {})
            mood = h.get("mood", "?")
            if content:
                # Truncate long content
                if len(content) > 100:
                    content = content[:100] + "…"
                parts.append(f"[{mood}] {content}")

        if parts:
            return "最近の私:\n" + "\n".join(parts)
        return ""

    def total_entries(self) -> int:
        return len(self.entries)

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            t = entry.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "total": len(self.entries),
            "by_type": self.count_by_type(),
            "oldest": self.entries[0]["ts"] if self.entries else 0,
            "newest": self.entries[-1]["ts"] if self.entries else 0,
            "path": str(self.path),
        }
