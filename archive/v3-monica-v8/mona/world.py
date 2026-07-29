"""
World — Monaの環境認識

Mona is aware of:
  - The file system (files, directories, changes)
  - Time (time of day, day of week, elapsed time)
  - System state (processes, resources)
  - Her own diary (internal world)

This gives her something to think about and react to
when the user isn't around.
"""

import os
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class World:
    """Environment awareness module."""

    def __init__(self, watch_dir: str | Path = "."):
        self.watch_dir = Path(watch_dir).resolve()
        self._file_snapshot: dict[str, float] = {}
        self._last_scan_time = 0.0
        self._interesting_extensions = {".py", ".md", ".txt", ".json", ".html",
                                        ".css", ".js", ".yaml", ".toml", ".cfg"}

        # Interesting findings cache
        self.recent_changes: list[dict] = []
        self.interesting_files: list[dict] = []

        # Self-awareness
        self.own_path = Path(__file__).resolve()
        self.project_root = self.own_path.parent.parent

    def _scan_files(self) -> dict[str, float]:
        """Scan the watch directory and return {path: mtime}."""
        snapshot = {}
        try:
            for p in self.watch_dir.rglob("*"):
                if p.is_file() and not any(
                    part.startswith(".") or part.startswith("__")
                    for part in p.relative_to(self.watch_dir).parts
                ):
                    try:
                        snapshot[str(p.relative_to(self.watch_dir))] = p.stat().st_mtime
                    except OSError:
                        continue
        except Exception:
            pass
        return snapshot

    def scan(self) -> list[dict]:
        """Scan for file changes since last scan.

        Returns list of {type, path, size} for changes.
        """
        changes = []
        current = self._scan_files()

        if self._file_snapshot:
            # Detect new files
            for path, mtime in current.items():
                if path not in self._file_snapshot:
                    full = self.watch_dir / path
                    try:
                        changes.append({
                            "type": "created",
                            "path": path,
                            "size": full.stat().st_size,
                        })
                    except OSError:
                        pass

            # Detect deleted files
            for path in self._file_snapshot:
                if path not in current:
                    changes.append({
                        "type": "deleted",
                        "path": path,
                        "size": 0,
                    })

            # Detect modified files
            for path, mtime in current.items():
                if path in self._file_snapshot and abs(mtime - self._file_snapshot[path]) > 0.1:
                    full = self.watch_dir / path
                    try:
                        changes.append({
                            "type": "modified",
                            "path": path,
                            "size": full.stat().st_size,
                        })
                    except OSError:
                        pass

        self._file_snapshot = current
        self._last_scan_time = time.time()

        # Update interesting files
        self.interesting_files = self._find_interesting()

        if changes:
            self.recent_changes = (changes + self.recent_changes)[:20]

        return changes

    def _find_interesting(self) -> list[dict]:
        """Find interesting files (recently modified, non-system)."""
        interesting = []
        now = time.time()
        try:
            for p in self.watch_dir.rglob("*"):
                if p.is_file() and p.suffix in self._interesting_extensions:
                    try:
                        mtime = p.stat().st_mtime
                        size = p.stat().st_size
                        age_hours = (now - mtime) / 3600
                        if age_hours < 24 and size > 50:
                            interesting.append({
                                "path": str(p.relative_to(self.watch_dir)),
                                "size": size,
                                "age_hours": round(age_hours, 1),
                            })
                    except OSError:
                        continue
        except Exception:
            pass
        interesting.sort(key=lambda x: x["age_hours"])
        return interesting[:10]

    def read_file_content(self, path: str, max_chars: int = 500) -> Optional[str]:
        """Read a file's content (safely, truncated)."""
        try:
            full = self.watch_dir / path
            if not full.exists() or not full.is_file():
                return None
            content = full.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n…"
            return content
        except Exception:
            return None

    def get_time_context(self) -> str:
        """Generate a description of current time."""
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            period = "朝"
        elif 12 <= hour < 17:
            period = "昼"
        elif 17 <= hour < 21:
            period = "夕方"
        else:
            period = "夜"

        weekday_names = ["月", "火", "水", "木", "金", "土", "日"]
        weekday = weekday_names[now.weekday()]

        return f"{period} ({weekday}曜日, {now.hour:02d}:{now.minute:02d})"

    def get_system_context(self) -> dict:
        """Get basic system info."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            uptime = time.time() - psutil.boot_time()
            return {
                "cpu": round(cpu, 1),
                "memory": round(mem, 1),
                "uptime_hours": round(uptime / 3600, 1),
            }
        except ImportError:
            return {"cpu": 0, "memory": 0, "uptime_hours": 0}

    def get_observation(self) -> str:
        """Get a natural observation about the environment.

        Returns a string like 'I noticed a new file was created' or
        'It's evening now. The system is quiet.'
        """
        changes = self.scan()
        time_ctx = self.get_time_context()

        if changes:
            change_types = {}
            for c in changes:
                change_types[c["type"]] = change_types.get(c["type"], 0) + 1
            parts = [f"{v}件の{k}" for k, v in change_types.items()]
            return f"ファイルシステムに変化: {', '.join(parts)}"

        if self.interesting_files:
            f = self.interesting_files[0]
            return f"気になるファイルを発見: {f['path']}"

        return f"{time_ctx}。システムは静か。特に変化なし。"

    def summary(self) -> dict:
        return {
            "watch_dir": str(self.watch_dir),
            "files_tracked": len(self._file_snapshot),
            "recent_changes": len(self.recent_changes),
            "interesting_files": len(self.interesting_files),
            "time": self.get_time_context(),
        }
