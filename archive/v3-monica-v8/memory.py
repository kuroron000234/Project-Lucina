import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "memories.json"


class Memory:
    def __init__(self):
        self.short: list[dict] = []
        self.long: list[dict] = []
        self._load()

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text())
                self.short = data.get("short", [])
                self.long = data.get("long", [])
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        MEMORY_FILE.write_text(json.dumps(
            {"short": self.short[-20:], "long": self.long[-50:]},
            ensure_ascii=False, indent=2,
        ))

    def add(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self.short.append(entry)

    def context(self, limit=6) -> str:
        lines = []
        for e in (self.short + self.long)[-limit:]:
            ts = e.get("timestamp", "")[-8:]
            c = str(e.get("content", "") or "")[:120]
            imp = " ⭐" if e.get("important") else ""
            lines.append(f"[{ts}]{imp} {c}")
        return "\n".join(lines)

    def mark_important(self, index=-1):
        if self.short:
            idx = index if index >= 0 else len(self.short) - 1
            self.short[idx]["important"] = True

    def compress(self, llm_fn) -> str:
        if len(self.short) < 3:
            return ""
        recent = self.short[-5:]
        text = "\n".join(
            f"- {e.get('content', '')[:100]}"
            for e in recent if e.get("content")
        )
        summary = llm_fn([
            {"role": "system", "content": "以下の出来事を1文で要約してください。"},
            {"role": "user", "content": text},
        ], max_tokens=100)
        if summary:
            self.long.append({
                "timestamp": datetime.now().isoformat(),
                "summary": summary.strip(),
                "source_count": len(recent),
            })
            self.short = self.short[-2:]
            self.save()
        return summary or ""
