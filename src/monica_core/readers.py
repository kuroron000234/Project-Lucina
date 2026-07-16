import json
import random
import re
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

AUTHORS = {
    148: "夏目漱石",
    135: "太宰治",
    181: "森鷗外",
    448: "宮沢賢治",
    119: "芥川龍之介",
    364: "谷崎潤一郎",
    15:  "江戸川乱歩",
    194: "中島敦",
    173: "小林多喜二",
    96:  "梶井基次郎",
}

CHARS_PER_HOUR = 10000


def _fetch(url: str, encoding: str = "utf-8") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Monica/1.0"})
    raw = urllib.request.urlopen(req, timeout=15).read()
    return raw.decode(encoding, errors="replace")


def _clean_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"[｜《》\n\r]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(?:[^。]*?)[。、]", "", text, count=1)
    return text.strip()


def _card_url(author_id: int, book_id: int) -> str:
    return f"https://www.aozora.gr.jp/cards/{author_id:06d}/card{book_id}.html"


def _author_url(author_id: int) -> str:
    return f"https://www.aozora.gr.jp/index_pages/person{author_id}.html"


def discover_works(author_id: int) -> list[dict]:
    """List works by an author (title + card URL)."""
    html = _fetch(_author_url(author_id))
    works = []
    seen = set()
    for m in re.finditer(r'<a href="([^"]*card(\d+)\.html)"[^>]*>(.*?)</a>', html):
        card_rel = m.group(1)
        book_id = int(m.group(2))
        title_raw = m.group(3)
        title = re.sub(r"<[^>]+>", "", title_raw).strip()
        if title and book_id not in seen:
            seen.add(book_id)
            works.append({"book_id": book_id, "title": title, "author_id": author_id})
    return works


def get_text(author_id: int, book_id: int) -> dict | None:
    """Fetch full text of a work. Returns {title, author, text, total_chars} or None."""
    try:
        card_html = _fetch(_card_url(author_id, book_id))
        # Find text file link
        for m in re.finditer(r'href="((?:\./)?files/(\d+_\d+)\.html)"', card_html):
            file_rel = m.group(1)
            version = m.group(2)
            text_url = f"https://www.aozora.gr.jp/cards/{author_id:06d}/{file_rel}"
            try:
                raw_text = _fetch(text_url, encoding="cp932")
                clean = _clean_html(raw_text)
                return {
                    "title": _get_title(card_html),
                    "author": AUTHORS.get(author_id, f"著者{author_id}"),
                    "text": clean,
                    "total_chars": len(clean),
                    "author_id": author_id,
                    "book_id": book_id,
                }
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_title(card_html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", card_html, re.DOTALL)
    if m:
        t = m.group(1).strip()
        t = re.sub(r"\s*[－—\-].*", "", t).strip()
        t = re.sub(r"^図書カード[：:]\s*", "", t)
        return t
    return "無題"


def pick_random_work() -> dict | None:
    """Pick a random work from a random author."""
    keys = list(AUTHORS.keys())
    random.shuffle(keys)
    for author_id in keys:
        works = discover_works(author_id)
        if works:
            random.shuffle(works)
            result = get_text(author_id, works[0]["book_id"])
            if result:
                return result
    return None


class ReadingHandler:
    def __init__(self, data_dir: str | Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / "reading_state.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                pass
        return {"author_id": None, "book_id": None, "title": "", "author": "",
                "position": 0, "total_chars": 0}

    def _save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))

    def start_new_book(self, author_id: int | None = None) -> str | None:
        """Pick a book and return the first chunk."""
        if author_id is None:
            author_id = random.choice(list(AUTHORS.keys()))
        works = discover_works(author_id)
        if not works:
            return None
        random.shuffle(works)
        candidates = []
        for w in works:
            result = get_text(author_id, w["book_id"])
            if result and result["total_chars"] > 2000:
                candidates.append(result)
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x["total_chars"])
        chosen = candidates[0]
        self.state = {
            "author_id": author_id,
            "book_id": chosen["book_id"],
            "title": chosen["title"],
            "author": chosen["author"],
            "position": 0,
            "total_chars": chosen["total_chars"],
        }
        self._save()
        return self._chunk(chosen["text"], 0, CHARS_PER_HOUR)

    def continue_reading(self, duration_min: int) -> str | None:
        """Continue current book, return next chunk."""
        if not self.state["book_id"]:
            return self.start_new_book()
        result = get_text(self.state["author_id"], self.state["book_id"])
        if not result:
            return self.start_new_book()
        pos = self.state["position"]
        chars = int(CHARS_PER_HOUR * duration_min / 60)
        chunk = self._chunk(result["text"], pos, chars)
        if not chunk:
            # Finished the book
            self.state["title"] = ""
            self._save()
            return None
        self.state["position"] = pos + chars
        self.state["total_chars"] = result["total_chars"]
        self._save()
        return chunk

    def _chunk(self, text: str, start: int, length: int) -> str | None:
        if start >= len(text):
            return None
        end = min(start + length, len(text))
        return text[start:end]

    def progress_str(self) -> str:
        if not self.state["book_id"]:
            return "今は読んでいる本がない"
        pct = min(100, int(self.state["position"] / max(1, self.state["total_chars"]) * 100))
        return f"{self.state['title']}（{self.state['author']}）{pct}%完了"
