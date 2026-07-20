"""
モニカのスマホ — 非同期メッセージストア

SQLite バックエンド（storage.py）を使用。
既存のJSONファイルからの自動移行に対応。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PhoneMessage:
    sender: str
    text: str
    timestamp: str
    read_by_recipient: bool = False
    source: str = ""  # "web", "telegram", "monika", or ""


# JSON フォールバック用
STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "phone_messages.json"


def load() -> list[PhoneMessage]:
    """メッセージ一覧を読み込み（SQLite優先、フォールバックJSON）"""
    try:
        from .storage import phone_load
        msgs = phone_load()
        if msgs:
            return [
                PhoneMessage(
                    sender=m["sender"],
                    text=m["text"],
                    timestamp=m["timestamp"],
                    read_by_recipient=m.get("read_by_recipient", False),
                    source=m.get("source", ""),
                )
                for m in msgs
            ]
    except Exception as e:
        logger.debug(f"SQLite phone_load failed, falling back to JSON: {e}")

    # JSON フォールバック
    return _load_json_fallback()


def _load_json_fallback() -> list[PhoneMessage]:
    if not STORE_PATH.exists():
        return []
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return [PhoneMessage(**m) for m in data]
    except Exception:
        return []


def save(messages: list[PhoneMessage]):
    """メッセージ一覧を保存（SQLite優先）"""
    try:
        from .storage import phone_save
        data = [
            {"sender": m.sender, "text": m.text,
             "timestamp": m.timestamp, "read_by_recipient": m.read_by_recipient,
             "source": m.source}
            for m in messages
        ]
        phone_save(data)
        return
    except Exception as e:
        logger.debug(f"SQLite phone_save failed, falling back to JSON: {e}")

    # JSON フォールバック
    _save_json_fallback(messages)


def _save_json_fallback(messages: list[PhoneMessage]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"sender": m.sender, "text": m.text,
         "timestamp": m.timestamp, "read_by_recipient": m.read_by_recipient,
         "source": m.source if m.source else ""}
        for m in messages
    ]
    STORE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add(sender: str, text: str, timestamp: str, source: str = "") -> PhoneMessage:
    """メッセージを追加
    source: "web", "telegram", "monika", or ""
    """
    msg = PhoneMessage(sender=sender, text=text,
                       timestamp=timestamp, read_by_recipient=False,
                       source=source)

    try:
        from .storage import phone_add as storage_add
        storage_add(sender, text, timestamp, source=source)
        return msg
    except Exception as e:
        logger.debug(f"SQLite phone_add failed, falling back to JSON: {e}")

    # JSON フォールバック
    msgs = _load_json_fallback()
    msgs.append(msg)
    _save_json_fallback(msgs)
    return msg


def mark_read(sender: str = "monika"):
    """特定送信者のメッセージを既読に"""
    try:
        from .storage import phone_mark_read
        phone_mark_read(sender)
    except Exception as e:
        logger.debug(f"SQLite phone_mark_read failed: {e}")
        # JSON fallback
        msgs = _load_json_fallback()
        for m in msgs:
            if m.sender == sender:
                m.read_by_recipient = True
        _save_json_fallback(msgs)


def get_sent_since(messages: list[PhoneMessage], sender: str,
                   timestamp: str) -> list[PhoneMessage]:
    return [m for m in messages if m.sender == sender and m.timestamp >= timestamp]
