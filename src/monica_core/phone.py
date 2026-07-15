import json
from dataclasses import dataclass, field
from pathlib import Path


STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "phone_messages.json"


@dataclass
class PhoneMessage:
    sender: str
    text: str
    timestamp: str
    read_by_recipient: bool = False


def load() -> list[PhoneMessage]:
    if not STORE_PATH.exists():
        return []
    data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return [PhoneMessage(**m) for m in data]


def save(messages: list[PhoneMessage]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"sender": m.sender, "text": m.text,
         "timestamp": m.timestamp, "read_by_recipient": m.read_by_recipient}
        for m in messages
    ]
    STORE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add(sender: str, text: str, timestamp: str) -> PhoneMessage:
    msgs = load()
    msg = PhoneMessage(sender=sender, text=text,
                       timestamp=timestamp, read_by_recipient=False)
    msgs.append(msg)
    save(msgs)
    return msg


def get_sent_since(messages: list[PhoneMessage], sender: str,
                   timestamp: str) -> list[PhoneMessage]:
    return [m for m in messages if m.sender == sender and m.timestamp >= timestamp]
