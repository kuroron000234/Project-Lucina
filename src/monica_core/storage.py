"""
SQLite ストレージバックエンド — JSONファイルに代わる統一データ永続化

すべてのデータは単一の SQLite データベース (data/monica.db) に保存される。
- トランザクションによる安全な同時アクセス
- 既存 JSON からの自動移行（初回アクセス時）
- バックアップ機能
- レイジー初期化（import時は副作用なし）
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "monica.db"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# スレッドセーフな接続管理
_local = threading.local()

# マイグレーションフラグ（初回アクセス時に一度だけ実行）
_migrated = False
_migrated_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """スレッドローカルな接続を取得"""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


def _ensure_initialized():
    """必要に応じてDB初期化とJSON移行を実行（初回アクセス時に一度だけ）"""
    global _migrated
    if _migrated:
        return
    with _migrated_lock:
        if _migrated:
            return
        _init_db()
        # JSONデータがあれば移行（データベースが空の場合のみ）
        _try_migrate_from_json()
        _migrated = True


def _init_db():
    """データベースのテーブル作成"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulation_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS phone_messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sender    TEXT NOT NULL,
            text      TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            read_by_recipient INTEGER DEFAULT 0,
            source    TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS memories (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            text           TEXT NOT NULL,
            embedding_blob TEXT,
            timestamp      TEXT NOT NULL,
            activity_type  TEXT DEFAULT 'other',
            tags           TEXT DEFAULT '[]',
            importance     INTEGER DEFAULT 5
        );

        CREATE TABLE IF NOT EXISTS day_log (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            time    TEXT NOT NULL,
            message TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_deltas (
            activity TEXT PRIMARY KEY,
            energy   REAL DEFAULT 0,
            hunger   REAL DEFAULT 0,
            fatigue  REAL DEFAULT 0,
            loneliness REAL DEFAULT 0,
            spirit   REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS model_counts (
            activity TEXT PRIMARY KEY,
            count    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS model_last_done (
            activity    TEXT PRIMARY KEY,
            last_done   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
        CREATE INDEX IF NOT EXISTS idx_phone_timestamp ON phone_messages(timestamp);
    """)
    conn.commit()

    # スキーママイグレーション: 既存DBに source カラムがない場合に追加
    _migrate_schema(conn)


def _migrate_schema(conn):
    """既存データベースのスキーマを最新に更新"""
    try:
        conn.execute("ALTER TABLE phone_messages ADD COLUMN source TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # カラムは既に存在する


def _try_migrate_from_json():
    """既存のJSONファイルからデータを移行（DBが空の場合のみ）"""
    conn = _get_conn()

    # 既にデータがある場合はスキップ
    row = conn.execute("SELECT COUNT(*) as cnt FROM phone_messages").fetchone()
    if row and row["cnt"] > 0:
        return

    _migrate_state_json(conn)
    _migrate_phone_json(conn)
    _migrate_memory_json(conn)
    conn.commit()


def _migrate_state_json(conn):
    """state.json の移行"""
    state_path = DATA_DIR / "state.json"
    if not state_path.exists():
        return
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return
        for key, value in state.items():
            if isinstance(value, (dict, list)):
                _set_state(key, json.dumps(value, ensure_ascii=False), conn)
            elif isinstance(value, (int, float, bool)):
                _set_state(key, str(value), conn)
            else:
                _set_state(key, str(value) if value is not None else "", conn)
        logger.info("Migrated state.json → SQLite")
        # 移行後、JSONをバックアップ
        _backup_json(state_path)
    except Exception as e:
        logger.warning(f"Failed to migrate state.json: {e}")


def _migrate_phone_json(conn):
    """phone_messages.json の移行"""
    msgs_path = DATA_DIR / "phone_messages.json"
    if not msgs_path.exists():
        return
    try:
        with open(msgs_path, encoding="utf-8") as f:
            msgs = json.load(f)
        if not isinstance(msgs, list):
            return
        cur = conn.cursor()
        for m in msgs:
            cur.execute(
                "INSERT INTO phone_messages (sender, text, timestamp, read_by_recipient, source) VALUES (?, ?, ?, ?, ?)",
                (m.get("sender", ""), m.get("text", ""), m.get("timestamp", ""),
                 1 if m.get("read_by_recipient", False) else 0,
                 m.get("source", "")),
            )
        conn.commit()
        logger.info(f"Migrated {len(msgs)} phone messages → SQLite")
        _backup_json(msgs_path)
    except Exception as e:
        logger.warning(f"Failed to migrate phone_messages.json: {e}")


def _migrate_memory_json(conn):
    """memory_store.json の移行"""
    mem_path = DATA_DIR / "memory_store.json"
    if not mem_path.exists():
        return
    try:
        with open(mem_path, encoding="utf-8") as f:
            mem_data = json.load(f)
        entries = mem_data if isinstance(mem_data, list) else mem_data.get("entries", [])
        if not entries:
            return
        cur = conn.cursor()
        for e in entries:
            emb = e.get("embedding")
            emb_blob = json.dumps(emb, ensure_ascii=False) if emb else None
            tags = json.dumps(e.get("tags", []), ensure_ascii=False)
            cur.execute(
                "INSERT INTO memories (text, embedding_blob, timestamp, activity_type, tags, importance) VALUES (?, ?, ?, ?, ?, ?)",
                (e.get("text", ""), emb_blob, e.get("timestamp", datetime.now().isoformat()),
                 e.get("activity_type", "other"), tags, e.get("importance", 5)),
            )
        conn.commit()
        logger.info(f"Migrated {len(entries)} memories → SQLite")
        _backup_json(mem_path)
    except Exception as e:
        logger.warning(f"Failed to migrate memory_store.json: {e}")


def _backup_json(path: Path):
    """JSONファイルを.bakにリネーム"""
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():  # 既存のバックアップは上書きしない
            path.rename(bak)
            logger.info(f"Backed up {path.name} → {bak.name}")


def _set_state(key: str, value: str, conn: Optional[sqlite3.Connection] = None):
    c = conn or _get_conn()
    c.execute(
        "INSERT OR REPLACE INTO simulation_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    if not conn:
        c.commit()


def _get_state(key: str, conn: Optional[sqlite3.Connection] = None) -> Optional[str]:
    c = conn or _get_conn()
    row = c.execute("SELECT value FROM simulation_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


# ── シミュレーション状態 ──

def save_simulation_state(state: dict):
    """シミュレーション状態を保存"""
    _ensure_initialized()
    conn = _get_conn()
    for key, value in state.items():
        if isinstance(value, (dict, list)):
            _set_state(key, json.dumps(value, ensure_ascii=False), conn)
        else:
            _set_state(key, str(value) if value is not None else "", conn)
    _set_state("saved_at", datetime.now().isoformat(), conn)
    conn.commit()


def load_simulation_state() -> Optional[dict]:
    """シミュレーション状態を読み込み"""
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM simulation_state").fetchall()
    if not rows:
        return None

    state = {}
    for row in rows:
        key = row["key"]
        val = row["value"]
        if key in ("state", "history", "day_log", "phone", "model_deltas", "model_counts", "last_done"):
            try:
                state[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                state[key] = val
        elif key in ("time", "current_activity", "current_room", "saved_at"):
            state[key] = val
        elif key in ("activity_remaining", "activity_total_duration"):
            try:
                state[key] = int(float(val))
            except (ValueError, TypeError):
                state[key] = 0
        else:
            state[key] = val
    return state


# ── 電話メッセージ ──

def phone_add(sender: str, text: str, timestamp: str, source: str = "") -> int:
    """メッセージを追加"""
    _ensure_initialized()
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO phone_messages (sender, text, timestamp, read_by_recipient, source) VALUES (?, ?, ?, 0, ?)",
        (sender, text, timestamp, source),
    )
    conn.commit()
    return cur.lastrowid


def phone_load() -> list[dict]:
    """全メッセージを読み込み"""
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, sender, text, timestamp, read_by_recipient, source FROM phone_messages ORDER BY id ASC"
    ).fetchall()
    return [
        {
            "sender": r["sender"],
            "text": r["text"],
            "timestamp": r["timestamp"],
            "read_by_recipient": bool(r["read_by_recipient"]),
            "source": r["source"] or "",
        }
        for r in rows
    ]


def phone_save(messages: list[dict]):
    """全メッセージを置き換え"""
    _ensure_initialized()
    conn = _get_conn()
    conn.execute("DELETE FROM phone_messages")
    cur = conn.cursor()
    for m in messages:
        cur.execute(
            "INSERT INTO phone_messages (sender, text, timestamp, read_by_recipient, source) VALUES (?, ?, ?, ?, ?)",
            (m["sender"], m["text"], m["timestamp"],
             1 if m.get("read_by_recipient", False) else 0,
             m.get("source", "")),
        )
    conn.commit()


def phone_mark_read(sender: str = "monika"):
    """特定の送信者のメッセージを既読にする"""
    _ensure_initialized()
    conn = _get_conn()
    conn.execute(
        "UPDATE phone_messages SET read_by_recipient = 1 WHERE sender = ? AND read_by_recipient = 0",
        (sender,),
    )
    conn.commit()


# ── 記憶 ──

def memory_add(text: str, activity_type: str = "other", tags: Optional[list] = None,
               importance: int = 5, embedding: Optional[list] = None, timestamp: Optional[str] = None):
    """記憶を追加"""
    _ensure_initialized()
    conn = _get_conn()
    emb_blob = json.dumps(embedding, ensure_ascii=False) if embedding else None
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    ts = timestamp or datetime.now().isoformat()
    conn.execute(
        "INSERT INTO memories (text, embedding_blob, timestamp, activity_type, tags, importance) VALUES (?, ?, ?, ?, ?, ?)",
        (text, emb_blob, ts, activity_type, tags_json, importance),
    )
    conn.commit()


def memory_search(query_embedding: Optional[list] = None, k: int = 5,
                  min_importance: int = 0) -> list[dict]:
    """記憶を検索（embedding類似度順）"""
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, text, embedding_blob, timestamp, activity_type, tags, importance "
        "FROM memories WHERE importance >= ? ORDER BY id DESC LIMIT ?",
        (min_importance, k * 3),
    ).fetchall()

    results = []
    for r in rows:
        emb = json.loads(r["embedding_blob"]) if r["embedding_blob"] else None
        results.append({
            "text": r["text"],
            "embedding": emb,
            "timestamp": r["timestamp"],
            "activity_type": r["activity_type"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "importance": r["importance"],
        })

    if query_embedding and results:
        def cosine_sim(a, b):
            if not a or not b:
                return 0
            return sum(x * y for x, y in zip(a, b))

        for r in results:
            r["_score"] = cosine_sim(query_embedding, r["embedding"]) if r["embedding"] else 0
        results.sort(key=lambda x: -x["_score"])
        results = results[:k]
        for r in results:
            del r["_score"]
    else:
        results = results[:k]

    return results


def memory_count() -> int:
    """記憶の総数を取得"""
    _ensure_initialized()
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
    return row["cnt"] if row else 0


def memory_trim(max_entries: int = 1000):
    """古い記憶をトリミング"""
    _ensure_initialized()
    conn = _get_conn()
    conn.execute(
        "DELETE FROM memories WHERE id NOT IN (SELECT id FROM memories ORDER BY id DESC LIMIT ?)",
        (max_entries,),
    )
    conn.commit()


def memory_get_all() -> list[dict]:
    """全記憶を取得（エクスポート用）"""
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT text, timestamp, activity_type, tags, importance FROM memories ORDER BY id ASC"
    ).fetchall()
    return [
        {
            "text": r["text"],
            "timestamp": r["timestamp"],
            "activity_type": r["activity_type"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "importance": r["importance"],
        }
        for r in rows
    ]


# ── モデル状態 ──

def model_save_deltas(deltas: dict[str, dict[str, float]]):
    _ensure_initialized()
    conn = _get_conn()
    for activity, params in deltas.items():
        conn.execute(
            "INSERT OR REPLACE INTO model_deltas (activity, energy, hunger, fatigue, loneliness, spirit) VALUES (?, ?, ?, ?, ?, ?)",
            (activity, params.get("energy", 0), params.get("hunger", 0),
             params.get("fatigue", 0), params.get("loneliness", 0), params.get("spirit", 0)),
        )
    conn.commit()


def model_load_deltas() -> dict[str, dict[str, float]]:
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM model_deltas").fetchall()
    return {
        r["activity"]: {
            "energy": r["energy"], "hunger": r["hunger"], "fatigue": r["fatigue"],
            "loneliness": r["loneliness"], "spirit": r["spirit"],
        }
        for r in rows
    }


def model_save_counts(counts: dict[str, int]):
    _ensure_initialized()
    conn = _get_conn()
    for activity, count in counts.items():
        conn.execute(
            "INSERT OR REPLACE INTO model_counts (activity, count) VALUES (?, ?)",
            (activity, count),
        )
    conn.commit()


def model_load_counts() -> dict[str, int]:
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM model_counts").fetchall()
    return {r["activity"]: r["count"] for r in rows}


def model_save_last_done(last_done: dict[str, Optional[datetime]]):
    _ensure_initialized()
    conn = _get_conn()
    for activity, dt in last_done.items():
        val = dt.isoformat() if dt else None
        conn.execute(
            "INSERT OR REPLACE INTO model_last_done (activity, last_done) VALUES (?, ?)",
            (activity, val),
        )
    conn.commit()


def model_load_last_done() -> dict[str, Optional[str]]:
    _ensure_initialized()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM model_last_done").fetchall()
    return {r["activity"]: r["last_done"] for r in rows}


# ── メンテナンス ──

def vacuum():
    """データベースを最適化"""
    _ensure_initialized()
    conn = _get_conn()
    conn.execute("VACUUM")
    conn.commit()


def backup(path: Optional[Path] = None) -> Path:
    """データベースのバックアップを作成"""
    _ensure_initialized()
    if path is None:
        path = DATA_DIR / f"monica_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    conn = _get_conn()
    backup_conn = sqlite3.connect(str(path))
    conn.backup(backup_conn)
    backup_conn.close()
    logger.info(f"Database backed up to {path}")
    return path
