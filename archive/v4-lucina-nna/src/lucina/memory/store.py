"""HierarchicalMemoryStore — 長期記憶ベクトルストア（仕様書 v1.4 §5.3）。

契約:
    - commit は Drive変化量 abs(delta) >= 0.3 の場合、分類器の判定結果に関わらず
      MemoryKind.EMOTIONAL を強制付与する（v3 §3.2のルール）。
    - 永続化: 本番は ChromaVectorStore（persist_directory 必須）。インメモリのみの
      実装を本番に使ってはならない。InMemoryVectorStore はテスト専用。
    - B3: commit の重要度（importance）はサプライズ値から与えられる。

MemoryCompressor: WorkingBuffer が閾値を超えた際、古い内容を要約で置き換える。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, Sequence

import numpy as np

from .schema import MemoryKind, MemoryRecord
from .working_buffer import WorkingBuffer

logger = logging.getLogger("lucina.memory")


class Classifier(Protocol):
    def classify(self, text: str) -> MemoryKind: ...


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class Summarizer(Protocol):
    def summarize(self, text: str) -> str: ...


# --------------------------------------------------------------------------- #
# ベクトルストア抽象
# --------------------------------------------------------------------------- #
class VectorStore(Protocol):
    def add(self, record: MemoryRecord) -> None: ...
    def query(self, embedding: np.ndarray, kind_filter: MemoryKind | None, top_k: int) -> list[MemoryRecord]: ...


class InMemoryVectorStore:
    """テスト専用のインメモリ実装（本番では不可。仕様書 §5.3）。"""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def query(self, embedding: np.ndarray, kind_filter: MemoryKind | None, top_k: int) -> list[MemoryRecord]:
        scored: list[tuple[float, MemoryRecord]] = []
        for rec in self._records:
            if kind_filter is not None and rec.kind != kind_filter:
                continue
            if rec.embedding is None:
                continue
            score = float(np.dot(embedding, rec.embedding) / (
                (np.linalg.norm(embedding) * np.linalg.norm(rec.embedding)) or 1.0
            ))
            scored.append((score, rec))
        scored.sort(key=lambda t: -t[0])
        return [rec for _, rec in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._records)


class ChromaVectorStore:
    """ChromaDB 永続化バックエンド（本番用。persist_directory 必須）。

    chromadb は遅延import（オプショナル依存。仕様書 §3）。
    """

    def __init__(self, persist_directory: str, collection: str = "memories"):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 本番依存
            raise RuntimeError(
                "ChromaDB がインストールされていません。`pip install -e '.[vector]'` を実行してください。"
            ) from exc
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, record: MemoryRecord) -> None:
        if record.embedding is None:
            raise ValueError("ChromaVectorStore には embedding 付きの MemoryRecord が必要です")
        self._col.upsert(
            ids=[record.id],
            embeddings=[record.embedding.tolist()],
            documents=[record.text],
            metadatas=[{
                "kind": record.kind.value,
                "importance": float(record.importance),
                "created_at": float(record.created_at),
            }],
        )

    def query(self, embedding: np.ndarray, kind_filter: MemoryKind | None, top_k: int) -> list[MemoryRecord]:
        where = {"kind": kind_filter.value} if kind_filter is not None else None
        res = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=max(1, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[MemoryRecord] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for rid, doc, meta in zip(ids, docs, metas):
            out.append(MemoryRecord(
                id=rid,
                text=doc or "",
                kind=MemoryKind(meta.get("kind", MemoryKind.SEMANTIC.value)) if meta else MemoryKind.SEMANTIC,
                importance=float((meta or {}).get("importance", 0.0)),
                created_at=float((meta or {}).get("created_at", 0.0)),
            ))
        return out


# --------------------------------------------------------------------------- #
# ストア本体
# --------------------------------------------------------------------------- #
class HierarchicalMemoryStore:
    def __init__(
        self,
        vector_store: VectorStore,
        *,
        classifier: Classifier | None = None,
        embedder: Embedder | None = None,
        drive_change_threshold: float = 0.3,
    ):
        self._vector_store = vector_store
        self._classifier = classifier
        self._embedder = embedder
        self._threshold = float(drive_change_threshold)
        self._last_snapshot: dict[str, float] | None = None

    async def commit(self, text: str, drive_snapshot: dict[str, float], *, importance: float = 0.0) -> MemoryRecord:
        """テキストを記憶としてコミットする。

        Drive変化量 abs(delta) >= threshold の場合は MemoryKind.EMOTIONAL を強制付与。
        importance（サプライズ由来）は圧縮時の保持優先度として使用する。
        """
        delta = self._drive_delta(drive_snapshot)
        kind = self._classifier.classify(text) if self._classifier else MemoryKind.SEMANTIC
        if delta >= self._threshold:
            kind = MemoryKind.EMOTIONAL
        # 埋め込み計算・ベクトルストア書き込みはブロッキングI/Oのためループをブロックしない
        embedding = (
            await asyncio.to_thread(self._embedder.embed, text) if self._embedder is not None else None
        )
        record = MemoryRecord(
            text=text,
            kind=kind,
            drive_snapshot=dict(drive_snapshot),
            importance=float(importance),
            embedding=embedding,
        )
        await asyncio.to_thread(self._vector_store.add, record)
        self._last_snapshot = dict(drive_snapshot)
        return record

    def retrieve(
        self,
        query_embedding: np.ndarray,
        kind_filter: MemoryKind | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        return self._vector_store.query(query_embedding, kind_filter, top_k)

    @property
    def embedder(self) -> Embedder | None:
        """クエリ埋め込み用の embedder を公開する（v1.12: 記憶の想起・文脈注入）。"""
        return self._embedder

    def close(self) -> None:
        """所有する embedder を明示解放する（VRAM・メモリリーク防止）。"""
        embedder = getattr(self, "_embedder", None)
        close = getattr(embedder, "close", None)
        if callable(close):
            close()

    def _drive_delta(self, snapshot: dict[str, float]) -> float:
        if self._last_snapshot is None:
            return 0.0
        keys = set(self._last_snapshot) | set(snapshot)
        return max(abs(float(snapshot.get(k, 0.0)) - float(self._last_snapshot.get(k, 0.0))) for k in keys)

    def __len__(self) -> int:
        """記録件数（ChromaDB バックエンドは問い合わせしないため 0 を返す）。"""
        if hasattr(self._vector_store, "__len__"):
            return int(len(self._vector_store))
        return 0


# --------------------------------------------------------------------------- #
# 圧縮
# --------------------------------------------------------------------------- #
class MemoryCompressor:
    """WorkingBuffer が閾値を超えた際、古い内容を要約で置き換える。

    summarizer 未指定時は警告ログを出して古い内容を破棄する（オーバーフローは例外なく防ぐ）。
    """

    def __init__(self, summarizer: Summarizer | None = None, *, token_estimator=None):
        self._summarizer = summarizer
        # 既定の見積もりは「要約=1行」扱い（本実装のトークン計数と整合）。
        # 実運用ではトークナイザベースの estimator を渡すこと。
        self._token_estimator = token_estimator or (lambda text: 1)

    def close(self) -> None:
        """要約器（裏でLlamaモデルを保持する場合）を明示解放する。"""
        summarizer = getattr(self, "_summarizer", None)
        close = getattr(summarizer, "close", None)
        if callable(close):
            close()

    async def compress(
        self, buffer: WorkingBuffer, context_window: int, ratio: float
    ) -> tuple[str, int, str] | None:
        """閾値以下になるまで古い内容を要約で置き換える。

        戻り値: (除去した原文, 除去トークン数, 生成した要約)。圧縮が発生しなければ None。
        """
        target = max(1, int(context_window * ratio))
        removed_all: list[str] = []
        removed_tokens_total = 0
        last_summary = ""
        guard = 0
        while buffer.token_count > target:
            guard += 1
            if guard > 100:  # 安全弁（理論上は到達しない）
                logger.warning("圧縮が収束しません。バッファを強制リセットします")
                buffer.reset()
                break
            remove_tokens = buffer.token_count - target
            removed_text, removed_tokens = buffer.take_oldest(remove_tokens)
            if self._summarizer is None:
                logger.warning("要約器が未設定のため、古い内容を破棄しました（%d tokens）", removed_tokens)
                summary, summary_tokens = "", 1
            else:
                summary = self._summarizer.summarize(removed_text)
                summary_tokens = self._token_estimator(summary)
            buffer.prepend(summary, summary_tokens)
            removed_all.append(removed_text)
            removed_tokens_total += removed_tokens
            last_summary = summary
        if not removed_all:
            return None
        return "".join(removed_all), removed_tokens_total, last_summary
