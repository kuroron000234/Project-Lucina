"""本番用アダプタ（実モデル・実ライブラリ対応・遅延import）。

- LlamaTokenizerAdapter: LlamaBackend を Tokenizer プロトコルとして公開（DriveVocabExpander用）
- SentenceTransformerEmbedder: 語彙拡張・埋め込み検索の共通モデル（仕様書 §2手順3）
- LlamaSummarizer: 要約専用軽量モデル（MemoryCompressor 用）
"""

from __future__ import annotations

import logging

import numpy as np

from .backends import LlamaBackend

logger = logging.getLogger("lucina.inference.adapters")


class LlamaTokenizerAdapter:
    def __init__(self, backend: LlamaBackend):
        """エンジンと同一の LlamaBackend を共有する（実モデルを多重ロードしない）。"""
        self._backend = backend
        self._bos_id = self._detect_bos()

    def _detect_bos(self) -> int | None:
        """GGUFメタデータに従いBOSを付加するモデル（Gemma/Llama系等）のBOS IDを特定する。

        Qwen系は add_bos_token=false のため BOS が付かない。
        """
        try:
            llm = self._backend._llm  # noqa: SLF001 - トークナイザ設定の参照
            bos = llm.token_bos()
            # BOS が実際に付加されるかどうかは実エンコードで判定する
            probe = llm.tokenize("寂しい".encode("utf-8"))
            if probe and probe[0] == bos:
                return int(bos)
        except Exception:  # noqa: BLE001 - BOS非対応モデルでは None のまま
            return None
        return None

    def encode(self, text: str) -> list[int]:
        """BOS を除去したトークン列を返す。

        語彙トークン列・セグメント追跡に使う。生成トークン列には BOS は現れないため、
        BOS 付きの語彙列では部分列一致（③）が絶対に成立しなくなる（収束率0%問題）。
        """
        ids = self._backend.encode(text)
        if self._bos_id is not None and ids and ids[0] == self._bos_id:
            return ids[1:]
        return ids

    def decode(self, token_id: int) -> str:
        return self._backend.decode(token_id)

    def words(self, max_words: int = 0) -> list[str]:
        """語彙拡張の候補: トークンの復号文字列（起動時1回のみ）。

        max_words > 0 の場合、先頭からその数だけ復号する（実モデルの語彙は数万〜
        数十万あり、全復号は起動を遅くするため。max_candidates と併用）。
        """
        limit = self._backend.vocab_size() if max_words <= 0 else min(max_words, self._backend.vocab_size())
        out: list[str] = []
        seen: set[str] = set()
        for i in range(limit):
            w = self._backend.decode(i).strip()
            if w and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def vocab_size(self) -> int:
        return self._backend.vocab_size()


class SentenceTransformerEmbedder:
    """埋め込みモデル（語彙拡張・記憶検索用）。

    device: 既定は cpu。語彙拡張は起動時1回の軽量処理であり、GPU を使うと
    8GB VRAM の Llama モデルと競合するため CPU を推奨（モデル選定時の多重ロード対策）。
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 本番依存
            raise RuntimeError(
                "sentence-transformers がインストールされていません。"
                "`pip install -e '.[embed]'` を実行してください。"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float64)

    def embed_many(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float64)

    def close(self) -> None:
        """モデルを解放する（多重ロード時も確実にメモリを返す）。"""
        model = getattr(self, "_model", None)
        if model is not None:
            model = None  # noqa: B018 - 参照を外してGCに任せる
            self._model = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass


class LlamaSummarizer:
    """要約専用の軽量モデルによる要約。プロンプトベースの生成要約。

    モデルは遅延ロードする（初回 summarize 時）。起動時にファイルが無くても
    run_agent は起動できる。生成失敗時は抽出的要約へフォールバックする。
    """

    _PROMPT_TEMPLATE = "以下の文章を簡潔に要約してください。\n\n{snippet}\n\n要約:"

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._backend: LlamaBackend | None = None

    def _ensure_backend(self) -> LlamaBackend:
        if self._backend is None:
            self._backend = LlamaBackend(self._model_path, n_ctx=self._n_ctx, n_gpu_layers=-1)
        return self._backend

    def summarize(self, text: str) -> str:
        try:
            backend = self._ensure_backend()
            snippet = text[-1500:] if len(text) > 1500 else text
            prompt = self._PROMPT_TEMPLATE.format(snippet=snippet)
            # チャットテンプレート前提モデルではラップしてから生成する（英語遷移防止）
            fmt = getattr(backend, "format_chat_prompt", None)
            if callable(fmt):
                prompt = fmt(prompt)
            out = backend.complete(prompt, max_tokens=64)
            if out.strip():
                return out.strip()
        except Exception:  # noqa: BLE001 - 要約失敗時は抽出的要約へフォールバック
            logger.warning("要約モデルの生成に失敗したため抽出的要約へフォールバックします", exc_info=True)
        return text[:120]

    def close(self) -> None:
        """遅延ロードした要約用バックエンドを明示解放する。"""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.close()
            self._backend = None
