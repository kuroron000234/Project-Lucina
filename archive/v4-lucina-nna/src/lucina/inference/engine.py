"""InferenceEngine — LLM呼び出しラッパー（仕様書 v1.4 §5.4）。

契約:
    - 本体の推論呼び出しは必ず executor（run_in_executor）経由で行い、
      呼び出し元のイベントループをブロックしない。
    - 単一フライト保証（C2）: 単一 Llama インスタンスは並行 generate に非対応のため、
      生成は必ず asyncio.Semaphore(1) で直列化する。
    - 戻り値: (生成トークン文字列, サプライズ値[0,1])。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .backends import TokenBackend
from .entropy import surprise_from_logits
from .logits import DriveLogitsProcessor, VocabMap


class InferenceEngine:
    def __init__(
        self,
        llm_path: str,
        executor: ThreadPoolExecutor,
        *,
        backend: TokenBackend | None = None,
        vocab_map: VocabMap | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._llm_path = llm_path
        self._executor = executor
        self._config = config or {}
        self._backend = backend if backend is not None else _lazy_backend(llm_path, self._config)
        self._vocab_map: VocabMap = vocab_map or {}
        inf = self._config.get("inference", {})
        self._logits_processor = DriveLogitsProcessor(
            coefficient=float(inf.get("logit_bias_coefficient", 2.5))
        )
        self._entropy_scaling = float(inf.get("entropy_scaling", 5.0))
        self._sampling = str(inf.get("sampling", "multinomial"))
        self._temperature = float(inf.get("temperature", 0.8))
        self._rng = np.random.default_rng(int(inf.get("seed", 42)))
        self._single_flight = asyncio.Semaphore(1)
        self.logger = None  # 差分ログ用（§8）。core が StructuredLogger を設定する

    def set_vocab_map(self, vocab_map: VocabMap) -> None:
        self._vocab_map = vocab_map

    def format_prompt(self, text: str, system: str | None = None, **tpl_kwargs: Any) -> str:
        """初期プロンプトをモデルのチャットテンプレートでラップする。

        新世代モデルは生テキストでは英語モード等へ遷移するため、初期プロンプトは
        テンプレート適用済み文字列として投入する。非対応バックエンド（モック等）は
        原文をそのまま返す。

        config の inference.chat_template_kwargs（例: llm-jp-4 の reasoning_effort: off）を
        テンプレート変数として既定で渡し、呼び出し側の kwargs が上書きする。
        """
        kwargs = dict(self._config.get("inference", {}).get("chat_template_kwargs", {}))
        kwargs.update(tpl_kwargs)
        fmt = getattr(self._backend, "format_chat_prompt", None)
        if callable(fmt):
            return fmt(text, system=system, **kwargs)
        return text

    def close(self) -> None:
        """所有するバックエンドを明示解放する（実験ループでの VRAM リーク防止）。"""
        backend = getattr(self, "_backend", None)
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    @property
    def vocab_map(self) -> VocabMap:
        return self._vocab_map

    async def generate_next_token(self, context: list[str], drive_state: dict) -> tuple[str, float]:
        """(生成トークン, サプライズ値[0,1]) を返す。イベントループをブロックしない。"""
        async with self._single_flight:
            loop = asyncio.get_running_loop()
            prompt = "".join(context)

            raw_logits: np.ndarray = await loop.run_in_executor(
                self._executor, self._backend.next_token_logits, prompt
            )
            raw_logits = np.asarray(raw_logits, dtype=np.float64)
            biased = self._logits_processor.apply(raw_logits, drive_state, self._vocab_map)
            self._emit_logit_diff(raw_logits, biased)

            token_id = await loop.run_in_executor(self._executor, self._sample, biased)
            token_text: str = await loop.run_in_executor(self._executor, self._backend.decode, int(token_id))
            surprise = surprise_from_logits(biased, self._entropy_scaling)
            return token_text, surprise

    async def generate_decision(self, context: list[str], options: list[str]) -> str:
        """制約付きデコード: options のいずれかに必ず収束するようロジットをマスクしてサンプリングする。

        モデル駆動スケジューリング（v1.9）の中核。各オプションのトークン列で prefix-trie を構築し、
        現在の trie ノードから遷移可能なトークン以外のロジットを -inf にしてサンプリングする。
        終端ノードに到達した時点でそのオプションを返す。Driveバイアスは適用しない（決断は
        人格バイアスから独立した「意志」として扱う。空の drive_state を渡すため実質無バイアス）。

        バックエンドが何を出力しようとしても選択肢外には出られないため、結果は常に options の
        いずれかになる（greedy では決定論的、multinomial では温度付きで揺れる）。
        """
        async with self._single_flight:
            loop = asyncio.get_running_loop()
            prompt = "".join(context)
            pairs = [(opt, list(self._backend.encode(opt))) for opt in options]
            pairs = [(opt, seq) for opt, seq in pairs if seq]
            if not pairs:
                return options[0] if options else ""
            trie = _build_decision_trie(pairs)
            node = trie
            max_steps = max(len(seq) for _, seq in pairs) + 2
            for _ in range(max_steps):
                allowed = list(node["children"].keys())
                if not allowed:
                    break
                raw = np.asarray(
                    await loop.run_in_executor(self._executor, self._backend.next_token_logits, prompt),
                    dtype=np.float64,
                )
                # 決断は人格バイアスから独立させる（drive_state={} → バイアスは実質ゼロ）
                biased = self._logits_processor.apply(raw, {}, self._vocab_map)
                masked = np.full(biased.shape, -np.inf)
                masked[allowed] = biased[allowed]
                token_id = await loop.run_in_executor(self._executor, self._sample, masked)
                node = node["children"][int(token_id)]
                prompt += self._backend.decode(int(token_id))
                if node.get("terminal"):
                    return str(node["option"])
            # 理論上到達しない（trie は常に終端に到達する）: 安全のため先頭オプションを返す
            return pairs[0][0]

    def tokenize(self, text: str) -> list[int]:
        return self._backend.encode(text)

    async def tokenize_async(self, text: str) -> list[int]:
        """ブロッキングなトークナイズを executor 経由で実行（イベントループをブロックしない）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._backend.encode, text)

    # ------------------------------------------------------------------ #
    def _sample(self, logits: np.ndarray) -> int:
        if self._sampling == "greedy":
            return int(np.argmax(logits))
        # multinomial（温度付き）
        z = logits / max(self._temperature, 1e-9)
        z = z - np.max(z)
        exp = np.exp(z)
        probs = exp / np.sum(exp)
        return int(self._rng.choice(probs.size, p=probs))

    def _emit_logit_diff(self, before: np.ndarray, after: np.ndarray) -> None:
        if self.logger is None:
            return
        # §8: バイアス適用前後の差分ログ（対象語彙の先頭トークンIDの差分を記録）
        target_ids = sorted({
            int(seq[0]) for seqs in self._vocab_map.values() for seq in seqs if seq
        })
        self.logger.logit_diff(
            before=before,
            after=after,
            target_ids=target_ids,
            coefficient=self._logits_processor.coefficient,
        )


def _build_decision_trie(pairs: list[tuple[str, list[int]]]) -> dict:
    """制約付きデコード用の prefix-trie を構築する（v1.9）。

    各ノードは {"children": {token_id: node}, "terminal": bool, "option": str}。
    短いオプションが長いオプションの接頭辞になる場合（例: 待機 と 待機中）、短い方の
    終端に先に到達して返る（prefix 一致の自然な挙動）。
    """
    root: dict = {"children": {}}
    for option, seq in pairs:
        node = root
        for tid in seq:
            node = node["children"].setdefault(int(tid), {"children": {}})
        node["terminal"] = True
        node["option"] = option
    return root


def _lazy_backend(llm_path: str, config: dict[str, Any]) -> TokenBackend:
    """本番バックエンド生成（遅延）。モデル未選定なら明確なエラーを出す。"""
    from .backends import LlamaBackend

    if not llm_path or llm_path.endswith("SELECTED_MODEL.gguf"):
        raise RuntimeError(
            "model.path が未選定です。§1のモデル選定フェーズを完了し、config の model.path を設定してください"
        )
    model_cfg = config.get("model", {})
    return LlamaBackend(
        model_path=llm_path,
        n_ctx=int(model_cfg.get("context_window", 8192)),
        n_gpu_layers=int(model_cfg.get("n_gpu_layers", -1)),
    )
