"""トークン生成バックエンド（仕様書 v1.4 §5.4）。

TokenBackend: ブロッキング前提のインターフェース。InferenceEngine は必ず
ThreadPoolExecutor 経由（run_in_executor）で呼び出し、イベントループをブロックしない（C2）。

LlamaBackend: llama-cpp-python の本番実装。遅延import（オプショナル依存）。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger("lucina.inference.backends")


class TokenBackend(Protocol):
    """1トークン分のロジットを返すブロッキングなバックエンド。"""

    def next_token_logits(self, context_text: str) -> np.ndarray: ...
    def decode(self, token_id: int) -> str: ...
    def encode(self, text: str) -> list[int]: ...
    def vocab_size(self) -> int: ...


class LlamaBackend:
    """llama-cpp-python ベースの本番バックエンド（遅延import）。

    close() で llama.cpp のモデル・CUDAコンテキストを明示解放する。
    解放しないと試行を重ねる実験（校正・モデル選定）で VRAM が枯渇する（C2）。
    """

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - 本番依存
            raise RuntimeError(
                "llama-cpp-python がインストールされていません。"
                "`pip install -e '.[llm]'` を実行してください（GPUはCMAKE_ARGS指定）。"
            ) from exc
        # NOTE: logits_all=False では eval() がロジットを保存しない（llama-cpp-python 0.3.34）。
        # ロジットバイアス適用には全トークンのロジットが必要なため logits_all=True で構築する。
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=True)
        self._chat_formatter = self._build_chat_formatter()

    def _build_chat_formatter(self):
        """GGUFメタデータの tokenizer.chat_template からチャットフォーマッタを構築する。

        新世代モデル（Qwen3系・Gemma3系など）はチャットテンプレート前提であり、
        生テキストを与えると英語モード等へ遷移してしまう（実験の収束判定が崩れる）。
        テンプレートを持たないモデルは None を返し、従来の生テキスト挙動にフォールバックする。
        """
        try:
            llm = self._llm
            meta = getattr(llm, "metadata", None) or {}
            template = meta.get("tokenizer.chat_template")
            if not template:
                return None
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            # token_eos()/token_bos() はトークンID（int）を返す。テンプレートの
            # {{ eos_token }} 変数には文字列が必要なため detokenize で復号する。
            eos = llm.detokenize([llm.token_eos()]).decode("utf-8", errors="replace")
            bos = llm.detokenize([llm.token_bos()]).decode("utf-8", errors="replace")
            return Jinja2ChatFormatter(
                template=str(template),
                eos_token=eos,
                bos_token=bos,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001 - テンプレート非対応モデルは従来挙動へフォールバック
            # テンプレート破損等の異常系は黙って生テキスト運用にすると英語遷移・退化出力になるため
            # 警告を残す（運用時の切り分け用）。テンプレート非存在（if not template）は正常系でログ不要。
            logger.warning(
                "チャットテンプレートの構築に失敗したため生テキスト運用にフォールバックします: %s", exc
            )
            return None

    def format_chat_prompt(
        self, text: str, system: str | None = None, **tpl_kwargs: Any
    ) -> str:
        """初期プロンプトをモデルのチャットテンプレートでラップする。

        text を user メッセージとしてフォーマットし、assistant の生成開始位置まで
        含めたプロンプト文字列を返す。テンプレート非対応モデルでは text をそのまま返す。
        生成トークンはラップせず追記される（テンプレートは初期プロンプトに1回だけ適用）。

        tpl_kwargs はテンプレート変数として渡される（例: llm-jp-4 の reasoning_effort）。
        モデルのテンプレートが参照しない変数は無害に無視される。
        """
        formatter = getattr(self, "_chat_formatter", None)
        if formatter is None:
            return text
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        resp = formatter(messages=messages, **tpl_kwargs)
        return str(resp.prompt)

    def close(self) -> None:
        """llama.cpp のモデル・CUDAコンテキストを明示解放する（呼び出し後は再利用不可）。"""
        llm = getattr(self, "_llm", None)
        if llm is not None:
            llm.close()
            self._llm = None

    def __del__(self) -> None:
        # 明示的な close() が呼ばれなかった場合の保険（実験ループでのリーク防止）
        try:
            self.close()
        except Exception:  # noqa: BLE001 - デストラクタ内の例外は握りつぶす
            pass

    def next_token_logits(self, context_text: str) -> np.ndarray:
        ids = self._llm.tokenize(context_text.encode("utf-8"))
        self._llm.reset()
        self._llm.eval(ids)
        # logits_all=True の scores は (n_ctx, n_vocab)。
        # プロンプト末尾トークンのロジット = scores[n_tokens - 1]（次のトークン予測）。
        n_prompt = self._llm.n_tokens
        if n_prompt <= 0:
            return np.zeros(self.vocab_size(), dtype=np.float64)
        return np.asarray(self._llm.scores[n_prompt - 1], dtype=np.float64)

    def decode(self, token_id: int) -> str:
        return self._llm.detokenize([int(token_id)]).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        return list(self._llm.tokenize(text.encode("utf-8")))

    def complete(self, prompt: str, max_tokens: int = 64) -> str:
        """自由生成（要約等）。llama-cpp-python の create_completion を利用。"""
        res = self._llm.create_completion(
            prompt, max_tokens=int(max_tokens), temperature=0.3, echo=False, stop=["\n\n"],
        )
        return str(res["choices"][0]["text"])

    def vocab_size(self) -> int:
        return int(self._llm.n_vocab())
