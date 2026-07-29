"""LLM Cognitive Layer — 言語・想像・解釈能力

LLM は「人格」ではなく「認知能力」として機能する。
以下を行い、行動選択は行わない：

  ✅ 候補生成    — 「この状況で取り得る行動は？」
  ✅ 未来予測    — 「この行動をとるとどうなるか？」
  ✅ 結果解釈    — 「この結果は何を意味するか？」
  ❌ 行動選択    — EFE が担当
  ❌ 価値評価    — Core が担当
  ❌ 自己更新    — Core が担当

Ollama (Qwen3-35B-A3B) をデフォルトバックエンドとし、
LLM なしでも動作可能な Fallback モードを持つ。
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCognitiveLayer:
    """LLM を認知能力として統合するラッパー。

    Parameters
    ----------
    backend : str
        "ollama" または "fallback"。"ollama" が接続できない場合も自動で fallback する。
    model : str
        使用するモデル名 (Ollama の場合)。
    ollama_url : str
        Ollama API のベース URL。
    temperature : float
        生成時の温度 (0.0=決定論的, 1.0=多様)。
    """
    def __init__(
        self,
        backend: str = "ollama",
        model: str = "qwen3.5:35b-a3b",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.temperature = temperature
        self._backend = backend
        self._available = False
        self._init_attempted = False

    # --- Connection Management ---

    def is_available(self) -> bool:
        """LLM が実際に利用可能か確認する。"""
        if not self._init_attempted:
            self._check_connection()
        return self._available

    def _check_connection(self):
        """Ollama サーバーに接続を試みる。"""
        self._init_attempted = True
        if self._backend == "fallback":
            self._available = False
            return
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self.ollama_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = [m["name"] for m in data.get("models", [])]
                    # Check if our model or a close match exists
                    if any(self.model in m for m in models):
                        self._available = True
                        logger.info(f"LLM connected: {self.model}")
                    else:
                        # Try to use any available model
                        if models:
                            self.model = models[0]
                            self._available = True
                            logger.warning(
                                f"Requested model not found, using {self.model}"
                            )
                        else:
                            logger.warning("No models in Ollama")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            self._available = False

    # --- Core LLM Call ---

    def _call_ollama(self, prompt: str, system: str = "") -> Optional[str]:
        """Ollama API を呼び出し、生成テキストを返す。"""
        if not self.is_available():
            return None
        try:
            import urllib.request

            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "temperature": self.temperature,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

    # --- Candidate Generation ---

    def generate_candidates(
        self,
        context: str,
        available_actions: Optional[list[str]] = None,
    ) -> list[str]:
        """現在の状況から取り得る行動の候補を生成する。

        Parameters
        ----------
        context : str
            現在の状況の説明。
        available_actions : list[str] | None
            利用可能な行動のリスト（None の場合は LLM に任せる）。

        Returns
        -------
        list[str]
            候補行動のリスト（最大 5 件）。
        """
        return self._generate(context, available_actions)

    def _generate(self, context: str, available: Optional[list[str]] = None) -> list[str]:
        """LLM または Fallback で候補を生成する。"""
        # Try LLM first
        if self.is_available():
            candidates = self._generate_with_llm(context, available)
            if candidates and len(candidates) > 0:
                return candidates

        # Fallback: return available actions or basic defaults
        if available:
            return available[:5]
        return ["A", "B", "C", "rest", "explore"]

    def _generate_with_llm(
        self, context: str, available: Optional[list[str]] = None
    ) -> list[str]:
        """LLM を使って候補を生成する。"""
        avail_str = f"\n利用可能な行動: {', '.join(available)}" if available else ""
        system = (
            "あなたは自律エージェントの「想像力」モジュールです。\n"
            "与えられた状況で取り得る行動を簡潔に列挙してください。\n"
            "各行動は1単語または短いフレーズで、1行に1つだけ書いてください。\n"
            "3〜5個の候補を出力してください。"
        )
        prompt = (
            f"現在の状況: {context}"
            f"{avail_str}\n\n"
            "取り得る行動:"
        )
        response = self._call_ollama(prompt, system)
        if not response:
            return []

        # Parse: extract lines that look like actions
        candidates = []
        for line in response.strip().split("\n"):
            # Remove leading numbers, bullets, whitespace
            cleaned = re.sub(r'^[\s\d\.\-\)\*·•]+\s*', '', line).strip()
            if cleaned and len(cleaned) < 50 and cleaned not in candidates:
                candidates.append(cleaned)

        return candidates[:5]

    # --- Outcome Prediction ---

    def predict_outcome(
        self,
        action: str,
        context: str,
        world_model_prediction: Optional[dict] = None,
    ) -> dict:
        """行動の結果を予測する。

        LLM が利用可能な場合は言語的な予測も生成し、
        利用不可の場合は World Model の数値予測をそのまま返す。

        戻り値には常に world_model_prediction の内容が含まれる。
        """
        result = {}
        if world_model_prediction:
            result.update(world_model_prediction)

        if self.is_available():
            system = (
                "あなたは自律エージェントの「予測」モジュールです。"
                "行動の結果を簡潔に予測してください。"
            )
            prompt = (
                f"状況: {context}\n"
                f"取ろうとしている行動: {action}\n\n"
                f"予測される結果:"
            )
            text_prediction = self._call_ollama(prompt, system)
            if text_prediction:
                result["text_prediction"] = text_prediction

        return result

    # --- Result Interpretation ---

    def interpret_result(
        self,
        action: str,
        outcome: str,
        prediction: dict,
        context: str,
    ) -> str:
        """観測結果を解釈する。

        LLM が利用可能な場合は言語的な解釈を生成し、
        利用不可の場合は単純なテンプレート文を返す。
        """
        if not self.is_available():
            return f"Action '{action}' resulted in '{outcome}'."

        # Calculate surprise for richer interpretation
        surprise = prediction.get(outcome, 0)
        surprise_level = (
            "very surprising" if surprise < 0.3
            else "moderately surprising" if surprise < 0.7
            else "expected"
        )

        system = (
            "あなたは自律エージェントの「解釈」モジュールです。\n"
            "観測結果が何を意味するかを1文で簡潔に解釈してください。\n"
            "推測や過剰な解釈は避けてください。"
        )
        prompt = (
            f"状況: {context}\n"
            f"行動: {action}\n"
            f"結果: {outcome}\n"
            f"予測の確信度: {surprise_level}\n\n"
            f"解釈:"
        )
        interpretation = self._call_ollama(prompt, system)
        return interpretation or f"Action '{action}' resulted in '{outcome}'."

    # --- Learning from LLM output (semantic PE) ---

    def semantic_surprise(self, observation: str, prediction: str) -> float:
        """LLM の解釈と実際の結果の意味的な一致度を測る。

        将来 Phase 8+ で semantic PE として使用する。
        Phase 3 では常に 0.0（未使用）を返す。
        """
        return 0.0

    # --- State ---

    def summary(self) -> dict:
        return {
            "model": self.model,
            "available": self.is_available(),
            "backend": self._backend,
            "temperature": self.temperature,
        }
