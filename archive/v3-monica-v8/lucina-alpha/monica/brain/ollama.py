"""Ollama API 接続モジュール"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class OllamaConfig:
    """Ollama接続設定"""
    model: str = "qwen3.5:35b-a3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 120
    think: bool = False  # 推論モデルの思考表示オフ（Trueでthinkingフィールドに思考過程が入る）


@dataclass
class GenerationResult:
    """生成結果"""
    text: str
    tokens_generated: int
    total_duration_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


class OllamaBrain:
    """Ollama との通信を担当"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        think: Optional[bool] = None,
    ) -> GenerationResult:
        """OllamaのチャットAPIを呼び出す"""
        url = f"{self.config.host}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            },
            "stream": False,
            "think": think if think is not None else self.config.think,
        }

        if system:
            payload["system"] = system

        start = time.time()
        try:
            r = requests.post(url, json=payload, timeout=self.config.timeout)
            elapsed = time.time() - start
            data = r.json()

            if "error" in data:
                return GenerationResult(
                    text="",
                    tokens_generated=0,
                    total_duration_ms=elapsed * 1000,
                    tokens_per_second=0,
                    success=False,
                    error=data["error"],
                )

            msg = data.get("message", {})
            response_text = msg.get("content", "") or ""
            # フォールバック: contentが空ならthinkingを使う（think=true時など）
            if not response_text:
                response_text = msg.get("thinking", "") or ""
            eval_count = data.get("eval_count", 0)

            return GenerationResult(
                text=response_text.strip(),
                tokens_generated=eval_count,
                total_duration_ms=elapsed * 1000,
                tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
                success=True,
            )

        except requests.ConnectionError:
            return GenerationResult(
                text="",
                tokens_generated=0,
                total_duration_ms=(time.time() - start) * 1000,
                tokens_per_second=0,
                success=False,
                error="Ollamaに接続できません。`ollama serve` が実行されているか確認してください。",
            )
        except requests.Timeout:
            return GenerationResult(
                text="",
                tokens_generated=0,
                total_duration_ms=(time.time() - start) * 1000,
                tokens_per_second=0,
                success=False,
                error="応答がタイムアウトしました。モデルが大きすぎるか、リソースが不足している可能性があります。",
            )
        except Exception as e:
            return GenerationResult(
                text="",
                tokens_generated=0,
                total_duration_ms=(time.time() - start) * 1000,
                tokens_per_second=0,
                success=False,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Ollamaが起動しているか確認"""
        try:
            r = requests.get(f"{self.config.host}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False

    def list_models(self) -> list[str]:
        """利用可能なモデル一覧を取得"""
        try:
            r = requests.get(f"{self.config.host}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
