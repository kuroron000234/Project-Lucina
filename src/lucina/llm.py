"""
LLMクライアント — OllamaネイティブAPI（Thinkingモード対応）
"""

import json
import logging
import requests

logger = logging.getLogger("llm")


class LLMResponse:
    """LLM応答（thinking + content）"""
    def __init__(self, thinking: str = "", content: str = ""):
        self.thinking = thinking
        self.content = content

    def __str__(self) -> str:
        return self.content


class LLM:
    def __init__(
        self,
        model: str = "g4-midnight-macaw-v2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        num_ctx: int = 4096,
        think: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.think = False
        logger.info(f"LLM: {model} @ {base_url}/api/chat (thinking={think})")

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send chat messages and return response content (backward compatible)."""
        resp = self.chat_with_thinking(messages, **kwargs)
        return resp.content

    def chat_with_thinking(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Send chat messages and return both thinking and content."""
        est_tokens = sum(len(m.get("content", "")) // 2 for m in messages)
        budget = self.num_ctx - est_tokens - 256
        max_tok = max(128, min(self.max_tokens, budget))

        payload = {
            "model": self.model,
            "messages": messages,
            "think": kwargs.get("think", self.think),
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", max_tok),
                "num_ctx": self.num_ctx,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            thinking = msg.get("thinking", "") or ""
            content = msg.get("content", "") or ""
            return LLMResponse(thinking=thinking, content=content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return LLMResponse(thinking="", content="")

    def extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}
