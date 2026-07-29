"""共有LLMクライアント — Zen API / OpenRouter を自動判別"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

_EMBED_DIM = 64


def get_api_config() -> tuple[str, str, str]:
    key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    if key.startswith("sk-or-"):
        base = "https://openrouter.ai/api/v1"
    else:
        base = "https://opencode.ai/zen/v1"
    model = os.environ.get("MONIKA_MODEL", "tencent/hy3:free")
    return key, base, model


def call_llm(prompt: str, max_tokens: int = 200, temperature: float = 0.8,
             max_retries: int = 2, system_prompt: str | None = None) -> str | None:
    key, base, model = get_api_config()
    if not key:
        return None
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    last_err = None
    for _ in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                body = json.loads(resp.read())
                return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(3)
                continue
        except Exception as e:
            last_err = e
            time.sleep(1)
    return None


def embed(text: str) -> list[float] | None:
    key, base, _ = get_api_config()
    if not key or base.endswith("openrouter.ai/api/v1"):
        return None
    try:
        data = json.dumps({"input": text, "model": "text-embedding-3-small"}).encode()
        req = urllib.request.Request(
            f"{base}/embeddings", data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
            return body["data"][0]["embedding"]
    except Exception:
        return None


def trigram_hash(text: str, dim: int = _EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    norm = text.lower()
    grams = set()
    for i in range(len(norm) - 2):
        grams.add(norm[i:i + 3])
    for g in grams:
        h = 0
        for ch in g:
            h = (h * 31 + ord(ch)) % dim
        vec[h] += 1.0
    total = sum(v * v for v in vec) ** 0.5
    if total > 0:
        vec = [v / total for v in vec]
    return vec
