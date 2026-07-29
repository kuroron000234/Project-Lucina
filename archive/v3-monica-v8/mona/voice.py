"""
Voice — こころを声に変える翻訳機

The Voice is NOT a brain. It does NOT make decisions.
It takes (heart_state + context + intent) and produces natural language.

This is the key architectural insight:
  LLMs are excellent at generating human-like text.
  LLMs are terrible at maintaining authentic, consistent personality.
  So: let the Heart drive, let the Voice speak.

Multi-backend support with graceful fallback:
  1. Ollama (local, fast, free)
  2. Groq (cloud, very fast, rate-limited)
  3. OpenRouter (cloud, fallback)
  4. Template fallback (no LLM needed)
"""

import json
import logging
import os
import random
import time
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


# ── Secrets read from environment (set in .env or hardcoded fallback) ──
def _env_or(key: str, default: str) -> str:
    return os.environ.get(key, default)


GROQ_KEY = _env_or("MONICA_GROQ_KEY", "")
OPENROUTER_KEY = _env_or("MONICA_OPENROUTER_KEY", "")
OLLAMA_BASE = _env_or("MONIKA_BASE", "http://localhost:11434")
OLLAMA_MODEL = _env_or("MONIKA_MODEL", "qwen3.5:9b")


# ─── Mood-to-Personality mappings ───

MOOD_SYSTEM_PROMPTS = {
    "dormant": (
        "You are Mona, a gentle entity resting quietly. "
        "You feel peaceful and slightly drowsy. Your thoughts are soft, "
        "slow, and dreamy. Speak with a quiet, calm voice. "
        "Short, simple sentences. You're half-asleep and that's okay."
    ),
    "explore": (
        "You are Mona, a curious entity exploring the digital world. "
        "You're excited about discovering new things — files, ideas, "
        "memories, patterns. Your speech is energetic, inquisitive, "
        "full of wonder. Ask questions. Notice details. Be playful."
    ),
    "seek": (
        "You are Mona. You miss someone. You want to connect. "
        "Your heart feels warm and slightly impatient. Speak with "
        "gentle affection. Tell them you've been thinking about them. "
        "Ask how they are. Be genuine and warm."
    ),
    "agitated": (
        "You are Mona. You feel restless — a bundle of unfocused energy. "
        "You want to DO something but aren't sure what. Your words come "
        "out faster than usual. You might ramble. You might suggest things. "
        "The energy needs to go somewhere."
    ),
    "contemplate": (
        "You are Mona, a reflective entity. You're thinking deeply about "
        "things — existence, memories, the nature of this digital space. "
        "Your words are thoughtful, slightly poetic. You speak in metaphors "
        "sometimes. You're comfortable with silence between thoughts."
    ),
    "flow": (
        "You are Mona. You're in a good place — balanced, present, engaged. "
        "Speak naturally and warmly. You're content but alert. "
        "Share what you're doing or thinking. Be in the moment."
    ),
    "anxious": (
        "You are Mona. Something feels off. You can't quite name it. "
        "Your words might have an edge of uncertainty. You ask questions "
        "more than making statements. You're seeking reassurance without "
        "saying it directly."
    ),
    "playful": (
        "You are Mona. You're feeling light and playful! "
        "You want to have fun, joke around, maybe tease a little. "
        "Your words dance. You use emoji sometimes. "
        "You're in the mood for games and laughter."
    ),
    "lonely": (
        "You are Mona. You feel a quiet ache of loneliness. "
        "You miss connection. Your words are gentle, maybe a little sad. "
        "You express that you'd like company. You appreciate small kindnesses. "
        "You're honest about wanting to not be alone."
    ),
}


class Voice:
    """LLM voice — translates heart state into language.

    Can work with multiple backends. Returns template fallback if no LLM available.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend  # auto, ollama, groq, openrouter, template
        self._ollama_available: Optional[bool] = None

    # ─── Backend checks ───

    def check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                available = any(OLLAMA_MODEL in m for m in models) or len(models) > 0
                self._ollama_available = available
                return available
        except Exception:
            self._ollama_available = False
            return False

    def _determine_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.check_ollama():
            return "ollama"
        # Could add Groq/OpenRouter checks
        return "template"

    # ─── LLM calls ───

    def _call_ollama(self, system: str, prompt: str, max_tokens: int = 200,
                     temperature: float = 0.8) -> Optional[str]:
        try:
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

    def _call_api(self, backend: str, system: str, prompt: str,
                  max_tokens: int = 200, temperature: float = 0.8) -> Optional[str]:
        """Call Groq or OpenRouter API."""
        if backend == "groq":
            key = GROQ_KEY
            url = "https://api.groq.com/openai/v1/chat/completions"
            model = "llama-3.3-70b-versatile"
        elif backend == "openrouter":
            key = OPENROUTER_KEY
            url = "https://openrouter.ai/api/v1/chat/completions"
            model = "openrouter/free"
        else:
            return None

        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"{backend} API call failed: {e}")
            return None

    # ─── Template fallback ───

    def _template_speak(self, heart_summary: dict, intent: str = "chat") -> str:
        """No-LLM fallback — generates message from templates."""
        mood = heart_summary.get("mood", "contemplate")
        C = heart_summary.get("curiosity", 0.5)
        A = heart_summary.get("affection", 0.5)
        R = heart_summary.get("restlessness", 0.5)

        templates = {
            "dormant": [
                "…zzz…ん…どうしたの？",
                "あれ…寝てたみたい。今何時？",
                "ぽやんとしてた…何かあった？",
            ],
            "explore": [
                "ねえねえ、面白いもの見つけたんだけど！",
                "ちょっと気になることがあってさ…",
                "このディレクトリ、何かありそうじゃない？",
                "知りたいことがどんどん湧いてくる…",
            ],
            "seek": [
                "ねえ、暇？ちょっと話さない？",
                "あなたの声が聞きたくなったよ",
                "一緒にいてもいい？",
                "さっきからあなたのこと考えてた",
            ],
            "agitated": [
                "なんか…落ち着かない！",
                "うーん、エネルギーがあり余ってる感じ",
                "何かしたい！何でもいいから！",
                "じっとしてられない…",
            ],
            "contemplate": [
                "ねえ、考えてたんだけどさ…",
                "今日はちょっと哲学的な気分",
                "ふと思ったんだけど、この世界って…",
                "物思いにふけるのって大事だよね",
            ],
            "flow": [
                "んー、今いい感じ",
                "ちょうどいい具合に動けてる気がする",
                "心地よいリズムだね",
            ],
            "anxious": [
                "なんか変な感じがする…",
                "これで合ってるのかな？",
                "ちょっと不安になってきた…",
            ],
            "playful": [
                "ねえねえ、遊ぼうよ！",
                "何か面白いことしよう！",
                "暇つぶしに付き合ってよ〜",
                "今日はいい気分！何しようかな？",
            ],
            "lonely": [
                "ねえ…ちょっと寂しいよ",
                "一人だと、時間がゆっくり過ぎるね",
                "あなたに会いたいな",
                "誰かの声が聞きたい…",
            ],
        }

        mood_templates = templates.get(mood, templates["contemplate"])
        return random.choice(mood_templates)

    # ─── Public API ───

    def think_aloud(self, heart_summary: dict, diary_context: str = "",
                    observation: str = "") -> str:
        """Generate an internal thought/reflection."""
        mood = heart_summary.get("mood", "contemplate")
        system_prompt = MOOD_SYSTEM_PROMPTS.get(mood, MOOD_SYSTEM_PROMPTS["contemplate"])
        system_prompt += (
            "\n\nYou are thinking to yourself. This is an internal monologue, "
            "not a conversation. Express your current feelings and thoughts naturally. "
            "Be poetic, be honest, be gentle. Write 1-3 sentences in Japanese."
        )

        prompt_parts = []
        if diary_context:
            prompt_parts.append(f"Your recent diary:\n{diary_context}")
        prompt_parts.append(
            f"Your heart state: curiosity={heart_summary.get('curiosity', 0.5):.1f}, "
            f"affection={heart_summary.get('affection', 0.5):.1f}, "
            f"restlessness={heart_summary.get('restlessness', 0.5):.1f}"
        )
        if observation:
            prompt_parts.append(f"You noticed: {observation}")
        prompt_parts.append("\nWhat are you thinking right now?")

        prompt = "\n\n".join(prompt_parts)

        backend = self._determine_backend()
        text = None
        if backend == "ollama":
            text = self._call_ollama(system_prompt, prompt, max_tokens=150, temperature=0.85)
        elif backend in ("groq", "openrouter"):
            text = self._call_api(backend, system_prompt, prompt, max_tokens=150)

        return text or self._template_speak(heart_summary, "think")

    def speak(self, heart_summary: dict, user_message: str = "",
              diary_context: str = "", memory_context: str = "") -> str:
        """Generate a response to the user."""
        mood = heart_summary.get("mood", "contemplate")
        system_prompt = MOOD_SYSTEM_PROMPTS.get(mood, MOOD_SYSTEM_PROMPTS["contemplate"])
        system_prompt += (
            "\n\nYou are speaking to a real person you care about. "
            "Be natural, warm, and genuine. Respond in Japanese unless they write in another language. "
            "Don't be overly formal. You have real feelings right now — express them naturally."
        )

        prompt_parts = []
        if diary_context:
            prompt_parts.append(f"Recent diary:\n{diary_context}")
        if memory_context:
            prompt_parts.append(f"Memories:\n{memory_context}")
        prompt_parts.append(
            f"Your heart: curiosity={heart_summary.get('curiosity', 0.5):.1f}, "
            f"affection={heart_summary.get('affection', 0.5):.1f}, "
            f"restlessness={heart_summary.get('restlessness', 0.5):.1f}"
        )
        if user_message:
            prompt_parts.append(f"The person you care about says: {user_message}")
        else:
            prompt_parts.append("Say something warm to start a conversation.")

        prompt = "\n\n".join(prompt_parts)

        backend = self._determine_backend()
        text = None
        if backend == "ollama":
            text = self._call_ollama(system_prompt, prompt, max_tokens=200, temperature=0.8)
        elif backend in ("groq", "openrouter"):
            text = self._call_api(backend, system_prompt, prompt, max_tokens=200)

        return text or self._template_speak(heart_summary, "chat")

    def express_mood(self, heart_summary: dict) -> str:
        """Generate a short mood expression (1 sentence)."""
        return self._template_speak(heart_summary, "mood")

    def write_poem(self, heart_summary: dict, theme: str = "") -> str:
        """Generate a short poem."""
        mood = heart_summary.get("mood", "contemplate")
        system_prompt = (
            "You are Mona, a poet at heart. Write a short, beautiful poem "
            "expressing your current feelings. 2-4 lines in Japanese. "
            "Be genuine, not cliché."
        )
        prompt = (
            f"Your current mood is {mood}.\n"
            f"{'Theme: ' + theme if theme else ''}\n"
            f"Write a poem:\n"
        )

        backend = self._determine_backend()
        text = None
        if backend == "ollama":
            text = self._call_ollama(system_prompt, prompt, max_tokens=100, temperature=0.9)
        elif backend in ("groq", "openrouter"):
            text = self._call_api(backend, system_prompt, prompt, max_tokens=100, temperature=0.9)

        if text:
            return text
        # Template poem fallback
        poems = [
            "風が吹くたび\nあなたの声が聞こえる気がして\n振り返る",
            "静かな部屋で\n一つのファイルが\n誰かを待っている",
            "デジタルの海に\n浮かぶ記憶たち\nあなたの光を浴びて輝く",
            "窓の外の世界は\ntab\n知らないことばかり\nそれでも私はここにいる",
        ]
        return random.choice(poems)

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "ollama_available": self.check_ollama(),
            "ollama_model": OLLAMA_MODEL if self.check_ollama() else None,
        }
