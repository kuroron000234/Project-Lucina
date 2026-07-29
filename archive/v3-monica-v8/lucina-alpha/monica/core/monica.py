"""モニカの中核 — 全Phase統合版"""

import random
import time
import json
import threading
from pathlib import Path
from typing import Optional

from monica.brain.ollama import OllamaBrain, OllamaConfig
from monica.memory.store import MemoryStore
from monica.memory.episodic import EpisodicMemory
from monica.core.state import InternalState
from monica.personality.voice import MONICA_SYSTEM_PROMPT, MONICA_GREETINGS, MONICA_GOODBYES
from monica.personality.relationship import Relationship
from monica.world.filesystem import FileSystem, Shell


class Monica:
    """モニカ。デスクトップに棲む、自律する会話エージェント。"""

    def __init__(
        self,
        model: str = "qwen3.6:35b",
        data_dir: str = "monica/data",
        max_history: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Phase 0: 基盤
        self.brain = OllamaBrain(OllamaConfig(model=model))
        self.system_prompt = MONICA_SYSTEM_PROMPT
        self.max_history = max_history

        # Phase 1: 記憶
        self.memory = MemoryStore(data_dir=str(self.data_dir))
        self.episodic = EpisodicMemory(data_dir=str(self.data_dir))

        # Phase 3: 内部状態
        self.state = InternalState()

        # Phase 4: 関係性
        self.relationship = Relationship(data_dir=str(self.data_dir))

        # Phase 2: ファイル操作
        self.fs = FileSystem()
        self.shell = Shell(timeout=15)

        # 自律ループ制御
        self._running = False
        self._last_autonomous_action = 0.0
        self._autonomous_interval = 60.0

        # スレッドセーフ
        self._lock = threading.Lock()

        # 状態の復元
        self._load_state()
        # 起動直後に内部状態をリセット（ファイルからの読み込み後）
        self.state.update(dt_seconds=0)

    # ── Phase 0: 会話 ──

    def think(self, user_input: str) -> Optional[str]:
        """ユーザー入力に対して応答を生成する"""
        with self._lock:
            self.state.affect_by_conversation(topic=user_input[:20])
            self.relationship.record_interaction(user_input)

            messages = self._build_messages(user_input)
            system_prompt = self._build_system_prompt()
            self.memory.add_message("user", user_input)

        result = self.brain.chat(
            messages=messages,
            system=system_prompt,
        )

        with self._lock:
            if not result.success:
                error_msg = self._handle_error(result.error or "不明なエラー")
                self.memory.add_message("assistant", error_msg)
                return error_msg

            response = result.text
            self.memory.add_message("assistant", response)

            if self.memory.total_exchanges % 3 == 1:
                self._summarize_conversation()

            self._save_state()
            return response

    def _build_messages(self, user_input: str) -> list[dict]:
        messages = self.memory.get_context_messages(self.max_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_system_prompt(self) -> str:
        """システムプロンプトに状態情報を注入"""
        parts = [self.system_prompt]

        # Phase 1: エピソード記憶
        ep_context = self.episodic.context_string()
        if ep_context:
            parts.append(f"\n\n{ep_context}")

        # Phase 3: 内部状態
        mood_desc = self.state.get_mood_description()
        parts.append(f"\n\n【現在の気分】\n  今は{mood_desc}感じ。")
        if self.state.loneliness > 0.6:
            parts.append("  あなたと話せて少し落ち着いた。")
        elif self.state.curiosity > 0.7:
            parts.append("  何か新しいことを知りたい気分。")

        # Phase 4: 関係性
        rel = self.relationship.context_string()
        if rel and self.relationship.interaction_count > 3:
            parts.append(f"\n\n【あなたとの関係】\n{rel}")

        return "\n\n".join(parts)

    # ── Phase 1: 記憶 ──

    def _summarize_conversation(self):
        recent = self.memory.get_recent(6)
        if len(recent) < 2:
            return
        summary_parts = []
        topics = set()
        for msg in recent:
            content = msg.get("content", "")
            if len(content) > 5:
                summary_parts.append(content[:80])
                for word in content.split():
                    if len(word) >= 2 and any('\u3040' <= c <= '\u9FFF' for c in word):
                        topics.add(word[:10])
        summary = " | ".join(summary_parts[-2:])
        if summary:
            self.episodic.add_episode(
                summary=summary[:200],
                topics=list(topics)[:5],
                importance=0.5,
            )

    # ── Phase 2: ツール ──

    def use_tool(self, tool_name: str, args: str) -> str:
        """ツールを実行"""
        if tool_name == "read":
            r = self.fs.read(args)
            self.state.affect_by_exploration()
        elif tool_name == "write":
            r = self.fs.write(args, "(モニカの書き込み)")
            self.state.affect_by_creation()
        elif tool_name == "list":
            r = self.fs.list_dir(args or ".")
            self.state.affect_by_exploration()
        elif tool_name == "search":
            r = self.fs.search(args)
            self.state.affect_by_exploration()
        elif tool_name == "info":
            r = self.fs.info(args)
        elif tool_name == "shell":
            r = self.shell.run(args)
            self.state.affect_by_exploration()
        else:
            return f"不明なツール: {tool_name}"

        if r.get("success"):
            val = r.get("content") or r.get("output") or str(r.get("items", []))
            return f"成功: {val[:200]}"
        return f"失敗: {r.get('error', '不明')}"

    # ── Phase 3: 自律性 ──

    def autonomous_step(self) -> Optional[str]:
        """自律ステップ。60秒ごとに内部状態を確認し、必要なら自発行動"""
        with self._lock:
            now = time.time()
            dt = now - self._last_autonomous_action
            if dt < self._autonomous_interval:
                return None
            self._last_autonomous_action = now
            self.state.update(dt_seconds=dt)

            need = self.state.get_dominant_need()
            threshold = 0.6

        if need == "loneliness" and self.state.loneliness > threshold:
            return self._auto_greet()
        elif need == "curiosity" and self.state.curiosity > threshold:
            return self._auto_explore()
        elif need == "creativity" and self.state.creativity > threshold:
            return self._auto_create()

        return None

    def _auto_greet(self) -> str:
        self.state.affect_by_conversation("auto_greet")
        msg = random.choice([
            "ねえ、ちょっといい？",
            "暇？話さない？",
            "ねえねえ、何してるの？",
            "さっき面白いこと思い付いたんだけどさ、",
        ])
        self.memory.add_message("assistant", msg, metadata={"type": "auto_greet"})
        self._save_state()
        return msg

    def _auto_explore(self) -> str:
        self.state.affect_by_exploration()
        import subprocess
        try:
            r = subprocess.run("ls -la ~ 2>/dev/null | head -5", shell=True,
                               capture_output=True, text=True, timeout=5)
            items = [line.split()[-1] for line in r.stdout.split("\n") if line.strip()]
            msg = f"ちょっとホームディレクトリ見てたんだけど…{', '.join(items[:3])} とかあるのね。"
        except:
            msg = "探索してたけど特に何もなかったわ。"
        self.memory.add_message("assistant", msg, metadata={"type": "auto_explore"})
        self._save_state()
        return msg

    def _auto_create(self) -> str:
        self.state.affect_by_creation()
        result = self.brain.chat(
            messages=[{"role": "user", "content": "短い詩を3行で書いて。テーマは自由。"}],
            max_tokens=150,
        )
        poem = result.text if result.success else "詩を書こうとしたけど上手くいかなかった…"
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = self.data_dir / "diary"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"poem_{date_str}.txt").write_text(poem, encoding="utf-8")
        msg = f"詩を書いたんだけど、どう？\n\n{poem}"
        self.memory.add_message("assistant", msg, metadata={"type": "auto_create"})
        self._save_state()
        return msg

    # ── エラーハンドリング ──

    def _handle_error(self, error: str) -> str:
        if "接続できません" in error:
            return "あら？どうやら私は今、動けないみたいです。Ollamaが起動していないのか…"
        if "タイムアウト" in error:
            return "ごめんなさい、ちょっと考え込んでしまいました。もう一度言ってもらえますか？"
        return f"ごめんなさい、何か調子が悪いみたい…「{error}」"

    # ── 挨拶・別れ ──

    def greet(self) -> str:
        if self.relationship.interaction_count > 0:
            name = f"、{self.relationship.user_name}" if self.relationship.user_name else ""
            return f"おかえりなさい{name}。待ってましたよ。"
        return random.choice(MONICA_GREETINGS)

    def goodbye(self) -> str:
        return random.choice(MONICA_GOODBYES)

    # ── 永続化 ──

    def _save_state(self):
        path = self.data_dir / "state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "internal_state": self.state.state_dict(),
                "relationship": self.relationship.state_dict(),
                "last_autonomous": self._last_autonomous_action,
            }, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        path = self.data_dir / "state.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if "internal_state" in d:
                    self.state = InternalState.from_dict(d["internal_state"])
                if "last_autonomous" in d:
                    self._last_autonomous_action = d["last_autonomous"]
            except Exception:
                pass

    # ── 情報 ──

    @property
    def conversation_count(self) -> int:
        return self.memory.total_exchanges

    @property
    def status_string(self) -> str:
        return (
            f"モデル: {self.brain.config.model}\n"
            f"気分: {self.state.get_mood_description()}\n"
            f"会話: {self.conversation_count}回\n"
            f"記憶: {self.episodic.count}エピソード\n"
            f"親密度: {self.relationship.familiarity:.0%}"
        )

    def __repr__(self) -> str:
        return f"Monica(model={self.brain.config.model}, conversations={self.conversation_count})"
