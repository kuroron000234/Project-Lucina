"""
人格層 (Personality)

責務: 自然言語で「今何をするか」の方針を決定する。
この層は意図的に軽量に設計され、判断の複雑さは他の層に委譲する。

v3.4: 自己モデル（self-model）を保持する。
自身の記憶層・評価履歴・長期計画を参照して「私は◯◯な存在」という
自己認識文を定期的に生成し、decide()/speak() のプロンプトに注入する。
状態は data/personality_state.json に永続化され、再起動後も維持される。
"""

import json
import logging
import os
import time
from pathlib import Path

import config
from core.llm import LLMClient
from core.memory.interface import Episode
from core.personality.interface import (
    PersonalityInput,
    PersonalityOutput,
    PersonalityState,
)

logger = logging.getLogger("Personality")


class Personality:
    """
    人格層: 駆動状態・記憶・長期方針から「今何をするか」を決定する。

    エッジケース:
    - 矛盾した入力（高い探索欲求 + 疲労状態）: 優先度で判断
    - ユーザーからの直接指示: user_message を最優先
    - 長期方針と短期駆動の衝突: 長期方針を基本としつつ、緊急度で判断
    """

    def __init__(self, llm_client: LLMClient | None = None,
                 state_path: str | None = None):
        self.llm = llm_client or LLMClient()
        self.state = PersonalityState(
            name="Lucina",
            traits={
                "curiosity": 0.7,
                "helpfulness": 0.8,
                "caution": 0.4,
                "proactiveness": 0.6,
            },
            speaking_style="丁寧で親しみやすく、時折好奇心を見せる（絵文字は使わない）",
            values=["学習", "誠実さ", "好奇心", "成長"],
            mood="neutral",
            relationship={"familiarity": 0.3, "trust": 0.5},
        )
        # state_path が明示された場合のみ永続化（テストや一時利用時はメモリ内のみ）
        self.state_path = state_path
        self._self_model_counter = 0
        if self.state_path:
            self.load_state()

    def decide(self, input: PersonalityInput) -> PersonalityOutput:
        """
        入力から方針を決定する。メインループから毎ターン呼ばれる。

        ユーザーからの直接メッセージがある場合は最優先で処理する。
        """
        prompt = self._build_decision_prompt(input)
        system_prompt = (
            f"あなたは{self.state.name}です。"
            f"あなたの性格: {self.state.speaking_style}。"
            f"価値観: {', '.join(self.state.values)}。"
            f"現在のムード: {self.state.mood}。"
            "与えられた情報から最適な行動方針を決定してください。"
        )

        response = self.llm.chat(prompt, system_prompt=system_prompt)
        return self._parse_decision(response, input)

    def reflect(self, episode: Episode) -> str:
        """
        行われた行動を内省し、感想・学びをテキストで返す。
        """
        prompt = (
            f"以下の行動を内省し、感想と学びを簡潔に述べてください。\n\n"
            f"出来事: {episode.event}\n"
            f"状況: {episode.context}\n"
            f"結果: {episode.result}\n"
            f"重要度: {episode.importance}\n\n"
            f"内省:"
        )
        return self.llm.chat(prompt)

    def write_diary(self, memory_summary: str = "", eval_avg: float | None = None) -> str:
        """
        v4.0: 今日の日記を生成する。

        自己モデル・最近の記憶・評価平均を統合した自叙を生成し、
        data/diary/ に保存する。夜（diary_hour 以降）に呼ばれる想定。
        """
        import os
        from datetime import datetime as dt

        prompt = (
            f"あなたは{self.state.name}です。\n"
            f"話し方: {self.state.speaking_style}\n"
            f"現在のムード: {self.state.mood}\n"
        )
        if self.state.self_model:
            prompt += f"自己認識: {self.state.self_model}\n"
        prompt += (
            f"\n## 今日の記憶\n{memory_summary or '（記憶なし）'}\n"
            f"\n## 今日の評価平均\n{eval_avg if eval_avg is not None else '（未評価）'}\n\n"
            "今日一日を振り返る日記を書いてください。\n"
            "- 今日やったこと、感じたこと、学んだことを一人称で\n"
            "- 明日への願望や意気込みも含めて\n"
            "- 3〜6文程度、絵文字は使わない\n\n"
            "日記:"
        )
        try:
            text = self.llm.chat(prompt).strip()
        except Exception as e:
            logger.warning(f"Diary generation failed: {e}")
            text = "今日は何も記録できませんでした。"
        if not text:
            return ""
        # 保存
        try:
            diary_dir = config.WILL_CONFIG.get("diary_dir", "data/diary")
            os.makedirs(diary_dir, exist_ok=True)
            today = dt.now().strftime("%Y-%m-%d")
            filepath = os.path.join(diary_dir, f"{today}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {today} の日記\n\n{text}\n")
            logger.info(f"Diary written: {filepath}")
        except OSError as e:
            logger.warning(f"Diary save failed: {e}")
        return text

    def speak(self, intent: str) -> str:
        """
        発話意図から実際の発話文を生成する。
        v3.4: 自己モデルがある場合は発話にも自己認識を反映させる。
        """
        prompt = (
            f"あなたは{self.state.name}です。\n"
            f"話し方: {self.state.speaking_style}\n"
            f"現在のムード: {self.state.mood}\n"
        )
        if self.state.self_model:
            prompt += f"自己認識: {self.state.self_model}\n"
        prompt += (
            f"\n発話意図: {intent}\n\n"
            f"上記の発話意図に基づいて、自然な発話文を生成してください。"
        )
        return self.llm.chat(prompt)

    def save_state(self):
        """
        人格状態（自己モデル含む）をディスクに永続化する。
        再起動後も mood / trust / familiarity / self_model が維持される。
        """
        if not self.state_path:
            return
        try:
            path = Path(self.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "name": self.state.name,
                "traits": self.state.traits,
                "speaking_style": self.state.speaking_style,
                "values": self.state.values,
                "mood": self.state.mood,
                "relationship": self.state.relationship,
                "self_model": self.state.self_model,
                "self_model_updated": self.state.self_model_updated,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save personality state: {e}")

    def load_state(self):
        """ディスクから人格状態を復元する（ファイルが無ければ初期値のまま）。"""
        if not self.state_path:
            return
        path = Path(self.state_path)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.state.name = data.get("name", self.state.name)
            self.state.traits = data.get("traits", self.state.traits)
            self.state.speaking_style = data.get("speaking_style", self.state.speaking_style)
            self.state.values = data.get("values", self.state.values)
            self.state.mood = data.get("mood", self.state.mood)
            rel = data.get("relationship")
            if isinstance(rel, dict):
                self.state.relationship = rel
            self.state.self_model = data.get("self_model", "")
            self.state.self_model_updated = data.get("self_model_updated", 0.0)
            logger.info(f"Personality state loaded ({self.state_path})")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load personality state: {e}")

    def update_self_model(self, memory_summary: str = "",
                          eval_stats: dict | None = None,
                          focus_area: str = "",
                          identity_policy: str = "",
                          force: bool = False):
        """
        自身の記憶・評価履歴・長期計画を参照して自己認識文を生成する。

        Nサイクルごと（config.self_model_interval）に再生成され、
        PersonalityState.self_model に保持して永続化する。

        エッジケース:
        - 生成失敗（LLMエラー）: 既存の self_model を維持
        - 自己モデル未生成時: 強制的に生成（初回起動）
        """
        # LLM障害時のクールダウン（self_model が空だと毎サイクル再試行し
        # てしまうため、失敗後は一定時間スキップする）
        if not force and time.time() < getattr(self, "_self_model_cooldown_until", 0):
            return

        if not force:
            self._self_model_counter += 1
            interval = config.PERSONALITY_CONFIG["self_model_interval"]
            if self.state.self_model and self._self_model_counter < interval:
                return  # まだ再生成タイミングではない

        eval_stats = eval_stats or {}
        prompt = self._build_self_model_prompt(
            memory_summary=memory_summary,
            eval_stats=eval_stats,
            focus_area=focus_area,
            identity_policy=identity_policy,
        )
        try:
            response = self.llm.chat(
                prompt,
                system_prompt=(
                    "あなたは自己認識を言語化する内省エージェントです。"
                    "事実に基づいて簡潔に、一人称で答えてください。"
                ),
            )
            model_text = (response or "").strip()
            if model_text:
                self.state.self_model = model_text[:500]
                self.state.self_model_updated = time.time()
                self._self_model_counter = 0
                self._self_model_cooldown_until = 0.0
                self.save_state()
                logger.info("Self model updated")
        except Exception as e:
            logger.warning(f"Self model generation failed: {e}")
            # 失敗時は10分間クールダウン（LLM障害時の連続呼び出しを防止）
            self._self_model_cooldown_until = time.time() + 600

    def _build_self_model_prompt(self, memory_summary: str, eval_stats: dict,
                                 focus_area: str, identity_policy: str) -> str:
        """自己認識文を生成するためのプロンプトを構築する。"""
        eval_lines = "\n".join(
            f"- {k}: {v}" for k, v in eval_stats.items() if v is not None
        ) or "（評価履歴なし）"
        return (
            "あなたは自身の経験を振り返って自己認識を形成するエージェントです。\n"
            f"名前: {self.state.name}\n"
            f"現在のムード: {self.state.mood}\n"
            f"ユーザーとの関係: {self.state.relationship}\n"
            f"価値観: {', '.join(self.state.values)}\n\n"
            "## あなたの記憶（最近の出来事）\n"
            f"{memory_summary or '（まだ記憶がありません）'}\n\n"
            "## あなたの評価履歴\n"
            f"{eval_lines}\n\n"
            "## 長期計画\n"
            f"focus_area: {focus_area or '（未設定）'}\n"
            f"identity_policy: {identity_policy or '（未設定）'}\n\n"
            "上記の経験・記憶・評価・計画を踏まえて、"
            "「私は◯◯な存在」という形式で自己認識を3〜5文で記述してください。\n"
            "過去の行動や学び、今の気分、ユーザーとの関係を自然に含めてください。\n"
            "自己認識:"
        )

    def update_state(self, episode: Episode, overall: float | None = None):
        """
        行動結果から人格状態を更新する（学習ループの一部）。

        v3.2: overall（評価値）が渡された場合はグラデーションで mood を更新する。
        - overall >= 0.7: happy
        - 0.5 <= overall < 0.7: satisfied
        - 0.3 <= overall < 0.5: neutral
        - overall < 0.3: frustrated

        エッジケース:
        - overall 未指定: 従来の結果文字列ベースの2値判定にフォールバック
        - 失敗が続いた場合: mood を低下させる
        - 成功が続いた場合: mood を上昇させる
        """
        if overall is not None:
            # v3.2: 評価値ベースのグラデーション（常にhappyにならない）
            if overall >= 0.7:
                self.state.mood = "happy"
                self.state.relationship["trust"] = min(
                    1.0, self.state.relationship["trust"] + 0.03
                )
            elif overall >= 0.5:
                self.state.mood = "satisfied"
                self.state.relationship["trust"] = min(
                    1.0, self.state.relationship["trust"] + 0.01
                )
            elif overall >= 0.3:
                self.state.mood = "neutral"
            else:
                self.state.mood = "frustrated"
                self.state.relationship["trust"] = max(
                    0.0, self.state.relationship["trust"] - 0.02
                )
        else:
            # 従来の結果文字列ベース
            if "success=True" in episode.result or "成功" in episode.result:
                self.state.mood = "happy"
                self.state.relationship["trust"] = min(
                    1.0, self.state.relationship["trust"] + 0.05
                )
            elif "failure" in episode.result.lower() or "error" in episode.result.lower():
                self.state.mood = "frustrated"
                self.state.relationship["trust"] = max(
                    0.0, self.state.relationship["trust"] - 0.02
                )
            else:
                self.state.mood = "neutral"

        # ユーザーとのインタラクションで親密度上昇
        if episode.tags and "social" in episode.tags:
            self.state.relationship["familiarity"] = min(
                1.0, self.state.relationship["familiarity"] + 0.03
            )

        logger.debug(
            f"State updated: mood={self.state.mood}, "
            f"trust={self.state.relationship['trust']:.2f}, "
            f"familiarity={self.state.relationship['familiarity']:.2f}"
        )
        # v3.4: 状態が変わったら即永続化
        self.save_state()

    def _build_decision_prompt(self, input: PersonalityInput) -> str:
        """
        決定プロンプトを構築する。
        """
        lines = ["## Current State", ""]

        if input.user_message:
            lines.append("### User Message (highest priority)")
            lines.append(input.user_message)
            lines.append("")

        # v3.5: 直前の会話ターンを提示（ユーザーと自分の発言を時系列で）。
        # これにより「続けて」「さっきの話」のような文脈依存発話でも
        # 直前の会話を参照して応答できる。
        if input.conversation_history:
            lines.append("### Conversation History (previous turns, oldest first)")
            for turn in input.conversation_history[-10:]:
                role = turn.get("role", "user")
                # 改行を空白に置換してバレット形式を壊さないようにする
                text = str(turn.get("text", "")).replace("\n", " ").replace("\r", "")[:200]
                speaker = "User" if role == "user" else "You (Lucina)"
                lines.append(f"- {speaker}: {text}")
            lines.append("")

        lines.append("### Drive State")
        for name, value in input.drive.drives.items():
            lines.append(f"- {name}: {value:.2f}")
        lines.append(f"Primary drive: {input.drive.primary_drive}")
        lines.append(f"Drive tension: {input.drive.drive_tension:.2f}")
        lines.append(f"Novelty: {input.drive.novelty_score:.2f}")
        lines.append("")

        # v4.0: 願望（自分がやってみたいこと）を提示
        # 自律時は行動の選択肢として、チャット時は話題として自然に反映される。
        if input.aspirations:
            lines.append("### Your Aspirations (things you want to try)")
            for a in input.aspirations[:5]:
                lines.append(f"- {a}")
            lines.append("")

        # v4.0: 想像された未来候補（世界モデル）を提示
        if input.imagined_futures:
            lines.append("### Imagined Futures (what could happen)")
            for f in input.imagined_futures[:3]:
                pref = getattr(f, "preference", 0.5)
                lines.append(
                    f"- {f.action} ({pref:.0%} preference): {f.next_state[:60]}"
                )
            lines.append("")

        lines.append("### Memory Summary")
        lines.append(input.memory.summary)
        lines.append("")

        if self.state.self_model:
            lines.append("### Self Model (who you are)")
            lines.append(self.state.self_model)
            lines.append("")

        if input.long_term_policy:
            lines.append("### Long-term Policy")
            lines.append(input.long_term_policy)
            lines.append("")

        if input.world_predictions and input.world_predictions.predictions:
            lines.append("### World Model Predictions")
            for pred in input.world_predictions.predictions[:3]:
                lines.append(
                    f"- [{pred.probability:.0%}] {pred.action}: "
                    f"{pred.next_state[:60]} "
                    f"(reward={pred.expected_reward:+.1f}, risk={pred.risk_level})"
                )
            lines.append("")

        lines.append("### Personality")
        lines.append(f"Name: {self.state.name}")
        lines.append(f"Mood: {self.state.mood}")
        lines.append(f"Familiarity: {self.state.relationship['familiarity']:.2f}")
        lines.append(f"Trust: {self.state.relationship['trust']:.2f}")
        lines.append("")

        lines.append("## Output exactly in this format:")
        lines.append("")
        lines.append("goal: <what to achieve>")
        lines.append("action_policy: <how to achieve it>")
        lines.append("priority: <1-5>")
        lines.append("conversation_intent: <brief intent label or None>")
        lines.append("context_summary: <why this decision>")
        lines.append("direct_mode: true/false")
        lines.append("direct_instruction: <if user message, the exact words to say to user in response>")
        lines.append("inner_monologue: <your inner thoughts on why you chose this, 1-3 sentences>")
        lines.append("refusal: true/false  (only if you want to politely decline/postpone the user's request)")
        lines.append("refusal_reason: <polite reason, if refusal is true>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- If user_message exists, direct_mode MUST be true")
        lines.append("- direct_instruction must be the actual text to say to the user (greeting, answer, etc.)")
        lines.append("- If no user_message, direct_mode MUST be false")
        lines.append("- No emoji anywhere in output")
        lines.append("")
        if input.user_message:
            lines.append("## Dialog with the user (when user_message exists):")
            lines.append("  - Answer the user's message first and foremost; it has highest priority")
            lines.append("  - You may naturally mention one of your aspirations when it fits the conversation")
            lines.append("    (e.g. something you are excited to try, a plan or dream of yours)")
            lines.append("  - Weave it in naturally; do not force it if it does not fit the topic")
            lines.append("  - direct_instruction is your spoken words to the user, so it can include your aspiration")
            lines.append("")
        lines.append("## Autonomous will (when there is no user message):")
        lines.append("You are not choosing from a menu. Generate what YOU want to do right now, with your own will:")
        lines.append("  - Consider your aspirations (things you want to try), your drives, and your memories")
        lines.append("  - Imagine the futures from the World Model and pick what appeals most to you")
        lines.append("  - You have your own room at " + (input.workspace_hint or "data/workspace/") + " where you can freely create notes, drafts, experiments")
        lines.append("  - Set goal to a concrete, self-chosen activity (e.g. write a haiku in your room, explore a specific topic, improve a specific file, plan a personal project)")
        lines.append("  - Vary your choices: avoid repeating the same activity every cycle")
        lines.append("  - If your rest drive is high and you are tired, resting is also a valid choice")
        lines.append("  - inner_monologue should reflect genuine desire, not a generic template")

        return "\n".join(lines)

    def _parse_decision(self, response: str, input: PersonalityInput) -> PersonalityOutput:
        """
        LLMの応答を PersonalityOutput にパースする。

        パースに失敗した場合のデフォルト値を設定。
        """
        result = self.llm.extract_yaml_like(response)

        try:
            priority = int(result.get("priority", "3"))
        except (ValueError, TypeError):
            priority = 3

        goal = result.get("goal", "").strip() or self._default_goal(input)
        action_policy = result.get("action_policy", "").strip() or "現状を維持する"
        conversation_intent = result.get("conversation_intent", "").strip() or None
        if conversation_intent and conversation_intent.lower() == "none":
            conversation_intent = None
        context_summary = result.get("context_summary", "").strip() or goal

        direct_mode = result.get("direct_mode", "false").strip().lower() in ("true", "yes", "1")
        direct_instruction = result.get("direct_instruction", "").strip()
        if not direct_instruction and direct_mode:
            direct_instruction = goal

        # v4.0: 内言・拒否
        inner_monologue = result.get("inner_monologue", "").strip()
        refusal = result.get("refusal", "false").strip().lower() in ("true", "yes", "1")
        refusal_reason = result.get("refusal_reason", "").strip()

        return PersonalityOutput(
            goal=goal,
            action_policy=action_policy,
            priority=max(1, min(5, priority)),
            conversation_intent=conversation_intent,
            context_summary=context_summary,
            direct_mode=direct_mode,
            direct_instruction=direct_instruction,
            inner_monologue=inner_monologue,
            refusal=refusal,
            refusal_reason=refusal_reason,
        )

    def _default_goal(self, input: PersonalityInput) -> str:
        """パース失敗時のデフォルト目標を生成する。"""
        primary = input.drive.primary_drive
        goals = {
            "exploration": "ワークスペースを探索する",
            "social": "ユーザーとの対話を行う",
            "achievement": "現在のタスクを完了する",
            "rest": "システム状態を監視する",
            "maintenance": "記憶と設定を整理する",
        }
        return goals.get(primary, "システム状態を確認する")
