"""
人格層 (Personality)

責務: 自然言語で「今何をするか」の方針を決定する。
この層は意図的に軽量に設計され、判断の複雑さは他の層に委譲する。

Phase 1: LLM呼び出しによる方針決定
"""

import logging

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

    def __init__(self, llm_client: LLMClient | None = None):
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

    def speak(self, intent: str) -> str:
        """
        発話意図から実際の発話文を生成する。
        """
        prompt = (
            f"あなたは{self.state.name}です。\n"
            f"話し方: {self.state.speaking_style}\n"
            f"現在のムード: {self.state.mood}\n\n"
            f"発話意図: {intent}\n\n"
            f"上記の発話意図に基づいて、自然な発話文を生成してください。"
        )
        return self.llm.chat(prompt)

    def update_state(self, episode: Episode):
        """
        行動結果から人格状態を更新する（学習ループの一部）。

        エッジケース:
        - 失敗が続いた場合: mood を低下させる
        - 成功が続いた場合: mood を上昇させる
        """
        # 結果に基づいてムードを更新
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

    def _build_decision_prompt(self, input: PersonalityInput) -> str:
        """
        決定プロンプトを構築する。
        """
        lines = ["## Current State", ""]

        if input.user_message:
            lines.append("### User Message (highest priority)")
            lines.append(input.user_message)
            lines.append("")

        lines.append("### Drive State")
        for name, value in input.drive.drives.items():
            lines.append(f"- {name}: {value:.2f}")
        lines.append(f"Primary drive: {input.drive.primary_drive}")
        lines.append(f"Drive tension: {input.drive.drive_tension:.2f}")
        lines.append(f"Novelty: {input.drive.novelty_score:.2f}")
        lines.append("")

        lines.append("### Memory Summary")
        lines.append(input.memory.summary)
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
        lines.append("")
        lines.append("Rules:")
        lines.append("- If user_message exists, direct_mode MUST be true")
        lines.append("- direct_instruction must be the actual text to say to the user (greeting, answer, etc.)")
        lines.append("- If no user_message, direct_mode MUST be false")
        lines.append("- No emoji anywhere in output")
        lines.append("")
        lines.append("When no user_message, choose a meaningful autonomous activity:")
        lines.append("  - If exploration drive is high: explore the codebase, check git status, read interesting files")
        lines.append("  - If achievement drive is high: improve the codebase (refactor, fix TODOs, add tests)")
        lines.append("  - If maintenance drive is high: backup project, clean memory, check logs for errors")
        lines.append("  - If social drive is high and familiarity is low: optionally say hello to user")
        lines.append("  - If all drives are low: rest (monitor system, light maintenance only)")
        lines.append("  - Avoid repeating the same activity every cycle -- vary your actions")

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

        return PersonalityOutput(
            goal=goal,
            action_policy=action_policy,
            priority=max(1, min(5, priority)),
            conversation_intent=conversation_intent,
            context_summary=context_summary,
            direct_mode=direct_mode,
            direct_instruction=direct_instruction,
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
