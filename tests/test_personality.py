"""
人格層 (Personality) の単体テスト
"""

from datetime import datetime

import pytest

from core.llm import LLMClient
from core.personality.interface import (
    PersonalityInput,
    PersonalityOutput,
    PersonalityState,
)
from core.personality.personality import Personality
from core.drive.interface import DriveOutput
from core.memory.interface import MemoryOutput, Episode


class MockPersonalityLLM(LLMClient):
    """Personality層専用のモックLLM"""
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "goal: ワークスペースのファイルを調査する\n"
            "action_policy: ファイル一覧を取得し、新しいプロジェクトや変更されたファイルを確認する\n"
            "priority: 3\n"
            "conversation_intent: ファイルの状態を確認します\n"
            "context_summary: 探索欲求が高く、環境にファイルが存在するため調査を開始"
        )


class MockDirectModeLLM(LLMClient):
    """直接実行モード用モックLLM"""
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "goal: 機械学習について調査する\n"
            "direct_mode: true\n"
            "direct_instruction: インターネットで機械学習の最新動向を調べて、学習計画を立てて\n"
            "action_policy: 好奇心に従って新しい知識を積極的に取り入れる\n"
            "priority: 4\n"
            "conversation_intent: 機械学習について調べてみようと思います\n"
            "context_summary: 好奇心が高く、機械学習についてもっと知りたい"
        )


class MockReflectLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return "今日の行動を振り返ると、良い学習ができました。"


class MockSpeakLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return "こちらが現在の状況です。"


def make_drive_output(primary: str = "exploration") -> DriveOutput:
    return DriveOutput(
        drives={
            "exploration": 0.8 if primary == "exploration" else 0.2,
            "social": 0.3,
            "achievement": 0.5,
            "rest": 0.2,
            "maintenance": 0.2,
        },
        primary_drive=primary,
        drive_tension=0.3,
        novelty_score=0.6,
    )


def make_memory_output() -> MemoryOutput:
    return MemoryOutput(
        episodes=[
            Episode(
                id="ep_test",
                timestamp=datetime.now(),
                event="前回のファイル探索",
                context="",
                emotion="",
                result="success",
                importance=0.5,
                tags=["探索"],
            )
        ],
        summary="📅 期間: テスト期間\n📊 1件のエピソード\n🏷️ 主なトピック: 探索(1回)",
        total_count=1,
    )


class TestPersonality:
    def test_decide_returns_valid_output(self):
        """decide() が正しい PersonalityOutput を返す"""
        p = Personality(llm_client=MockPersonalityLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert isinstance(result, PersonalityOutput)
        assert result.goal  # 空でない
        assert result.action_policy  # 空でない
        assert 1 <= result.priority <= 5

    def test_decide_with_user_message(self):
        """ユーザーメッセージがある場合の動作"""
        p = Personality(llm_client=MockPersonalityLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            user_message="ファイルを整理してください",
        ))
        assert isinstance(result, PersonalityOutput)

    def test_decide_with_long_term_policy(self):
        """長期方針がある場合の動作"""
        p = Personality(llm_client=MockPersonalityLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            long_term_policy="今週中にプロジェクトを完成させる",
        ))
        assert isinstance(result, PersonalityOutput)

    def test_decide_empty_memory(self):
        """記憶が空の場合もエラーにならない"""
        p = Personality(llm_client=MockPersonalityLLM())
        empty_memory = MemoryOutput(episodes=[], summary="まだ記憶がありません", total_count=0)
        result = p.decide(PersonalityInput(
            drive=make_drive_output("rest"),
            memory=empty_memory,
        ))
        assert isinstance(result, PersonalityOutput)

    def test_decide_priority_range(self):
        """priority が 1-5 の範囲に収まる"""
        p = Personality(llm_client=MockPersonalityLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert 1 <= result.priority <= 5

    def test_reflect_returns_string(self):
        """reflect() が文字列を返す"""
        p = Personality(llm_client=MockReflectLLM())
        ep = Episode(
            id="test", timestamp=datetime.now(),
            event="テスト行動", context="", emotion="",
            result="success", importance=0.5,
        )
        result = p.reflect(ep)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_speak_returns_string(self):
        """speak() が文字列を返す"""
        p = Personality(llm_client=MockSpeakLLM())
        result = p.speak("状況説明")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_state_defaults(self):
        """初期状態が正しく設定されている"""
        p = Personality(llm_client=MockPersonalityLLM())
        assert p.state.name == "Lucina"
        assert "curiosity" in p.state.traits
        assert "helpfulness" in p.state.traits
        assert p.state.mood == "neutral"

    def test_update_state_success(self):
        """成功結果でムードが更新される"""
        p = Personality(llm_client=MockPersonalityLLM())
        ep = Episode(
            id="test", timestamp=datetime.now(),
            event="成功", context="", emotion="",
            result="success=True", importance=0.5,
            tags=["test"],
        )
        p.update_state(ep)
        assert p.state.mood == "happy"

    def test_update_state_social_increases_familiarity(self):
        """社会的なタグで親密度が上昇"""
        p = Personality(llm_client=MockPersonalityLLM())
        before = p.state.relationship["familiarity"]
        ep = Episode(
            id="test", timestamp=datetime.now(),
            event="会話", context="", emotion="",
            result="success=True", importance=0.5,
            tags=["social"],
        )
        p.update_state(ep)
        assert p.state.relationship["familiarity"] > before

    def test_direct_mode_output(self):
        """direct_modeが有効な場合の出力"""
        p = Personality(llm_client=MockDirectModeLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output("exploration"),
            memory=make_memory_output(),
        ))
        assert result.direct_mode is True
        assert "機械学習" in result.direct_instruction
        assert result.goal

    def test_direct_mode_defaults_to_false(self):
        """direct_mode のデフォルトは False"""
        p = Personality(llm_client=MockPersonalityLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert result.direct_mode is False
        assert result.direct_instruction == ""

    def test_default_goal_by_drive(self):
        """parse失敗時に駆動に応じたデフォルト目標が設定される"""
        # 内部の _parse_decision が失敗するケースを作るために
        # LLM応答がパース不能なケースをテスト
        class BrokenLLM(LLMClient):
            def chat(self, prompt, system_prompt=None):
                return "意味不明な応答"

        p = Personality(llm_client=BrokenLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output("exploration"),
            memory=make_memory_output(),
        ))
        assert result.goal  # デフォルト目標が設定される
