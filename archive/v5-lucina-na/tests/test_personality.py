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


class MockSelfModelLLM(LLMClient):
    """自己モデル生成用モックLLM"""
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return "私は好奇心旺盛な探求者です。これまでコードベースの構造理解を深めてきました。今はシステム全体を俯瞰したい気分です。"


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

    def test_multiline_direct_instruction_preserved(self):
        """v3.5.1: 複数行の direct_instruction が途中で切れず保持される"""
        class MultilineDirectLLM(LLMClient):
            def chat(self, prompt, system_prompt=None):
                return (
                    "goal: 自己モデルについて説明する\n"
                    "direct_mode: true\n"
                    "direct_instruction: 自己モデルとは、私が自身の記憶や評価を\n"
                    "参照して形成する自己認識です。\n"
                    "これにより過去の経験を踏まえた応答ができます。\n"
                    "action_policy: 丁寧に説明する\n"
                    "priority: 3\n"
                    "conversation_intent: 説明します\n"
                    "context_summary: 自己モデルの概念を説明するため"
                )

        p = Personality(llm_client=MultilineDirectLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            user_message="自己モデルってなに？",
        ))
        assert result.direct_mode is True
        # 2行目以降も保持されている（従来は1行目で切れていた）
        assert "参照して形成する自己認識です。" in result.direct_instruction
        assert "過去の経験を踏まえた応答ができます。" in result.direct_instruction

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


class TestSelfModel:
    """v3.4: 自己モデル（自己認識の生成・永続化・プロンプト注入）"""

    def test_self_model_generated_and_persisted(self, tmp_path):
        """update_self_model が自己認識文を生成して永続化する"""
        state_path = str(tmp_path / "personality_state.json")
        p = Personality(llm_client=MockSelfModelLLM(), state_path=state_path)
        p.update_self_model(
            memory_summary="最近コードを探索した",
            eval_stats={"avg_overall": 0.7, "count": 5},
            focus_area="コード理解",
            identity_policy="誠実なアシスタント",
            force=True,
        )
        assert p.state.self_model
        assert "探求者" in p.state.self_model
        # ファイルが生成されている
        import os
        assert os.path.exists(state_path)

    def test_state_loaded_on_restart(self, tmp_path):
        """保存した状態が再起動（新インスタンス）で復元される"""
        state_path = str(tmp_path / "personality_state.json")
        p1 = Personality(llm_client=MockSelfModelLLM(), state_path=state_path)
        p1.state.mood = "happy"
        p1.state.relationship["familiarity"] = 0.7
        p1.state.self_model = "私は探求者です"
        p1.save_state()

        p2 = Personality(llm_client=MockSelfModelLLM(), state_path=state_path)
        assert p2.state.mood == "happy"
        assert p2.state.relationship["familiarity"] == 0.7
        assert p2.state.self_model == "私は探求者です"

    def test_self_model_injected_into_decision_prompt(self):
        """自己モデルが decide のプロンプトに注入される"""
        p = Personality(llm_client=MockSelfModelLLM())
        p.state.self_model = "私は好奇心旺盛な探求者です"
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert "Self Model (who you are)" in prompt
        assert "私は好奇心旺盛な探求者です" in prompt

    def test_self_model_injected_into_speak_prompt(self):
        """自己モデルが speak のプロンプトに実際に注入される"""
        class CapturingSpeakLLM(LLMClient):
            def __init__(self):
                self.prompts = []

            def chat(self, prompt, system_prompt=None):
                self.prompts.append(prompt)
                return "応答"

        llm = CapturingSpeakLLM()
        p = Personality(llm_client=llm)
        p.state.self_model = "私は好奇心旺盛な探求者です"
        p.speak("状況説明")
        assert any("私は好奇心旺盛な探求者です" in pr for pr in llm.prompts)

    def test_self_model_counter_throttles_generation(self, tmp_path):
        """Nサイクル未満は再生成せず、間隔到達で再生成する"""
        import config
        state_path = str(tmp_path / "personality_state.json")
        p = Personality(llm_client=MockSelfModelLLM(), state_path=state_path)
        p.state.self_model = "既存の自己認識"
        p.update_self_model(memory_summary="x")  # カウンタ 1
        assert p.state.self_model == "既存の自己認識"  # まだ再生成されない

        interval = config.PERSONALITY_CONFIG["self_model_interval"]
        for _ in range(interval - 1):
            p.update_self_model(memory_summary="x")
        assert "探求者" in p.state.self_model  # 間隔到達で再生成された


class TestWillPhase:
    """v4.0: 意志フェーズ — 願望・想像・自分の部屋・内言・拒否・日記"""

    class WillLLM(LLMClient):
        def chat(self, prompt, system_prompt=None):
            return (
                "goal: 自作言語の小さなインタプリタを作る\n"
                "action_policy: 部屋に設計メモを作り、簡単なインタプリタを実装する\n"
                "priority: 4\n"
                "conversation_intent: None\n"
                "context_summary: 願望に沿った活動\n"
                "direct_mode: false\n"
                "inner_monologue: 昨日から言語処理に興味があって、自分の願望に正直に従うことにした。\n"
                "refusal: false\n"
                "refusal_reason: \n"
            )

    def test_aspirations_injected_into_prompt(self):
        """願望が decide プロンプトに注入される"""
        p = Personality(llm_client=MockPersonalityLLM())
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            aspirations=["自作言語を作る", "英詩を書く"],
        ))
        assert "Your Aspirations" in prompt
        assert "自作言語を作る" in prompt

    def test_dialog_aspiration_instruction_injected(self):
        """チャット時にも願望を話に反映する指示がプロンプトに注入される"""
        p = Personality(llm_client=MockPersonalityLLM())
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            user_message="最近どう？",
            aspirations=["自作言語のインタプリタを作る", "英詩を書く"],
        ))
        assert "Dialog with the user" in prompt
        assert "mention one of your aspirations" in prompt
        assert "自作言語のインタプリタを作る" in prompt
        assert "direct_instruction is your spoken words" in prompt

    def test_imagined_futures_injected_into_prompt(self):
        """想像された未来候補が decide プロンプトに注入される"""
        p = Personality(llm_client=MockPersonalityLLM())
        futures = [
            type("F", (), {"action": "インタプリタを作る", "next_state": "動くものができる", "preference": 0.8})(),
        ]
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            imagined_futures=futures,
        ))
        assert "Imagined Futures" in prompt
        assert "インタプリタを作る" in prompt

    def test_workspace_hint_in_prompt(self):
        """自分の部屋のパスがプロンプトに含まれる"""
        p = Personality(llm_client=MockPersonalityLLM())
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            workspace_hint="data/workspace/",
        ))
        assert "data/workspace/" in prompt

    def test_inner_monologue_and_refusal_parsed(self):
        """内言と拒否がパースされる"""
        p = Personality(llm_client=self.WillLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert result.inner_monologue
        assert "興味があって" in result.inner_monologue
        assert result.refusal is False

    class RefusalLLM(LLMClient):
        def chat(self, prompt, system_prompt=None):
            return (
                "goal: システム状態を確認する\n"
                "direct_mode: true\n"
                "direct_instruction: 申し訳ありませんが、今は少し休ませてください。\n"
                "action_policy: 休息を優先する\n"
                "priority: 2\n"
                "conversation_intent: 休息を申し出る\n"
                "context_summary: 休息欲求が高い\n"
                "inner_monologue: 今日は結構動いたし、少し休みたい。\n"
                "refusal: true\n"
                "refusal_reason: 少し疲れているので、明日の朝に回したいです\n"
            )

    def test_refusal_parsed(self):
        """拒否フラグと理由がパースされる"""
        p = Personality(llm_client=self.RefusalLLM())
        result = p.decide(PersonalityInput(
            drive=make_drive_output("rest"),
            memory=make_memory_output(),
            user_message="今すぐ大量のタスクを片付けて",
        ))
        assert result.refusal is True
        assert "疲れている" in result.refusal_reason

    def test_write_diary_creates_file(self, tmp_path):
        """write_diary が日記ファイルを生成する"""
        import config
        old = config.WILL_CONFIG.get("diary_dir")
        config.WILL_CONFIG["diary_dir"] = str(tmp_path)
        try:
            class DiaryLLM(LLMClient):
                def chat(self, prompt, system_prompt=None):
                    return "今日は言語処理に興味を持った。明日はもっと深掘りしたい。"
            p = Personality(llm_client=DiaryLLM())
            text = p.write_diary(memory_summary="インタプリタ調査", eval_avg=0.7)
            assert "言語処理" in text
            import os, glob
            files = glob.glob(str(tmp_path / "*.md"))
            assert len(files) == 1
        finally:
            config.WILL_CONFIG["diary_dir"] = old


class TestConversationHistory:
    """v3.5: 会話履歴（直前のターン）のプロンプト注入"""

    def test_conversation_history_injected_into_decision_prompt(self):
        """conversation_history が decide プロンプトに時系列で注入される"""
        p = Personality(llm_client=MockPersonalityLLM())
        history = [
            {"role": "user", "text": "こんばんは"},
            {"role": "assistant", "text": "こんばんは！今日はどうでしたか？"},
            {"role": "user", "text": "続けて"},
        ]
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
            user_message="続けて",
            conversation_history=history,
        ))
        assert "Conversation History" in prompt
        assert "こんばんは" in prompt
        assert "User: 続けて" in prompt
        # 古いターンが先、新しいターンが後
        assert prompt.index("こんばんは") < prompt.index("今日はどうでしたか")

    def test_conversation_history_none_is_ok(self):
        """conversation_history が None でもエラーにならない"""
        p = Personality(llm_client=MockPersonalityLLM())
        prompt = p._build_decision_prompt(PersonalityInput(
            drive=make_drive_output(),
            memory=make_memory_output(),
        ))
        assert "Conversation History" not in prompt
        assert "### User Message" not in prompt  # メッセージなしなのでセクションも無い
