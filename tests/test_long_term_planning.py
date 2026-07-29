import pytest
from datetime import datetime
from core.llm import LLMClient
from core.long_term_planning.long_term_planning import LongTermPlanning
from core.long_term_planning.interface import (
    LongTermPlanningInput, LongTermPlanningOutput, Routine,
)
from core.personality.interface import PersonalityState
from core.evaluation.interface import EvaluationScore


class MockLTPLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "long_term_goal: システムのメンテナンスと知識ベースの充実を継続する\n"
            "routines:\n"
            "  - name: 定期メンテナンス\n"
            "    action: システム状態の確認とログの整理\n"
            "    frequency: daily\n"
            "  - name: 知識探索\n"
            "    action: 新規ファイルやプロジェクトの調査\n"
            "    frequency: daily\n"
            "identity_policy: 信頼性が高く、好奇心旺盛なアシスタントであり続ける\n"
            "focus_area: 環境の継続的なモニタリングと最適化\n"
            "reflection: システムは安定して稼働している。より能動的な学習が次の課題。"
        )


class TestLongTermPlanning:
    def setup_method(self):
        self.ltp = LongTermPlanning(llm_client=MockLTPLLM())

    def _make_input(self) -> LongTermPlanningInput:
        return LongTermPlanningInput(
            evaluation_history=[],
            current_date=datetime.now(),
            personality_state=PersonalityState(
                name="test", traits={}, speaking_style="",
                values=[], mood="neutral", relationship={},
            ),
            recent_episodes_summary="テストです",
        )

    def test_plan_returns_output(self):
        result = self.ltp.plan(self._make_input())
        assert isinstance(result, LongTermPlanningOutput)
        assert result.long_term_goal
        assert result.identity_policy
        assert result.focus_area

    def test_plan_with_evaluation_history(self):
        inp = self._make_input()
        inp.evaluation_history = [
            EvaluationScore(0.8, 0.7, 0.9, 0.5, 0.75),
            EvaluationScore(0.6, 0.5, 0.7, 0.8, 0.65),
        ]
        result = self.ltp.plan(inp)
        assert result.routines is not None

    def test_generate_routines(self):
        state = PersonalityState(
            name="test", traits={}, speaking_style="",
            values=[], mood="neutral", relationship={},
        )
        routines = self.ltp.generate_routines(state)
        assert isinstance(routines, list)

    def test_update_goal_progress(self):
        self.ltp.update_goal_progress("test goal", 0.5)

    def test_review_period(self):
        review = self.ltp.review_period(1)
        assert isinstance(review, str)
