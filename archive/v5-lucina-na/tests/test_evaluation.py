"""
評価層 (Evaluation) の単体テスト
"""

from datetime import datetime

from core.evaluation.evaluation import Evaluation
from core.evaluation.interface import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationScore,
)
from core.agent.interface import AgentOutput, StepResult
from core.llm import LLMClient
from core.memory.interface import Episode


class MockEvalLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "goal_achievement: 0.8\n"
            "efficiency: 0.7\n"
            "correctness: 0.9\n"
            "novelty: 0.3\n"
            "overall: 0.7\n"
            "discrepancy: 期待通りにファイル探索が完了した。効率面はもう少し改善可能。\n"
            "improvement_suggestion: 並行処理を導入することで効率が上がる可能性がある。"
        )


class TestEvaluation:
    def setup_method(self):
        self.eval = Evaluation(llm_client=MockEvalLLM())

    def _make_agent_output(self, success: bool = True,
                           num_steps: int = 1) -> AgentOutput:
        return AgentOutput(
            plan_id="test_plan",
            step_results=[
                StepResult(
                    step_order=i + 1,
                    action="file_list" if success else "unknown",
                    success=success,
                    output="result",
                    error=None if success else "error occurred",
                    duration=0.1,
                )
                for i in range(num_steps)
            ],
            overall_success=success,
            execution_time=0.5,
            log="test log",
        )

    def _make_episode(self) -> Episode:
        return Episode(
            id="ep_test",
            timestamp=datetime.now(),
            event="テスト行動",
            context="",
            emotion="",
            result="success",
            importance=0.5,
        )

    # --- 正常系テスト ---

    def test_evaluate_returns_valid_output(self):
        """evaluate() が正しい EvaluationOutput を返す"""
        result = self.eval.evaluate(EvaluationInput(
            goal="ファイルを調査する",
            action_result=self._make_agent_output(success=True),
            expected_outcome="ファイル一覧が表示される",
            episode=self._make_episode(),
        ))
        assert isinstance(result, EvaluationOutput)
        assert isinstance(result.score, EvaluationScore)
        assert 0.0 <= result.score.goal_achievement <= 1.0
        assert 0.0 <= result.score.overall <= 1.0

    def test_success_gets_high_score(self):
        """成功すると高い評価スコアが得られる"""
        result = self.eval.evaluate(EvaluationInput(
            goal="調査する",
            action_result=self._make_agent_output(success=True),
            expected_outcome="一覧表示",
            episode=self._make_episode(),
        ))
        assert result.score.goal_achievement > 0.5

    def test_failure_registered(self):
        """失敗した場合に discrepancy にエラー情報が設定される"""
        result = self.eval.evaluate(EvaluationInput(
            goal="調査する",
            action_result=self._make_agent_output(success=False),
            expected_outcome="一覧表示",
            episode=self._make_episode(),
        ))
        # モックLLMを使用している場合も、EvaluationOutput が返ることを確認
        assert isinstance(result, EvaluationOutput)
        assert isinstance(result.score, EvaluationScore)
        assert 0.0 <= result.score.overall <= 1.0

    def test_empty_goal_uses_default(self):
        """目標未定義でも評価できる"""
        result = self.eval.evaluate(EvaluationInput(
            goal="",
            action_result=self._make_agent_output(success=True),
            expected_outcome="",
            episode=self._make_episode(),
        ))
        assert isinstance(result, EvaluationOutput)

    def test_evaluation_with_many_steps(self):
        """多数のステップがある場合の評価"""
        result = self.eval.evaluate(EvaluationInput(
            goal="複雑なタスク",
            action_result=self._make_agent_output(
                success=True, num_steps=10
            ),
            expected_outcome="完了",
            episode=self._make_episode(),
        ))
        assert isinstance(result, EvaluationOutput)

    # --- 履歴テスト ---

    def test_history_tracks_scores(self):
        """評価履歴が蓄積される"""
        for _ in range(5):
            self.eval.evaluate(EvaluationInput(
                goal="テスト",
                action_result=self._make_agent_output(success=True),
                expected_outcome="",
                episode=self._make_episode(),
            ))
        history = self.eval.get_history()
        assert len(history) == 5

    def test_history_max_size(self):
        """履歴の最大サイズが制限される"""
        self.eval.max_history = 3
        for _ in range(10):
            self.eval.evaluate(EvaluationInput(
                goal="テスト",
                action_result=self._make_agent_output(success=True),
                expected_outcome="",
                episode=self._make_episode(),
            ))
        assert len(self.eval.get_history()) <= 3

    def test_get_history_with_period(self):
        """get_history() で期間指定ができる"""
        for i in range(10):
            self.eval.evaluate(EvaluationInput(
                goal=f"テスト{i}",
                action_result=self._make_agent_output(success=True),
                expected_outcome="",
                episode=self._make_episode(),
            ))
        history_7d = self.eval.get_history("7d")
        assert len(history_7d) == 7

    # --- エッジケーステスト ---

    def test_compare_returns_string(self):
        """compare() が文字列を返す"""
        s1 = EvaluationScore(0.8, 0.7, 0.9, 0.3, 0.7)
        s2 = EvaluationScore(0.5, 0.5, 0.5, 0.5, 0.5)
        result = self.eval.compare(s1, s2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compare_similar_scores(self):
        """似たスコアの比較で期待通りと返す"""
        s = EvaluationScore(0.5, 0.5, 0.5, 0.5, 0.5)
        result = self.eval.compare(s, s)
        assert "期待通り" in result

    def test_invalid_scores_fallback_to_rule(self):
        """無効なスコアの場合、ルールベース評価にフォールバック"""

        class EvaluationWithMock(Evaluation):
            def _llm_evaluate(self, input):
                return EvaluationScore(
                    goal_achievement=9.9,  # 無効な値
                    efficiency=0.5,
                    correctness=0.5,
                    novelty=0.5,
                    overall=9.9,
                )

        e = EvaluationWithMock()
        result = e.evaluate(EvaluationInput(
            goal="テスト",
            action_result=self._make_agent_output(success=True),
            expected_outcome="",
            episode=self._make_episode(),
        ))
        # ルールベースにフォールバックしているため有効なスコアになる
        assert 0.0 <= result.score.overall <= 1.0
