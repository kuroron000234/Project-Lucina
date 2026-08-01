"""
学習層 (Learning) の単体テスト
"""

from core.learning.learning import Learning
from core.learning.interface import LearningInput, LearningOutput
from core.evaluation.interface import EvaluationOutput, EvaluationScore
from core.drive.interface import DriveOutput


class TestLearning:
    def setup_method(self):
        self.learning = Learning()

    def _make_evaluation(self, overall: float = 0.7) -> EvaluationOutput:
        return EvaluationOutput(
            score=EvaluationScore(
                goal_achievement=overall,
                efficiency=0.6,
                correctness=0.8,
                novelty=0.3,
                overall=overall,
            ),
            discrepancy="",
            improvement_suggestion="",
        )

    def _make_drive_snapshot(self) -> DriveOutput:
        return DriveOutput(
            drives={
                "exploration": 0.45,   # base 0.35 + memory空+0.1
                "social": 0.35,
                "achievement": 0.35,
                "rest": 0.35,
                "maintenance": 0.35,
            },
            primary_drive="exploration",
            drive_tension=0.2,
            novelty_score=0.5,
        )

    # --- 正常系テスト ---

    def test_learn_returns_valid_output(self):
        """learn() が正しい LearningOutput を返す"""
        result = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.8),
            evaluation_history=[self._make_evaluation(0.7).score for _ in range(5)],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
        ))
        assert isinstance(result, LearningOutput)
        assert isinstance(result.drive_adjustments, dict)
        assert isinstance(result.learning_summary, str)

    def test_learn_skips_on_insufficient_data(self):
        """履歴が3件未満の場合、調整をスキップ"""
        result = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.8),
            evaluation_history=[],  # 履歴なし
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
        ))
        assert result.drive_adjustments == {}
        assert "skipped" in result.learning_summary

    # --- 駆動調整テスト ---

    def test_drive_adjustments_are_clipped(self):
        """駆動調整値が最大調整量を超えない"""
        result = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.1),  # 非常に低い評価
            evaluation_history=[self._make_evaluation(s).score for s in [0.3, 0.4, 0.5, 0.6, 0.7]],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
            driving_drive="exploration",
        ))
        for name, delta in result.drive_adjustments.items():
            assert -0.2 <= delta <= 0.2, f"{name}={delta} exceeds clip limit"

    def test_high_reward_increases_primary_drive(self):
        """高い報酬で主駆動（primary）が正の調整になる（ゼロサム）"""
        result = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.9),
            evaluation_history=[self._make_evaluation(s).score for s in [0.3, 0.4, 0.5, 0.6, 0.7]],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
            driving_drive="exploration",
        ))
        assert result.drive_adjustments["exploration"] > 0
        # ゼロサム: 合計は0
        assert abs(sum(result.drive_adjustments.values())) < 1e-9
        # 他の駆動は負（クレジット割り当て）
        assert result.drive_adjustments["social"] < 0

    # --- 重要度更新テスト ---

    def test_importance_delta_range(self):
        """重要度更新値が範囲内に収まる"""
        # 非常に高い評価
        result_high = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(1.0),
            evaluation_history=[self._make_evaluation(0.5).score for _ in range(5)],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
        ))
        assert -0.2 <= result_high.memory_importance_update <= 0.2

        # 非常に低い評価
        result_low = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.0),
            evaluation_history=[self._make_evaluation(0.5).score for _ in range(5)],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_002",
        ))
        assert -0.2 <= result_low.memory_importance_update <= 0.2

    # --- 学習曲線テスト ---

    def test_learning_curve_tracks_progress(self):
        """学習曲線が評価ごとに更新される"""
        for score in [0.3, 0.5, 0.7, 0.9]:
            self.learning.learn(LearningInput(
                evaluation=self._make_evaluation(score),
                evaluation_history=[
                    self._make_evaluation(0.5).score for _ in range(5)
                ],
                drive_snapshot=self._make_drive_snapshot(),
                episode_id="ep_test",
            ))
        curve = self.learning.get_learning_curve()
        assert len(curve) > 0

    # --- エッジケーステスト ---

    def test_personality_adjustments_none_with_few_data(self):
        """データ不足時は personality_adjustments が None"""
        result = self.learning.learn(LearningInput(
            evaluation=self._make_evaluation(0.8),
            evaluation_history=[self._make_evaluation(0.5).score for _ in range(3)],
            drive_snapshot=self._make_drive_snapshot(),
            episode_id="ep_001",
        ))
        assert result.personality_adjustments is None

    def test_adjust_drive_parameters_without_history(self):
        """履歴なしで adjust_drive_parameters() を呼べる"""
        result = self.learning.adjust_drive_parameters([])
        assert result == {}
