"""
学習層 (Learning)

責務: 評価結果からシステム全体のパラメータを調整する。
駆動層の重み・記憶の重要度・人格特性を更新する。

Phase 2 Step 2: 移動平均 + 微分調整
"""

import logging

import config
from core.learning.interface import (
    LearningInput,
    LearningOutput,
)

logger = logging.getLogger("Learning")


class Learning:
    """
    学習層: 評価結果から各層のパラメータ調整値を生成する。

    学習則 (Phase 2):
    - 単純移動平均: drive_adjustments[d] = 0.1 * (avg_reward - current_drive[d])
    - 重要度更新: delta = 0.1 * (overall - 0.5)

    エッジケース:
    - 学習データ不足: evaluation_history < 3件 なら調整を保留
    - スコア急変: 1回の評価で大幅調整はしない（クリッピング）
    - 共適応問題: 評価と学習が互いに追いかけっこにならないよう、学習率を抑制
    """

    def __init__(self):
        self.learning_rate = config.DRIVE_CONFIG["learning_rate"]
        self.learning_curve: list[float] = []
        self.max_adjustment = 0.2  # 1回の最大調整量

    def learn(self, input: LearningInput) -> LearningOutput:
        """
        評価結果から学習し、各層のパラメータ調整値を出力する。

        学習データが十分でない場合（履歴3件未満）は調整を保留。
        """
        self.learning_curve.append(input.evaluation.score.overall)

        # 学習データ不足チェック
        if len(input.evaluation_history) < 3:
            logger.debug("Learning data insufficient (< 3), skipping adjustments")
            return LearningOutput(
                drive_adjustments={},
                memory_importance_update=0.0,
                personality_adjustments=None,
                learning_summary=f"(skipped - only {len(input.evaluation_history)} history entries)",
            )

        # 駆動パラメータ調整値
        drive_adjustments = self._compute_drive_adjustments(
            input.evaluation, input.evaluation_history, input.drive_snapshot
        )

        # エピソード重要度更新
        importance_delta = self._compute_importance_delta(input.evaluation)

        # 人格特性調整（Phase 2 では簡易版）
        personality_adjustments = self._compute_personality_adjustments(
            input.evaluation, input.evaluation_history
        )

        # 学習曲線の更新
        self.learning_curve.append(input.evaluation.score.overall)

        summary = (
            f"overall={input.evaluation.score.overall:.2f}, "
            f"avg={self._moving_avg(input.evaluation_history, 5):.2f}, "
            f"imp_delta={importance_delta:.3f}, "
            f"adjustments={len(drive_adjustments)} drives"
        )
        logger.debug(f"Learning: {summary}")

        return LearningOutput(
            drive_adjustments=drive_adjustments,
            memory_importance_update=importance_delta,
            personality_adjustments=personality_adjustments,
            learning_summary=summary,
        )

    def adjust_drive_parameters(self, history: list) -> dict[str, float]:
        """
        駆動層のパラメータを調整する。
        歴史的な評価履歴から駆動調整値を計算。
        """
        if len(history) < 3:
            return {}

        # 最新の評価から調整
        latest = history[-1]
        avg_reward = self._moving_avg(history, 5)

        adjustments = {}
        for drive_name in ["exploration", "social", "achievement", "rest", "maintenance"]:
            delta = self.learning_rate * (latest.overall - avg_reward)
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta))
            adjustments[drive_name] = delta

        return adjustments

    def get_learning_curve(self) -> list[float]:
        """
        学習曲線（時系列の総合スコア）を返す。
        """
        return self.learning_curve

    def _compute_drive_adjustments(self, evaluation: Any, history: list, drive_snapshot: Any) -> dict[str, float]:
        """
        駆動調整値を計算する。

        式: adjustment = learning_rate * (reward - predicted)
        ここでは簡易的に avg_reward を baseline として使用。
        """
        avg_reward = self._moving_avg(history, 5)
        reward = evaluation.score.overall

        adjustments = {}
        for drive_name, drive_value in drive_snapshot.drives.items():
            # 報酬が平均より高い → 現在の駆動方向は正しい → 強化
            # 報酬が平均より低い → 現在の駆動方向は間違い → 抑制
            delta = self.learning_rate * (reward - avg_reward)
            # クリッピング
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta))
            adjustments[drive_name] = delta

        return adjustments

    def _compute_importance_delta(self, evaluation: Any) -> float:
        """
        エピソード重要度の増減を計算する。

        式: delta = 0.1 * (overall - 0.5)
        overall > 0.5 → 重要度上昇（良い経験）
        overall < 0.5 → 重要度低下（悪い経験・忘れてよい）
        """
        delta = 0.1 * (evaluation.score.overall - 0.5)
        return max(-0.2, min(0.2, delta))

    def _compute_personality_adjustments(self, evaluation: Any, history: list) -> dict | None:
        """
        人格特性の微調整値を計算する。
        Phase 2 では簡易版。
        """
        if len(history) < 5:
            return None

        # 直近の傾向を分析
        recent = [s.novelty for s in history[-5:]]
        avg_novelty = sum(recent) / len(recent)

        adjustments = {}
        # 新規性が高い → 好奇心を強化
        if avg_novelty > 0.6:
            adjustments["curiosity_delta"] = 0.05
        elif avg_novelty < 0.2:
            adjustments["curiosity_delta"] = -0.02

        return adjustments if adjustments else None

    def _moving_avg(self, history: list, window: int) -> float:
        """移動平均を計算する。"""
        if not history:
            return 0.5
        scores = [s.overall if hasattr(s, 'overall') else (s.get('overall', 0.5) if isinstance(s, dict) else 0.5) for s in history]
        recent = scores[-window:] if len(scores) >= window else scores
        return sum(recent) / len(recent)
