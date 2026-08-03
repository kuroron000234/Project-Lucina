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

        v3.2:
        - 学習曲線の重複appendバグを修正（1回のみ）
        - 駆動調整はゼロサム（主駆動 +delta、他 −delta/4）でクレジット割り当て
        - 学習ゲート: 同一 eval_type の履歴3件未満 or 報酬分散 < 0.02 なら調整を保留
        - 自律サイクル由来の調整は全体を0.5倍（対話を信頼）
        """
        self.learning_curve.append(input.evaluation.score.overall)

        # 学習データ不足チェック（同一 eval_type の履歴が必要）
        eval_type = input.evaluation.score.eval_type
        same_type = [
            s for s in input.evaluation_history
            if getattr(s, "eval_type", "rule") == eval_type
        ]
        # v3.3: 学習ゲートを緩和（自律行動の報酬分散が小さいため）
        # DRIVE_CONFIG.learning_gate が優先、なければ LEARNING_CONFIG の値を使う（後方互換）
        gate_cfg = config.DRIVE_CONFIG.get("learning_gate", {})
        min_same = gate_cfg.get("min_history",
                                 config.LEARNING_CONFIG.get("history_min_same_type", 3))
        if len(same_type) < min_same:
            logger.debug("Learning data insufficient, skipping adjustments")
            return LearningOutput(
                drive_adjustments={},
                memory_importance_update=0.0,
                personality_adjustments=None,
                learning_summary=f"(skipped - only {len(same_type)} same-type history entries)",
            )

        # 報酬分散ゲート（報酬が一定なら学ぶことがない）
        # 浮動小数の境界誤差を吸収するため6桁に丸めて比較
        variance = round(self._reward_variance(same_type), 6)
        # v3.3: ゲート閾値を DRIVE_CONFIG.learning_gate から取得（緩和）
        gate_cfg = config.DRIVE_CONFIG.get("learning_gate", {})
        variance_gate = gate_cfg.get("variance_threshold",
                                      config.LEARNING_CONFIG.get("variance_gate", 0.02))
        drive_adjustments = {}
        if variance >= variance_gate:
            # v5.0: Phase 3 — サプライズによる学習率変調。
            # 予測が外れた（高サプライズ）時は学ぶべき時なので学習率を上げる。
            effective_lr = self._modulated_learning_rate(
                getattr(input, "surprise", None)
            )
            drive_adjustments = self._compute_drive_adjustments(
                input.evaluation, same_type, input.drive_snapshot,
                input.driving_drive, input.source,
                learning_rate=effective_lr,
            )

        # エピソード重要度更新
        importance_delta = self._compute_importance_delta(input.evaluation)

        # 人格特性調整（Phase 2 では簡易版）
        personality_adjustments = self._compute_personality_adjustments(
            input.evaluation, same_type
        )

        summary = (
            f"overall={input.evaluation.score.overall:.2f}, "
            f"avg={self._moving_avg(same_type, 5):.2f}, "
            f"imp_delta={importance_delta:.3f}, "
            f"adjustments={len(drive_adjustments)} drives, "
            f"variance={variance:.3f}"
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
        駆動層のパラメータを調整する（レガシー互換メソッド）。
        歴史的な評価履歴から駆動調整値を計算。

        v3.2: このメソッドは main.py からは呼ばれない。ゼロサム・クレジット
        割り当ては _compute_drive_adjustments() を使用する（driving_drive と
        source を考慮できるため）。本メソッドは後方互換のため残している。
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

    def _modulated_learning_rate(self, surprise: float | None) -> float:
        """
        v5.0: サプライズに応じて学習率を変調する（高サプライズ = 学習率上昇）。

        effective = min(base * (1 + modulation * surprise), base * cap)
        surprise は正規化済み 0.0〜1.0 を想定。None なら変調しない。
        """
        if surprise is None:
            return self.learning_rate
        s = max(0.0, min(1.0, float(surprise)))
        mod = 1.0 + config.SURPRISE_CONFIG.get("learning_modulation", 1.5) * s
        cap = config.SURPRISE_CONFIG.get("learning_modulation_cap", 2.0)
        return min(self.learning_rate * mod, self.learning_rate * cap)

    def _compute_drive_adjustments(self, evaluation: Any, history: list,
                                   drive_snapshot: Any,
                                   driving_drive: str | None = None,
                                   source: str = "autonomous",
                                   learning_rate: float | None = None) -> dict[str, float]:
        """
        駆動調整値を計算する（v3.2: ゼロサム・クレジット割り当て）。

        主駆動（driving_drive）に +delta、他の駆動に −delta/4 を割り当てる。
        合計は常に0（ゼロサム）になり、駆動baseの単調インフレを防ぐ。

        自律サイクル（source=autonomous）由来の調整は全体を0.5倍し、
        ユーザー対話ほど信頼できる報酬信号でないことを表現する。

        driving_drive が rest の場合は調整をスキップ（trivial行動に報酬を与えない）。
        """
        avg_reward = self._moving_avg(history, 5)
        reward = evaluation.score.overall

        # クレジット対象の駆動を決定
        target = driving_drive or drive_snapshot.primary_drive
        if target == "rest":
            logger.debug("Skipping drive adjustment: driving_drive is rest")
            return {}
        if target not in drive_snapshot.drives:
            target = drive_snapshot.primary_drive

        lr = learning_rate if learning_rate is not None else self.learning_rate
        delta = lr * (reward - avg_reward)
        delta = max(-self.max_adjustment, min(self.max_adjustment, delta))

        # ゼロサム割り当て: 主駆動 +delta、他駆動 -delta/4
        adjustments = {}
        other_count = max(len(drive_snapshot.drives) - 1, 1)
        for drive_name in drive_snapshot.drives:
            if drive_name == target:
                adjustments[drive_name] = delta
            else:
                adjustments[drive_name] = -delta / other_count

        # 自律サイクルは全体を0.5倍（ゼロサムは維持される）
        if source == "autonomous":
            adjustments = {k: v * 0.5 for k, v in adjustments.items()}

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

    def _reward_variance(self, history: list, window: int = 5) -> float:
        """
        直近window件の報酬（overall）の分散を計算する。
        報酬が一定（分散<ゲート）なら学習する情報がない。
        """
        if not history:
            return 0.0
        scores = [s.overall if hasattr(s, 'overall') else (s.get('overall', 0.5) if isinstance(s, dict) else 0.5) for s in history]
        recent = scores[-window:] if len(scores) >= window else scores
        if len(recent) < 2:
            return 0.0
        mean = sum(recent) / len(recent)
        return sum((x - mean) ** 2 for x in recent) / len(recent)
