"""
評価層 (Evaluation)

責務: 行動結果を目標・期待と比較して点数化する。
学習層と長期行動計画層にフィードバックを提供する。

Phase 2 Step 1: LLM評価 + ルールベース評価のハイブリッド
"""

import json
import logging
import os

import config
from core.evaluation.interface import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationScore,
)
from core.llm import LLMClient

logger = logging.getLogger("Evaluation")


class Evaluation:
    """
    評価層: 行動結果を多次元で評価する。

    v3.2:
    - eval_type（llm/rule）と source（dialog/autonomous）を各スコアにタグ付け
    - 評価履歴をアトミックに永続化（tmp+rename）し、再起動後も学習が機能する
    - use_llm=False でルールベース評価のみ実行（tier2のコスト抑制）

    エッジケース:
    - 目標未定義: goal が空ならデフォルト "探索" とみなす
    - 結果が空: 何もしなかった場合の評価（コスト=0、達成度=0）
    - 評価不能: overall = 0.5 の中間値
    - 永続化ファイル破損: 空履歴で復帰（学習は再蓄積）
    """

    def __init__(self, llm_client: LLMClient | None = None,
                 storage_path: str | None = None):
        self.llm = llm_client or LLMClient()
        self.history: list[EvaluationScore] = []
        self.max_history = config.EVALUATION_CONFIG["history_size"]
        self.weights = config.EVALUATION_CONFIG["weights"]
        self.storage_path = storage_path
        if storage_path:
            self.history = self._load_history()

    def evaluate(self, input: EvaluationInput) -> EvaluationOutput:
        """
        行動結果を総合評価する。

        LLM評価を試み、パースに失敗した場合はルールベース評価にフォールバック。
        """
        # LLM評価を試行（use_llm=False ならルールベースのみ）
        llm_result = self._llm_evaluate(input) if input.use_llm else None

        # LLM評価が有効ならそれを、失敗したらルールベース評価を使用
        if llm_result and self._is_valid_score(llm_result):
            score = llm_result
            evaluation_type = "llm"
        else:
            score = self._rule_based_evaluate(input)
            evaluation_type = "rule"

        # v3.2: レジームタグを付与（学習層が同一タイプ内で統計を取るため）
        score.eval_type = evaluation_type
        score.source = getattr(input.episode, "source", "autonomous")

        discrepancy = self._compute_discrepancy(input, score)
        improvement = self._generate_improvement_suggestion(input, score)

        # 履歴に追加
        self.history.append(score)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # v3.2: アトミック永続化
        if self.storage_path:
            self._persist_history()

        logger.debug(
            f"Evaluation ({evaluation_type}): overall={score.overall:.2f}, "
            f"achievement={score.goal_achievement:.2f}"
        )

        return EvaluationOutput(
            score=score,
            discrepancy=discrepancy,
            improvement_suggestion=improvement,
        )

    def compare(self, actual: EvaluationScore, expected: EvaluationScore) -> str:
        """
        期待スコアと実績スコアの差を分析する。
        """
        diffs = []
        for attr in ["goal_achievement", "efficiency", "correctness", "novelty", "overall"]:
            a = getattr(actual, attr)
            e = getattr(expected, attr)
            diff = a - e
            if abs(diff) > 0.1:
                direction = "上回った" if diff > 0 else "下回った"
                diffs.append(f"{attr}: 期待{e:.2f}→実績{a:.2f} ({direction})")

        if not diffs:
            return "ほぼ期待通りの結果でした。"
        return "差異分析:\n" + "\n".join(diffs)

    def get_history(self, period: str = "all") -> list[EvaluationScore]:
        """
        評価履歴を取得する。

        period:
        - "all": 全履歴
        - "7d": 直近7日分（実際は履歴サイズで制限）
        """
        if period == "7d":
            # Phase 1 では単純に直近の履歴を返す
            return self.history[-7:] if len(self.history) >= 7 else self.history
        return self.history

    def _llm_evaluate(self, input: EvaluationInput) -> EvaluationScore | None:
        """LLM評価を試行する。"""
        try:
            prompt = (
                f"## 評価タスク\n\n"
                f"**目標**: {input.goal or '探索'}\n"
                f"**期待結果**: {input.expected_outcome}\n"
                f"**実際の結果**: {input.action_result.log}\n\n"
                f"以下の各項目を0.0〜1.0で評価してください。\n"
                f"- goal_achievement: 目標は達成されたか\n"
                f"- efficiency: 効率的だったか（リソース・時間の使い方）\n"
                f"- correctness: 正確だったか（エラーの有無）\n"
                f"- novelty: 新しい要素があったか\n"
                f"- overall: 総合評価\n\n"
                f"出力形式:\n"
                f"goal_achievement: <0.0-1.0>\n"
                f"efficiency: <0.0-1.0>\n"
                f"correctness: <0.0-1.0>\n"
                f"novelty: <0.0-1.0>\n"
                f"overall: <0.0-1.0>"
            )
            response = self.llm.chat(prompt)
            return self._parse_score(response)
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return None

    def _parse_score(self, response: str) -> EvaluationScore | None:
        """LLM応答から EvaluationScore をパースする。"""
        result = self.llm.extract_yaml_like(response)

        try:
            return EvaluationScore(
                goal_achievement=float(result.get("goal_achievement", 0.5)),
                efficiency=float(result.get("efficiency", 0.5)),
                correctness=float(result.get("correctness", 0.5)),
                novelty=float(result.get("novelty", 0.5)),
                overall=float(result.get("overall", 0.5)),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse evaluation score: {e}")
            return None

    def _rule_based_evaluate(self, input: EvaluationInput) -> EvaluationScore:
        """
        ルールベースの評価。

        LLM評価が使えない場合のフォールバック。
        """
        result = input.action_result

        # 目標達成度: 全ステップ成功なら高評価
        goal_achievement = 1.0 if result.overall_success else 0.3

        # 効率性: ステップ数と実行時間から判断
        num_steps = max(len(result.step_results), 1)
        avg_time = result.execution_time / num_steps if num_steps > 0 else 0
        if avg_time < 1:
            efficiency = 0.9
        elif avg_time < 5:
            efficiency = 0.7
        elif avg_time < 15:
            efficiency = 0.5
        else:
            efficiency = 0.3

        # 正確性: エラーの有無
        error_count = sum(1 for s in result.step_results if s.error)
        correctness = max(0.0, 1.0 - (error_count / num_steps))

        # 新規性: Phase 1 ではデフォルト値
        novelty = 0.3

        # 総合 = 加重平均
        overall = (
            self.weights["goal_achievement"] * goal_achievement
            + self.weights["efficiency"] * efficiency
            + self.weights["correctness"] * correctness
            + self.weights["novelty"] * novelty
        )

        return EvaluationScore(
            goal_achievement=max(0.0, min(1.0, goal_achievement)),
            efficiency=max(0.0, min(1.0, efficiency)),
            correctness=max(0.0, min(1.0, correctness)),
            novelty=novelty,
            overall=max(0.0, min(1.0, overall)),
        )

    def _is_valid_score(self, score: EvaluationScore) -> bool:
        """評価スコアが有効な範囲かチェック。"""
        for val in [
            score.goal_achievement,
            score.efficiency,
            score.correctness,
            score.novelty,
            score.overall,
        ]:
            if not (0.0 <= val <= 1.0):
                return False
        return True

    def _persist_history(self):
        """
        評価履歴をアトミックに永続化する（tmp+rename）。

        スーパーバイザによる強制終了（SIGTERM/SIGKILL）でもファイルが
        破損しないよう、一時ファイルへの書き込み後に rename する。
        """
        try:
            tmp_path = f"{self.storage_path}.tmp"
            data = {"version": 1, "entries": [
                {
                    "goal_achievement": s.goal_achievement,
                    "efficiency": s.efficiency,
                    "correctness": s.correctness,
                    "novelty": s.novelty,
                    "overall": s.overall,
                    "eval_type": s.eval_type,
                    "source": s.source,
                }
                for s in self.history
            ]}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.storage_path)
        except Exception as e:
            logger.warning(f"Failed to persist evaluation history: {e}")

    def _load_history(self) -> list[EvaluationScore]:
        """
        永続化された評価履歴を型安全に読み込む。

        破損・欠落時は空リストで復帰する（学習は再蓄積される）。
        """
        try:
            if not os.path.exists(self.storage_path):
                return []
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            result = []
            for e in entries:
                try:
                    result.append(EvaluationScore(
                        goal_achievement=float(e.get("goal_achievement", 0.5)),
                        efficiency=float(e.get("efficiency", 0.5)),
                        correctness=float(e.get("correctness", 0.5)),
                        novelty=float(e.get("novelty", 0.5)),
                        overall=float(e.get("overall", 0.5)),
                        eval_type=e.get("eval_type", "rule"),
                        source=e.get("source", "autonomous"),
                    ))
                except (ValueError, TypeError):
                    continue
            return result[-self.max_history:]
        except Exception as e:
            logger.warning(f"Failed to load evaluation history: {e}")
            return []

    def _compute_discrepancy(self, input: EvaluationInput, score: EvaluationScore) -> str:
        """
        期待と実績のズレの説明を生成する。
        """
        parts = []
        if input.action_result.overall_success:
            if score.goal_achievement < 0.5:
                parts.append("成功したが、目標達成度は低め")
        else:
            parts.append("全体として失敗")

        if score.efficiency < 0.4:
            parts.append("効率性に改善の余地あり")

        if score.novelty < 0.2:
            parts.append("新しい試みが不足")

        return "\n".join(parts) if parts else "期待通りの結果"

    def _generate_improvement_suggestion(
        self, input: EvaluationInput, score: EvaluationScore
    ) -> str:
        """
        改善提案を生成する。
        """
        suggestions = []
        if not input.action_result.overall_success:
            suggestions.append("エラー原因の分析と、該当ステップの代替手段を検討する")
        if score.efficiency < 0.4:
            suggestions.append("より少ないステップ数で目標を達成できないか検討する")
        if score.novelty < 0.3:
            suggestions.append("新しいアプローチやツールを試す")
        if len(input.action_result.step_results) > 5:
            suggestions.append("ステップ数を削減できないか計画を見直す")

        return "\n".join(suggestions[:3]) if suggestions else "現状維持でよい"
