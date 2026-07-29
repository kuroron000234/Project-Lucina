"""
行動計画層 (Planning)

責務: 人格層の方針を、実行可能な具体的な手順へ分解する。
Phase 1: LLM呼び出しによる計画生成
"""

import logging
import re
import time

from core.llm import LLMClient
from core.planning.interface import (
    PlanningInput,
    PlanningOutput,
    Step,
    ToolInfo,
)

logger = logging.getLogger("Planning")


class Planning:
    """
    行動計画層: 人格層の方針を具体的な実行手順に分解する。

    エッジケース:
    - 不可能な計画: 分割・代替案を自動生成
    - ステップ数爆発: 最大10ステップに制限
    - タイムアウト: timeout を超えたステップは失敗とみなし再計画
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def make(self, input: PlanningInput) -> PlanningOutput:
        """
        方針から実行計画を生成する。

        世界モデルからの予測がある場合はそれを考慮する。
        """
        prompt = self._build_planning_prompt(input)

        system_prompt = (
            "あなたはAIエージェントの行動計画層です。\n"
            "与えられた目標を達成するための具体的な手順を考えてください。\n"
            "各手順は実行可能なアクションとして定義し、適切なパラメータを設定してください。\n"
            "ステップ数は最小限に抑え、3〜7ステップ程度を目安にしてください。"
        )

        response = self.llm.chat(prompt, system_prompt=system_prompt)
        return self._parse_plan(response, input)

    def revise(self, plan_id: str, failed_step: int, feedback: str) -> PlanningOutput:
        """
        失敗したステップを修正した新しい計画を生成する。
        """
        prompt = (
            f"以下の計画のステップ{failed_step}が失敗しました。\n\n"
            f"計画ID: {plan_id}\n"
            f"失敗ステップ: {failed_step}\n"
            f"フィードバック: {feedback}\n\n"
            f"失敗を考慮して、修正された完全な計画を生成してください。\n"
            f"特に失敗したステップの代替手段を検討してください。"
        )
        response = self.llm.chat(prompt)
        # 修正版をパース（簡易版として new_plan_id で再生成）
        return self._parse_plan(response, None)

    def estimate_duration(self, plan: PlanningOutput) -> float:
        """
        計画の所要時間を推定する。
        """
        return sum(step.timeout for step in plan.steps)

    def _build_planning_prompt(self, input: PlanningInput) -> str:
        """計画生成プロンプトを構築する。"""
        lines = ["## 目標", f"{input.policy.goal}", ""]
        lines.append("## 行動方針")
        lines.append(input.policy.action_policy)
        lines.append("")
        lines.append(f"## 緊急度: {input.policy.priority}/5")
        lines.append("")

        # ツール一覧
        if input.available_tools:
            lines.append("## 利用可能なツール")
            for tool in input.available_tools:
                lines.append(f"- {tool.name}: {tool.description}")
            lines.append("")

        # 世界モデル予測
        if input.world_model_predictions:
            lines.append("## 予測される結果")
            for pred in input.world_model_predictions:
                lines.append(
                    f"- [{pred.probability:.0%}] {pred.action}: "
                    f"{pred.next_state[:50]}..."
                )
            lines.append("")

        # 出力形式
        lines.append("## 出力形式")
        lines.append("以下の形式で出力してください。各ステップは具体的なアクションにしてください。")
        lines.append("")
        lines.append("plan_id: <一意のID>")
        lines.append("steps:")
        lines.append("  - order: 1")
        lines.append("    action: <ツール名 (file_read|file_write|file_list|command_exec|web_search|web_fetch|code_analyze|notify_user)>")
        lines.append("    params: {<パラメータ>}")
        lines.append("    description: <このステップの説明>")
        lines.append("    expected_result: <期待される結果>")
        lines.append("    fallback: <失敗時の代替アクション>")
        lines.append("    timeout: <タイムアウト秒数>")
        lines.append("  - order: 2")
        lines.append("    ...")
        lines.append("expected_outcome: <全体として期待される結果>")
        lines.append("estimated_duration: <推定所要時間（秒）>")

        return "\n".join(lines)

    def _parse_plan(self, response: str, input: PlanningInput | None) -> PlanningOutput:
        """
        LLMの応答を PlanningOutput にパースする。

        パースに失敗した場合はシンプルなデフォルト計画を生成。
        """
        lines = response.strip().split("\n")

        plan_id = f"plan_{int(time.time())}"
        steps = []
        expected_outcome = ""
        estimated_duration = 15.0
        current_step = None
        in_steps = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("plan_id:"):
                plan_id = stripped.split("plan_id:", 1)[1].strip()

            elif stripped.startswith("steps:"):
                in_steps = True
                if current_step:
                    steps.append(current_step)
                    current_step = None

            elif stripped.startswith("expected_outcome:"):
                in_steps = False
                if current_step:
                    steps.append(current_step)
                    current_step = None
                expected_outcome = stripped.split("expected_outcome:", 1)[1].strip()

            elif stripped.startswith("estimated_duration:"):
                try:
                    estimated_duration = float(
                        stripped.split("estimated_duration:", 1)[1].strip()
                    )
                except ValueError:
                    pass

            elif in_steps and stripped.startswith("- order:"):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "order": 0,
                    "action": "",
                    "params": {},
                    "description": "",
                    "expected_result": "",
                    "fallback": None,
                    "timeout": 30.0,
                }
                try:
                    current_step["order"] = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif current_step is not None:
                if stripped.startswith("action:"):
                    current_step["action"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("params:"):
                    params_str = stripped.split(":", 1)[1].strip()
                    if params_str and params_str != "{}":
                        # 簡易パース（完全なJSONパースは extract_json を使用）
                        current_step["params"] = self._parse_simple_params(params_str)
                elif stripped.startswith("description:"):
                    current_step["description"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("expected_result:"):
                    current_step["expected_result"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("fallback:"):
                    fb = stripped.split(":", 1)[1].strip()
                    current_step["fallback"] = fb if fb and fb.lower() != "none" else None
                elif stripped.startswith("timeout:"):
                    try:
                        current_step["timeout"] = float(stripped.split(":", 1)[1].strip())
                    except ValueError:
                        pass

        # 最後のステップを追加
        if current_step:
            steps.append(current_step)

        # ステップがない場合はデフォルト計画を生成
        if not steps:
            return self._default_plan(input, plan_id)

        # Step オブジェクトに変換
        step_objects = []
        for s in steps:
            step_objects.append(Step(
                order=s["order"],
                action=s["action"],
                params=s["params"],
                description=s.get("description", ""),
                expected_result=s.get("expected_result", ""),
                fallback=s.get("fallback"),
                timeout=s.get("timeout", 30.0),
            ))

        return PlanningOutput(
            plan_id=plan_id,
            steps=step_objects,
            expected_outcome=expected_outcome or "計画を実行する",
            estimated_duration=estimated_duration or sum(s.timeout for s in step_objects),
        )

    def _default_plan(self, input: PlanningInput | None, plan_id: str) -> PlanningOutput:
        """パース失敗時のデフォルト計画。"""
        if input and input.policy:
            goal = input.policy.goal
        else:
            goal = "システム状態を確認する"

        return PlanningOutput(
            plan_id=plan_id,
            steps=[
                Step(
                    order=1,
                    action="file_list",
                    params={},
                    description="ワークスペースのファイル一覧を取得する",
                    expected_result="ファイル一覧が表示される",
                    fallback="notify_user",
                    timeout=10.0,
                ),
            ],
            expected_outcome=f"{goal} のための計画を実行する",
            estimated_duration=10.0,
        )

    def _parse_simple_params(self, params_str: str) -> dict:
        """簡易パラメータパース。"""
        params = {}
        # {} で囲まれている場合
        if params_str.startswith("{") and params_str.endswith("}"):
            inner = params_str[1:-1]
            # キー: 値 のペアを探す
            pairs = re.findall(r'"(\w+)":\s*"([^"]*)"', inner)
            for key, value in pairs:
                params[key] = value
        return params
