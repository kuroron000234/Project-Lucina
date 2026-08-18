"""
行動計画層 (Planning)

責務: 人格層の方針を、実行可能な具体的な手順へ分解する。
Phase 1: LLM呼び出しによる計画生成
"""

import json
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

        # ツール一覧（v4.1: 各ツールの必須パラメータと具体例を明示し、
        # 空パラメータのステップ生成を防ぐ）
        if input.available_tools:
            lines.append("## 利用可能なツール")
            lines.append("各ツールのパラメータは必ず具体的な値（パス・内容・クエリ等）を埋めてください。")
            lines.append("params を空の {} にしたステップは実行時に失敗するため、絶対に生成しないでください。")
            lines.append("")
            for tool in input.available_tools:
                line = f"- {tool.name}: {tool.description}"
                params = tool.parameters or {}
                required = params.get("required", [])
                example = params.get("example", {})
                if required:
                    line += f"  [必須パラメータ: {', '.join(required)}]"
                if example:
                    line += f"  例: {json.dumps(example, ensure_ascii=False)}"
                lines.append(line)
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
        lines.append("params は必ず JSON 形式の辞書で、そのツールの必須パラメータを必ず含めてください。")
        lines.append("")
        lines.append("plan_id: <一意のID>")
        lines.append("steps:")
        lines.append("  - order: 1")
        lines.append("    action: <ツール名>")
        lines.append("    params: {\"path\": \"data/workspace/report.md\", \"content\": \"レポート本文...\"}")
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

        i = 0
        while i < len(lines):
            line = lines[i]
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
                    # v4.1: 複数行にまたがる JSON を蓄積してからパースする。
                    # ローカルLLMは params: { で改行して値を書くことが多く、
                    # 行単位パースでは空になるため、ブレースが閉じるまで蓄積する。
                    if params_str.count("{") > params_str.count("}"):
                        acc = [params_str]
                        while i + 1 < len(lines):
                            i += 1
                            nxt = lines[i]
                            acc.append(nxt)
                            if nxt.count("{") <= nxt.count("}") and nxt.count("}") >= 1:
                                break
                        params_str = "\n".join(acc)
                    if params_str and params_str.strip() not in ("", "{}"):
                        current_step["params"] = self._parse_simple_params(params_str)
                    else:
                        current_step["params"] = {}
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

            i += 1

        # 最後のステップを追加
        if current_step:
            steps.append(current_step)

        # ステップがない場合はデフォルト計画を生成
        if not steps:
            return self._default_plan(input, plan_id)

        # v4.1: パラメータの実体チェック（スキーマ準拠）。
        # LLM が params を省略・空文字を含むステップを出した場合は実行不能なので
        # 除去する。ただし必須パラメータを持たないツール（file_list 等）の
        # 空 params は正当なので残す（一律除去は regression になる）。
        required_map = {}
        if input and input.available_tools:
            for t in input.available_tools:
                required_map[t.name] = (t.parameters or {}).get("required", [])
        steps = [s for s in steps
                 if self._step_params_are_concrete(s, required_map)]

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

        if not step_objects:
            return self._default_plan(input, plan_id)

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
        """
        パラメータパース。

        v4.1: まず extract_json で正規JSONとして解釈し、失敗時のみ
        簡易正規表現パースにフォールバックする。
        """
        params_str = params_str.strip()
        # 正規JSONとして解釈を試みる（複数行・入れ子対応）
        if params_str.startswith("{"):
            parsed = self.llm.extract_json(params_str)
            if parsed:
                return parsed
        # フォールバック: 簡易パース
        params = {}
        if params_str.startswith("{") and params_str.endswith("}"):
            inner = params_str[1:-1]
            pairs = re.findall(r'"(\w+)":\s*"([^"]*)"', inner)
            for key, value in pairs:
                params[key] = value
        return params

    def _step_params_are_concrete(self, step: dict, required_map: dict) -> bool:
        """
        v4.1: ステップの params が実行可能な実値を持っているか判定する。

        - ツールスキーマに必須パラメータがある場合: 欠けている/空値なら除去。
        - 必須パラメータが無いツール（file_list 等）: 空 params でも保持。
        - スキーマ不明のツール: 空 params は除去（0バイトゴミ生成ガード）。
        """
        action = step.get("action")
        params = step.get("params") or {}

        if action in required_map:
            required = required_map[action]
            if not required:
                return True  # 必須パラメータ無し → 空 params でも正当
            for key in required:
                value = params.get(key)
                if value is None:
                    logger.warning(
                        f"Planning: dropping step missing required param '{key}' "
                        f"(action={action})"
                    )
                    return False
                if isinstance(value, str) and not value.strip():
                    logger.warning(
                        f"Planning: dropping step with empty required param '{key}' "
                        f"(action={action})"
                    )
                    return False
                if isinstance(value, (list, dict)) and not value:
                    logger.warning(
                        f"Planning: dropping step with empty required param '{key}' "
                        f"(action={action})"
                    )
                    return False
            return True

        # スキーマ不明のツール: 空 params のまま実行すると0バイトゴミ等になるため除去
        if not params:
            logger.warning(
                f"Planning: dropping step with empty params "
                f"(action={action}, order={step.get('order')})"
            )
            return False
        return True
