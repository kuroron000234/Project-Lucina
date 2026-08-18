"""
行動計画層 (Planning) の単体テスト
"""

import pytest

from core.llm import LLMClient
from core.planning.interface import (
    PlanningInput,
    PlanningOutput,
    Step,
    ToolInfo,
)
from core.planning.planning import Planning
from core.personality.interface import PersonalityOutput


class MockPlanningLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "plan_id: plan_test_001\n"
            "steps:\n"
            "  - order: 1\n"
            "    action: file_list\n"
            "    params: {}\n"
            "    description: ワークスペースのファイル一覧を取得\n"
            "    expected_result: ファイル一覧が表示される\n"
            "    fallback: notify_user\n"
            "    timeout: 10.0\n"
            "  - order: 2\n"
            "    action: notify_user\n"
            "    params: {\"message\": \"調査完了\"}\n"
            "    description: 結果を通知\n"
            "    expected_result: 通知完了\n"
            "    timeout: 5.0\n"
            "expected_outcome: 調査完了\n"
            "estimated_duration: 15.0"
        )


class ParamsInjectionLLM(LLMClient):
    """プロンプトにツールパラメータ仕様が注入されているかを確認するためのLLM。"""
    last_prompt = ""

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        ParamsInjectionLLM.last_prompt = prompt
        return "意味不明な応答"  # パース失敗→デフォルト計画


class EmptyParamsLLM(LLMClient):
    """全ステップが空パラメータの応答を返すLLM（除去検証用）。"""

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "plan_id: plan_empty_001\n"
            "steps:\n"
            "  - order: 1\n"
            "    action: workspace_write\n"
            "    params: {}\n"
            "    description: メモ作成\n"
            "    expected_result: 作成される\n"
            "    timeout: 10.0\n"
            "  - order: 2\n"
            "    action: file_write\n"
            "    params: {\"path\": \"data/workspace/x.md\", \"content\": \"\"}\n"
            "    description: 空内容書き込み\n"
            "    expected_result: 作成される\n"
            "    timeout: 10.0\n"
            "expected_outcome: 計画実行\n"
            "estimated_duration: 20.0"
        )


def make_personality_output() -> PersonalityOutput:
    return PersonalityOutput(
        goal="ワークスペースのファイルを調査する",
        action_policy="ファイル一覧を取得して確認する",
        priority=3,
        context_summary="探索欲求に基づく調査",
    )


def make_tool_info() -> list[ToolInfo]:
    return [
        ToolInfo(name="file_list", description="リスト取得", parameters={}),
        ToolInfo(name="file_read", description="読み込み", parameters={}),
        ToolInfo(name="notify_user", description="通知", parameters={}),
    ]


class TestPlanning:
    def test_make_returns_valid_output(self):
        """make() が正しい PlanningOutput を返す"""
        p = Planning(llm_client=MockPlanningLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
            available_tools=make_tool_info(),
        ))
        assert isinstance(result, PlanningOutput)
        assert result.plan_id
        assert len(result.steps) > 0
        assert result.steps[0].order == 1

    def test_steps_have_required_fields(self):
        """各ステップに必須フィールドが揃っている"""
        p = Planning(llm_client=MockPlanningLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
        ))
        for step in result.steps:
            assert step.action, f"Step {step.order} has no action"
            assert isinstance(step.params, dict)
            assert step.description or step.expected_result, f"Step {step.order} has no description"

    def test_default_plan_on_parse_failure(self):
        """パース失敗時にデフォルト計画が生成される"""

        class BrokenLLM(LLMClient):
            def chat(self, prompt, system_prompt=None):
                return "意味不明な応答"

        p = Planning(llm_client=BrokenLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
        ))
        assert isinstance(result, PlanningOutput)
        assert result.steps  # デフォルト計画が作られる

    def test_make_without_tools(self):
        """ツール情報なしでも計画が生成される"""
        p = Planning(llm_client=MockPlanningLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
            available_tools=None,
        ))
        assert isinstance(result, PlanningOutput)

    def test_estimate_duration(self):
        """estimate_duration() が正しい値を返す"""
        p = Planning()
        plan = PlanningOutput(
            plan_id="test",
            steps=[
                Step(order=1, action="file_read", params={},
                     description="", expected_result="", timeout=10.0),
                Step(order=2, action="file_list", params={},
                     description="", expected_result="", timeout=20.0),
            ],
            expected_outcome="",
        )
        duration = p.estimate_duration(plan)
        assert duration == 30.0  # 10 + 20

    def test_revise_returns_new_plan(self):
        """revise() が新しい計画を返す"""
        p = Planning(llm_client=MockPlanningLLM())
        result = p.revise(
            plan_id="plan_001",
            failed_step=1,
            feedback="ファイルが見つかりません",
        )
        assert isinstance(result, PlanningOutput)

    def test_prompt_includes_tool_param_schemas(self):
        """v4.1: プロンプトにツールの必須パラメータと例が注入される"""
        ParamsInjectionLLM.last_prompt = ""
        p = Planning(llm_client=ParamsInjectionLLM())
        tools = [
            ToolInfo(name="workspace_write", description="メモ作成",
                     parameters={"required": ["content"], "example": {"content": "設計メモ"}}),
            ToolInfo(name="file_read", description="読み込み",
                     parameters={"required": ["path"], "example": {"path": "x.md"}}),
        ]
        p.make(PlanningInput(
            policy=make_personality_output(),
            available_tools=tools,
        ))
        assert "必須パラメータ: content" in ParamsInjectionLLM.last_prompt
        assert "例: {" in ParamsInjectionLLM.last_prompt
        assert "空の {}" in ParamsInjectionLLM.last_prompt

    def test_steps_with_empty_params_are_dropped(self):
        """v4.1: 必須パラメータが欠けているステップは除去されデフォルト計画にフォールバック"""
        p = Planning(llm_client=EmptyParamsLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
            available_tools=[
                ToolInfo(name="workspace_write", description="メモ作成",
                         parameters={"required": ["content"]}),
                ToolInfo(name="file_write", description="書き込み",
                         parameters={"required": ["path", "content"]}),
            ],
        ))
        # 全ステップが必須パラメータ欠落（{} / content 空）→ デフォルト計画にフォールバック
        assert result.steps
        # フォールバックは file_list（必須パラメータなし）のはず
        assert result.steps[0].action == "file_list"

    def test_no_required_param_tools_keep_empty_params(self):
        """v4.1: 必須パラメータのないツール(file_list等)の空 params は除去されない"""

        class NoParamLLM(LLMClient):
            def chat(self, prompt, system_prompt=None):
                return (
                    "plan_id: plan_noparam\n"
                    "steps:\n"
                    "  - order: 1\n"
                    "    action: file_list\n"
                    "    params: {}\n"
                    "    description: 一覧取得\n"
                    "    expected_result: 一覧\n"
                    "    timeout: 10.0\n"
                    "expected_outcome: 完了\n"
                    "estimated_duration: 10.0"
                )

        p = Planning(llm_client=NoParamLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
            available_tools=[
                ToolInfo(name="file_list", description="一覧取得",
                         parameters={"required": []}),
            ],
        ))
        assert len(result.steps) == 1
        assert result.steps[0].action == "file_list"
        assert result.steps[0].params == {}  # 正当な空 params は保持

    def test_params_with_content_are_kept(self):
        """v4.1: 実値のある params は保持される"""

        class GoodParamsLLM(LLMClient):
            def chat(self, prompt, system_prompt=None):
                return (
                    "plan_id: plan_good\n"
                    "steps:\n"
                    "  - order: 1\n"
                    "    action: workspace_write\n"
                    "    params: {\"name\": \"note.md\", \"content\": \"設計メモ\"}\n"
                    "    description: メモ作成\n"
                    "    expected_result: 作成される\n"
                    "    timeout: 10.0\n"
                    "expected_outcome: 完了\n"
                    "estimated_duration: 10.0"
                )

        p = Planning(llm_client=GoodParamsLLM())
        result = p.make(PlanningInput(
            policy=make_personality_output(),
        ))
        assert len(result.steps) == 1
        assert result.steps[0].params.get("content") == "設計メモ"
