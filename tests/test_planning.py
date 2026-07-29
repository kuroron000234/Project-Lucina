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
