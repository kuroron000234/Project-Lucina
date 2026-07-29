"""
エージェント層 (Agent) の単体テスト
"""

import tempfile
import os

from core.agent.agent import Agent
from core.agent.interface import AgentInput, StepResult, TOOL_REGISTRY
from core.planning.interface import PlanningOutput, Step


class TestAgent:
    def setup_method(self):
        self.agent = Agent()
        self.tmpdir = tempfile.mkdtemp()

    def _make_plan(self, steps: list[Step]) -> PlanningOutput:
        return PlanningOutput(
            plan_id="test_plan_001",
            steps=steps,
            expected_outcome="テスト計画",
            estimated_duration=30.0,
        )

    # --- 正常系テスト ---

    def test_execute_empty_plan(self):
        """空の計画を実行できる"""
        plan = self._make_plan([])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.overall_success is True
        assert len(result.step_results) == 0

    def test_file_list_tool(self):
        """file_list ツールが動作する"""
        plan = self._make_plan([
            Step(order=1, action="file_list", params={"path": self.tmpdir},
                 description="リスト取得", expected_result="一覧表示"),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert len(result.step_results) == 1
        assert result.step_results[0].success is True
        assert result.overall_success is True

    def test_file_write_and_read(self):
        """file_write → file_read のサイクルが動作する"""
        test_file = os.path.join(self.tmpdir, "test.txt")
        plan = self._make_plan([
            Step(order=1, action="file_write",
                 params={"path": test_file, "content": "Hello, World!"},
                 description="ファイル書き込み", expected_result="書き込み成功"),
            Step(order=2, action="file_read",
                 params={"path": test_file},
                 description="ファイル読み込み", expected_result="読み込み成功"),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.overall_success is True
        assert len(result.step_results) == 2
        assert result.step_results[0].success is True
        assert result.step_results[1].success is True

    def test_command_exec_tool(self):
        """command_exec ツールが動作する"""
        plan = self._make_plan([
            Step(order=1, action="command_exec",
                 params={"command": "echo 'test'"},
                 description="コマンド実行", expected_result="test"),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.step_results[0].success is True
        assert "test" in result.step_results[0].output

    def test_notify_user_tool(self):
        """notify_user ツールが動作する"""
        plan = self._make_plan([
            Step(order=1, action="notify_user",
                 params={"message": "テスト通知"},
                 description="通知", expected_result="通知成功"),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.step_results[0].success is True

    # --- エッジケーステスト ---

    def test_unknown_tool_returns_error(self):
        """未知のツールを呼ぶとエラーになる"""
        plan = self._make_plan([
            Step(order=1, action="unknown_tool", params={},
                 description="未知のツール", expected_result=""),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.step_results[0].success is False
        assert "Unknown" in (result.step_results[0].error or "")

    def test_file_not_found_error(self):
        """存在しないファイルの読み込みでエラーになる"""
        plan = self._make_plan([
            Step(order=1, action="file_read",
                 params={"path": "/nonexistent/file.txt"},
                 description="存在しないファイル", expected_result=""),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.step_results[0].success is False

    def test_partial_success(self):
        """部分成功: 一部のステップが失敗しても全体は success=False"""
        plan = self._make_plan([
            Step(order=1, action="file_list", params={"path": self.tmpdir},
                 description="成功", expected_result=""),
            Step(order=2, action="file_read",
                 params={"path": "/nonexistent"},
                 description="失敗", expected_result=""),
        ])
        result = self.agent.execute(AgentInput(plan=plan))
        assert result.step_results[0].success is True
        assert result.step_results[1].success is False
        assert result.overall_success is False

    def test_speak_output(self):
        """speak() がテキストを返す"""
        text = self.agent.speak("テスト発話")
        assert text == "テスト発話"

    def test_call_tool_valid(self):
        """call_tool() で有効なツールが呼べる"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test")
            fname = f.name
        result = self.agent.call_tool("file_read", {"path": fname})
        assert result.get("success") is True
        os.unlink(fname)

    def test_call_tool_invalid(self):
        """call_tool() で無効なツールはエラー"""
        import pytest
        with pytest.raises(ValueError):
            self.agent.call_tool("nonexistent", {})

    def test_execute_step(self):
        """execute_step() が1ステップを実行する"""
        step = Step(order=1, action="file_list",
                    params={"path": self.tmpdir},
                    description="", expected_result="")
        result = self.agent.execute_step(step)
        assert isinstance(result, StepResult)
        assert result.step_order == 1

    def test_self_modify_requires_task(self):
        """self_modify は task が必須"""
        result = self.agent.call_tool("self_modify", {})
        assert result["success"] is False
        assert "task is required" in (result.get("error") or "")

    def test_direct_execute_requires_instruction(self):
        """direct_execute は instruction が必須"""
        result = self.agent.call_tool("direct_execute", {})
        assert result["success"] is False
        assert "instruction is required" in (result.get("error") or "")

    def test_direct_execute_has_tool(self):
        """direct_execute がツール一覧に存在する"""
        assert "direct_execute" in self.agent.tools

    def test_self_modify_has_tool(self):
        """self_modify がツール一覧に存在する"""
        assert "self_modify" in self.agent.tools
        assert "backup" in self.agent.tools
