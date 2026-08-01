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

    # --- v4.0: 意志フェーズ — 自分の部屋 ---

    def test_workspace_write_creates_file(self):
        """workspace_write が自分の部屋にファイルを作る"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            result = self.agent.call_tool("workspace_write", {
                "name": "test_note.md",
                "content": "# ノート\n実験メモ",
            })
            assert result["success"] is True
            assert "test_note.md" in result["output"]
            assert os.path.exists(os.path.join(config.WILL_CONFIG["workspace_dir"], "test_note.md"))
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_workspace_write_with_subdir(self):
        """workspace_write がサブディレクトリにも書ける"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            result = self.agent.call_tool("workspace_write", {
                "name": "poem.md", "content": "poem", "subdir": "poems",
            })
            assert result["success"] is True
            assert os.path.exists(os.path.join(config.WILL_CONFIG["workspace_dir"], "poems", "poem.md"))
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_workspace_write_path_traversal_blocked(self):
        """パストラバーサルは防止される"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            result = self.agent.call_tool("workspace_write", {
                "name": "../../evil.md", "content": "x",
            })
            assert result["success"] is True
            assert ".." not in result["output"]
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_workspace_list(self):
        """workspace_list がファイル一覧を返す"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            self.agent.call_tool("workspace_write", {"name": "a.md", "content": "x"})
            result = self.agent.call_tool("workspace_list", {})
            assert result["success"] is True
            assert "a.md" in result["output"]
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    # --- v4.1: 自己検証・空パラメータガード ---

    def test_workspace_write_empty_content_fails(self):
        """v4.1: 空内容の workspace_write は失敗する（0バイトゴミ防止）"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            result = self.agent.call_tool("workspace_write", {"name": "x.md", "content": ""})
            assert result["success"] is False
            assert "content is required" in (result.get("error") or "")
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_workspace_write_file_name_alias(self):
        """v4.1.1: file_name / filename キーでもファイル名が指定できる"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            result = self.agent.call_tool("workspace_write", {
                "file_name": "causal_sim.py", "content": "# sim"})
            assert result["success"] is True
            assert "causal_sim.py" in result["output"]
            assert os.path.exists(
                os.path.join(config.WILL_CONFIG["workspace_dir"], "causal_sim.py"))
            # filename キーも同様に動作
            result2 = self.agent.call_tool("workspace_write", {
                "filename": "note2.md", "content": "note"})
            assert result2["success"] is True
            assert "note2.md" in result2["output"]
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_workspace_write_empty_content_via_execute_fails(self):
        """v4.1: 空内容のステップは execute 全体で失敗になる"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            plan = self._make_plan([
                Step(order=1, action="workspace_write",
                     params={"name": "x.md", "content": ""},
                     description="メモ作成", expected_result="作成される"),
            ])
            result = self.agent.execute(AgentInput(plan=plan))
            assert result.overall_success is False
            assert result.step_results[0].success is False
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_zero_byte_write_converted_to_failure(self):
        """v4.1: 0バイト書き込みは自己検証で失敗に転換される"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            # file_write は空でも書き込めてしまうため、自己検証が失敗に転換する
            f = os.path.join(self.tmpdir, "empty.txt")
            plan = self._make_plan([
                Step(order=1, action="file_write",
                     params={"path": f, "content": ""},
                     description="空書き込み", expected_result=""),
            ])
            result = self.agent.execute(AgentInput(plan=plan))
            assert result.overall_success is False
            assert result.step_results[0].success is False
            assert "0 bytes" in (result.step_results[0].error or "")
        finally:
            config.WILL_CONFIG["workspace_dir"] = old

    def test_ten_byte_write_not_falsely_flagged(self):
        """v4.1: 10バイト等の正当な書き込みは0バイトと誤検出されない"""
        # 部分一致（"0 bytes" in output）だと "10 bytes" に誤マッチする。
        # 正規表現でバイト数を抽出する実装を検証する。
        assert self.agent._wrote_zero_bytes("Written 0 bytes to /tmp/x") is True
        assert self.agent._wrote_zero_bytes("Written 10 bytes to /tmp/x") is False
        assert self.agent._wrote_zero_bytes("Written 20 bytes to /tmp/x") is False
        assert self.agent._wrote_zero_bytes(
            "Created in your room: x.md (0 bytes)") is True
        assert self.agent._wrote_zero_bytes(
            "Created in your room: x.md (13 bytes)") is False
        assert self.agent._wrote_zero_bytes("") is False
        assert self.agent._wrote_zero_bytes(None) is False

    def test_normal_write_stays_successful(self):
        """v4.1: 通常サイズの書き込みは自己検証後も成功のまま"""
        import config
        import tempfile
        old = config.WILL_CONFIG.get("workspace_dir")
        config.WILL_CONFIG["workspace_dir"] = tempfile.mkdtemp()
        try:
            f = os.path.join(self.tmpdir, "ok.txt")
            plan = self._make_plan([
                Step(order=1, action="file_write",
                     params={"path": f, "content": "0123456789"},  # 10 bytes
                     description="10バイト書き込み", expected_result=""),
            ])
            result = self.agent.execute(AgentInput(plan=plan))
            assert result.overall_success is True
            assert result.step_results[0].success is True
        finally:
            config.WILL_CONFIG["workspace_dir"] = old
