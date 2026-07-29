import pytest
from datetime import datetime
from environment.environment import Environment
from environment.interface import EnvironmentInput, ActionResult


class TestEnvironment:
    def setup_method(self):
        self.env = Environment()

    def test_observe_returns_output(self):
        result = self.env.observe(EnvironmentInput(trigger="startup"))
        assert result.timestamp is not None
        assert result.system_state is not None
        assert result.system_state.cpu_percent >= 0
        assert result.system_state.memory_percent >= 0
        assert result.system_state.current_directory != ""

    def test_observe_with_user_message(self):
        result = self.env.observe(EnvironmentInput(
            trigger="user_interrupt", user_message="hello"
        ))
        assert result.user_input == "hello"

    def test_observe_without_user_message(self):
        result = self.env.observe(EnvironmentInput(trigger="periodic"))
        assert result.user_input is None

    def test_execute_action_unknown(self):
        result = self.env.execute_action("nonexistent", {})
        assert result.success is False
        assert "Unknown action" in (result.error or "")

    def test_execute_action_file_list(self):
        result = self.env.execute_action("file_list", {"path": "."})
        assert result.success is True
        assert isinstance(result.output, str)

    def test_observe_network_state(self):
        result = self.env.observe(EnvironmentInput(trigger="periodic"))
        assert result.network is not None
