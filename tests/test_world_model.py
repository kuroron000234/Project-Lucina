import pytest
from datetime import datetime
from core.llm import LLMClient
from core.world_model.world_model import WorldModel
from core.world_model.interface import WorldModelInput, WorldModelOutput, Prediction
from environment.interface import EnvironmentOutput, SystemState, NetworkState


class MockWorldModelLLM(LLMClient):
    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "- action: exploration\n"
            "  next_state: 新しいファイルや更新されたファイルが発見される。システム負荷は低い。\n"
            "  probability: 0.8\n"
            "  expected_reward: 0.6\n"
            "  risk_level: low\n"
            "  reasoning: 現在のCPU負荷は低く、ファイル探索に適した状態。\n"
            "- action: rest\n"
            "  next_state: システム状態は変化しない。エネルギーを節約できる。\n"
            "  probability: 0.5\n"
            "  expected_reward: 0.3\n"
            "  risk_level: low\n"
            "  reasoning: 特に負荷がかかっていないため休息の必要性は低い。"
        )


class TestWorldModel:
    def setup_method(self):
        self.wm = WorldModel(llm_client=MockWorldModelLLM())

    def _make_env_state(self):
        return EnvironmentOutput(
            timestamp=datetime.now(),
            user_input=None,
            system_state=SystemState(
                cpu_percent=30.0, memory_percent=50.0,
                active_window="terminal", uptime=1000.0,
                current_directory="/home",
            ),
            files=[],
            network=None,
        )

    def test_predict_returns_predictions(self):
        from core.drive.interface import DriveOutput
        env = self._make_env_state()
        drive = DriveOutput(
            drives={"exploration": 0.7, "social": 0.3, "achievement": 0.4, "rest": 0.2, "maintenance": 0.2},
            primary_drive="exploration",
            drive_tension=0.2,
            novelty_score=0.5,
        )
        result = self.wm.predict(WorldModelInput(
            environment=env,
            drive=drive,
            active_goal="explore the workspace",
        ))
        assert isinstance(result, WorldModelOutput)
        assert len(result.predictions) > 0
        for p in result.predictions:
            assert p.action
            assert p.next_state
            assert 0.0 <= p.probability <= 1.0
            assert p.risk_level in ("low", "medium", "high")

    def test_confidence_default(self):
        assert 0.0 <= self.wm.confidence("state_x", "action_y") <= 1.0

    def test_update(self):
        env = self._make_env_state()
        from core.drive.interface import DriveOutput
        drive = DriveOutput(
            drives={"exploration": 0.5, "social": 0.3, "achievement": 0.4, "rest": 0.2, "maintenance": 0.2},
            primary_drive="exploration",
            drive_tension=0.15,
            novelty_score=0.3,
        )
        result = self.wm.predict(WorldModelInput(
            environment=env, drive=drive, active_goal="test"
        ))
        if result.predictions:
            from core.memory.interface import Episode
            ep = Episode(
                id="test", timestamp=datetime.now(),
                event="test", context="", emotion="",
                result="ok", importance=0.5,
            )
            self.wm.update(actual=ep, prediction=result.predictions[0])
