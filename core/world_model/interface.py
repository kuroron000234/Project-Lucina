from dataclasses import dataclass, field


@dataclass
class Prediction:
    action: str
    next_state: str
    probability: float
    expected_reward: float
    risk_level: str
    reasoning: str


@dataclass
class WorldModelInput:
    environment: "EnvironmentOutput"
    drive: "DriveOutput"
    active_goal: str
    candidate_action: str | None = None  # derived from primary_drive if not set


@dataclass
class WorldModelOutput:
    predictions: list[Prediction]


class WorldModel:
    def predict(self, input: WorldModelInput) -> WorldModelOutput: ...
    def simulate(self, state: "EnvironmentOutput", plan: "PlanningOutput") -> list[Prediction]: ...
    def update(self, actual: "Episode", prediction: Prediction): ...
    def confidence(self, state: str, action: str) -> float: ...
