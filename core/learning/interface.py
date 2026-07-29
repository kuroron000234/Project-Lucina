from dataclasses import dataclass, field


@dataclass
class LearningInput:
    evaluation: "EvaluationOutput"
    evaluation_history: list["EvaluationScore"]
    drive_snapshot: "DriveOutput"
    episode_id: str


@dataclass
class LearningOutput:
    drive_adjustments: dict[str, float]
    memory_importance_update: float
    personality_adjustments: dict | None = None
    learning_summary: str = ""


class Learning:
    def learn(self, input: LearningInput) -> LearningOutput: ...
    def adjust_drive_parameters(self, history: list) -> dict[str, float]: ...
    def get_learning_curve(self) -> list[float]: ...
