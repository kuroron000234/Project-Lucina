from dataclasses import dataclass, field


@dataclass
class Step:
    order: int
    action: str
    params: dict
    description: str
    expected_result: str
    fallback: str | None = None
    timeout: float = 30.0


@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, type]
    examples: list[str] = field(default_factory=list)


@dataclass
class PlanningInput:
    policy: "PersonalityOutput"
    world_model_predictions: list["Prediction"] | None = None
    available_tools: list[ToolInfo] | None = None


@dataclass
class PlanningOutput:
    plan_id: str
    steps: list[Step]
    expected_outcome: str
    fallback_plan: list[Step] | None = None
    estimated_duration: float = 0.0


class Planning:
    def make(self, input: PlanningInput) -> PlanningOutput: ...
    def revise(self, plan_id: str, failed_step: int, feedback: str) -> PlanningOutput: ...
    def estimate_duration(self, plan: PlanningOutput) -> float: ...
