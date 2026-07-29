from dataclasses import dataclass


@dataclass
class EvaluationScore:
    goal_achievement: float
    efficiency: float
    correctness: float
    novelty: float
    overall: float


@dataclass
class EvaluationInput:
    goal: str
    action_result: "AgentOutput"
    expected_outcome: str
    episode: "Episode"


@dataclass
class EvaluationOutput:
    score: EvaluationScore
    discrepancy: str
    improvement_suggestion: str


class Evaluation:
    def evaluate(self, input: EvaluationInput) -> EvaluationOutput: ...
    def compare(self, actual: EvaluationScore, expected: EvaluationScore) -> str: ...
    def get_history(self, period: str = "all") -> list[EvaluationScore]: ...
