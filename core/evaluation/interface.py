from dataclasses import dataclass


@dataclass
class EvaluationScore:
    goal_achievement: float
    efficiency: float
    correctness: float
    novelty: float
    overall: float
    # v3.2: 評価レジームのタグ（学習層が同一タイプ内で統計を取るため）
    eval_type: str = "rule"      # "llm" | "rule"
    source: str = "autonomous"   # "dialog" | "autonomous"


@dataclass
class EvaluationInput:
    goal: str
    action_result: "AgentOutput"
    expected_outcome: str
    episode: "Episode"
    # v3.2: tier2ではLLM評価をスキップしルールベースのみ使用（コスト抑制）
    use_llm: bool = True


@dataclass
class EvaluationOutput:
    score: EvaluationScore
    discrepancy: str
    improvement_suggestion: str


class Evaluation:
    def evaluate(self, input: EvaluationInput) -> EvaluationOutput: ...
    def compare(self, actual: EvaluationScore, expected: EvaluationScore) -> str: ...
    def get_history(self, period: str = "all") -> list[EvaluationScore]: ...
