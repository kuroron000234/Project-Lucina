from dataclasses import dataclass, field


@dataclass
class LearningInput:
    evaluation: "EvaluationOutput"
    evaluation_history: list["EvaluationScore"]
    drive_snapshot: "DriveOutput"
    episode_id: str
    # v3.2: ゼロサム・クレジット割り当て用
    driving_drive: str | None = None   # 行動を選んだ駆動（Noneならprimary_drive使用）
    source: str = "autonomous"         # "dialog" | "autonomous"
    # v5.0: Phase 3 — 実測サプライズ（正規化済み 0.0〜1.0）。学習率を変調する
    surprise: float | None = None


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
