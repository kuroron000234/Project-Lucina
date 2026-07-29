from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Routine:
    name: str
    action: str
    frequency: str
    interval_hours: float | None = None
    last_executed: datetime | None = None
    enabled: bool = True


@dataclass
class LongTermPlanningInput:
    evaluation_history: list["EvaluationScore"]
    current_date: datetime
    personality_state: "PersonalityState"
    recent_episodes_summary: str


@dataclass
class LongTermPlanningOutput:
    long_term_goal: str
    routines: list[Routine]
    identity_policy: str
    focus_area: str
    reflection: str


class LongTermPlanning:
    def plan(self, input: LongTermPlanningInput) -> LongTermPlanningOutput: ...
    def generate_routines(self, personality: "PersonalityState") -> list[Routine]: ...
    def review_period(self, days: int) -> str: ...
    def update_goal_progress(self, goal: str, progress: float): ...
