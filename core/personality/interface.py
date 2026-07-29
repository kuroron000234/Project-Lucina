from dataclasses import dataclass, field


@dataclass
class PersonalityState:
    name: str
    traits: dict[str, float]
    speaking_style: str
    values: list[str]
    mood: str
    relationship: dict[str, float]


@dataclass
class PersonalityInput:
    drive: "DriveOutput"
    memory: "MemoryOutput"
    long_term_policy: str | None = None
    user_message: str | None = None
    world_predictions: "WorldModelOutput | None" = None


@dataclass
class PersonalityOutput:
    goal: str
    action_policy: str
    priority: int
    conversation_intent: str | None = None
    context_summary: str = ""
    direct_mode: bool = False
    direct_instruction: str = ""


class Personality:
    def decide(self, input: PersonalityInput) -> PersonalityOutput: ...
    def reflect(self, episode: "Episode") -> str: ...
    def speak(self, intent: str) -> str: ...
    def update_state(self, episode: "Episode"): ...
