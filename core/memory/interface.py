from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Episode:
    id: str
    timestamp: datetime
    event: str
    context: str
    emotion: str
    result: str
    importance: float
    tags: list[str] = field(default_factory=list)


@dataclass
class EpisodeSummary:
    period: tuple[datetime, datetime]
    key_events: list[str] = field(default_factory=list)
    learned_patterns: list[str] = field(default_factory=list)
    importance_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class MemoryInput:
    query: str
    top_k: int = 5
    min_importance: float = 0.0
    time_range: tuple[datetime, datetime] | None = None


@dataclass
class MemoryOutput:
    episodes: list[Episode]
    summary: str
    total_count: int


class Memory:
    def search(self, input: MemoryInput) -> MemoryOutput: ...
    def save(self, episode: Episode) -> str: ...
    def update_importance(self, episode_id: str, new_importance: float): ...
    def summarize(self, episodes: list[Episode]) -> str: ...
    def forget(self, threshold: float = 0.1): ...
    def get_statistics(self) -> dict: ...
