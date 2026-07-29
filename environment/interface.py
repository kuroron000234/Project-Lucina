from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EnvironmentInput:
    trigger: str
    user_message: str | None = None


@dataclass
class SystemState:
    cpu_percent: float
    memory_percent: float
    active_window: str | None
    uptime: float
    current_directory: str


@dataclass
class FileInfo:
    path: str
    name: str
    size: int
    modified: datetime
    type: str


@dataclass
class NetworkState:
    is_connected: bool
    ip_address: str | None
    signal_strength: float | None


@dataclass
class ActionResult:
    success: bool
    output: str
    error: str | None = None
    duration: float = 0.0
    side_effects: dict | None = None


@dataclass
class EnvironmentOutput:
    timestamp: datetime
    user_input: str | None
    system_state: SystemState
    files: list[FileInfo] = field(default_factory=list)
    network: NetworkState | None = None
    sensors: dict[str, float] = field(default_factory=dict)


class Environment:
    def observe(self, input: EnvironmentInput) -> EnvironmentOutput: ...
    def execute_action(self, action: str, params: dict) -> ActionResult: ...
