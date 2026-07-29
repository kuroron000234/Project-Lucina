from dataclasses import dataclass, field


DRIVE_DEFINITIONS: dict = {
    "exploration": {
        "label": "探索欲求",
        "triggers": ["環境変化", "低刺激", "未踏領域"],
        "satiation": 0.3,
    },
    "social": {
        "label": "社会欲求",
        "triggers": ["長時間孤独", "共有したい出来事"],
        "satiation": 0.3,
    },
    "achievement": {
        "label": "達成欲求",
        "triggers": ["未完了タスク", "スキル向上機会"],
        "satiation": 0.4,
    },
    "rest": {
        "label": "休息欲求",
        "triggers": ["高負荷継続", "エラー多発"],
        "satiation": 0.2,
    },
    "maintenance": {
        "label": "メンテナンス欲求",
        "triggers": ["設定不備", "メモリ散乱"],
        "satiation": 0.2,
    },
}


@dataclass
class DriveInput:
    environment: "EnvironmentOutput"
    memory_summary: str
    adjustments: dict[str, float] | None = None


@dataclass
class DriveOutput:
    drives: dict[str, float]
    primary_drive: str
    drive_tension: float
    novelty_score: float


class Drive:
    def generate(self, input: DriveInput) -> DriveOutput: ...
    def update_parameters(self, adjustments: dict[str, float]): ...
    def get_drive_profile(self) -> dict: ...
