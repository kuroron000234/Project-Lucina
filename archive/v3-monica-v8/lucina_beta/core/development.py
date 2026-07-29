"""Development: 発達・成長カリキュラム

各 Phase の機能を「最初から全て利用可能」にするのではなく、
発達カリキュラムに従って段階的に解放する。

成長は Skill Level だけではない。
認知能力の成長、価値観の変化、関係の変化、自己認識の変化。
"""


class Development:
    """発達段階の管理と能力の段階的解放。

    各発達段階で利用可能な能力を制御する。
    """

    # 発達段階の定義
    STAGES = {
        0: {
            "name": "Infant",
            "description": "World learning only",
            "capabilities": ["learn_world"],
        },
        1: {
            "name": "Child",
            "description": "Internal state awareness",
            "capabilities": ["learn_world", "internal_state"],
        },
        2: {
            "name": "Adolescent",
            "description": "Active inference (EFE)",
            "capabilities": ["learn_world", "internal_state", "efe"],
        },
        3: {
            "name": "Young Adult",
            "description": "LLM cognitive abilities",
            "capabilities": ["learn_world", "internal_state", "efe", "llm"],
        },
        4: {
            "name": "Adult",
            "description": "Memory and self-model",
            "capabilities": ["learn_world", "internal_state", "efe", "llm",
                             "memory", "self_model"],
        },
        5: {
            "name": "Social Adult",
            "description": "Other model and relationships",
            "capabilities": ["learn_world", "internal_state", "efe", "llm",
                             "memory", "self_model", "other_model", "relationship"],
        },
        6: {
            "name": "Mature",
            "description": "Full cognitive suite",
            "capabilities": ["learn_world", "internal_state", "efe", "llm",
                             "memory", "self_model", "other_model", "relationship",
                             "values", "identity", "metacognition"],
        },
        7: {
            "name": "Enlightened",
            "description": "Meta-cognition and autonomous agency",
            "capabilities": ["learn_world", "internal_state", "efe", "llm",
                             "memory", "self_model", "other_model", "relationship",
                             "values", "identity", "metacognition",
                             "autonomy", "meta_world"],
        },
    }

    def __init__(self, initial_stage: int = 0):
        self.current_stage = initial_stage
        self._experience_points = 0
        self._stage_thresholds = {
            0: 0,
            1: 30,    # 30 experiences — Monicaが早めに成長できるように
            2: 100,
            3: 200,
            4: 400,
            5: 700,
            6: 1200,
            7: 2000,
        }

    @property
    def name(self) -> str:
        return self.STAGES[self.current_stage]["name"]

    @property
    def description(self) -> str:
        return self.STAGES[self.current_stage]["description"]

    @property
    def capabilities(self) -> list[str]:
        return list(self.STAGES[self.current_stage]["capabilities"])

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def add_experience(self, n: int = 1) -> None:
        """経験値を追加し、条件を満たせば発達する。"""
        self._experience_points += n
        self._check_development()

    def _check_development(self) -> bool:
        """発達条件をチェックし、条件を満たす最高段階に設定する。"""
        advanced = False
        for stage, threshold in sorted(self._stage_thresholds.items()):
            if stage > self.current_stage and self._experience_points >= threshold:
                self.current_stage = stage
                advanced = True
        return advanced

    @property
    def experience_to_next(self) -> int:
        """次の発達段階までに必要な経験値。"""
        next_stage = self.current_stage + 1
        if next_stage not in self._stage_thresholds:
            return 0
        needed = self._stage_thresholds[next_stage]
        return max(0, needed - self._experience_points)

    @property
    def total_experience(self) -> int:
        return self._experience_points

    def summary(self) -> dict:
        return {
            "stage": self.current_stage,
            "name": self.name,
            "description": self.description,
            "experience": self._experience_points,
            "to_next": self.experience_to_next,
            "capabilities": self.capabilities,
        }
