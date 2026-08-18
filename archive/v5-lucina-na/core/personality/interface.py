from dataclasses import dataclass, field


@dataclass
class PersonalityState:
    name: str
    traits: dict[str, float]
    speaking_style: str
    values: list[str]
    mood: str
    relationship: dict[str, float]
    # v3.4: 自己モデル — 自身の記憶・評価履歴・長期計画を参照して生成した
    # 「私は◯◯な存在」という自己認識文（decide/speak のプロンプトに注入される）
    self_model: str = ""
    self_model_updated: float = 0.0


@dataclass
class PersonalityInput:
    drive: "DriveOutput"
    memory: "MemoryOutput"
    long_term_policy: str | None = None
    user_message: str | None = None
    world_predictions: "WorldModelOutput | None" = None
    # v3.5: 直前の会話ターン（[{"role": "user"/"assistant", "text": ...}]）。
    # WebUI が保持した会話履歴を LLM に渡し、前の会話を参照できるようにする。
    conversation_history: list | None = None
    # v4.0: 意志フェーズ
    # 願望（自分がやってみたいこと）— メニューから選ばず、これから自分で目標を生成する
    aspirations: list | None = None
    # 想像された未来候補（世界モデルが生成）— 好みとの一致度で行動を選ぶ
    imagined_futures: list | None = None
    # 自分の部屋（自由に使えるワークスペースのパス）
    workspace_hint: str = ""
    # v5.0: Phase 3 — 直近の実測サプライズ（予測誤差 0.0〜1.0）。
    # 行動選択のバイアスとしてプロンプトに注入する（能動的推論: 高サプライズ=探索）
    surprise: float | None = None


@dataclass
class PersonalityOutput:
    goal: str
    action_policy: str
    priority: int
    conversation_intent: str | None = None
    context_summary: str = ""
    direct_mode: bool = False
    direct_instruction: str = ""
    # v4.0: 意志フェーズ
    # 内言（この決定に至った「なぜ」の独白）
    inner_monologue: str = ""
    # 拒否（休息欲求・不機嫌時に理由付きで先延ばしを提案する）
    refusal: bool = False
    refusal_reason: str = ""


class Personality:
    def decide(self, input: PersonalityInput) -> PersonalityOutput: ...
    def reflect(self, episode: "Episode") -> str: ...
    def speak(self, intent: str) -> str: ...
    def update_state(self, episode: "Episode"): ...
