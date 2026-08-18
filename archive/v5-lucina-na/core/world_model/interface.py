from dataclasses import dataclass, field


@dataclass
class Prediction:
    action: str
    next_state: str
    probability: float
    expected_reward: float
    risk_level: str
    reasoning: str
    # v5.0: Phase 3 — 予測の不確実性 σ（サプライズ計算 S=(x−μ)²/σ²+ln σ に使用）
    uncertainty: float = 0.3


@dataclass
class WorldModelInput:
    environment: "EnvironmentOutput"
    drive: "DriveOutput"
    active_goal: str
    candidate_action: str | None = None  # derived from primary_drive if not set
    # v3.2: tier2ではLLMシミュレーションをスキップしルールベース予測のみ（コスト抑制）
    use_llm: bool = True


@dataclass
class WorldModelOutput:
    predictions: list[Prediction]


@dataclass
class ImaginedFuture:
    """
    v4.0: 想像された未来の候補。

    世界モデルが「もし◯◯したらどうなる？」を複数生成し、
    人格層が自分の好み（期待報酬）との一致度で選択する。
    アクティブ推論における「好ましい未来の事前分布」に相当する。
    """
    action: str        # 想像上の行動（例: 「自作言語のインタプリタを作る」）
    next_state: str    # その結果どうなるか
    preference: float  # どれだけ自分がそれを望むか (0.0-1.0)
    reasoning: str = ""  # なぜそれを望むか


class WorldModel:
    def predict(self, input: WorldModelInput) -> WorldModelOutput: ...
    def simulate(self, state: "EnvironmentOutput", plan: "PlanningOutput") -> list[Prediction]: ...
    def update(self, actual: "Episode", prediction: Prediction): ...
    def confidence(self, state: str, action: str) -> float: ...
    def imagine(self, input: WorldModelInput) -> list[ImaginedFuture]: ...
