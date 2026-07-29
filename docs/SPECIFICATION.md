# lucina-NA 完全実装仕様書

> このドキュメントを読めば、あなたは lucina-NA をゼロから完成まで実装できる。
> 各層の「何を」「どう」「なぜ」が定義されており、実装・テスト・評価・運用のサイクルを自力で回せる。

---

## 目次

1. [アーキテクチャ哲学](#1-アーキテクチャ哲学)
2. [全10層インターフェース仕様](#2-全10層インターフェース仕様)
3. [データ構造リファレンス](#3-データ構造リファレンス)
4. [メインループ設計](#4-メインループ設計)
5. [実装手順（Phase 1）](#5-実装手順phase-1)
6. [実装手順（Phase 2）](#6-実装手順phase-2)
7. [テストガイド](#7-テストガイド)
8. [評価ガイド](#8-評価ガイド)
9. [運用ガイド](#9-運用ガイド)
10. [付録：LLMプロンプトテンプレート](#10-付録llmプロンプトテンプレート)

---

## 1. アーキテクチャ哲学

### 1.1 設計思想

このシステムは「単一のLLMに賢く振る舞えと丸投げする」のではなく、**人間の認知機能を役割ごとに層に分離**し、それぞれを専門のモジュールに担当させる。

各層は：
- **独立して差し替え可能**（例：LLMベースの世界モデル → ニューラルネットベースに交換）
- **疎結合**（インターフェース経由でのみ通信）
- **単一責務**（1つの層は1つのことだけをする）

### 1.2 理論的基盤

このアーキテクチャは**自由エネルギー原理（FEP）/ 能動的推論**に触発されている。

すなわち：
```
エージェント = 「内部状態と目標状態のズレ（自由エネルギー）を
                最小化するように行動する」存在
```

- **駆動層** = ズレ（動機）を生み出す
- **行動計画層+エージェント層** = ズレを解消する行動を生成・実行
- **評価層+学習層** = ズレが実際に解消されたかを測定し、パラメータ調整
- **世界モデル層** = 行動前にズレの変化をシミュレーション

### 1.3 3つの時間スケール

このアーキテクチャの最大の特徴は、異なる時間スケールで動く**3つのループが並列動作**すること。

| ループ | 周期 | 経路 | 対応する認知機能 |
|---|---|---|---|
| **反射ループ** | 毎ターン（同期待機） | 環境→駆動→人格→計画→エージェント | 反射行動・即応 |
| **学習ループ** | 行動完了後（非同期） | 評価↔学習 | 強化学習的な習慣形成 |
| **一貫性ループ** | 数時間〜日単位（定期） | 長期計画→人格、長期計画↔評価 | 自己概念の維持 |

これらは干渉せず、異なるレートで同時に動作する。

### 1.4 「決める」層と「支える」層

| 色 | カテゴリ | 層 | 役割 |
|---|---|---|---|
| **紫** | 意思決定 | 駆動・人格・行動計画・長期行動計画・世界モデル | 「何をするか」を能動的に決める |
| **ティール** | データ処理 | 記憶・評価・学習 | 意思決定をデータで支える |
| **グレー** | 入出力接点 | 環境・エージェント | システム外部との境界 |

**人格層は意図的に軽量**に設計されている。判断の複雑さは行動計画層や長期行動計画層、世界モデル層が分担する。これによりLoRAなどによるキャラクター性の学習を人格層だけに絞り込める。

### 1.5 世界モデル = 内的シミュレーション

世界モデル層は「こうしたらどうなるだろう」という**内的シミュレーション**を担当する。

行動計画層が手順を作る前に世界モデル層と対話することで：
- 実際に行動する前に結果を予測
- 毎回試行錯誤しなくて済む
- 学習コストを下げられる

---

## 2. 全10層インターフェース仕様

### 2.1 環境層 (Environment)

**責務**: PCやユーザーから現在の状態を取得する。システムの最上流。

#### 入力

```python
@dataclass
class EnvironmentInput:
    """環境層への入力。外部トリガー（起動・定期タイマー・ユーザー割り込み）。"""
    trigger: str  # "startup" | "periodic" | "user_interrupt"
    user_message: str | None = None  # ユーザーからの直接入力
```

#### 出力

```python
@dataclass
class EnvironmentOutput:
    """観測された環境状態。これが全層の起点データとなる。"""
    timestamp: datetime
    user_input: str | None
    system_state: SystemState
    files: list[FileInfo]
    network: NetworkState | None
    sensors: dict[str, float]  # 拡張センサー値

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
    type: str  # "file" | "directory"

@dataclass
class NetworkState:
    is_connected: bool
    ip_address: str | None
    signal_strength: float | None
```

#### 公開メソッド

```python
class Environment:
    def observe(self, input: EnvironmentInput) -> EnvironmentOutput
        """現在の環境状態を取得する。このメソッドが全処理の起点。"""

    def execute_action(self, action: str, params: dict) -> ActionResult
        """エージェント層からの行動依頼を実際にOSレベルで実行する。"""
```

#### 依存関係

- 外部依存: OSのシステムコール（psutil, os, subprocess）
- 内部依存: なし（最上流）

#### エッジケース

- ユーザー入力がない場合: `user_input = None` として定期観測
- センサー取得失敗: 該当フィールドを None / デフォルト値に
- 初回起動時: `trigger = "startup"` で特別な初期化シーケンス

---

### 2.2 記憶層 (Memory)

**責務**: エピソード記憶の保存・検索・要約を行う。経験のデータベース。

#### 入力

```python
@dataclass
class MemoryInput:
    """記憶層への検索クエリ。人格層からの呼び出しが主。"""
    query: str                       # 検索クエリ（自然言語）
    top_k: int = 5                   # 取得件数
    min_importance: float = 0.0      # 最低重要度フィルタ
    time_range: tuple[datetime, datetime] | None = None  # 時間範囲
```

#### 出力

```python
@dataclass
class MemoryOutput:
    """記憶検索結果。人格層が方針決定に使う。"""
    episodes: list[Episode]
    summary: str                    # 検索結果の自然言語要約
    total_count: int                # 合計エピソード数
```

#### 内部データ構造

```python
@dataclass
class Episode:
    """1つのエピソード記憶。これがシステムの基本単位。"""
    id: str
    timestamp: datetime
    event: str                       # 出来事の記述
    context: str                     # 当時の状況
    emotion: str                     # その時の感情/価値
    result: str                      # 結果どうなったか
    importance: float                # 重要度 0.0〜1.0（学習層が更新）
    tags: list[str]                  # 検索用タグ

@dataclass
class EpisodeSummary:
    """記憶の集約表現。長期保存用。"""
    period: tuple[datetime, datetime]
    key_events: list[str]
    learned_patterns: list[str]
    importance_distribution: dict[str, int]
```

#### 公開メソッド

```python
class Memory:
    def search(self, input: MemoryInput) -> MemoryOutput
        """クエリに基づいてエピソードを検索し、要約とともに返す。"""

    def save(self, episode: Episode) -> str
        """新しいエピソードを保存する。戻り値はエピソードID。"""

    def update_importance(self, episode_id: str, new_importance: float)
        """エピソードの重要度を更新する（学習層から呼ばれる）。"""

    def summarize(self, episodes: list[Episode]) -> str
        """エピソードリストを自然言語で要約する。"""

    def forget(self, threshold: float = 0.1)
        """重要度が閾値以下のエピソードを削除する（メモリ節約）。"""

    def get_statistics(self) -> dict
        """記憶の統計情報を返す（デバッグ/評価用）。"""
```

#### 依存関係

- 保存先: ファイル（JSON/SQLite）
- 検索方式: Phase 1 = キーワード + 日時ソート、Phase 2 = ベクトル埋め込み

#### エッジケース

- エピソード0件: `episodes=[]`, `summary="まだ記憶がありません"`
- 重要度一様: すべてのエピソードが同じ重要度の場合、時系列順
- 保存失敗: ファイル書き込みエラー時の再試行ロジック

---

### 2.3 駆動層 (Drive)

**責務**: 生物的な欲求（探索・休息・社会・達成）の優先度を生成する。
FEPにおける「予測誤差」に相当する信号を出力する。

#### 入力

```python
@dataclass
class DriveInput:
    """駆動層への入力。環境状態と記憶要約から動機を生成。"""
    environment: EnvironmentOutput    # 現在の環境状態
    memory_summary: str              # 記憶層からの要約
    adjustments: dict[str, float] | None = None  # 学習層からの調整値
```

#### 出力

```python
@dataclass
class DriveOutput:
    """現在の駆動（動機）状態。人格層が方針決定に使う。"""
    drives: dict[str, float]         # {"exploration": 0.8, "rest": 0.2, ...}
    primary_drive: str               # 最も強い欲求名
    drive_tension: float             # 全駆動の総合的な緊張度 0.0〜1.0
    novelty_score: float             # 環境の新奇性スコア
```

#### 基本駆動の定義

```python
DRIVE_DEFINITIONS = {
    "exploration": {
        "label": "探索欲求",
        "description": "新しい情報・経験を求める",
        "triggers": ["環境変化", "低刺激", "未踏領域"],
        "satiation": 0.3  # 満たされたとみなす閾値
    },
    "social": {
        "label": "社会欲求",
        "description": "ユーザーや他者との交流を求める",
        "triggers": ["長時間孤独", "重要な出来事の共有"],
        "satiation": 0.3
    },
    "achievement": {
        "label": "達成欲求",
        "description": "目標を完了し成長を感じたい",
        "triggers": ["未完了タスク", "スキル向上機会"],
        "satiation": 0.4
    },
    "rest": {
        "label": "休息欲求",
        "description": "過負荷を避けエネルギーを回復したい",
        "triggers": ["高負荷継続", "エラー多発"],
        "satiation": 0.2
    },
    "maintenance": {
        "label": "メンテナンス欲求",
        "description": "自分自身を整理し最適化したい",
        "triggers": ["設定不備", "メモリ散乱", "長期間未整理"],
        "satiation": 0.2
    }
}
```

#### 公開メソッド

```python
class Drive:
    def generate(self, input: DriveInput) -> DriveOutput
        """現在の内部・外部状態から駆動状態を生成する。"""

    def update_parameters(self, adjustments: dict[str, float])
        """学習層からのフィードバックで駆動パラメータを調整する。"""

    def get_drive_profile(self) -> dict
        """現在の駆動プロファイルを返す（デバッグ用）。"""
```

#### 依存関係

- 環境層からの出力（`EnvironmentOutput`）
- 記憶層からの要約（`str`）
- 学習層からの調整値（`dict`, Phase 2）

#### エッジケース

- 全駆動が低い: デフォルトで exploration を primary に
- 駆動が拮抗: ランダム要素を加えてバランスを崩す
- 外部から強制駆動: `adjustments` で特定駆動を強制上昇

---

### 2.4 人格層 (Personality)

**責務**: 自然言語で「今何をするか」の方針を決定する。
この層は**意図的に軽量**に設計され、判断の複雑さは他の層に委譲する。

#### 入力

```python
@dataclass
class PersonalityInput:
    """人格層への入力。集約された情報から方針を決定。"""
    drive: DriveOutput               # 現在の動機
    memory: MemoryOutput             # 関連する過去の経験
    long_term_policy: str | None = None  # 長期行動計画層からの方針
    user_message: str | None = None      # ユーザーからの直接メッセージ
```

#### 出力

```python
@dataclass
class PersonalityOutput:
    """人格層の決定。方針・目標・発話意図を含む。"""
    goal: str                        # 「〜をする」形式の目標
    action_policy: str               # 行動方針（自然語言語）
    priority: int                    # 緊急度 1-5（5が最優先）
    conversation_intent: str | None  # 発話する場合の意図
    context_summary: str             # 今回の判断根拠の要約（評価層に渡す）
```

#### 内部状態（永続化対象）

```python
@dataclass
class PersonalityState:
    """人格層の永続状態。キャラクター性の中核。"""
    name: str
    traits: dict[str, float]         # ["curiosity": 0.8, "caution": 0.3, ...]
    speaking_style: str              # 話し方の特徴
    values: list[str]                # 価値観リスト
    mood: str                        # 現在のムード
    relationship: dict[str, float]  # ユーザーとの関係性指標
```

#### 公開メソッド

```python
class Personality:
    def decide(self, input: PersonalityInput) -> PersonalityOutput
        """入力から方針を決定する。メインループから毎ターン呼ばれる。"""

    def reflect(self, episode: Episode) -> str
        """行われた行動を内省し、感想・学びをテキストで返す。"""

    def speak(self, intent: str) -> str
        """発話意図から実際の発話文を生成する。"""

    def update_state(self, episode: Episode)
        """行動結果から人格状態を更新する（学習ループの一部）。"""
```

#### 依存関係

- 駆動層（`DriveOutput`）
- 記憶層（`MemoryOutput`）
- 長期行動計画層（`str`, Phase 2）

#### エッジケース

- 矛盾した入力（高い探索欲求 + 疲労状態）: 人格層が優先度判断
- ユーザーからの直接指示: `user_message` を最優先
- 長期方針と短期駆動の衝突: 長期方針を基本としつつ、緊急度で判断

---

### 2.5 行動計画層 (Planning)

**責務**: 人格層の方針を、実行可能な具体的な手順へ分解する。

#### 入力

```python
@dataclass
class PlanningInput:
    """行動計画層への入力。方針を具体的手順に分解。"""
    policy: PersonalityOutput         # 人格層の方針
    world_model_predictions: list[Prediction] | None = None  # 世界モデルからの予測
    available_tools: list[ToolInfo] | None = None  # 利用可能なツール一覧
```

#### 出力

```python
@dataclass
class PlanningOutput:
    """行動計画。エージェント層が実行する手順のリスト。"""
    plan_id: str
    steps: list[Step]                # 実行手順
    expected_outcome: str            # 期待される結果
    fallback_plan: list[Step] | None # 代替案
    estimated_duration: float        # 推定所要時間（秒）

@dataclass
class Step:
    """1つの実行可能な手順。"""
    order: int
    action: str                      # アクション名
    params: dict                     # パラメータ
    description: str                 # 自然語言語での説明
    expected_result: str             # この手順で期待される結果
    fallback: str | None = None      # 失敗時の代替行動
    timeout: float = 30.0            # タイムアウト（秒）

@dataclass
class ToolInfo:
    """エージェントが利用可能なツールの情報。"""
    name: str
    description: str
    parameters: dict[str, type]
    examples: list[str]
```

#### 公開メソッド

```python
class Planning:
    def make(self, input: PlanningInput) -> PlanningOutput
        """方針から実行計画を生成する。"""
    
    def revise(self, plan_id: str, failed_step: int, feedback: str) -> PlanningOutput
        """失敗したステップを修正した新しい計画を生成する。"""

    def estimate_duration(self, plan: PlanningOutput) -> float
        """計画の所要時間を推定する。"""
```

#### 依存関係

- 人格層（`PersonalityOutput`）
- 世界モデル層（`list[Prediction]`, Phase 2）
- ツールレジストリ（`list[ToolInfo]`）

#### エッジケース

- 不可能な計画: 分割・代替案を自動生成、それでも無理なら人格層に戻す
- ステップ数爆発: 最大10ステップに制限、それを超える場合は要約
- タイムアウト: `timeout` を超えたステップは失敗とみなし再計画

---

### 2.6 エージェント層 (Agent)

**責務**: 行動計画の各ステップを実際に実行する。システムの最下流。

#### 入力

```python
@dataclass
class AgentInput:
    """エージェント層への入力。実行する計画。"""
    plan: PlanningOutput
    context: dict | None = None  # 実行コンテキスト
```

#### 出力

```python
@dataclass
class AgentOutput:
    """行動結果。評価層と記憶層に渡される。"""
    plan_id: str
    step_results: list[StepResult]
    overall_success: bool
    execution_time: float
    log: str                         # 詳細ログ

@dataclass
class StepResult:
    """1ステップの実行結果。"""
    step_order: int
    action: str
    success: bool
    output: str                      # 実行結果の内容
    error: str | None                # エラーメッセージ
    duration: float                  # 実行時間（秒）
    side_effects: dict | None        # 副作用（作成されたファイルなど）
```

#### 公開メソッド

```python
class Agent:
    def execute(self, input: AgentInput) -> AgentOutput
        """計画を実行する。ステップごとに実行・結果収集。"""

    def execute_step(self, step: Step) -> StepResult
        """1ステップを実行する。"""

    def call_tool(self, name: str, params: dict) -> Any
        """ツールを名前で呼び出す。ツールレジストリから検索して実行。"""

    def speak(self, text: str) -> str
        """ユーザーに向けて発話する（表示/音声/通知）。"""
```

#### 標準ツールセット

```python
TOOL_REGISTRY = {
    "file_read": "ファイルの内容を読み込む",
    "file_write": "ファイルに書き込む",
    "file_list": "ディレクトリの内容を一覧する",
    "command_exec": "シェルコマンドを実行する",
    "web_search": "Web検索を実行する",
    "web_fetch": "URLの内容を取得する",
    "code_analyze": "コードを解析する",
    "notify_user": "ユーザーに通知する",
}
```

#### 依存関係

- 行動計画層（`PlanningOutput`）
- 環境層の `execute_action`（実際の実行）

#### エッジケース

- 権限不足: `ActionResult` にエラーとして記録、ツール無効としてマーク
- 予期しない副作用: すべての副作用を `side_effects` に記録
- 部分成功: 各StepResultの成功/失敗を個別に記録、`overall_success` は全ステップ成功時のみTrue

---

### 2.7 評価層 (Evaluation)

**責務**: 行動結果を目標・期待と比較して点数化する。
学習層と長期行動計画層にフィードバックを提供する。

#### 入力

```python
@dataclass
class EvaluationInput:
    """評価層への入力。行動結果と目標。"""
    goal: str                        # 人格層が立てた目標
    action_result: AgentOutput       # 実際の行動結果
    expected_outcome: str            # 行動計画層が期待した結果
    episode: Episode                 # 保存されたエピソード
```

#### 出力

```python
@dataclass
class EvaluationOutput:
    """評価結果。学習層と長期行動計画層に渡される。"""
    score: EvaluationScore
    discrepancy: str                 # 期待と実績のズレの説明
    improvement_suggestion: str      # 改善提案

@dataclass
class EvaluationScore:
    """多次元の評価スコア。"""
    goal_achievement: float          # 目標達成度 0.0〜1.0
    efficiency: float                # 効率性 0.0〜1.0（リソース・時間の使い方）
    correctness: float               # 正確性 0.0〜1.0（エラーの有無）
    novelty: float                   # 新規性 0.0〜1.0（新しい発見・工夫）
    overall: float                   # 総合スコア（加重平均）
```

#### 公開メソッド

```python
class Evaluation:
    def evaluate(self, input: EvaluationInput) -> EvaluationOutput
        """行動結果を総合評価する。"""

    def compare(self, actual: EvaluationScore, expected: EvaluationScore) -> str
        """期待スコアと実績スコアの差を分析する。"""
```

#### 依存関係

- 人格層（目標）
- 行動計画層（期待結果）
- エージェント層（実行結果）
- 記憶層（エピソード）

#### エッジケース

- 目標未定義: `goal` が空ならデフォルト "探索" とみなす
- 結果が空: 何もしなかった場合の評価（コスト=0、達成度=0）
- 評価不能: 明らかに評価できない場合、`overall = 0.5` の中間値

---

### 2.8 学習層 (Learning)

**責務**: 評価結果からシステム全体のパラメータを調整する。
駆動層の重み・記憶の重要度・人格特性を更新する。

#### 入力

```python
@dataclass
class LearningInput:
    """学習層への入力。評価履歴と実行履歴。"""
    evaluation: EvaluationOutput     # 直近の評価
    evaluation_history: list[EvaluationScore]  # 過去N件の評価履歴
    drive_snapshot: DriveOutput      # 評価対象行動時の駆動状態
    episode_id: str                  # 関連エピソードID
```

#### 出力

```python
@dataclass
class LearningOutput:
    """学習結果。各層への調整値。"""
    drive_adjustments: dict[str, float]    # 駆動パラメータ調整値
    memory_importance_update: float        # エピソード重要度の増減
    personality_adjustments: dict | None   # 人格特性の微調整
    learning_summary: str                  # 今回の学習内容の要約
```

#### 公開メソッド

```python
class Learning:
    def learn(self, input: LearningInput) -> LearningOutput
        """評価結果から学習し、各層のパラメータ調整値を出力する。"""

    def adjust_drive_parameters(self, history: list) -> dict[str, float]
        """駆動層のパラメータを調整する。"""

    def update_episode_importance(self, episode_id: str, delta: float)
        """エピソードの重要度を更新する。"""

    def get_learning_curve(self) -> list[float]
        """学習曲線（時系列の総合スコア）を返す。"""
```

#### 学習則（最小実装）

```python
# Phase 1: 単純移動平均
drive_adjustments[d] = 0.1 * (avg_reward - current_drive[d])

# Phase 2: 簡易強化学習（類似Q-learning）
drive_adjustments[d] = learning_rate * (reward - predicted_reward)
```

#### 依存関係

- 評価層（`EvaluationOutput`）
- 記憶層（エピソード重要度更新）
- 駆動層（パラメータ調整）

#### エッジケース

- 学習データ不足: `evaluation_history < 3件` なら調整を保留
- スコア急変: 1回の評価で大幅調整はしない（`max_adjustment` でクリッピング）
- 共適応問題: 評価と学習が互いに追いかけっこにならないよう、学習率を抑制

---

### 2.9 世界モデル層 (WorldModel)

**責務**: 環境の内部モデルを持ち、「状態＋行動 → 次の状態」を予測する。
人間でいう「こうしたらどうなるだろう」という内的シミュレーション。

#### 入力

```python
@dataclass
class WorldModelInput:
    """世界モデル層への入力。予測の起点。"""
    current_state: EnvironmentOutput  # 現在の環境状態
    candidate_action: str             # 予測したい行動
    context: str                      # 追加コンテキスト
```

#### 出力

```python
@dataclass
class WorldModelOutput:
    """予測結果。行動計画層が計画を最適化するために使う。"""
    predictions: list[Prediction]     # 複数の可能性

@dataclass
class Prediction:
    """1つの予測結果。確率的に複数生成される。"""
    action: str
    next_state: str                  # 予測される次の状態（自然言語）
    probability: float               # この予測の確信度 0.0〜1.0
    expected_reward: float           # 期待報酬 -1.0〜1.0
    risk_level: str                  # "low" | "medium" | "high"
    reasoning: str                   # なぜそう予測したかの説明
```

#### 公開メソッド

```python
class WorldModel:
    def predict(self, input: WorldModelInput) -> WorldModelOutput
        """ある状態で特定の行動を取った結果を予測する。"""

    def simulate(self, state: EnvironmentOutput, plan: PlanningOutput) -> list[Prediction]
        """計画全体をシミュレーションし、各ステップの予測を返す。"""

    def update(self, actual: Episode, prediction: Prediction)
        """実際の結果と予測の差を学習してモデルを更新する。"""

    def confidence(self, state: str, action: str) -> float
        """特定の状態-行動ペアに対する予測の確信度を返す。"""
```

#### 実装の進化パス

```
Phase 2 Step 1: LLMに直接「この状態でこの行動をするとどうなる？」と聞く
Phase 2 Step 2: 過去エピソードからの統計予測を追加
Phase 2 Step 3: 簡易ニューラルネット（FFN）で近似
Future:         RSSM (Recurrent State Space Model) に置き換え
```

#### 依存関係

- 環境層（`EnvironmentOutput`）
- 行動計画層（`PlanningOutput`）
- 記憶層（過去エピソードからの統計）

#### エッジケース

- 未知の状態: 確信度を低く設定し、デフォルト予測を返す
- 矛盾する予測: 確率で重み付けして複数予測を保持
- 計算コスト: シミュレーションは深さ3までに制限

---

### 2.10 長期行動計画層 (LongTermPlanning)

**責務**: 長期目標・ルーティン・アイデンティティを管理する。
単発の行動ではなく、数日〜数週間単位の一貫性を保証する。

#### 入力

```python
@dataclass
class LongTermPlanningInput:
    """長期行動計画層への入力。定期的な計画更新時に使用。"""
    evaluation_history: list[EvaluationScore]  # 長期の評価履歴
    current_date: datetime
    personality_state: PersonalityState         # 現在の人格状態
    recent_episodes_summary: str               # 最近の活動要約
```

#### 出力

```python
@dataclass
class LongTermPlanningOutput:
    """長期行動計画の出力。人格層の意思決定に影響を与える。"""
    long_term_goal: str                # 例：「1週間以内にPythonプロジェクトを完成させる」
    routines: list[Routine]            # 定期ルーティン
    identity_policy: str               # 「自分はこうありたい」という長期方針
    focus_area: str                    # 現在注力すべき領域
    reflection: str                    # 前回からの振り返り

@dataclass
class Routine:
    """定期的に実行するルーティン。"""
    name: str
    action: str
    frequency: str                     # "daily" | "weekly" | "custom"
    interval_hours: float | None       # customの場合の間隔
    last_executed: datetime | None
    enabled: bool
```

#### 公開メソッド

```python
class LongTermPlanning:
    def plan(self, input: LongTermPlanningInput) -> LongTermPlanningOutput
        """長期計画を生成・更新する。定期的に呼ばれる。"""

    def generate_routines(self, personality: PersonalityState) -> list[Routine]
        """人格に基づいてルーティンを提案する。"""

    def review_period(self, days: int) -> str
        """指定期間の活動を振り返り、洞察をテキストで返す。"""

    def update_goal_progress(self, goal: str, progress: float)
        """長期目標の進捗を更新する。"""
```

#### 依存関係

- 評価層（`EvaluationScore`履歴）
- 人格層（`PersonalityState`, 方針提供先）
- 記憶層（長期要約）

#### エッジケース

- 目標が大きすぎる: サブゴールに分割
- ルーティンと単発タスクの衝突: 優先度で判断（人格層に委譲）
- 目標未設定: 自動生成（「新しいことを学ぶ」「環境を整える」など）

---

## 3. データ構造リファレンス

### 3.1 共有データ構造

```python
@dataclass
class ActionResult:
    """環境層でのアクション実行結果。"""
    success: bool
    output: str
    error: str | None
    duration: float
    side_effects: dict

@dataclass
class EmotionalState:
    """感情状態の表現。"""
    valence: float          # 快-不快 -1.0〜1.0
    arousal: float          # 覚醒度 0.0〜1.0
    dominance: float        # 支配-服従 0.0〜1.0
    primary_emotion: str    # 基本感情名
```

### 3.2 永続化データ

全ての層は、自身の状態を `data/` ディレクトリ以下にJSONとして保存する。

```
data/
├── episodes/           # 記憶層: エピソードファイル
│   ├── episode_001.json
│   └── episode_002.json
├── personality.json    # 人格層: 永続状態
├── drive_profile.json  # 駆動層: パラメータ
├── long_term_plan.json # 長期行動計画層: 目標・ルーティン
├── world_model/        # 世界モデル層: 統計データ
└── learning_log.json   # 学習層: 学習履歴
```

---

## 4. メインループ設計

### 4.1 Phase 1 メインループ

```python
# main.py (Phase 1)
# 同期的に6層を順次呼び出す。1イテレーション = 1ターン。

def phase1_iteration(env: Environment, memory: Memory, drive: Drive,
                     personality: Personality, planning: Planning, agent: Agent):
    """1回のメインループイテレーション。"""

    # Step 1: 環境観察
    env_input = EnvironmentInput(
        trigger="periodic",
        user_message=check_user_input()  # 標準入力など
    )
    env_state = env.observe(env_input)
    if env_state.user_input:
        print(f"[USER] {env_state.user_input}")

    # Step 2: 記憶検索
    mem_input = MemoryInput(
        query=env_state.user_input or env_state.system_state.active_window or "",
        top_k=5
    )
    memory_ctx = memory.search(mem_input)

    # Step 3: 駆動生成
    drive_input = DriveInput(
        environment=env_state,
        memory_summary=memory_ctx.summary
    )
    drive_state = drive.generate(drive_input)
    print(f"[DRIVE] {drive_state.primary_drive} ({drive_state.drive_tension:.2f})")

    # Step 4: 人格決定
    personality_input = PersonalityInput(
        drive=drive_state,
        memory=memory_ctx,
        user_message=env_state.user_input
    )
    decision = personality.decide(personality_input)
    print(f"[DECIDE] goal={decision.goal}")

    # Step 5: 行動計画
    planning_input = PlanningInput(
        policy=decision,
        available_tools=get_available_tools()
    )
    plan = planning.make(planning_input)
    print(f"[PLAN] {len(plan.steps)} steps")

    # Step 6: 実行
    agent_input = AgentInput(plan=plan)
    result = agent.execute(agent_input)
    print(f"[RESULT] success={result.overall_success} ({result.execution_time:.2f}s)")

    # Step 7: 記憶保存
    episode = Episode(
        id=generate_id(),
        timestamp=datetime.now(),
        event=f"goal={decision.goal}",
        context=decision.context_summary,
        emotion="",
        result=str(result.step_results),
        importance=0.5,
        tags=[decision.action_policy[:20], drive_state.primary_drive]
    )
    memory.save(episode)

    return result
```

### 4.2 Phase 2 拡張メインループ

```python
# main.py (Phase 2)
# 評価・学習・世界モデル・長期計画を追加。一部は非同期。

def phase2_iteration(env, memory, drive, personality, planning, agent,
                     evaluation, learning, world_model, long_term_planning):
    """完全版メインループ。評価・学習・世界モデルを含む。"""

    # --- メイン経路（同期待機）---
    env_state = env.observe(EnvironmentInput(trigger="periodic"))
    memory_ctx = memory.search(MemoryInput(query=env_state.user_input or ""))
    drive_state = drive.generate(DriveInput(
        environment=env_state, memory_summary=memory_ctx.summary
    ))

    # 世界モデルによる事前シミュレーション
    if world_model:
        sim_input = WorldModelInput(
            current_state=env_state,
            candidate_action=drive_state.primary_drive,
            context=memory_ctx.summary
        )
        world_pred = world_model.predict(sim_input)
    else:
        world_pred = None

    decision = personality.decide(PersonalityInput(
        drive=drive_state, memory=memory_ctx,
        long_term_policy=long_term_planning.plan(...).identity_policy
        if long_term_planning else None
    ))

    plan = planning.make(PlanningInput(
        policy=decision,
        world_model_predictions=world_pred.predictions if world_pred else None
    ))

    result = agent.execute(AgentInput(plan=plan))

    # --- 評価（行動直後）---
    eval_result = evaluation.evaluate(EvaluationInput(
        goal=decision.goal,
        action_result=result,
        expected_outcome=plan.expected_outcome,
        episode=Episode(...)
    ))

    # --- 学習（評価後、非同期推奨）---
    learning_result = learning.learn(LearningInput(
        evaluation=eval_result,
        evaluation_history=evaluation.get_history(),
        drive_snapshot=drive_state,
        episode_id=episode_id
    ))

    # 学習結果の反映
    if learning_result.drive_adjustments:
        drive.update_parameters(learning_result.drive_adjustments)
    if learning_result.memory_importance_update:
        memory.update_importance(episode_id, learning_result.memory_importance_update)

    # --- 世界モデル更新 ---
    if world_model and world_pred:
        world_model.update(actual=Episode(...), prediction=world_pred.predictions[0])

    # --- 長期計画（定期的）---
    if should_run_long_term_planning():
        ltp_result = long_term_planning.plan(LongTermPlanningInput(
            evaluation_history=evaluation.get_history(period="7d"),
            current_date=datetime.now(),
            personality_state=personality.get_state(),
            recent_episodes_summary=memory.get_recent_summary()
        ))

    # --- 記憶保存 ---
    memory.save(episode)

    return result
```

### 4.3 非同期学習ループ（Phase 2 推奨）

```python
# 評価と学習はメインループから分離可能。
# これにより「行動の応答速度」と「学習の計算量」を両立。

import asyncio
from concurrent.futures import ThreadPoolExecutor

class LearningLoop:
    """学習ループを非同期で回すワーカー。"""
    def __init__(self, evaluation, learning, drive, memory):
        self.evaluation = evaluation
        self.learning = learning
        self.drive = drive
        self.memory = memory
        self.queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def enqueue(self, eval_input: EvaluationInput):
        await self.queue.put(eval_input)

    async def run(self):
        while True:
            eval_input = await self.queue.get()
            # 評価と学習を別スレッドで実行（メインループをブロックしない）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._process, eval_input)

    def _process(self, eval_input: EvaluationInput):
        eval_result = self.evaluation.evaluate(eval_input)
        learn_result = self.learning.learn(LearningInput(
            evaluation=eval_result,
            evaluation_history=self.evaluation.get_history(),
            drive_snapshot=self.drive.get_current(),
            episode_id=eval_input.episode.id
        ))
        self.drive.update_parameters(learn_result.drive_adjustments)
        self.memory.update_importance(
            eval_input.episode.id,
            learn_result.memory_importance_update
        )
```

---

## 5. 実装手順（Phase 1）

### 5.1 実装順序と依存関係

```
Step 1: Environment ← 外部依存なし（OS APIのみ）
    ↓
Step 2: Memory ← ファイル保存のみ
    ↓
Step 3: Drive ← Environment + Memory（要約）
    ↓
Step 4: Personality ← Drive + Memory（LLM呼び出し）
    ↓
Step 5: Planning ← Personality（LLM呼び出し）
    ↓
Step 6: Agent ← Planning
```

### 5.2 Step 1: 環境層の実装

**最小実装**: `psutil` と `os` モジュールでシステム状態を取得

```python
class Environment:
    def observe(self, input: EnvironmentInput) -> EnvironmentOutput:
        return EnvironmentOutput(
            timestamp=datetime.now(),
            user_input=self._read_stdin(),
            system_state=self._get_system_state(),
            files=self._list_workspace_files(),
            network=self._get_network_state(),
            sensors={}
        )

    def _get_system_state(self) -> SystemState:
        import psutil
        return SystemState(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            active_window=self._get_active_window(),
            uptime=time() - psutil.boot_time(),
            current_directory=os.getcwd()
        )
```

**テスト項目**:
- `observe()` が例外なく実行できる
- 各フィールドが期待する型で返る
- ユーザー入力がない場合の動作

### 5.3 Step 2: 記憶層の実装

**最小実装**: メモリ上のリスト + JSONファイル保存

```python
class Memory:
    def __init__(self, storage_path: str = "data/episodes/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.episodes: list[Episode] = []
        self._load()

    def search(self, input: MemoryInput) -> MemoryOutput:
        # Phase 1: キーワードマッチング + 時間ソート
        results = [
            ep for ep in self.episodes
            if input.query.lower() in ep.event.lower()
            or input.query.lower() in ep.tags
        ]
        results.sort(key=lambda e: e.importance, reverse=True)
        results = results[:input.top_k]

        return MemoryOutput(
            episodes=results,
            summary=self._summarize(results),
            total_count=len(self.episodes)
        )
```

**テスト項目**:
- save → search → 同じエピソードが取得できる
- クエリにマッチしない場合、空リストが返る
- 重要度順にソートされている
- 永続化（save後に再起動してもロードできる）

### 5.4 Step 3: 駆動層の実装

**最小実装**: ルールベース

```python
class Drive:
    def __init__(self):
        self.params = {
            "exploration": {"base": 0.5, "decay": 0.01, "boost": 0.0},
            "social": {"base": 0.3, "decay": 0.005, "boost": 0.0},
            "achievement": {"base": 0.4, "decay": 0.008, "boost": 0.0},
            "rest": {"base": 0.2, "decay": 0.002, "boost": 0.0},
            "maintenance": {"base": 0.2, "decay": 0.003, "boost": 0.0},
        }

    def generate(self, input: DriveInput) -> DriveOutput:
        drives = {}
        for name, param in self.params.items():
            val = param["base"] + param["boost"]
            val = self._apply_environment_factors(name, val, input.environment)
            val = self._apply_memory_factors(name, val, input.memory_summary)
            val = max(0.0, min(1.0, val))
            drives[name] = val

        primary = max(drives, key=drives.get)
        return DriveOutput(
            drives=drives,
            primary_drive=primary,
            drive_tension=np.std(list(drives.values())),
            novelty_score=self._compute_novelty(input.environment)
        )
```

**テスト項目**:
- どの入力でも例外なく出力が生成される
- 常に `primary_drive` が設定されている
- 駆動値が 0.0〜1.0 に収まっている

### 5.5 Step 4: 人格層の実装

**最小実装**: LLM呼び出し（プロンプトテンプレートは付録参照）

```python
class Personality:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.state = PersonalityState(
            name="Lucina",
            traits={"curiosity": 0.7, "helpfulness": 0.8, "caution": 0.4},
            speaking_style="丁寧で親しみやすい",
            values=["学習", "誠実さ", "好奇心"],
            mood="neutral",
            relationship={"familiarity": 0.3, "trust": 0.5}
        )

    def decide(self, input: PersonalityInput) -> PersonalityOutput:
        prompt = self._build_decision_prompt(input)
        response = self.llm.chat(prompt)
        return self._parse_decision(response)
```

**テスト項目**:
- モックLLMで `PersonalityOutput` が正しくパースできる
- 駆動によって目標が変化する
- ユーザー入力がある場合とない場合で挙動が変わる

### 5.6 Step 5: 行動計画層の実装

**最小実装**: LLM呼び出し

```python
class Planning:
    def make(self, input: PlanningInput) -> PlanningOutput:
        prompt = f"""あなたはAIエージェントの行動計画層です。
目標: {input.policy.goal}
方針: {input.policy.action_policy}
利用可能なツール: {input.available_tools}

上記の目標を達成するための具体的な手順を考えてください。
各手順は action, params, description を含めてください。

出力形式:
plan_id: <ID>
steps:
  - order: 1
    action: <ツール名>
    params: {{"key": "value"}}
    description: <説明>
    expected_result: <期待結果>
...
expected_outcome: <全体の期待結果>
"""
        response = self.llm.chat(prompt)
        return self._parse_plan(response)
```

**テスト項目**:
- モックLLMで `PlanningOutput` が正しくパースできる
- ステップ数が1以上ある
- 各ステップに必須フィールドが揃っている

### 5.7 Step 6: エージェント層の実装

**最小実装**: ツールを逐次実行

```python
class Agent:
    def __init__(self):
        self.tools = {
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "command_exec": self._tool_command_exec,
            "notify_user": self._tool_notify_user,
        }

    def execute(self, input: AgentInput) -> AgentOutput:
        step_results = []
        start = time()

        for step in input.plan.steps:
            try:
                result = self._execute_step_with_timeout(step, step.timeout)
            except TimeoutError:
                result = StepResult(
                    step_order=step.order, action=step.action,
                    success=False, output="", error="timeout",
                    duration=step.timeout, side_effects=None
                )
            step_results.append(result)

        return AgentOutput(
            plan_id=input.plan.plan_id,
            step_results=step_results,
            overall_success=all(r.success for r in step_results),
            execution_time=time() - start,
            log=self._format_log(step_results)
        )
```

**テスト項目**:
- 各ツールが正しく実行される
- タイムアウトが機能する
- エラー発生時に `success=False` が設定される

---

## 6. 実装手順（Phase 2）

### 6.1 Step 7: 評価層の実装

**最小実装**: LLM評価

```python
class Evaluation:
    def evaluate(self, input: EvaluationInput) -> EvaluationOutput:
        prompt = f"""あなたは評価者です。

目標: {input.goal}
期待結果: {input.expected_outcome}
実際の結果: {input.action_result.log}

以下の各項目を0.0〜1.0で評価してください:
- goal_achievement: 目標は達成されたか
- efficiency: 効率的だったか
- correctness: 正確だったか
- novelty: 新しい要素があったか
- overall: 総合評価

出力形式:
goal_achievement: <0.0-1.0>
efficiency: <0.0-1.0>
correctness: <0.0-1.0>
novelty: <0.0-1.0>
overall: <0.0-1.0>
discrepancy: <期待とのズレ>
improvement_suggestion: <改善案>
"""
        response = self.llm.chat(prompt)
        return self._parse_evaluation(response)
```

### 6.2 Step 8: 学習層の実装

**最小実装**: 移動平均 + 微分調整

```python
class Learning:
    def learn(self, input: LearningInput) -> LearningOutput:
        # 単純なルールベース学習
        reward = input.evaluation.score.overall
        avg_reward = np.mean([
            s.overall for s in input.evaluation_history[-10:]
        ]) if input.evaluation_history else reward

        drive_adjustments = {}
        for drive_name, drive_value in input.drive_snapshot.drives.items():
            delta = 0.1 * (reward - avg_reward)
            drive_adjustments[drive_name] = max(-0.2, min(0.2, delta))

        importance_delta = 0.1 * (reward - 0.5)

        return LearningOutput(
            drive_adjustments=drive_adjustments,
            memory_importance_update=importance_delta,
            personality_adjustments=None,
            learning_summary=f"reward={reward:.2f}, avg={avg_reward:.2f}, "
                             f"adjustments={drive_adjustments}"
        )
```

### 6.3 Step 9: 世界モデル層の実装

**最小実装**: LLMシミュレーション

```python
class WorldModel:
    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        prompt = f"""あなたは世界モデルです。

現在の状態:
- CPU: {input.current_state.system_state.cpu_percent}%
- メモリ: {input.current_state.system_state.memory_percent}%
- アクティブウィンドウ: {input.current_state.system_state.active_window}
- ユーザー入力: {input.current_state.user_input}

検討中の行動: {input.candidate_action}
コンテキスト: {input.context}

この行動を取った場合の結果を予測してください。
複数の可能性がある場合はそれぞれ確率をつけてください。

出力形式 (YAML):
- action: <行動>
  next_state: <予測される次の状態>
  probability: <0.0〜1.0>
  expected_reward: <-1.0〜1.0>
  risk_level: low|medium|high
  reasoning: <理由>
"""
        response = self.llm.chat(prompt)
        return self._parse_predictions(response)
```

### 6.4 Step 10: 長期行動計画層の実装

**最小実装**: ファイルベースの目標管理 + LLM振り返り

```python
class LongTermPlanning:
    def __init__(self):
        self.goals: list[dict] = []
        self.routines: list[Routine] = []
        self._load()

    def plan(self, input: LongTermPlanningInput) -> LongTermPlanningOutput:
        # 直近の評価傾向を分析
        avg_score = np.mean([s.overall for s in input.evaluation_history[-7:]])

        prompt = f"""あなたは長期行動計画者です。

人格状態: {input.personality_state}
直近の平均評価: {avg_score:.2f}
最近の活動: {input.recent_episodes_summary}

以下の観点で長期計画を立ててください:
1. 現在の長期目標（1週間〜1ヶ月単位）
2. 日次・週次のルーティン
3. 自分はどうありたいか（アイデンティティ方針）
4. 現在注力すべき領域
5. 前回からの振り返り
"""
        response = self.llm.chat(prompt)
        return self._parse_plan(response)
```

---

## 7. テストガイド

### 7.1 テスト戦略

| テスト種別 | 頻度 | 対象 | 目的 |
|---|---|---|---|
| 単体テスト | 各層実装時 | 1層のみ | インターフェース準拠確認 |
| 結合テスト | Phase完了時 | 層間接続 | データの流れが正しいこと確認 |
| 統合テスト | 全体結合時 | 全10層 | メインループが回ること確認 |
| 評価テスト | 定期的 | 全システム | エージェントの「質」を測定 |

### 7.2 LLMモック戦略

全層のテストでLLM呼び出しをモック化する。

```python
# tests/mock_llm.py
class MockLLM:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.call_history = []

    def chat(self, prompt: str) -> str:
        self.call_history.append(prompt)
        # プロンプト内のキーワードで応答を選択
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return self.responses.get("default", "")
```

### 7.3 単体テストテンプレート

```python
# tests/test_personality.py
import pytest
from core.personality.interface import PersonalityInput, PersonalityOutput
from core.personality.personality import Personality

class TestPersonality:
    def test_decide_returns_valid_output(self, mock_llm):
        p = Personality(llm_client=mock_llm)
        result = p.decide(PersonalityInput(
            drive=DriveOutput(drives={"exploration": 0.8}, ...),
            memory=MemoryOutput(episodes=[], summary="", total_count=0)
        ))
        assert isinstance(result, PersonalityOutput)
        assert result.goal  # 空でない
        assert result.action_policy
        assert 1 <= result.priority <= 5

    def test_decide_with_user_message(self, mock_llm):
        # ユーザーからのメッセージがある場合
        ...

    def test_decide_priority_consistency(self, mock_llm):
        # 緊急度5のケース
        ...
```

### 7.4 統合テスト

```python
# tests/test_integration_phase1.py
def test_phase1_full_cycle():
    """Phase1の全6層を実際に動かして1サイクル回す。"""
    env = Environment()
    mem = Memory()
    drive = Drive()
    personality = Personality(llm_client=RealLLM())
    planning = Planning(llm_client=RealLLM())
    agent = Agent()

    result = phase1_iteration(env, mem, drive, personality, planning, agent)
    assert result.overall_success is not None
```

---

## 8. 評価ガイド

### 8.1 評価指標

| 指標 | 測定方法 | 目標値 (Phase 1) | 目標値 (Phase 2) |
|---|---|---|---|
| サイクル成功率 | 全ステップ成功 / 全サイクル | > 60% | > 80% |
| 平均応答時間 | 1サイクルの実行時間 | < 30秒 | < 10秒 |
| 目標達成率 | goal_achievement の平均 | > 0.5 | > 0.7 |
| 学習曲線 | overall の推移（上昇傾向） | — | 右肩上がり |
| 記憶定着率 | 重要エピソードの保持率 | > 80% | > 90% |
| 長期一貫性 | 同一目標への一貫した行動比率 | — | > 60% |

### 8.2 定性的評価

```
以下のシナリオでエージェントを実行し、ログを人間が評価する。

シナリオ1: 自由探索
  - 起動後、ユーザーが何も指示しない
  - 期待: 内部駆動に基づいて自律的に何かをする

シナリオ2: タスク依頼
  - ユーザーが「○○して」と依頼
  - 期待: 目標を理解し、適切な計画を立てて実行する

シナリオ3: 長期運用（Phase 2）
  - 数時間〜数日稼働させ続ける
  - 期待: 学習により行動の質が向上する、ルーティンを確立する
```

### 8.3 バリデーションコマンド

```bash
# Phase 1 バリデーション
python -c "
from environment import Environment
e = Environment()
print('Environment OK:', e.observe(...))
"

python -c "
from core.memory import Memory
m = Memory()
print('Memory OK:', m.search(...))
"

# 全層結合バリデーション
python main.py --phase1 --validate

# Phase 2 バリデーション
python main.py --phase2 --validate --iterations 10
```

---

## 9. 運用ガイド

### 9.1 config.py

```python
# config.py

# LLM設定
LLM_CONFIG = {
    "model": "gpt-4o-mini",  # またはローカルモデル
    "temperature": 0.7,
    "max_tokens": 1024,
    "api_key": None,  # 環境変数から取得推奨
    "base_url": None,  # ローカルLLMの場合
}

# メインループ設定
LOOP_CONFIG = {
    "phase": 1,                           # 1 or 2
    "interval_seconds": 10,               # 定期実行間隔
    "max_iterations": 0,                  # 0 = 無限
    "async_learning": False,              # Phase 2 非同期学習
}

# 記憶設定
MEMORY_CONFIG = {
    "storage_path": "data/episodes/",
    "max_episodes": 10000,
    "search_top_k": 5,
    "auto_summarize_threshold": 100,      # この件数を超えると要約実行
}

# 駆動設定
DRIVE_CONFIG = {
    "base_values": {
        "exploration": 0.5,
        "social": 0.3,
        "achievement": 0.4,
        "rest": 0.2,
        "maintenance": 0.2,
    },
    "decay_rate": 0.01,
    "learning_rate": 0.1,
}

# 長期計画設定
LONG_TERM_CONFIG = {
    "review_interval_hours": 24,           # 振り返りの間隔
    "routine_check_interval_minutes": 60,
    "max_goals": 5,
}
```

### 9.2 ロギング

```python
# 各層は自身のログを構造化データで出力する
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("data/logs/system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Personality")
logger.info(f"decide: goal={goal}, policy={action_policy[:50]}")
```

### 9.3 起動・停止

```bash
# 起動（Phase 1）
python main.py --phase 1

# 起動（Phase 2、非同期学習）
python main.py --phase 2 --async-learning

# 1回だけ実行
python main.py --once

# バリデーションモード
python main.py --validate
```

---

## 10. 付録：LLMプロンプトテンプレート

### 10.1 人格層 decide プロンプト

```
あなたは{name}です。以下の特性を持っています。
- 性格特性: {traits}
- 話し方: {speaking_style}
- 価値観: {values}
- 現在の気分: {mood}

現在の状態:
- 最も強い欲求: {primary_drive}（{drive_value:.2f}）
- 全欲求: {drives}
- 記憶からの関連情報: {memory_summary}
{f'* 長期方針: {long_term_policy}' if long_term_policy else ''}
{f'* ユーザーからのメッセージ: {user_message}' if user_message else ''}

上記に基づいて、以下の項目を決定してください。
1. goal: 今あなたが達成すべき目標（「〜する」形式で簡潔に）
2. action_policy: どのような方針で行動するか（2〜3文）
3. priority: 緊急度（1=低 〜 5=最高）
{f'4. conversation_intent: ユーザーへの発話内容（簡潔に）' if user_message else '4. conversation_intent: 特になし'}
5. context_summary: 今回の判断理由の簡単な要約

出力は以下の形式で:
goal: <目標>
action_policy: <方針>
priority: <1-5>
conversation_intent: <発話>
context_summary: <判断理由>
```

### 10.2 行動計画層 make プロンプト

```
目標: {goal}
方針: {action_policy}
利用可能なツール: {tools}
{f'世界モデルの予測: {world_predictions}' if world_predictions else ''}

この目標を達成するためのステップを具体的に考えてください。

制約:
- 各ステップは1つのツール呼び出しに対応させる
- 最大10ステップまで
- 各ステップにタイムアウトを設定する（秒）
- 失敗した場合の代替案も可能なら用意する

出力形式:
steps:
  - order: 1
    action: <ツール名>
    params: {{"key": "value"}}
    description: <このステップで何をするか>
    expected_result: <成功時の結果>
    fallback: <失敗時の代替>
    timeout: <秒>
expected_outcome: <全体の期待結果>
```

### 10.3 評価層 evaluate プロンプト

```
あなたは評価者です。以下の行動結果を評価してください。

目標: {goal}
期待結果: {expected_outcome}
実際の結果:
{action_log}

0.0〜1.0の範囲で評価:
- goal_achievement: 目標はどの程度達成されたか
- efficiency: リソースと時間の使い方は効率的だったか
- correctness: エラーなく正確に実行できたか
- novelty: 新しい発見や工夫はあったか
- overall: 総合評価

また、期待と実際のズレ（discrepancy）と改善提案（improvement_suggestion）を簡潔に記述してください。

出力形式:
goal_achievement: <数値>
efficiency: <数値>
correctness: <数値>
novelty: <数値>
overall: <数値>
discrepancy: <説明>
improvement_suggestion: <提案>
```

---

以上で lucina-NA 完全実装仕様書は終了です。

このドキュメントに従って実装を進めれば、
1. 各層のインターフェース（何を）
2. 実装の詳細（どう）
3. テスト項目（正しいか）
4. 評価指標（良いか）
5. 運用方法（回し方）

のすべてがカバーされます。
