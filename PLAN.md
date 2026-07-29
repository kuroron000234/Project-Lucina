# lucina-NA 完全実装計画書

## 0. プロジェクト概要

**lucina-NA（New Agent）** は、10層のレイヤーアーキテクチャを持つAIエージェントシステム。
エピソード記憶、世界モデルによるシミュレーション、学習による適応を備え、
環境からの入力を元に自律的に行動・評価・学習する。

---

## 1. アーキテクチャ図（テキスト再現）

```
                    [環境層]  ← PC・ネットワークからの状態取得
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     [記憶層] ←──→ [駆動層] ←── [学習層]
        │        双方向    │   一方向    │
        │              │              │
        │              ↓              │
        └──────────→ [人格層] ──→ [評価層]
                 双方向    │   一方向    │
                          │              ↑
              ┌───────────┼──────────┐  │
              │           │          │  │
           [世界モデル] ←→ [行動計画]  [長期行動計画] ─┘
               双方向      │      一方向
                          │
                       [エージェント層]  ← 自然言語で実行
```

### 凡例
- **グレー**: 入出力の接点（環境層, エージェント層）
- **紫**: 判断・生成（駆動層, 人格層, 行動計画層, 長期行動計画層, 世界モデル層）
- **ティール**: データ・評価・学習（記憶層, 学習層, 評価層）

---

## 2. 全10層の定義

| # | 層 | 色 | 責務（1行） | 主な接続 |
|---|---|---|---|---|
| 1 | 環境層 | グレー | PCやユーザーから状態を取得する | →駆動層 |
| 2 | 記憶層 | ティール | エピソード・知識を保存・検索する | ↔人格層 |
| 3 | 駆動層 | 紫 | 現在の欲求・優先度を生成する | →人格層, ←学習層 |
| 4 | 人格層 | 紫 | 自然言語で方針を決定する | →行動計画層, →評価層, ↔記憶層, ←長期行動計画層 |
| 5 | 行動計画層 | 紫 | 方針を実行可能な手順へ分解する | →エージェント層, ↔世界モデル層 |
| 6 | 長期行動計画層 | 紫 | 長期目標・ルーチンを管理する | →人格層, →評価層 |
| 7 | 世界モデル層 | 紫 | 環境を内部シミュレーションし行動結果を予測する | ↔行動計画層 |
| 8 | エージェント層 | グレー | 自然言語で指示を実際に実行する | ←行動計画層 |
| 9 | 評価層 | ティール | 行動結果を評価する | ↔学習層, ←人格層, ←長期行動計画層 |
| 10 | 学習層 | ティール | パラメータを更新する | ↔評価層, →駆動層 |

---

## 3. 実装フェーズ（全2フェーズ、6+4層）

### Phase 1: 「動くAI」を作る（6層）
**目標**: シンプルなLLM呼び出しで各層を実装し、環境→行動→記憶のサイクルを回す

**実装順序**:

```
Environment (1) → Memory (2) → Drive (3)
                                      ↘
                          Agent (6) ← Planning (5) ← Personality (4)
```

| Step | 層 | やること | 最小実装 |
|---|---|---|---|
| 1 | 環境層 | PCの状態取得（時刻、センサー、ファイル、USER入力） | 簡易Sensorクラス |
| 2 | 記憶層 | エピソードの保存／検索／要約 | 簡易リスト + JSON保存 |
| 3 | 駆動層 | 生物的な欲求（探索/休息/社会）の優先度を生成 | ルールベース →
| 4 | 人格層 | 方針決定、発話生成、内省 | LLM呼び出し（1回） |
| 5 | 行動計画層 | 方針→手順の具体化 | LLM呼び出し（1回） |
| 6 | エージェント層 | 実際の行動実行（関数呼び出し/ファイル操作/発話） | コマンド実行 + 発話 |

### Phase 2: 「学習するAI」にする（4層）
**目標**: 評価・学習・予測・長期計画を追加し、行動の質を向上させる

```
Phase 1 のシステム
         ↓
Evaluation (7) → Learning (8) → WorldModel (9) → LongTermPlanning (10)
```

| Step | 層 | やること | 最小実装 |
|---|---|---|---|
| 7 | 評価層 | 行動結果の成功/報酬/コストを計算 | LLM評価 → ルール評価 |
| 8 | 学習層 | 評価結果から駆動パラメータを調整 | 勾配なし簡易更新則 |
| 9 | 世界モデル層 | 状態→行動→結果の確率予測 | LLMシミュレーション |
| 10 | 長期行動計画層 | 日次/週次ルーチン、長期目標の管理 | カレンダー + 目標リスト |

---

## 4. 各層の責務・入出力・データ構造

### 4-1. 環境層 (Environment)

**責務**: PCやユーザーから状態を取得する

```python
class EnvironmentInput:
    pass  # 外部からのトリガー

class EnvironmentOutput:
    timestamp: datetime
    sensors: dict       # システム状態（CPU, メモリ, 時間帯など）
    user_input: str | None
    files: list[FileInfo]

class Environment:
    def observe() -> EnvironmentOutput
    def execute_action(action: str) -> ActionResult
```

### 4-2. 記憶層 (Memory)

**責務**: エピソード・知識を保存・検索する

```python
class MemoryInput:
    query: str
    context: dict

class MemoryOutput:
    episodes: list[Episode]
    summary: str

class Episode:
    id: str
    timestamp: datetime
    event: str
    emotion: str
    result: str
    importance: float  # 0.0〜1.0

class Memory:
    def save(episode: Episode)
    def search(query: str, limit: int = 5) -> list[Episode]
    def summarize(episodes: list[Episode]) -> str
    def forget(threshold: float)  # 重要度の低いものを削除
```

### 4-3. 駆動層 (Drive)

**責務**: 現在の欲求・優先度を生成する

```python
class DriveInput:
    environment: EnvironmentOutput
    memory_context: str  # 記憶層からの要約
    learning_signal: dict | None  # 学習層からの調整値

class DriveOutput:
    drives: dict[str, float]  # 例: {"exploration": 0.8, "rest": 0.2, "social": 0.5}
    primary_drive: str

class Drive:
    def generate(input: DriveInput) -> DriveOutput
    def update_parameters(signal: dict)
```

### 4-4. 人格層 (Personality)

**責務**: 自然言語で方針を決定する

```python
class PersonalityInput:
    drive: DriveOutput
    memory: MemoryOutput
    long_term_policy: str | None  # 長期行動計画層からの方針

class PersonalityOutput:
    goal: str                # 現在の目標
    action_policy: str       # 行動方針（自然言語）
    conversation_intent: str | None  # 発話意図

class Personality:
    def decide(input: PersonalityInput) -> PersonalityOutput
    def reflect(episode: Episode) -> str  # 内省
    def speak(intent: str) -> str  # 発話生成
```

### 4-5. 行動計画層 (Planning)

**責務**: 方針を実行可能な手順へ分解する

```python
class PlanningInput:
    policy: PersonalityOutput
    world_prediction: dict | None  # 世界モデルからの予測

class PlanningOutput:
    plan: list[Step]  # 手順リスト
    expected_outcome: str

class Step:
    order: int
    action: str
    params: dict
    fallback: str | None

class Planning:
    def make(input: PlanningInput) -> PlanningOutput
    def revise(feedback: str) -> PlanningOutput  # 失敗時の再計画
```

### 4-6. エージェント層 (Agent)

**責務**: 自然言語で指示を実際に実行する

```python
class AgentInput:
    plan: PlanningOutput

class AgentOutput:
    result: ActionResult
    log: str

class ActionResult:
    success: bool
    output: str
    error: str | None
    duration: float

class Agent:
    def execute(input: AgentInput) -> AgentOutput
    def call_tool(name: str, params: dict) -> Any
    def speak(text: str) -> str
```

### 4-7. 評価層 (Evaluation)

**責務**: 行動結果を評価する

```python
class EvaluationInput:
    goal: str
    action_result: ActionResult
    episode: Episode

class EvaluationOutput:
    score: EvaluationScore

class EvaluationScore:
    success: bool          # 目標達成？ 
    reward: float          # -1.0〜1.0
    cost: float            # 0.0〜1.0（リソース消費）
    feedback: str          # 自然言語でのフィードバック

class Evaluation:
    def evaluate(input: EvaluationInput) -> EvaluationOutput
    def compare(actual: EvaluationScore, expected: EvaluationScore) -> str
```

### 4-8. 学習層 (Learning)

**責務**: パラメータを更新する

```python
class LearningInput:
    evaluation_history: list[EvaluationScore]
    drive_history: list[DriveOutput]

class LearningOutput:
    drive_adjustments: dict  # 駆動層への調整値
    memory_importance_update: list[tuple[str, float]]  # エピソード重要度更新

class Learning:
    def learn(input: LearningInput) -> LearningOutput
    def adjust_drive_parameters(history: list) -> dict
    def update_episode_importance(history: list) -> list[tuple[str, float]]
```

### 4-9. 世界モデル層 (WorldModel)

**責務**: 環境を内部シミュレーションし行動結果を予測する

```python
class WorldModelInput:
    state: EnvironmentOutput
    candidate_plan: PlanningOutput | None

class WorldModelOutput:
    predictions: list[Prediction]

class Prediction:
    state: str
    action: str
    next_state: str
    probability: float  # 0.0〜1.0
    expected_reward: float

class WorldModel:
    def predict(input: WorldModelInput) -> WorldModelOutput
    def simulate(plan: PlanningOutput) -> list[Prediction]  # 複数のactionをシミュレート
    def update(actual: Episode, predicted: Prediction)  # 予測の修正
```

### 4-10. 長期行動計画層 (LongTermPlanning)

**責務**: 長期目標・ルーチンを管理する

```python
class LongTermPlanningInput:
    evaluation_history: list[EvaluationScore]
    current_date: datetime

class LongTermPlanningOutput:
    long_term_goal: str
    routines: list[Routine]
    identity_policy: str  # 長期的な人格方針

class Routine:
    time: str
    action: str
    frequency: str  # "daily" | "weekly" | "custom"

class LongTermPlanning:
    def plan(input: LongTermPlanningInput) -> LongTermPlanningOutput
    def review(period: str) -> str  # 一定期間の振り返り
```

---

## 5. メインループ設計

```python
# Phase 1 のメインループ
def main_loop_iteration():
    # Step 1: 環境観察
    env_state = environment.observe()

    # Step 2: 記憶検索
    memory_context = memory.search(query=env_state.user_input or "")

    # Step 3: 駆動生成
    drive_state = drive.generate(DriveInput(
        environment=env_state,
        memory_context=memory_context
    ))

    # Step 4: 人格決定
    personality_output = personality.decide(PersonalityInput(
        drive=drive_state,
        memory=memory_context
    ))

    # Step 5: 行動計画
    plan = planning.make(PlanningInput(
        policy=personality_output
    ))

    # Step 6: 実行
    result = agent.execute(AgentInput(plan=plan))

    # Step 7: 記憶に保存
    memory.save(Episode(
        event=f"goal={personality_output.goal}, result={result.result.success}",
        result=str(result.result),
        importance=0.5  # デフォルト値
    ))

    return result
```

```python
# Phase 2 拡張メインループ
def main_loop_iteration_v2():
    # Step 1-6: 同上
    env_state = environment.observe()
    memory_context = memory.search(...)
    drive_state = drive.generate(...)
    personality_output = personality.decide(...)
    plan = planning.make(...)
    result = agent.execute(...)

    # Step 7: 評価
    evaluation = evaluation.evaluate(EvaluationInput(
        goal=personality_output.goal,
        action_result=result,
        episode=episode
    ))

    # Step 8: 学習
    learning_signal = learning.learn(LearningInput(
        evaluation_history=[evaluation],
        drive_history=[drive_state]
    ))

    # Step 9: 世界モデル更新
    world_model.update(actual=episode, predicted=prediction)

    # Step 10: 長期計画レビュー（定期的）
    if should_review_long_term():
        long_term_policy = long_term_planning.plan(...)

    # Step 11: 記憶保存（重要度は学習結果から）
    episode.importance = adjust_by_learning(learning_signal)
    memory.save(episode)

    return result
```

---

## 6. ディレクトリ構造（完成形）

```
lucina-NA/
│
├── main.py                 # エントリポイント（メインループ）
├── config.py              # 全体設定
├── PLAN.md                # 本計画書
│
├── core/                  # アーキテクチャの核
│   ├── __init__.py
│   │
│   ├── memory/            # 記憶層
│   │   ├── __init__.py
│   │   ├── interface.py   # 入出力・データ構造の定義
│   │   ├── memory.py      # Memoryクラス
│   │   └── storage.py     # 永続化戦略
│   │
│   ├── drive/             # 駆動層
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── drive.py
│   │
│   ├── personality/       # 人格層
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── personality.py
│   │
│   ├── planning/          # 行動計画層
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── planning.py
│   │
│   ├── agent/             # エージェント層
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── agent.py
│   │
│   ├── evaluation/        # 評価層（Phase 2）
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── evaluation.py
│   │
│   ├── learning/          # 学習層（Phase 2）
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── learning.py
│   │
│   ├── world_model/       # 世界モデル層（Phase 2）
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   └── world_model.py
│   │
│   └── long_term_planning/ # 長期行動計画層（Phase 2）
│       ├── __init__.py
│       ├── interface.py
│       └── long_term_planning.py
│
├── environment/           # 環境層
│   ├── __init__.py
│   ├── interface.py
│   └── environment.py
│
├── data/                  # 保存データ
│   ├── episodes/         # エピソード記憶
│   ├── models/           # 学習済みパラメータ
│   └── config/           # 設定ファイル
│
├── tests/                 # テスト
│   ├── test_memory.py
│   ├── test_drive.py
│   ├── test_personality.py
│   ├── test_planning.py
│   ├── test_agent.py
│   ├── test_evaluation.py
│   ├── test_learning.py
│   ├── test_world_model.py
│   └── test_long_term_planning.py
│
└── docs/                  # ドキュメント
    ├── architecture.md
    └── interfaces.md
```

---

## 7. マイルストーンとチェックポイント

### Phase 1: Moving AI（目標: 2〜3週間）

| # | マイルストーン | 内容 | チェックポイント |
|---|---|---|---|
| M1 | Environment完成 | 環境層が状態を取得できる | `python -c "from environment import Environment; e=Environment(); print(e.observe())"` で結果表示 |
| M2 | Memory完成 | エピソードの保存/検索ができる | save→search→summarize のサイクルテスト |
| M3 | Drive完成 | 欲求の優先度が生成できる | 状態→drives の出力確認 |
| M4 | Personality完成 | LLMを使った方針決定ができる | 入力を変えてgoal/action_policyが変わる確認 |
| M5 | Planning完成 | 方針→手順の具体化ができる | goalからplanが生成される確認 |
| M6 | Agent完成 | 実際に行動を実行できる | planを実行して結果が返る確認 |
| **M7** | **Phase 1 結合** | **全6層がつながって動く** | `python main.py --phase1` で1サイクル動作確認 |

### Phase 2: Learning AI（目標: 2〜3週間）

| # | マイルストーン | 内容 | チェックポイント |
|---|---|---|---|
| M8 | Evaluation完成 | 行動結果の評価ができる | 成功/失敗に応じた報酬計算確認 |
| M9 | Learning完成 | 評価→駆動パラメータ調整ができる | 複数サイクルで駆動値が変化する確認 |
| M10 | WorldModel完成 | 行動前の予測ができる | シミュレーションと実結果の比較テスト |
| M11 | LongTermPlanning完成 | 長期目標・ルーチン管理 | 振り返りと長期方針生成確認 |
| **M12** | **Phase 2 結合** | **全10層がつながって動く** | `python main.py --phase2` で複数サイクル動作確認 |

---

## 8. 各層の実装パターン（推奨）

### Phase 1: LLM First方式
全層で最初は **LLMに任せる** シンプル実装から始める。

```
例: Personality.decide()
  ↓
prompt = f"""
あなたはAIエージェントの人格層です。
現在の欲求: {drive}
記憶: {memory}
長期方針: {long_term_policy}

上記に基づいて、以下の形式で出力してください。
goal: <目標>
action_policy: <方針>
conversation_intent: <発話意図>
"""
response = llm.chat(prompt)
parse(response) → PersonalityOutput
```

### Phase 2: 段階的具体化
動作確認後、各層を徐々に「本物の実装」に置き換える。

```
世界モデル層の進化:
  Step 1: LLMに "この状態でこの行動をするとどうなる？" と聞く
  Step 2: 過去エピソードからの統計予測を追加
  Step 3: 簡易ニューラルネットで近似
  Step 4: RSSM (Recurrent State Space Model) に置き換え（将来）
```

---

## 9. 開発の進め方（日次推奨）

```
各層の実装手順（1層あたり）:
  1. interface.py を書く（入出力・データ構造の定義）     ← 今ここ！
  2. テストを書く（interfaceの動作確認）
  3. 実装する（LLM呼び出し or ルールベース）
  4. テストを通す
  5. 結合テストで前後の層とつなぐ
```

---

## 10. 次のアクション（すぐやること）

1. ✅ プロジェクトフォルダ作成（済）
2. 🔲 **各層の interface.py を全部書く**
   - 全10層の入出力・データ構造をPythonの`dataclass`で定義
   - この時点では中身の実装は不要
3. 🔲 テスト用のモックLLMを用意する
4. 🔲 Phase 1 をStep1から順に実装
5. 🔲 結合テストでメインループを回す

---

*作成日: 2026-07-24*
*ベースアーキテクチャ: 3列×3行+2の10層グリッド構成*
