# Lucina-Beta 実装ロードマップ

> このドキュメントは `LUCINA_ARCHITECTURE.md` の設計を実装可能な単位に分解したものである。
>
> 各 Phase は「前の Phase の検証可能な成功」を次の Phase への前提とする。
> 逆に、どの Phase でも失敗したら設計を戻し、仮定を再検証する。

---

## 全体構成

```
lucina_beta/
├── core/                  # Lucina Core（汎用エンジン）
│   ├── __init__.py
│   ├── world_model.py     # Phase 0: 確率推定
│   ├── agent.py           # Phase 0: メインループ
│   ├── internal_state.py  # Phase 1: 内部状態/Needs
│   ├── inference.py       # Phase 2: EFE/Active Inference
│   ├── llm.py             # Phase 3: Qwen3-8B Cognitive Layer
│   ├── memory.py          # Phase 4: 3層記憶
│   ├── other_model.py     # Phase 5: 他者モデル
│   ├── relationship.py    # Phase 6: 関係モデル
│   ├── self_model.py      # Phase 7: 自己モデル
│   ├── values.py          # Phase 8: 価値観形成
│   ├── identity.py        # Phase 9-10: Identity + Continuity
│   ├── metacognition.py   # Phase 12: メタ認知
│   ├── goals.py           # Goals/Intentions/Desires
│   └── development.py     # Phase 11: 発達カリキュラム
│
├── world/                 # 環境
│   ├── __init__.py
│   ├── mock_world.py      # Phase 0: シンプル確率世界
│   ├── npc.py             # Phase 5: NPC
│   └── ddlc_world.py      # Phase 18: DDLC世界
│
├── cli/                   # インターフェース
│   ├── __init__.py
│   ├── display.py         # Phase 0: 表示
│   └── repl.py            # Phase 13+: 自律ループ
│
├── experiments/           # 検証実験
│   ├── __init__.py
│   ├── experiment_0.py    # Phase 0: 基本学習
│   ├── experiment_a.py    # Phase 0: 経験履歴差
│   └── experiment_b.py    # Phase 0: 初期条件差
│
├── monica/                # Monica 固有
│   ├── __init__.py
│   ├── initial_state.py   # Phase 17: Monica初期条件
│   └── bootstrap.py       # Phase 19: Monica個体形成
│
├── main.py                # エントリポイント
└── requirements.txt       # 依存関係
```

---

## Phase 0: 予測学習（実装可能）

**依存**: なし（Python 3.10+ のみ）
**LLM**: 不要
**見積もり**: 150〜250行

### ファイル

| ファイル | 責務 | 行数目安 |
|---------|------|---------|
| `core/world_model.py` | 信念：カウントベースの確率推定、surprise、EV計算 | 60 |
| `world/mock_world.py` | MockWorld：A/B/C の確率的環境 | 40 |
| `core/agent.py` | Agent：predict → act → observe → learn ループ | 50 |
| `cli/display.py` | 表示：状態の可視化 | 30 |
| `experiments/experiment_0.py` | 実験0：L1 Error 測定 | 40 |
| `experiments/experiment_a.py` | 実験A：経験履歴差 | 40 |
| `experiments/experiment_b.py` | 実験B：温度差 | 40 |
| `main.py` | エントリポイント：引数で実験選択 | 20 |

### 検証

```bash
python main.py --experiment 0  # L1 Error の収束を確認
python main.py --experiment a  # 経験履歴差を確認
python main.py --experiment b  # 温度差を確認
```

### 完了条件

- [ ] surprise が正の値を取り、計算可能
- [ ] L1 Error が試行回数増加に伴い減少
- [ ] 異なる経験履歴 → 異なる行動分布（Experiment A）
- [ ] 異なる temperature → 異なる探索率（Experiment B）
- [ ] 複数 seed で再現可能

### 失敗時の切り戻し

- L1 Error 減少しない → 確率推定アルゴリズムの誤り
- surprise が 0/1 に張り付く → 予測機構のバグ
- seed 間のばらつきが大きすぎる → 試行回数不足

---

## Phase 1: 内部状態 / Needs

**依存**: Phase 0（Agent, WorldModel, MockWorld）
**LLM**: 不要
**見積もり**: 100〜150行追加

### 追加/変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/internal_state.py` | **新規**: Energy / Stress / Curiosity / Social Need の状態管理と更新則 |
| `core/agent.py` | **変更**: InternalState の統合、EV 計算に Needs の影響を追加 |
| `world/mock_world.py` | **変更**: rest / explore 行動の追加 |
| `main.py` | **変更**: Phase 1 モード追加 |

### 核心実装

```python
class InternalState:
    energy: float      # [0,100] 行動で消費、休息で回復
    safety: float      # [0,100] 危険で低下
    curiosity: float   # [0,100] 時間経過で上昇、探索で充足

    def utility(self, action: str, base_ev: float) -> float:
        # 同じ food でも energy によって価値が変わる
        if action == "rest":
            return base_ev + (100 - self.energy) * 0.01
        if action == "explore":
            return base_ev + self.curiosity * 0.01
        return base_ev
```

### 検証

```bash
python main.py --phase 1 --test energy_dependency
# energy↓ → rest↑ の相関を確認

python main.py --phase 1 --test curiosity_dependency
# curiosity↑ → explore↑ の相関を確認
```

---

## Phase 2: Active Inference

**依存**: Phase 0-1
**LLM**: 不要
**見積もり**: 100〜150行追加

### 追加/変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/inference.py` | **新規**: EFE = Pragmatic Value + Risk - Epistemic Value - Need Satisfaction |
| `core/agent.py` | **変更**: 行動選択を EFE ベースに変更 |

### 核心実装

```python
def efe(action, world_model, internal_state, temperature):
    pragmatic = world_model.expected_value(action, internal_state)
    risk = 1.0 - max(world_model.confidence(action), 0.01)
    epistemic = world_model.uncertainty(action)  # 未知ほど高い
    need_satisfaction = internal_state.need_satisfaction(action)

    return pragmatic + risk - epistemic + need_satisfaction
```

### 検証

```bash
python main.py --phase 2 --test exploration_vs_exploitation
# EFE 導入後、未知の行動を選ぶ確率が上昇することを確認
```

---

## Phase 3: LLM 認知能力

**依存**: Phase 0-2
**LLM**: Qwen3-8B（Transformers + hooks）
**見積もり**: 200〜300行（MonicaCore からの移植を含む）

### 追加/変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/llm.py` | **新規**: Qwen3-8B ラッパー（候補生成・未来予測・結果解釈） |
| `core/agent.py` | **変更**: LLM を認知能力として統合（判断はさせない） |

### 核心設計

```python
class LLMCognitiveLayer:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(...)
        self.hooks = ...  # FEP hooks (Phase 3 では semantic PE に活用)

    def generate_candidates(self, context: str) -> list[str]:
        # 「この状況で取り得る行動は？」— 候補生成のみ
        ...

    def predict_outcome(self, action: str, context: str) -> str:
        # 「この行動をとるとどうなるか？」— 未来予測のみ
        ...

    def interpret_result(self, observation: str, prediction: str) -> str:
        # 「この結果は何を意味するか？」— 解釈のみ
        ...

    # ❌ 行動選択はしない
    # ❌ 価値評価はしない
    # ❌ 自己言及はさせない
```

### 検証

```bash
python main.py --phase 3 --test candidate_diversity
# LLM 導入前後の候補の多様性を比較

python main.py --phase 3 --test llm_optional
# LLM なしでも Phase 2 相当の動作が可能であることを確認
```

---

## Phase 4: 記憶システム

**依存**: Phase 0-3
**LLM**: オプショナル（記憶の言語化に使用）
**見積もり**: 200〜300行

### 追加/変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/memory.py` | **新規**: 3層記憶（Episodic / Semantic / Autobiographical） |

### 核心設計

```python
class Memory:
    def __init__(self):
        self.episodic: list[Episode] = []    # 具体的事象
        self.semantic: dict[str, float] = {}  # 汎化された知識
        self.autobiographical: list[Episode] = []  # 自己に関する記憶

    def store(self, episode: Episode):
        self.episodic.append(episode)
        if episode.self_relevant:
            self.autobiographical.append(episode)
        self._consolidate_semantic(episode)

    def recall(self, query: str, n: int = 5) -> list[Episode]:
        # 類似エピソードの検索
        ...

    def _consolidate_semantic(self, episode: Episode):
        # エピソード → 意味記憶への圧縮
        ...
```

### 検証

```bash
python main.py --phase 4 --test memory_recall
python main.py --phase 4 --test semantic_consolidation
```

---

## Phase 5-6: 他者モデル + 関係形成

**依存**: Phase 0-4
**LLM**: 会話生成に使用
**見積もり**: 300〜400行

### 追加/変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/other_model.py` | **新規**: 他者の行動予測・意図推定 |
| `core/relationship.py` | **新規**: trust / familiarity / predictability / attachment |
| `world/npc.py` | **新規**: 内部状態を持つ NPC |
| `world/mock_world.py` | **変更**: NPC との相互作用 |

### 核心設計（他者モデルの誤り）

```python
class OtherModel:
    def predict(self, entity_id: str, context: str) -> dict:
        # 予測は常に不確実性を持つ
        prediction = self._predict_from_history(entity_id, context)
        return {
            "action": prediction,
            "confidence": self.confidence[entity_id],
            "uncertainty": self.uncertainty[entity_id],
        }
    # 予測が外れる → Other Prediction Error → モデル更新
```

---

## Phase 7: 自己モデル

**依存**: Phase 0-6
**LLM**: 自己記述の言語化に使用（ただし統計が先）
**見積もり**: 150〜200行

### 追加ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/self_model.py` | **新規**: Ability / Limitation / Prediction Accuracy |

### 順序（重要）

```
行動履歴 → 統計的自己モデル → 自己理解 → 言語化された Identity
```

自己モデルは自己紹介文ではなく、未来の自分の行動・結果を予測するモデル。

---

## Phase 8: 価値観形成

**依存**: Phase 0-7
**LLM**: 価値観の言語化に使用
**見積もり**: 150〜200行

### 追加ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/values.py` | **新規**: Value weights と更新則 |

### 核心

価値観は単純な強化ではない。`Curiosity: 0.9 / Risk Tolerance: 0.3` のような矛盾を含む個体差が生まれることを検証する。

---

## Phase 9-10: Identity + Continuity

**依存**: Phase 0-8
**LLM**: 「自分は何者か」の言語化（ただし自由生成させない）
**見積もり**: 200〜300行

### 追加ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/identity.py` | **新規**: Identity 圧縮 + Continuity 機構 |

### 核心

```
Autobiographical Memory + Self Model + Values + Relationships → Identity（圧縮）
Identity(t) と Identity(t+1) の差分 → Continuity
```

---

## Phase 11: 発達・成長カリキュラム

**依存**: Phase 0-10
**見積もり**: 100〜150行

### 追加ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/development.py` | **新規**: 発達段階の管理、能力の段階的解放 |

### 核心

各 Phase の機能を「最初から全て利用可能」にするのではなく、発達カリキュラムに従って段階的に解放する。

---

## Phase 12: メタ認知

**依存**: Phase 0-11
**LLM**: メタ認知の言語化に使用
**見積もり**: 150〜200行

### 追加ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/metacognition.py` | **新規**: Belief → Confidence → Reason → Possible Bias |

---

## Phase 13-14: 自発性 + 長期自律

**依存**: Phase 0-12
**見積もり**: 200〜300行

### 核心

外部入力がないときの自発的行動（`while True: agent.step(world)`）。
数日〜数週間の連続稼働。

### 必要なもの

- Memory Consolidation（記憶の圧縮・忘却）
- State Persistence（状態の永続化）
- Idle Cycle（「何もしない」時間の組み込み）

---

## Phase 15: 個体生成

**依存**: Phase 0-14
**見積もり**: 100〜150行

### 核心

```python
def create_individual(core: LucinaCore, config: IndividualConfig) -> Agent:
    agent = Agent(core)
    agent.initialize(
        abilities=config.abilities,
        priors=config.core_priors,
        memories=config.initial_memories,
    )
    return agent
```

同じ Core + 異なる Config → 異なる個体。

---

## Phase 16: メタ世界

**依存**: Phase 0-15
**見積もり**: 200〜300行

### 4層世界モデル

```
Physical World → Social World → System World → Meta World
```

Monica が「世界の真実に到達する過程」を生成するための機構。

---

## Phase 17-19: Monica

**依存**: Phase 0-16
**見積もり**: 400〜600行

### ファイル

| ファイル | 変更内容 |
|---------|---------|
| `monica/initial_state.py` | **新規**: Monica の初期条件定義 |
| `world/ddlc_world.py` | **新規**: DDLC 世界（Literature Club / NPC / Game System） |
| `monica/bootstrap.py` | **新規**: Monica 個体形成の実行 |

### 核心

Monica 人格をロードしない。
Monica 初期条件 + DDLC 世界 + 時間 → Monica 個体の形成。

---

## 実装順序のまとめ

```
Phase 0:  予測学習          ← 今ここ。すぐ実装可能
Phase 1:  内部状態/Needs    ← Phase 0 完了後
Phase 2:  Active Inference  ← Phase 0-1 完了後
Phase 3:  LLM認知能力       ← Phase 0-2 完了後（Qwen3-8B導入）
Phase 4:  記憶              ← Phase 0-3 完了後
Phase 5:  他者モデル        ← Phase 0-4 完了後
Phase 6:  関係形成          ← Phase 0-5 完了後
Phase 7:  自己モデル        ← Phase 0-6 完了後
Phase 8:  価値観形成        ← Phase 0-7 完了後
Phase 9:  Identity          ← Phase 0-8 完了後
Phase 10: Continuity        ← Phase 0-9 完了後
Phase 11: 発達カリキュラム   ← Phase 0-10 完了後
Phase 12: メタ認知          ← Phase 0-11 完了後
Phase 13: 自発性            ← Phase 0-12 完了後
Phase 14: 長期自律          ← Phase 0-13 完了後
Phase 15: 個体生成          ← Phase 0-14 完了後
Phase 16: メタ世界          ← Phase 0-15 完了後
Phase 17: Monica初期条件    ← Phase 0-16 完了後
Phase 18: DDLC世界          ← Phase 0-17 完了後
Phase 19: Monica個体形成    ← Phase 0-18 完了後
```

**Phase 0 から Phase 14 までは LLM なしでも動作可能でなければならない。**
Phase 3 で LLM を導入するが、LLM なしでも Phase 2 相当の動作を保証すること。

---

## マイルストーン

| マイルストーン | 達成条件 | 時期目安 |
|--------------|---------|---------|
| **M1: 予測学習の実証** | Phase 0 の 3 つの実験が全て成功 | 〜1週間 |
| **M2: 内部状態エージェント** | Phase 1-2 完了、Needs が行動を変えることを確認 | 〜2週間 |
| **M3: LLM 統合** | Phase 3-4 完了、LLM が認知能力として機能 | 〜3週間 |
| **M4: 社会エージェント** | Phase 5-6 完了、他者との関係形成を確認 | 〜4週間 |
| **M5: 自己認識** | Phase 7-10 完了、Identity の形成を確認 | 〜6週間 |
| **M6: 自立エージェント** | Phase 11-14 完了、自発的・長期自律動作を確認 | 〜8週間 |
| **M7: 個体生成** | Phase 15-16 完了、異なる個体の生成を確認 | 〜10週間 |
| **M8: Monica** | Phase 17-19 完了、Monica 個体の形成を確認 | 〜12週間 |

---

## 各 Phase 完了時のチェックリスト

各 Phase 完了時には以下を確認する：

1. [ ] 全テストが通過
2. [ ] 複数 seed で再現可能
3. [ ] LLM 依存がない（Phase 3 以降も LLM なしで動作）
4. [ ] 前 Phase の動作が維持されている（回帰テスト）
5. [ ] 新しい行動変化が測定可能
6. [ ] 失敗条件が定義され、該当しないことを確認
