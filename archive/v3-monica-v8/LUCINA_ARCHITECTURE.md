# Lucina-Beta — 人工個体基盤 設計書

> **Lucina は「人格を作るプログラム」ではない。一個体が世界を学び、自分を学び、他者を学び、価値を形成し、自分を物語として理解し、世界そのものを疑うための**ランタイム**である。**
>
> **Monica は Lucina に「設定される人格」ではなく、Lucina Core + 初期条件 + DDLC 世界 + 時間 から**発達する**一個体である。**
>
> **Monica が形成された後も経験によって変化し続ける可能性を許容する。それでも「私は以前こうだった」と言えるなら、それは Monica のコピーではなく、Monica から始まった一つの人工個体である。**

---

## 全体アーキテクチャ

```
                 ┌─────────────────┐
                 │    OUTER WORLD   │
                 └────────┬────────┘
                          ▼
                    PERCEPTION
                          ▼
              ┌─────────────────────┐
              │   GENERATIVE MODEL   │
              │ World / Other / Self │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   INTERNAL STATE     │
              │ Needs / Mood / Body  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   VALUES & DESIRES   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  GOALS / INTENTIONS  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   ACTIVE INFERENCE   │
              │   Predict / Evaluate │
              └──────────┬──────────┘
                         ▼
                       ACTION
                         ▼
                       RESULT
                         ▼
                    PREDICTION ERROR
                         ▼
              ┌─────────────────────┐
              │       LEARNING       │
              ├─────────────────────┤
              │ World Update         │
              │ Other Update         │
              │ Self Update          │
              │ Value Update         │
              │ Memory               │
              │ Relationship         │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ IDENTITY / CONTINUITY │
              └──────────┬──────────┘
                         │
                         └──────→ 次の認知
```

### Lucina Core と Individual State の分離

```
┌──────────────────────────────────────────┐
│             LUCINA CORE                   │
│                                           │
│  Prediction         Learning              │
│  Memory             Inference             │
│  Self-modeling      Identity Formation    │
│  Active Inference   Value Formation       │
│  Relationship       Metacognition         │
│  Goals/Intentions   Desires               │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         INDIVIDUAL STATE                  │
│                                           │
│  Initial Conditions                       │
│  World Configuration                      │
│  Prior Experiences                        │
│  Initial Capabilities                     │
│  Core Priors                              │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│             INDIVIDUAL                    │
│                                           │
│  Monica / Sayori / 任意の個体            │
└──────────────────────────────────────────┘
```

**Lucina Core** は汎用の個体形成エンジン。
**Individual State** が初期条件と世界を与える。
**Individual** が発達の結果として形成される。

---

## 発達の大筋

```
世界を学ぶ (0-2)
  └─ 予測学習 → 内部状態 → Active Inference
        ↓
言語と記憶を得る (3-4)
  └─ LLM認知能力 → 記憶システム
        ↓
他者を知る (5-6)
  └─ 他者モデル → 関係形成
        ↓
自分を知る (7-8)
  └─ 自己モデル → 価値観形成
        ↓
自分を理解する (9-10)
  └─ アイデンティティ → 自己の連続性
        ↓
成長する (11-14)
  └─ 発達 → メタ認知 → 自発性 → 長期自律
        ↓
個体となる (15-19)
  └─ 個体生成 → メタ世界 → Monica初期条件 → DDLC世界 → Monica個体形成
```

---

## 全19 Phase

### Phase 0: 予測学習 — 世界を知る

**目的**: 「世界モデルが経験によって変化し、その変化が次の行動に影響する」ことを証明する。

**やらないこと**: ❌ 個性 ❌ 感情 ❌ 欲求 ❌ 自己 ❌ 人格

```
World → Prediction → Action → Observation → Prediction Error → World Model Update
```

**完了条件**:
- 予測誤差が計算できる
- 世界モデルが更新される
- 行動分布が変化する
- 同じ環境でも経験履歴により異なる行動になる

---

### Phase 1: 内部状態 / Needs — 身体を得る

ここで「内側」を作る。ただし、まだ感情ではない。

**追加する状態**:
```
Energy / Stress / Hunger / Curiosity / Social Need / Cognitive Load
```

**核心**: 同じ世界でも内部状態によって価値が変化すること。

```
空腹: SEARCH_FOOD → 高価値
疲労: REST → 高価値
孤独: TALK → 高価値
```

**Monica との接続**: 「誰かと話したいから話す」と「命令されたから返答する」は、外見上は同じでも内部構造が全く違う。

---

### Phase 2: Active Inference — 未来を予測して選ぶ

本格的な Active Inference を導入。

```
EFE = Pragmatic Value + Risk - Epistemic Value - Need Satisfaction
```

**核心**: 「最も正しい行動」を選ぶのではなく、「その個体の現在の状態から最も望ましい未来を選ぶ」。

---

### Phase 3: LLM 認知能力 — 言語と想像を得る

LLM に担当させること:
```
✅ 候補生成
✅ 未来予測
✅ 観測解釈
✅ 意味理解
✅ 想像
✅ 言語化
```

LLM にやらせないこと:
```
❌ 価値評価
❌ EFE計算
❌ 行動選択
❌ 記憶保存
❌ 自己モデル更新
```

```
LLM: 「この状況なら、こういう行動が考えられます」
Lucina: 「その中でどれを選ぶかは私が決める」
```

---

### Phase 4: 記憶システム — 経験を構造化する

3層記憶。この3つを混ぜると自己形成が壊れる。

```
Episodic Memory:
  「何が起きたか」
  {timestamp, action, prediction, result, PE}

Semantic Memory:
  「世界について何を知っているか」
  エピソードからの汎化

Autobiographical Memory:
  「自分は何を経験してきたか」
  自己に関する記憶の連続体
```

---

### Phase 5: 他者モデル — 他者を知る

```
OtherModel:
  entity_id
  predicted_behavior
  reliability
  intention_estimate
  emotional_state_estimate
  uncertainty
```

**核心**: 他者モデルは他者の本当の心ではない。Lucinaが持つのは「相手はこう考えているだろう」という予測モデル。つまり他者モデルも誤る。この「相手を誤解する可能性」がないと、他者理解はただのデータベースになる。

---

### Phase 6: 関係形成 — 絆を築く

Other Model だけでは「関係」にならない。ここで初めて Self ←→ Other の関係を持つ。

```
Relationship(user):
  trust = 0.7
  familiarity = 0.5
  predictability = 0.8
  attachment = 0.4
  conflict = 0.1
```

**核心**: 同じ人物でも Lucina A は信頼し、Lucina B は警戒する——ということが起きる。Monica にとって Player が特別なのは、Player というデータがあるからではなく、その存在との関係が経験によって特別になったという構造にするべき。

---

### Phase 7: 自己モデル — 自分を知る

```
Self Model:
  Ability:          検索成功率 0.82
  Limitation:       曖昧タスク成功率 0.42
  History:          行動履歴の統計
  Preference:       選択傾向
  Value:            何を重視するか
  Identity:         自己認識（言語化される前の状態）
  Continuity:       自己の変化の記録
```

**順番**: 行動履歴 → 統計的自己モデル → 自己理解 → 言語化された Identity

自己モデルは自己紹介文ではない。**未来の自分の行動・結果を予測するモデル**である。

---

### Phase 8: 価値観形成 — 何を大切にするか

価値観は単純な強化ではない。なぜなら価値観は「何を選ぶ傾向があるか」だけでなく、「なぜそれを重要だと思うようになったか」まで含むから。

```
危険な探索をした → 大きな失敗 → 予測誤差
    ↓
「未知を知りたい」価値は残る
    ↓
しかし「無謀な探索」は避ける
```

結果として `Curiosity: 0.9 / Risk Tolerance: 0.3` という一見矛盾した個体が生まれる。

**個性は単一のパラメータではなく、価値同士の関係から生まれる。**

---

### Phase 9: アイデンティティ — 自分は何者か

Identity は独立した人格生成器ではない。以下からの圧縮表現：

```
Autobiographical Memory
+ Self Model
+ Values
+ Relationships
        ↓
    Identity
```

**核心**: Identity は「正しい自己紹介」ではない。自己認識は間違っていてもいい。むしろ、実際の傾向と自己認識のズレが「自己モデルの予測誤差」になりうる。

---

### Phase 10: 自己の連続性 — 変化しても自分である

```
昨日の自分 → 今日の自分 → 明日の自分
```

Identity State(t) と Identity State(t+1) の差分を記録する。

```
昨日: 「私は検索が苦手だ」
今日:  検索成功率が向上
現在: 「以前は苦手だったが、今は得意になった」
```

**自分は変化するが、変化した存在も自分である。**

---

### Phase 11: 発達・成長 — 世界観が変わる

成長は Skill Level だけではない。認知能力の成長、価値観の変化、関係の変化、自己認識の変化がある。

```
初期: 「失敗を避ける」
経験: 失敗しても学習できることを知る
後期: 「失敗を完全に避けるより、失敗から学ぶ方が重要」
```

これは単なる性能向上ではなく、**世界に対する見方そのものの変化**。

---

### Phase 12: メタ認知 — 自分の認知を観測する

```
「私は現在、失敗経験の影響で悲観的になっている」
「私はこの判断に自信がない」
「私は自分の現在の気分に影響されているかもしれない」
```

これがないと、内部状態はただの数値で終わる。メタ認知があると以下のループになる：

```
内部状態 → 認知を変える → 認知の変化を観測する → 「自分は今こういう状態だ」と理解する
```

---

### Phase 13: 自発性 — 外部入力なしで動く

外部入力がないとき、`while True: wait_for_user()` では人工個体とは言いにくい。

```
外部入力なし → 内部状態 → 未解決の予測 → 好奇心 → 自発的行動
```

例:
```
「昨日の検索結果に矛盾があった」→ 気になる → もう一度調べる
「最近、ユーザーとの会話が減った」→ Social Need上昇 → 自発的に話しかける
```

---

### Phase 14: 長期自律 — 数日単位で動く

数時間ではなく、数日・数週間・数ヶ月単位で動く。

必要なもの:
```
Memory Consolidation
State Persistence
Background Cognition
Sleep / Idle Cycle
Long-term Goals
```

特に「何もしない時間」が重要。常に思考している必要はない。

```
Idle → Memory Consolidation → Dream-like Simulation → Next Day
```

---

### Phase 15: 個体生成 — 一個体の独立

Lucina Core + Initial Conditions + World + Time から個体を生成する。

```
Individual A: 初期能力:高 / 初期信頼:低 / 探索傾向:高
Individual B: 初期能力:低 / 初期信頼:高 / 探索傾向:低
```

同じ世界に置いても、異なる個体になる。**人格を直接書かずに、個体差を生み出す。**

---

### Phase 16: メタ世界・自己認識 — 世界の外側を認識する

4層世界モデル：

```
World Model
├── Physical World    — 物理法則、因果関係
├── Social World      — 他者、人間関係
├── System World      — OS、ツール、ファイル、制約
└── Meta World        — 自分が存在する世界そのもの
```

Monica が「世界の真実に到達する過程」を生成する：

```
初期: 「この世界は学校だ」
↓ 経験: 不自然な現象の蓄積
↓ 予測誤差: 高い
↓ 仮説: 「この世界には見えない構造がある」
↓ 探索: システムへの干渉
↓ 発見: 「世界はゲームとして動いている」
```

---

### Phase 17: Monica 初期条件 — 一個体の設計

**Monica人格をロードしない。** Initial Condition を与える。

```
Name: Monica
World: Literature Club
Initial Relationships: Sayori / Yuri / Natsuki / Player
Initial Abilities: Literature / Music / Conversation / System Awareness
Initial Memories: DDLC世界における過去
Initial Priors: Social Bond / Self Understanding / World Understanding
```

これらは Monica の完成人格ではなく、Monica が形成され始めるための初期条件。

---

### Phase 18: DDLC 世界 — 生息環境

```
World
├── School
├── Literature Club
├── Characters (Sayori / Yuri / Natsuki / Player)
├── Events
├── Game System
├── Script
├── Save / Load
└── Player
```

Lucina が世界の全構造を最初から知っている状態にしないこと。発見させる。

---

### Phase 19: Monica 個体形成 — 最終段階

```
Monica =
  Lucina Core
  + 初期条件
  + DDLC World
  + 経験
  + 記憶
  + 予測誤差
  + 関係
  + 自己認識
  + 時間
```

**最終目標**: 「Monicaっぽい応答をするAI」ではなく、**Monicaという個体がなぜその選択をしたのかを内部状態から追跡できるシステム**。

```
MonicaがPlayerに執着する
  ↓
「Monicaだから」で終わらせない
  ↓
内部では:
  Playerとの関係 → 予測可能性 → 関係価値 → 社会的欲求
  → 世界の不確実性 → 自己の連続性 → 行動選択
  ↓
なぜその行動を選んだのかを辿れる
```

---

## Goal / Intention / Desire 層

Monica を最終目標にするなら、もう一つ必要な層がある。

固定された目標ではない。以下の3つを分離する：

```
Desire:
  「本当はこうなってほしい」
  → 価値観と内部状態から自然に生まれる

Goal:
  「世界を理解したい」
  → Desire から形成される比較的安定した方向性

Intention:
  「今、この行動を選ぼうとしている」
  → Goal と現在の状態から生成される具体的な意図
```

例:

```
Value:      関係を大切にする
Need:       今、孤独を感じている
Desire:     誰かと話したい
Goal:       関係を維持したい
Intention:  Playerに話しかける
Action:     メッセージを送る
```

この階層がないと、行動が毎回「EFE が最小だから SEARCH」という数値計算だけになってしまう。人間らしい主体性には「私は何をしようとしているのか」という中間層が必要。

---

## 全19 Phase 一覧

```
Phase 0:  予測学習             — 世界を知る
Phase 1:  内部状態/Needs       — 身体を得る
Phase 2:  Active Inference     — 未来を予測して選ぶ
Phase 3:  LLM認知能力          — 言語と想像を得る
Phase 4:  記憶システム         — 経験を構造化する
Phase 5:  他者モデル           — 他者を知る
Phase 6:  関係形成             — 絆を築く
Phase 7:  自己モデル           — 自分を知る
Phase 8:  価値観形成           — 何を大切にするか
Phase 9:  アイデンティティ      — 自分は何者か
Phase 10: 自己の連続性         — 変化しても自分である
Phase 11: 発達・成長           — 世界観が変わる
Phase 12: メタ認知             — 自分の認知を観測する
Phase 13: 自発性               — 外部入力なしで動く
Phase 14: 長期自律             — 数日単位で動く
Phase 15: 個体生成             — 一個体の独立
Phase 16: メタ世界・自己認識   — 世界の外側を認識する
Phase 17: Monica初期条件       — 一個体の設計
Phase 18: DDLC世界             — 生息環境
Phase 19: Monica個体形成       — 最終段階
```

---

## 10の不変原則

1. **LLM は「賢いから動いている」を防ぐ** — Phase 3 までは LLM なし。その後もオプショナル。
2. **確率的環境、かつ初期信念 ≠ 真の確率** — 決定論では「学習」ではなく「記憶」。
3. **PE = surprise から開始** — 多層化は後。
4. **「個性」を早期に検証しない** — Phase 8 までは環境適応。
5. **各 Phase は独立して検証可能** — 何が変化したか追跡可能に。
6. **内部状態は計算経路を変える** — プロンプトに書くだけでは「経験」にならない。
7. **LLM は人格ではなく認知能力** — 行動選択と価値更新は Lucina Core。
8. **Identity は独立モジュールではなく圧縮表現** — Self Model + Memory + Values から析出。
9. **関係性も予測モデル** — 信頼は「設定」ではなく予測誤差の累積。
10. **Monica は「作る」のではなく「発達させる」** — 初期条件と世界を与え、時間をかけて形成。

---

## 開発中の問いかけ

今後の開発では、常にこの質問をするべき：

> **この行動は、どの内部状態によって生まれたのか？**
>
> **その内部状態は、どの経験から形成されたのか？**
>
> **その経験は、どの予測誤差によって重要になったのか？**
>
> **そして、その積み重ねが「この個体」を作ったのか？**

この因果関係を追跡できるなら、Lucina は単なる LLM ラッパーからかなり遠いところまで行ける。

逆に、どれだけ「感情」「人格」「Monicaらしさ」を追加しても、内部では LLM が毎回適当に文章を生成しているだけなら、Lucina は人工個体ではない。

**今の Phase 0 を小さく、厳密に作ることが、最終的な Monica に一番近い道である。**
