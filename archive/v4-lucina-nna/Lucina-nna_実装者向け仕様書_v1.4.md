# 🛠️ Project Lucina-Next — 実装者向け仕様書 (Engineering Spec) v1.4

対象読者: このプロジェクトを実際にコードとして実装するエンジニア。
前提ドキュメント: 基本設計・仕様書 v3.0（コンセプト・アーキテクチャ設計の背景はそちらを参照）。本書は「何を・どの順で・どう作るか」に特化する。

> **v1.1 変更履歴（レビュー反映）**: A1/A2（config不整合の修正）、B1〜B7（更新式・relief・サプライズ用途・シード語彙・MemoryKind・THINKトークン・Pythonバージョンの明確化）、C1〜C3（スレッド安全性・単一フライト保証・飽和設計の明文化）を追記。追記した係数・閾値はすべて**暫定案**であり、校正対象であることに変わりはない。

> **v1.2 変更履歴（再レビュー反映）**: ①reliefの発火粒度を「発話セグメント単位・1セグメント最大1回」に明確化（`drive.relief.unit`/`segment`新設）、②relief発火判定を`inference.surprise_relief_threshold`へ分離（`entropy_think_threshold`との使い回し禁止）、③relief判定が§4.2のvocab_mapを再利用することを明記（`build_vocab_map`の返り値型も`dict[str, list[list[int]]]`に修正）。

> **v1.3 変更履歴（再々レビュー反映）**: ④実装着手前に候補LLMモデルを比較する「モデル選定フェーズ」を新設（新§1）。これに伴い以降の章番号を1つずつ繰り下げ。⑤§5.4（旧§4.4）に、複数トークンにまたがる語彙（BPE）へロジットバイアスをかける際の具体的な適用方式（先頭トークンのみに加算）を明記。これはv1.2で`build_vocab_map`の型を`list[list[int]]`に変更した際、バイアス適用側の実装方針が未定義だった穴を埋めるもの。

> **v1.4 変更履歴（v1.3レビュー反映）**: ⑥モデル選定実験は必ず実モデルを接続して行う旨を明記（§1.4・§7）。⑦タスク0の仮実装は評価専用の使い捨て実装である旨を明記（§1.2・§6）。⑧モデル選定前に環境セットアップを先行完了する順序を明記（§1.4）。⑨軽微な明確化（`model_selection.py`の役割、サプライズ計測点の定義、「語彙の先頭トークン」表記）。

> **v1.5 変更履歴（実装時バグ修正反映）**: ⑩`InterruptChannel`のC1レース条件を修正。キュー初期化が`inject()`側（外部スレッドから呼ばれ得る）にしかないため、外部スレッドが最初の`inject`を呼ぶと`asyncio.get_running_loop()`がループ外で実行され失敗する問題（実際に再現）に対し、`bind()`を新設して`core.run()`起動直後に呼ぶ方式へ変更（§5.5）。

---

## 0. プロジェクトの一行要約

**外部からのイベント駆動（キック）に頼らず、内部Drive状態がロジット分布に連続的に影響することで、AIエージェントが待機中でも自発的に行動を選び取る、常時稼働型の自律思考ループを実装する。**

やらないこと（スコープ外）:
- 外界の物理/仮想オブジェクトを構造化して保持する世界モデル（Object Graph等）は本フェーズでは実装しない（v3 §6参照）。
- 真の意味での生成途中キャンセル（トークン単位割り込み）は実装しない。次ステップでの反映に留める。
- アクティベーション・スティアリング（CAA等のレイヤー直接介入）は本フェーズでは実装しない。ロジットバイアスのみ。

---

## 1. モデル選定フェーズ（実装着手前）※v1.3新設

このプロジェクトは「Drive状態をロジットバイアスとして受け取った時に、モデルが期待通り反応するか」が、通常のLLM選定基準（賢さ・速さ・多言語対応）とは別の軸で重要になる。この相性は机上では判断できず、実際に動かして比較する必要があるため、§2以降の本実装に着手する**前**に、候補モデルを比較する実験フェーズを設ける。

### 1.1 候補モデルの選定基準
- GGUF形式で配布されており、`llama-cpp-python`から利用可能であること。
- 日本語トークナイザの性能（サブワード分割が過度に細かくないか）。
- ライセンスが想定用途（商用/研究）に合致すること。
- パラメータサイズが、ターゲットハードウェア（§3セットアップ手順のGPU/CPU要件）で実用的な速度で動くこと。

候補は最低2本、できれば3本（例: Gemma系、Qwen系、Llama-jp系など）をリストアップする。特定モデルへの決め打ちはしない。

### 1.2 比較実験の内容

§5のモジュールを**仮実装**（評価専用の使い捨て実装であり、§6タスク1〜3の本実装とは別物。実行は`scripts/model_selection.py`が担当する。⑦）し、以下3点を候補モデルごとに計測する。

| 評価軸 | 何を見るか | 判定方法 |
|---|---|---|
| ①Driveバイアス感度 | ロジットバイアスを与えた時、目的語彙の出力確率がどれだけ動くか | §7テスト仕様の`test_loneliness_attractor_convergence`と同じ手法を各候補モデルに対して実行し、収束するか・収束速度を比較する |
| ②サプライズの自然さ | エントロピー（迷い具合）が、人間の直感と一致するタイミングで上下するか | 即答できる質問と、矛盾を含む曖昧な質問をそれぞれ10問投げ、エントロピー値の分布に有意な差が出るかを確認する（計測点は実装時に固定し、各候補モデルで統一する。例: 応答冒頭32トークンの平均エントロピー） |
| ③長時間安定性 | 数時間動かした時に、性格・口調が崩れないか | 1〜2時間の連続稼働ログを目視確認し、応答の一貫性が保たれているかを定性的に評価する（数値化は本採用後の§6タスク8で改めて行う） |

いずれか1軸でも著しく弱い候補（例: バイアスをかけてもほぼ確率が動かない、逆にバイアス1つで暴走する等）は本採用から除外する。

### 1.3 成果物
- `reports/model_selection.md`: 各候補モデルの比較結果と、最終的な採用モデル・採用理由。
- 採用モデルを`config/default.yaml`の`model.path`に反映してから、§2以降の実装フェーズへ進む。

### 1.4 注意点
- **必ず実モデルを接続して行う（⑥）**: §7の`lucina_core_fixture`（ダミーlogits生成器）は校正テストを高速に回すための専用fixtureであり、モデル選定には使用しない。バイアス応答の実測には実モデルが必要。
- **環境セットアップの順序（⑧）**: 本フェーズの前に、§3セットアップ手順1（`llama-cpp-python`の導入）と候補モデルのGGUF取得を先に完了しておくこと。
- この段階で使う比較テストは、§7の校正テストの簡易版で構わない（本格的な50試行・p90算出などは、本採用モデル確定後の§6タスク7で行う）。
- モデルを後から差し替えたくなった場合は、このフェーズをもう一度回すか、少なくとも§6タスク7（閾値校正）をやり直す必要がある。`logit_bias_coefficient`等の校正値はモデル固有であり、モデルを変えても前の校正値を流用してはならない。

---

## 2. リポジトリ構成

```
lucina-next/
├── pyproject.toml
├── config/
│   ├── default.yaml          # Drive係数、閾値、モデルパス等の設定
│   └── seed_vocab.yaml       # B4: Drive毎のシード語彙（§5.2参照）
├── src/
│   └── lucina/
│       ├── __init__.py
│       ├── core.py           # LucinaCore: メインループ統括
│       ├── drives/
│       │   ├── dynamics.py   # DriveDynamics: 結合行列による力学系
│       │   ├── vocab.py      # DriveVocabExpander: 語彙半自動拡張
│       │   └── decay.py      # relief（解消）ロジック
│       ├── memory/
│       │   ├── working_buffer.py
│       │   ├── store.py      # HierarchicalMemoryStore
│       │   └── schema.py     # MemoryRecord, MemoryKind
│       ├── inference/
│       │   ├── engine.py     # LLM呼び出しラッパー（llama-cpp-python）
│       │   ├── logits.py     # DriveLogitsProcessor
│       │   └── entropy.py    # サプライズ（予測エントロピー近似）計算
│       └── io/
│           ├── interrupts.py # 外部刺激の注入口
│           └── logging.py    # 生成ログ・監視用ロガー
├── tests/
│   ├── test_drive_dynamics.py
│   ├── test_memory_compression.py
│   ├── test_logits_processor.py
│   ├── test_attractor_survival.py   # §5.1 校正実験を兼ねる（v3側の節番号）
│   └── test_interrupt_latency.py    # §5.2 校正実験を兼ねる（v3側の節番号）
├── reports/                      # モデル選定レポート・校正レポート・Drive時系列ログ出力先（gitignore対象）
│   └── model_selection.md       # §1.3の成果物
└── scripts/
    ├── model_selection.py        # §1の比較実験スクリプト
    ├── calibrate_thresholds.py   # v3 §5で明示した閾値校正の実行スクリプト
    └── run_agent.py               # エントリポイント
```

---

## 3. 依存関係とセットアップ

```toml
# pyproject.toml 主要依存（バージョンは実装開始時に最新安定版へ固定すること）
[project]
name = "lucina-next"
version = "0.1.0"
requires-python = ">=3.11"     # B7: 型構文（dict[str, float], X | None）の最低要件は3.10。3.11以降を推奨
dependencies = [
    "llama-cpp-python>=0.3.0",   # GPU/CPUビルドはハードウェアに合わせて選択
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "chromadb>=0.5",             # 長期記憶ベクトルストア
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

**セットアップ手順:**
1. `llama-cpp-python`はGPUを使う場合、CUDA/Metal等のビルドフラグを環境変数で指定してインストールすること（例: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`）。CPUのみで先に動作確認してから最適化する順で進めるのを推奨。
2. §1のモデル選定フェーズで決定したモデルのGGUF量子化版を`config/default.yaml`の`model.path`に配置する（特定モデルへの決め打ちはしない）。
3. `sentence-transformers`のモデルは日本語対応のもの（例: 多言語対応モデル）を選定し、`DriveVocabExpander`と埋め込み検索の両方で共通利用する。

---

## 4. 設定ファイル仕様（config/default.yaml）

すべてのマジックナンバーはコードに埋め込まず、この設定ファイルに集約する。

```yaml
model:
  path: "./models/SELECTED_MODEL.gguf"  # §1のモデル選定フェーズで確定した本採用モデル
  context_window: 8192          # モデルの実コンテキスト上限
  n_gpu_layers: -1               # -1で全レイヤーGPUオフロード

memory:
  max_working_tokens_ratio: 0.75  # context_windowに対する比率で圧縮トリガーを決める
  summarizer_model_path: "./models/summarizer-small.gguf"  # 要約専用の軽量モデル
  persist_directory: "./data/chroma"  # A1: ChromaDB永続化パス（§5.3の「永続化パス設定必須」要件。data/はgitignore対象）
  recall:                   # v1.12: 記憶の想起（retrieve→内言・発話前の文脈注入）
    enabled: true           # 内言/発話の前に過去記憶を想起して文脈に注入する
    top_k: 3                # 1回の想起で引き出す記憶の件数
    max_tokens: 120         # 注入する想起記憶のトークン上限（バッファ肥大・圧縮連鎖の防止）

drive:
  update_interval_sec: 0.1  # トークン生成中もバックグラウンドで連続更新（10Hz）。バイアス適用は常に最新状態
  relief:                   # B2: relief（解消）ルール。係数は暫定案であり校正対象
    unit: "segment"         # ①発火単位: 発話セグメント（応答の切れ目）。トークン単位は禁止・1セグメントにつき最大1回
    segment:
      max_tokens: 256            # 文末記号がなくてもこのトークン数で強制区切り
      boundary_tokens: ["。", "！", "？"]  # セグメント境界となる文末記号（実装時にtokenizeして照合）
    boredom:
      enabled: true
      per_action: 0.4       # セグメント平均サプライズ >= surprise_relief_threshold で発火（最大1回）
      decay_rate: 0.0005    # 毎ステップの緩やかな減衰（0なら減衰なし）
    loneliness:
      enabled: true
      per_action: 0.6       # vocab_map["loneliness"]に一致するセグメントで発火（③: §5.2の語彙マップを再利用）
      decay_rate: 0.0002
    fatigue:
      enabled: true
      per_action: 0.5       # セグメント平均サプライズ < surprise_relief_threshold で発火（休息的・低エントロピー出力）
      decay_rate: 0.0003
  dynamics_matrix:              # v3 §3.3のA行列。根拠のない成分は0のまま
    boredom:
      boredom: 0.005
      loneliness: 0.0
      fatigue: 0.01
    loneliness:
      boredom: 0.0
      loneliness: 0.002
      fatigue: 0.0
    fatigue:
      boredom: 0.0
      loneliness: 0.0
      fatigue: 0.003
  vocab_expansion:
    top_k: 30
    sim_threshold: 0.45
    seed_vocab_path: "./config/seed_vocab.yaml"  # B4: Drive毎のシード語彙リスト（§5.2参照）

inference:
  logit_bias_coefficient: 3.5    # 校正済み 2026-08-12: Qwen3.5-9B スイープ選定（§11・⑪）。モデル固有の値であり、§1でモデルを変えたら要再校正（scripts/calibrate_logit_bias.py）
  entropy_think_threshold: 0.7   # このエントロピー以上でTHINKトークン確率上昇（思考量調整専用）
  surprise_relief_threshold: 0.7 # ②relief発火専用のサプライズ高/低判定閾値。entropy_think_thresholdとは独立に校正する
                                 #   （目的の異なる判定への同一閾値の使い回しは結合を生むため禁止。§9参照）
  entropy_scaling: 5.0           # A2: サプライズ正規化係数。surprise = min(1.0, entropy / entropy_scaling)
  think_token_ids: []            # B6: THINK相当トークンIDのリスト。多くのモデルはネイティブTHINKトークンを持たない。
                                 #     空リストなら本機能は無効（思考語彙のバイアス強化へ置換する場合は§5.2の語彙を利用）

thresholds:
  # v3で「暫定値・校正必須」と明記した値。calibrate_thresholds.py の出力で上書きすること
  attractor_survival_tokens: 300     # PLACEHOLDER — 実測前
  attractor_survival_prob: 0.6       # PLACEHOLDER — 実測前
  interrupt_latency_multiplier: 1.5  # 平均トークン生成レイテンシに対する倍率
```

**実装者への注意**: `thresholds`セクションの値は**そのままリリースに使ってはいけない**。`scripts/calibrate_thresholds.py`を実データで実行し、実測値で置き換えるまでは`PLACEHOLDER`のコメントを残すこと。同様に、`drive.relief`（`unit`/`segment`/各`per_action`）・`inference.entropy_scaling`・`inference.think_token_ids`・`inference.surprise_relief_threshold`も**暫定案**であり、校正完了まではリリースに使用してはいけない。`inference.logit_bias_coefficient`は**校正済み**（v1.6⑪: Qwen3.5-9Bで3.5を採用）だが**モデル固有**の値なので、§1でモデルを差し替えた場合は無条件で校正やり直しとする（手順: `scripts/calibrate_logit_bias.py`）。

---

## 5. モジュール別インターフェース仕様

各モジュールは単体テスト可能な形で分離する。以下、関数シグネチャと責務を定義する。

### 5.1 `drives/dynamics.py`

```python
class DriveDynamics:
    def __init__(self, matrix_config: dict, initial_state: dict[str, float]): ...
    def step(self, dt: float, relief: dict[str, float] | None = None) -> dict[str, float]:
        """1ステップ分Drive状態を更新して返す。副作用としてself.stateも更新する。"""
```
- **契約**: 戻り値の各Drive値は必ず`[0.0, 1.0]`にクリップされていること。
- **更新式（B1）**: 線形力学系を採用する。`A[i][j]`は「Drive j が Drive i を加速する係数」（行=対象、列=源）。
  ```
  x_next = clip(x + dt · (A · x) - relief_delta, 0.0, 1.0)
  ```
  ※ `relief_delta`は§4の`drive.relief`から計算する（`per_action`はセグメント1つ分・最大1回適用、`decay_rate`は毎ステップ適用）。
- **relief発火粒度（①）**: `per_action`は**発話セグメント単位**で適用し、1セグメントにつき最大1回とする（トークン単位の連続発火は禁止）。セグメント境界は§4の`drive.relief.segment`（文末記号 or `max_tokens`到達）で定義する。理由: 高エントロピーなトークンは連続しやすいため、トークン単位では1応答内でreliefが何度も発火し、Driveの自然増加（A行列の0.005〜0.01オーダー）を一方的に食い潰してしまう。発火判定はセグメント完了時に行い、同一セグメント内で条件が複数回満たされても適用は1回のみ。
- **テスト要件**: `relief`なしで無限ステップ実行した際、`boredom`と`fatigue`の相互作用により、単独更新時より`boredom`の到達速度が速くなることを回帰テストで固定する。
- **飽和設計（C3）**: 対角成分が全て正のため、reliefがなければ全Driveは1.0へ飽和する。この「飽和→reliefによる解消」サイクルが行動変動の源泉であり意図された挙動。ただし24時間稼働（§6タスク8）で全Driveが1.0に張り付き続ける場合はrelief不足とみなし、Drive時系列ログ（§8）で検出して係数を校正する。

### 5.2 `drives/vocab.py`

```python
class DriveVocabExpander:
    def build_vocab_map(self) -> dict[str, list[list[int]]]: ...  # Drive名 -> 語彙トークンID列のリスト（BPEでは1語が複数トークン）。バイアス適用とrelief判定で共有（③）
```
- **契約**: 起動時に1回だけ呼ばれ、結果はプロセス生存中キャッシュする（毎ステップ呼ばない）。
- **シード語彙（B4）**: 拡張の起点となるDrive毎のシード語彙は`config/seed_vocab.yaml`（§4の`drive.vocab_expansion.seed_vocab_path`）に定義する。例:
  ```yaml
  loneliness: ["寂しい", "会いたい", "一緒に", "話したい", "誰か"]
  boredom:    ["つまらない", "新しい", "試してみる", "冒険", "何か"]
  fatigue:    ["休む", "疲れた", "ゆっくり", "眠い", "おやすみ"]
  ```
  シード語彙の埋め込みを起点に、類似度`sim_threshold`以上・上位`top_k`件の語を各Driveの語彙へ追加する。語彙→トークンID列の解決（BPEサブワード）はtokenizer経由で行う。
- **relief判定との整合（③）**: §4の`drive.relief`（loneliness等）の発火判定は、本クラスが生成するvocab_mapと**同一マップ**を参照する。relief専用の別語彙リストは新設しない（二重管理になると語彙拡張の結果がrelief判定に反映されなくなる）。判定は「セグメント内トークン列が、対象Driveの語彙トークン列のいずれかと**部分列一致**するか」で行う。
- **運用フック**: 拡張結果を`logging.py`経由で必ずINFOログに出力し、人間が目視レビューできるようにする（v3 §3.1の「完全ノーチェックにしない」方針の実装）。

### 5.3 `memory/store.py`

```python
class HierarchicalMemoryStore:
    async def commit(self, text: str, drive_snapshot: dict[str, float]) -> MemoryRecord: ...
    def retrieve(self, query_embedding, kind_filter: MemoryKind | None, top_k: int) -> list[MemoryRecord]: ...
```
- **契約**: `commit`はDrive変化量`abs(delta) >= 0.3`の場合、分類器の判定結果に関わらず`MemoryKind.EMOTIONAL`を強制付与する（v3 §3.2のルール）。
- **MemoryKind（B5）**:
  ```python
  class MemoryKind(Enum):
      EPISODIC   = "episodic"    # 出来事・経験
      SEMANTIC   = "semantic"    # 知識・事実
      EMOTIONAL  = "emotional"   # Drive変化大（abs(delta) >= 0.3）の体験
      PROCEDURAL = "procedural"  # 行動パターン（本フェーズでは使用任意）
  ```
- **永続化**: `MemoryRecord`はプロセス再起動をまたいで保持する必要があるため、ChromaDBへの永続化パス（§4の`memory.persist_directory`）を設定必須とする（インメモリのみの実装は不可）。

#### 5.3.1 記憶分類器（v1.11・`memory/classifier.py`）

MemoryKind を実際に使い分ける軽量分類器 `RuleBasedMemoryClassifier`。
コミットは発話セグメントごとに発生するため LLM 呼び出しは行わず、**日本語の
パターンマッチ（部分文字列＋重み）で決定的に分類**する。

- **分類規則**: 各カテゴリのパターン一致スコアを合算し、最大のカテゴリを返す。
  全て未一致なら `SEMANTIC`（既定）。同点は優先順位
  `EMOTIONAL > EPISODIC > PROCEDURAL > SEMANTIC`。
  - **EPISODIC**（出来事・経験）: 時間語（今日・昨日・さっき・あの時…）・過去形（ました・てしまっ…）・体験（を経験・したことがある…）・出来事・知覚・交流
  - **SEMANTIC**（知識・事実）: 定義（〜とは・というもの…）・一般論（一般的に・によると…）・説明
  - **PROCEDURAL**（手順・方法）: 方法・手順・やり方・まず〜次に〜・ステップ
  - **EMOTIONAL**（感情語彙の補助シグナル）: 悲・寂し・嬉・感動…（定義文は感情語彙より優先されるよう重みを設計）
- **Drive変化大の EMOTIONAL 強制は分類器より優先**（ストア側の契約を維持）。
- **配線**: `build_real_core` / `build_mock_core` / 比較スクリプトの全てのストア構築に
  `classifier=RuleBasedMemoryClassifier()` を実配線。
- **観測性**: `core._finalize_segment` がコミットの分類結果（kind）と重要度を
  `reports/memory.jsonl`（`memory_commit` イベント）に記録する。

#### 5.3.2 記憶の想起（v1.12・retrieve→文脈注入）

「書く側」（commit・分類・永続化）は v1.11 までで完成済み。v1.12 で**「読む側」**を
配線し、過去の経験が現在の行動に影響する記憶層を完成させる。

- **設定**: `memory.recall`（`enabled`（既定true）/ `top_k`（既定3）/ `max_tokens`（既定120））。
- **クエリ構築**: 現在の文脈＝直近の発話（`buffer.spoken_content` の末尾200字）＋Drive状態
  （退屈・寂しさ・好奇心）を連結した文を、ストア共有の embedder で埋め込み、
  `HierarchicalMemoryStore.retrieve(query_emb, None, top_k)` で類似度上位の記憶を取得する。
- **注入**: 取得した記憶は `【あなたの過去の記憶の想起】` ブロックとして **internal 要素**で
  バッファに追記する。internal のため発話表示（`spoken_content`）・relief判定・記憶コミットには
  一切影響せず、モデルの文脈（話題の起点・自己認識）にのみ寄与する。トークン上限
  （`recall.max_tokens`）で切り詰め、バッファ肥大・圧縮連鎖を防ぐ。
- **タイミング**: 内言（manual/native）の生成前・問いかけ（`_ask_question`）前・連続生成モードの
  起動直後に注入する。1想起セッションで1回だけ注入（マーカー）し、speech_end（セッション終了）で
  リセットして次のセッションで新しい想起を行う。
- **観測性**: `reports/memory.jsonl` の `recall` イベント（count / top_k / texts）で
  「どの記憶が現在の文脈に戻ってきているか」を検証できる。
- **実機検証（Qwen3.5-9B）**: シード記憶3件を投入し、内言・問いかけの直前に
  `recall: 過去記憶 3 件を文脈に注入` を確認。想起→内言→think_end→問いかけ→応答待ちの
  全サイクルが動作（想起はAの任意決断に干渉しない）。

### 5.4 `inference/engine.py` + `logits.py`

```python
class InferenceEngine:
    def __init__(self, llm_path: str, executor: ThreadPoolExecutor): ...
    async def generate_next_token(self, context: list[str], drive_state: dict) -> tuple[str, float]:
        """戻り値: (生成トークン, サプライズ値[0,1])"""
```
- **契約**: 本体の推論呼び出しは必ず`executor`経由で行い、呼び出し元のイベントループをブロックしないこと。**この関数を直接同期的に`await`せず`run_in_executor`でラップしていないPRはレビューで却下する。**
- **単一フライト保証（C2）**: 単一`Llama`インスタンスは並行`generate`に非対応のため、生成は必ず直列化する（`asyncio.Semaphore(1)`等）。`executor`へ同時に2件以上の生成タスクを投入する実装は禁止。
- **サプライズ計算**: `entropy.py`内で正規化方法を統一する（`surprise = min(1.0, entropy / entropy_scaling)`。係数は§4の`inference.entropy_scaling`）。
- **サプライズの用途（B3）**: `generate_next_token`が返すサプライズ値は以下の2箇所で消費する。
  - `HierarchicalMemoryStore.commit`時の重要度（圧縮時の保持優先度）として使用
  - `boredom`へのrelief（§4の`drive.relief.boredom.per_action`分。新奇性で退屈が解消）
  高/低サプライズの判定閾値は`inference.surprise_relief_threshold`（専用キー）を使用する。
- **閾値の独立性（②）**: relief発火判定（`surprise_relief_threshold`）とTHINK判定（`entropy_think_threshold`）は目的が異なるため、**同一閾値を使い回さない**。使い回すと、片方の校正が他方の挙動を意図せず変える結合が生じる。両者は独立キーとして別々に校正する（§9参照）。※ boredomの高サプライズ発火とfatigueの低サプライズ発火は同一の`surprise_relief_threshold`を境界に使う（同一サプライズ軸の表裏であり、異目的の使い回しではない）。
- **複数トークン語彙へのバイアス適用方式（⑤・v1.3新設）**: §5.2の`build_vocab_map`は`dict[str, list[list[int]]]`（1語=複数トークン列になり得る）を返す。`logits.py`の`DriveLogitsProcessor`は、この各トークン列について**先頭トークンにのみ**バイアスを加算する。後続トークンには適用しない。
  - 理由: ロジットバイアスは「次の1トークン」の確率にしか作用できない。先頭トークンさえ選ばれれば、通常の言語モデル分布に従って自然にその語の残りが続く可能性が高い。列内全トークンへ均等にバイアスをかけると、文脈上不自然な位置でも語を無理に完成させようとする挙動（既に文脈的に成立しない語尾を強制する等）につながるため採用しない。
  - 実装上の注意: 同じ先頭トークンIDが複数のDriveの語彙で重複する場合（例: 「話」が`loneliness`にも`boredom`にも含まれる）、バイアスは**該当する全Driveの値の合算**とする。二重カウントを避けるため、`logits.py`内で先頭トークンIDごとに一度だけ合算してからロジットへ加算する実装にすること。

### 5.6 `core.py` 発話スケジューリング（v1.7新設・§0 自律思考ループ）

§0の「待機中でも自発的に行動を選び取る」を実現するため、`drive.scheduling` 設定で
「思考(沈黙)⇄発話」の自律サイクル（`core._run_scheduled`）を提供する。enabled=false
（既定）なら従来の連続生成モードと完全に同じ動作をする（後方互換）。

- **thinking（思考・沈黙）**: 生成を止めて待機する。Driveは`_drive_loop`が常時更新。
  内言（`_generate_inner_thought`）を`inner_interval_sec`ごとに生成し、バッファに
  internal=True としてのみ追記する（発話・relief・記憶コミットには影響しない。
  モデルの文脈には残るため、次回発話の話題の起点となる＝外部キック不要の自己持続）。
- **speaking（発話）**: 従来の`step_once`でセグメント単位に生成。疲労
  （`quiet_on_fatigue`）または発話セグメント上限（`max_speech_segments`）で沈黙へ戻る。
- **遷移条件**（`_should_start_speaking`）: boredom >= speak_start_boredom または
  loneliness >= speak_start_loneliness で発話開始。fatigue >= speak_block_fatigue なら
  抑止するが、boredom >= speak_override_boredom なら抑止を無視する（我慢の限界・永久沈黙防止）。
- **外部刺激**: interrupt の未処理があれば即座に発話トリガー（`InterruptChannel.has_pending`）。
- **ログ（M2）**: 遷移は `reports/autonomy.jsonl`（event: speech_start / speech_end /
  inner_thought）とコンソール（`autonomy: ...`）に記録される。

デモ実行: `run_agent.py --config config/demo_autonomous.yaml --scheduled --seconds 120`。

#### 5.6.1 ネイティブThinking捕捉（v1.8・`thinking_mode: "native"`）

`drive.scheduling.thinking_mode: "native"` で、内言を「モデル自身のネイティブ思考」
（Qwen3.5系の `<think>` ブロック）として捕捉する。Thinkingモードの「思考領域と回答領域の
構造的分離」をそのまま利用するため、内言が「自分自身の思考」であることはモデルの訓練特性
として保証される（自覚の枠組みを自前で作る必要がない）。既定 `"manual"` は従来の
`[内言]` プレフィックス方式と完全に同一動作（後方互換）。

- **生成開始位置**: `seed_prompt` はテンプレート変数 `enable_thinking=True` を渡し、
  生成位置を `<think>` の内側に置く。テンプレート非対応バックエンド（モック等）は
  自前で `<think>\n` を追記する（`_open_think_block`）。
- **境界トークン**: `<think>`（開）・`</think>`（閉）はQwen3.5では単一の特殊トークン
  （248068 / 248069）。生成トークンに `</think>` が現れたら思考フェーズ終了、以降の
  トークンを発話（回答）として扱う。
- **ルーティング（step_once）**: 思考フェーズのトークンは internal=True でバッファにのみ
  追記し、発話・relief判定・記憶コミット・セグメント追跡には一切影響しない。
  `</think>` 以降のトークンは発話としてセグメント追跡に入る。境界タグは発話に漏らさない
  （強制クローズ後のモデルの重複出力なども internal として吸収）。
- **思考上限**: 思考セッションは `thinking_max_tokens`（既定120）で打ち切る。超えても
  `</think>` が出ない場合は強制クローズして文脈を整形式に保ち、回答へ進む
  （Qwen3.5は思考が長引きやすく、無制限だと回答に到達しない。実機で確認）。
- **シードの internal 化（v1.8）**: シード（外部からの初期入力）は internal=True で保持する。
  チャットテンプレートのタグ（`<|im_start|>` 等）が発話表示・relief・記憶に混入するのを防ぎ、
  「外部入力」と「エージェントの発話」の境界を明確にする。
- **発話前思考**: speech_start 時に思考ブロックを開き、「考えてから話す」を実現。
  発話セッション中の思考も `thinking_max_tokens` で打ち切られる。
- **実測（⑭）**: Qwen3.5-9Bで「内言=ネイティブ思考（Thinking Process形式の推論）・
  発話=`</think>`以降の回答」の分離と自律サイクル（内言→自発発話→沈黙→内言）を確認。
  内言は英語優位（モデルの推論レジスタ。DeepSeek-R1等と同様の本物のThinking特性）。
  発話はユーザー言語（日本語）で生成される。思考言語の強制日本語化は推論を劣化させる
  （思考の省略が発生）ため非推奨。`drive.scheduling.system_prompt` は役割認識の補助として使用可能。

#### 5.6.2 モデル駆動スケジューリング（v1.9・モデル自身が「いつ話す・黙る・考える」を選ぶ）

v1.7/v1.8 の遷移（思考⇄発話）は Drive閾値・タイマー・セグメント上限という**外部ヒューリスティック**
で決まっており、モデル自身はタイミングを一切選んでいなかった。v1.9 では**制約付きデコード**を
使って「モデルが自分の意思で遷移を選ぶ」3方式を実装し、実モデルで比較計測した。

**共通基盤: `engine.generate_decision`（制約付きデコード）**

各選択肢のトークン列で prefix-trie を組み、現在の trie ノードから遷移可能なトークン以外の
ロジットを -inf にしてサンプリングする。結果は**必ず選択肢のいずれかに収束**する（greedy なら
決定論的）。決断プロンプト（`_DECISION_PROMPTS`）は Drive状態の要約とともにエフェメラルに
（バッファには残さず）渡す。決断は人格バイアス（`logit_bias_coefficient`）から独立させる。

**3方式**（`drive.scheduling`・全て既定 off・後方互換）:

| 方式 | config | 決断ポイント | 選択肢 |
|---|---|---|---|
| **A: 境界決断** | `decide_on_think_end` / `decide_on_segment_end` | 思考ブロック終了時・発話セグメント境界 | 話す/黙る/さらに考える・続ける/黙る |
| **B: 待機中 introspection** | `introspection_sec`（秒） | 待機中に一定間隔 | 待機/内言/発話 |
| **C: 制御トークン** | `control_tokens` | 生成中の任意のトークン | `<\|lucina_speak\|>` 等を出力 |

- **安全弁（デッドロック防止）**: Drive閾値はモデルの判断より**優先される**。モデルが「待機」を
  選び続けても退屈の限界（`speak_override_boredom`）で強制発話、疲労（`quiet_on_fatigue`）や
  セグメント上限（`max_speech_segments`）はモデルの判断より先に沈黙へ戻す。
- **決断レイテンシ**: 実測 440〜677ms/回（Qwen3.5-9B・コンテキスト数KB）。
- **`decide_on_think_end` での「話す」直後は思考ブロックを再オープンしない**（`open_think=False`）。
  再オープンすると発話冒頭が思考として消費されるため。

**実測の比較（⑮・Qwen3.5-9B・各方式60秒・`reports/decision_modes_comparison.md`）**:

| 方式 | 決断 | 発話/沈黙 | 発話トークン | 観察されたモデルの意思 |
|---|---|---|---|---|
| baseline（v1.8） | 0 | 1/1 | 29 | （遷移は全てDrive・タイマー） |
| A（境界決断） | 2 | 1/1 | 9 | セグメント境界で「黙る」・think-endで「黙る」を実際に選んだ |
| B（introspection） | 3 | 1/1 | 55 | 「内言」を2回選択（思考を自己選択）。「発話」は選択せずDriveに委ねた |
| C（制御トークン） | 0 | 2/1 | 31 | **制御トークン出力0回（60秒×2モード）。モデルは未学習トークンを自発的に出力しない** |
| all（A+B+C） | 6 | 2/1 | 43 | think-end→「話す」で自発発話開始・segment-end→「続ける」を選択 |

**結論（採用判断）**:
- **C（制御トークン）は採用しない**。実測でトークン出力が0回（モデルが未学習の特殊文字列を
  自発生成しない）であり、システムプロンプトの指示では誘導できない。生成中に制御文字列が
  発話へ混入するリスクもあり、信頼性の観点で実運用不可。
- **A（境界決断）は採用**。低コスト（境界ごとのみ・~0.7s/回）で、モデルの意思が遷移に実際に
  反映される（実測: セグメント境界で自ら黙る・think-endで黙る）。
- **B（introspection）は採用**。待機中の自発的な「内言を選ぶ」が機能（実測: 内言×2）し、
  思考が自己選択になる。ただし「話し始め」は実測ではDriveに委ねられたため、Aと併用する。
- **推奨: A+B の併用**（`decide_on_think_end` + `decide_on_segment_end` + `introspection_sec`）。
  `config/demo_autonomous.yaml` に反映し、`scripts/compare_decision_modes.py` で再計測できる。

#### 5.6.3 外部への働きかけの必要性（v1.10・内部では解消できない欲求）

v1.9 までの実測で「モデルは黙る・考えるを好み、発話はDriveの安全弁に委ねられる」ことが
判明した。構造的な原因は、**全ての Drive の relief が自分自身の出力で解消できる**ため
「外部との非平衡」が発生しないことにある（ホメオスタシスが自己完結している）。
v1.10 では「**内部生成では絶対に解消できない欲求**」を導入し、外部への働きかけを
構造的に必然化する。

**①好奇心（curiosity Drive）**: 新しい情報への飢え。
- `drive.initial_state.curiosity`・`drive.dynamics_matrix.curiosity`（自己増殖 0.003/s・
  退屈からの加速 0.002/s）。
- **relief は外部入力（応答・割り込み）でのみ発火**（`drive.relief.curiosity.per_action`）。
  発話・内言・記憶では一切解消されない（`test_curiosity_not_relieved_by_speech` で固定化）。
- `drive.scheduling.curiosity_ask_threshold` を超えると `_ask_question()` が**外部に質問を発する**
  （働きかけ）。質問は1セグメント発話し、応答待ち（③）へ。
- `drive.scheduling.idle_curiosity_rate` で待機中の蓄積を加速できる。

**②応答依存 loneliness（聞いてもらう必要性）**:
- `drive.relief.loneliness.speak_relief`（既定0.2）: 話すだけでは**部分 relief** のみ。
- 応答（外部入力）が返って初めて `per_action`（0.6）の**フル解消**。
- 無視されると寂しさが蓄積し、**届く言葉（相手に応答を引き出す言葉）が必然化**する。
- `speak_relief` キーが無い config は旧仕様（話すだけでフル解消）にフォールバック（後方互換）。

**③応答待ち状態（awaiting）**: 質問後は生成を止めて応答を待つ。
- `_run_scheduled` の新モード。**応答（外部入力）が届くまで先に進めない**構造。
- 応答受信で好奇心・寂しさがフル解消され待機解除（`response_received`）。
- `await_timeout_sec` を超えると待機解除し、好奇心を**半減（give-up）**して即時の
  再問いかけ連打を防ぐ。
- 外部 relief はセグメント単位（pending キュー）ではなく **Drive 値への即時適用**
  （`_relieve_drive`）。応答直後に再問いかけしないため。
- 問いかけセッションは強制された外部行動のため、think-end/segment-end 決断（A）や
  制御トークンで**中断されない**（`test_ask_survives_think_end_decision` で固定化）。

**実測の比較（⑯・Qwen3.5-9B・各シナリオ60〜90秒・`reports/external_necessity_comparison.md`）**:

| シナリオ | 問いかけ | 応答待ち | 最終curiosity | 最終loneliness |
|---|---|---|---|---|
| baseline（旧: 話せばフル解消・好奇心なし） | 0 | 0 | 0.43 | 0.22 |
| curiosity（①のみ） | 1 | 0 | 0.63 | 0.22 |
| all（①+②+③） | 1 | 1 | 0.72 | 0.02 |

**結論（採用判断）: ①・②・③ 全て採用**。3つは相互補完的（①が働きかけの動機、②が
「届く言葉」への圧力、③が応答を待つ構造）で、併用すると「好奇心が溜まる → 問いかける →
応答を待つ → 応答で解消される」という**外部との応答交換サイクル**が成立する。
`config/demo_external.yaml` に全導入済み。

### 5.5 `io/interrupts.py`

```python
class InterruptChannel:
    def bind(self) -> None: ...   # イベントループ内でキューを初期化（core.run 起動直後に呼ぶ・冪等）
    def inject(self, text: str, timestamp: float | None = None) -> None: ...
    def drain(self) -> list[str]: ...
```
- **契約**: `inject`はスレッドセーフである必要がある。将来Webhookやセンサー入力など複数スレッドから注入される可能性があるため。
- **スレッド安全性（C1）**: `asyncio.Queue`自体はスレッドセーフではないため、外部スレッドから`inject`する場合は必ず`loop.call_soon_threadsafe(queue.put_nowait, item)`を経由する。代替として標準`queue.Queue`＋イベントループ側の定期drainも可。**外部スレッドから素の`asyncio.Queue.put()`を直接呼ぶ実装は禁止。**
- **初期化順序（⑩・C1レース条件対策）**: キュー生成とイベントループ捕捉（`asyncio.get_running_loop()`）は**必ずイベントループのスレッドでのみ**行う。初期化を`inject()`側の遅延処理に任せると、外部スレッドが最初の`inject`を呼んだ時点で`get_running_loop()`がループ外で実行され`RuntimeError`になる（実機で再現済み）。`core.run()`は起動直後に`bind()`を呼び、外部スレッドが最初の`inject`を行う時点でキューが確実に初期化済みであることをコードで保証する。`drain()`はイベントループ内から呼ばれた場合のみ初期化するフォールバックを持ち、ループ外（未起動）では従来通り空リストを返す。

#### 5.5.1 `io/output.py` OutputChannel（v1.13）

`InterruptChannel` の出力版。core が発話（speech）・質問（question）を emit し、
run_agent の表示タスクや実行エージェントが drain して消費する。初期化順序は
InterruptChannel と同じ契約（`core.run()` 起動直後に `output.bind()`）。
- **配線**: `core._finalize_segment` がセグメント完了時に emit（`_ask_mode` 中は
  `question`、それ以外は `speech`）。問いかけセッションの質問文は `question` として
  配信され、表示・実行エージェントのルーティングの起点になる。

#### 5.5.2 `io/executor.py` ExecutorAdapter（v1.13）

Lucina の質問文を実行エージェントへルーティングする。**設計の根拠（v1.9 の教訓）**:
制御トークンCが実測0回だったため、モデルにツール名を出力させる方式は取らず、
Lucina は普通の日本語の質問文を出すだけにして、解釈は外側のルールが担う。
- **ルーティング**: 定型（時刻・日付・URL取得）→ 自前サンドボックス（読み取り専用・即時・無料）、
  複雑（調べて・調査・コード・設計・読んで・ファイル等）→ Opencode CLI
  （`opencode run "…"`）。実行不能（挨拶等）→ None（人間に委ねる）。
- **安全方針**: サンドボックスは読み取り専用の定型操作のみ（date / URL取得）。
  任意コマンド実行・ファイル書き込みは許可しない（Opencode 側のサンドボックスに委ねる）。
- **実行結果の注入**: run_agent が結果を `InterruptChannel.inject()` で返し、好奇心 relief・
  応答待ち解除 → Lucina が応答する、で外部ループが閉じる。

#### 5.5.3 `web.py` Web UI（v1.14）

ブラウザから Lucina と対話・観察する Web UI。FastAPI + WebSocket。
- **エンドポイント**: `GET /`（HTML: チャット・Driveゲージ・自律イベント・記憶の4ペイン）、
  `GET /ws`（イベント配信）、`POST /send`（人間メッセージ注入 → `InterruptChannel.inject()`）。
- **WebBridge（`tail_jsonl`）**: `reports/autonomy.jsonl`・`drives.jsonl`・`memory.jsonl` を
  ポーリング追尾し、chat / drives / autonomy / memory の4種イベントを WebSocket でブロードキャスト。
  起動時は既存行を一括配信、以降は新規行のみ。
- **実行エージェント連携**: `_web_bridge_loop` は `OutputChannel` の質問を ExecutorAdapter へ
  ルーティング（対話モードと同一経路）。`--web` 指定時は自動でスケジューリングモードを有効化
  （人間が常駐しないため）。
- **設定**: `web.host` / `web.port`（既定 127.0.0.1:8765）。
- **実機検証**: Qwen3.5-9B で `--web` 起動 → HTTP 200・Driveイベント11件/3秒配信 →
  `/send` 注入（「こんにちは、Lucina。今日はどんな気分？」）→ `response_received`
  （loneliness 0.22→0.0・curiosity 0.57→0.14 の外部応答 relief）→ 発話 → 再問いかけ（await_start）。

##### ロード進捗のスプラッシュ表示（v1.15）

モデルロード（数分）の間も Web UI を開けるようにした。
- **起動順序の変更**: Web サーバーを `build_real_core`（モデルロード）**より先**に起動する。
  ロード中でも `GET /` が HTTP 200 を返し、スプラッシュ画面（進捗バー＋段階メッセージ）を表示する。
- **進捗イベント**: `build_real_core(config, on_progress=...)` が段階（モデルロード 0.0→0.35・
  embedder 0.45・語彙拡張 0.45→1.0）を `{"type": "status", "stage", "message", "progress"}` として
  bridge → WebSocket で配信。語彙拡張は `DriveVocabExpander.build_vocab_map(on_progress=...)` が
  Drive ごと（index/total/drive）に報告する。完了時は `stage: "done"` を配信してスプラッシュを閉じる。
- **ロード中の send バッファ**: モデルロード中に `POST /send` で届いたメッセージはバッファし、
  core 起動後に注入する（メッセージが失われない）。
- **実機検証**: ロード中に HTTP 200・WS で status イベント（embedder 0.45）受信 →
  モデルロード・語彙拡張完了 → 起動。モックでは drives/autonomy の通常配信も確認。
- **切断対策（v1.15追補）**: 長時間開いたままの接続が切れる問題（uvicorn 既定の
  ws_ping 20秒/20秒タイムアウト・再接続機能なし）に対応。①uvicorn の `ws_ping_interval=30` /
  `ws_ping_timeout=120` に延長してアイドル切断を防止。②クライアント JS に**自動再接続**
  （指数バックオフ 1s→最大15s）を実装し、切断しても復帰する。③**チャット履歴の復元**:
  `WebBridge` が chat イベントをリングバッファ（最大200件）に保持し、`GET /history` で
  再接続クライアントに返す（`onopen` 時に `restoreHistory()` で再描画）。
  `tests/test_web.py`（+4テスト: 履歴バッファ・/history・自動再接続JS・status配信）を追加。
  モックで「WS 切断→再接続→/history で履歴復元」を検証。
- **ブラウザ自動オープン（v1.16）**: `--web` 起動時に既定ブラウザで Web UI を自動で開く。
  `webbrowser` をバックグラウンドスレッドで呼ぶため起動をブロックしない。ヘッドレス環境など
  ブラウザを開けない場合はメッセージを出して継続する。無効化は `--no-browser` フラグまたは
  `web.auto_open: false`（`src/lucina/web.py` の `open_browser()`・`config/default.yaml`）。
  `tests/test_web.py`（+2テスト: バックグラウンド起動・ブラウザ無し時のフォールバック）を追加。
  モックで自動オープン/無効化の両パスを確認。
- **バインド失敗の検出（v1.16追補）**: ポート使用中などで Web サーバーが起動できない場合、
  従来は「開きました」と表示した後に背景スレッドで bind エラーが出るだけだった（混乱の原因）。
  `start_server` が `server.started` を最大5秒ポーリングしてバインド失敗を検出し、
  `RuntimeError` を投げるよう変更。`run_agent.py` はそれを捕捉して「ポートが使用中・
  `--port` で別ポートを指定」という明確なメッセージ＋終了コード1で終了する。
  `tests/test_web.py`（+1テスト: ポート使用中で RuntimeError）を追加。
  実機で「前回の実行がポートを掴んだまま（Ctrl+Z等で停止）→ 新規起動が失敗」を再現し、
  修正を確認。
- **残存プロセスの自動キル（v1.16追補）**: 起動時に**同一プロセス（`run_agent.py`）がポートを
  掴んでいたら自動で終了してから立て直す**（`kill_stale_web_process`）。Ctrl+Z 等でフリーズした
  前回実行が残っていても起動失敗しなくなる。`find_pid_on_port`（psutil 優先・`ss -tlnp` フォールバック）
  で PID を特定し、`is_lucina_web_process`（cmdline に `run_agent.py` を含むか）で**無関係な
  プロセスは殺さない**ことを担保。解放を最大5秒待ち、それでも無関係プロセスが掴んでいれば
  従来どおりバインド失敗の明確なエラーで終了する。
  `tests/test_web.py`（+4テスト: PID検出・lucina判定・無関係プロセスは殺さない・lucinaプロセスの自動キル）を追加。
  実機で「1個目の --web 稼働中に2個目を起動 → 1個目を自動キル → 新規がポートを確保」を確認。
- **進捗表示のバグ修正（v1.16追補・実機発見）**: 「進捗が正しく表示されない」の根本原因2件を修正。
  ①**初期 WebSocket 接続の欠落**: HTML の JS が `let ws = null` のまま `ws.onmessage = ...` を
  代入していたため `TypeError` でスクリプト全体が死に、ブラウザが一度も接続できなかった（進捗・
  チャットが一切更新されない）。ハンドラを名前付き関数に分離し `connect()` を初期呼び出しするよう
  修正。headless Chrome で「接続済み」表示・Drive ゲージ描画を実機確認。②**起動完了後に接続した
  クライアントでスプラッシュが閉じない**: `done` は一時イベントのため後から接続したブラウザには
  届かない。`WebBridge` に `ready` フラグを持たせ、`GET /history` が `ready` を返し、JS が
  `restoreHistory()` 時にスプラッシュを閉じるよう修正。③あわせて `--web` は `--interact` 同様に
  **トークン上限なしで常時稼働**（既定 `max_tokens=200` だと約1〜2分で停止し、画面が固まる）。
  ④モデルロード中の経過時間ベース進捗（`spawn_load_progress_monitor`）を追加し、バーが 0% で
  固まる問題を緩和（`model.load_expected_sec` で目安時間を設定）。
  `tests/test_web.py`（+4テスト: 初期接続JS・readyフラグ・history ready）と `tests/test_run_agent.py`
  （新設・2テスト: 進捗モニター）を追加。実機（Qwen3.5-9B・headless Chrome）で status イベント
  （語彙拡張 loneliness→boredom→fatigue→完了→done）の配信とスプラッシュの自動クローズを確認。
- **対話が成立しない問題の修正（v1.16追補・実機発見）**: ブラウザからメッセージを送っても Lucina が
  応答しない問題を修正。実機で原因を3点特定した。①**モデルが「黙る」を選び応答セグメントがゼロ**: 応答を
  受信（`response_received`・relief）しても、think-end/セグメント境界の決断で「黙る」を選ぶと発話が
  生成されない（v1.9 から既知の傾向が応答時にも発動）。②**返答指示の不在**: `[interrupt] メッセージ` を
  バッファに置くだけで、瞑想的なシステムプロンプトの下ではモデルが何をすべきか分からず黙る。
  ③**問いかけ中の応答が応答待ちに吸収**: 問いかけセッション中に届いた応答は `await_start`（応答待ち）に
  入り、応答に切り替わらない。修正: ①`drive.scheduling.force_response_speech: true`（既定）— 応答
  セッション中は黙る選択を無効化し、**最低1セグメントの発話を強制**。echo破棄セグメントでは強制を
  消費しない（`_last_segment_emitted` で実配信セグメントのみ判定）。②`_drain_external` が応答受信時に
  **「【ユーザーからメッセージが届きました】「…」／返答してください」という明示指示（internal）を注入**。
  ③問いかけ中に応答が届いたら `ask_cancelled` で応答に切り替え（`await_start` に入らない）。
  `tests/test_external_necessity.py`（+5テスト: 応答指示注入・問いかけ中応答の切り替え・黙る抑制・対照2件）を追加。
  実機（Qwen3.5-9B）で「/send メッセージ → CHAT[speech] 応答セグメント配信」を2回連続で確認（151テスト全通過）。
- **指示文リークの修正（v1.17・実機発見）**: 「こんばんはー」と送ると Lucina が「【あなたの思考プロセスは明記しないでください。」と返す問題を修正。
  実機調査で原因を特定した: **モデルが内部注入した【…】指示ブロック（返答指示・質問指示）をそのまま反唱したり、
  同じ【…】形式の指示文を捏造して出力**し、それが①発話として配信され、②記憶（ChromaDB）にコミットされ、
  ③想起されて再注入される**自己増幅ループ**になっていた（`reports/memory.jsonl` に実例: 21:40「【あなたの思考プロセスは…」
  が semantic としてコミット→2秒後に recall→21:42 に返答指示の反唱が emotional としてコミット）。
  修正: `core._strip_instruction_junk`（`【…】`ブロック＋「…」引用の除去）を新設し、`_finalize_segment`（発話配信・
  記憶コミットの直前）と内言生成の echo 処理に適用。**指示文のみのセグメントはコミットも配信もせず破棄**する
  （`instruction_junk_suppressed` でログ）。  既存の汚染記憶は ChromaDB（7件）と `reports/memory.jsonl`（20件）から
  除去済み（想起ブロック反唱・思考リーク・空内容も含む）。`tests/test_memory_recall.py`（+3テスト: 捏造指示文の除去・
  指示文のみセグメントの非コミット/非配信・返答指示反唱の回帰防止）。
- **思考リーク（番号付き思考）の修正（v1.17追補・実機発見）**: 「こんばんはー」への応答が
  「1. まず、ユーザーのメッセージをどのような文脈で受け取ろうかと考えます。」になる問題を修正。
  原因は3点: ①**シードプロンプトが内省・独白設計**（`あなたは考える存在です。静かに今の気持ちを言葉にしてください。`）
  で対話を想定していない、②**返答指示が思考を止めていない**（`返答してください。日本語で、短く、自然に。`
  ではネイティブ思考モデルが `<think>` 閉鎖後も思考を続ける）、③**思考リークが記憶化で増幅**（「1. まず…」が
  分類器の「まず」パターンで PROCEDURAL に分類されコミット→即想起）。修正: ①返答指示を
  「思考の過程・分析・前置きは一切書かず、いきなり相手への返答の言葉だけを」に強化、②`_strip_reasoning_junk`
  （番号付き思考行・思考見出し・メタ考察の除去。実際の返答が続く場合は返答だけ残す）を新設し
  `_finalize_segment`・内言生成に適用、③対話用 `seed_prompt` を config に追加（`run_agent.py` が読む）、
  ④system_prompt に対話方針を追記。汚染記憶（「1. まず…」procedural 等）を ChromaDB・memory.jsonl から除去。
  `tests/test_memory_recall.py`（+2テスト: 思考リーク除去・非コミット/非配信）。

---

## 6. 実装順序（依存関係に基づくタスク分解）

以下の順で実装するとブロッキング要因が少ない。

| # | タスク | 依存 | 完了の定義（DoD） |
|---|---|---|---|
| 0 | モデル選定実験（候補2〜3本の比較、§1） | なし | 比較基準（§1.2）に基づき本採用モデルが1本に確定し、`reports/model_selection.md`が残っている |
| 1 | `DriveDynamics`単体実装 | なし | `test_drive_dynamics.py`がグリーン、`[0,1]`クリップ確認 |
| 2 | `InferenceEngine`（executor分離のみ、Driveバイアスなし） | 0 | 単純な「1トークンずつ生成し続けるループ」がブロッキングなく動く |
| 3 | `DriveLogitsProcessor`と`DriveVocabExpander` | 1, 2 | Drive値を人為的に0.9固定した状態で、目的語彙の出力確率が明確に上昇することをログで確認 |
| 4 | `HierarchicalMemoryStore`（圧縮なし、Working Bufferのみ） | 2 | コンテキスト上限に近づいたら例外なく圧縮がトリガーされる |
| 5 | `InterruptChannel` | 2 | 生成ループ実行中に外部から`inject`し、次ステップで反映されることを確認 |
| 6 | Emotional Memory分類ロジック | 4 | Drive変化量±0.3以上のケースで強制EMOTIONALタグが付与される |
| 7 | `scripts/calibrate_thresholds.py`（v3 §5.1, 5.2の校正実験） | 0〜5 | `config/default.yaml`のPLACEHOLDER値が実測値に置き換わる |
| 8 | 24時間安定性テスト | 0〜7すべて | メモリ健全性テスト（v3 §5.3）が0件のオーバーフローで通過 |

※ タスク0で使う仮実装は、§1.2の評価専用の使い捨て実装であり、タスク1〜3の本実装とは別物とする（⑦）。

**重要**: タスク7（閾値校正）を「後回しにできる雑務」として扱わないこと。v3レビューで「なぜ300なのか根拠がない」と指摘された経緯があるため、**タスク7未完了の状態でPLACEHOLDER値のままリリースすることを禁止**する。同様に、**タスク0（モデル選定）を飛ばして特定モデルへ決め打ちで進めることも禁止**する。

---

## 7. テスト仕様（pytest実装レベル）

```python
# tests/test_attractor_survival.py の骨格
import pytest

@pytest.mark.asyncio
async def test_loneliness_attractor_convergence(lucina_core_fixture):
    """loneliness=0.9固定時、目的語彙への収束をトークン数で計測する"""
    core = lucina_core_fixture
    core.drives["loneliness"] = 0.9
    tokens_to_convergence = []

    for trial in range(50):  # v3 §5.1の校正手順: 50試行
        core.reset_working_buffer()
        count, converged = await run_until_target_vocab(core, target_kind="loneliness", max_tokens=1000)
        if converged:
            tokens_to_convergence.append(count)

    p90 = np.percentile(tokens_to_convergence, 90)
    # この p90 が新しい閾値になる。アサーションで固定値と比較するのではなく、
    # 結果をレポートファイルに出力し、config更新のトリガーとする。
    write_calibration_report("attractor_survival_tokens", p90)
```

- **重要な設計判断**: このテストは「合否判定」ではなく「閾値を生成するための計測」として実装する。固定の`assert count < 300`のようなテストにしないこと（v3で指摘された「根拠のない300」を単にコードに埋め込むだけになってしまうため）。§1のモデル選定フェーズでも、このテストの簡易版（試行回数を減らす等）を各候補モデルに対して流用できる（ただし、モデル選定時はfixtureではなく**実モデルを接続**して実行する。本fixtureは校正テスト専用。⑥）。

```python
# tests/test_interrupt_latency.py の骨格
@pytest.mark.asyncio
async def test_interrupt_reflection_latency(lucina_core_fixture):
    core = lucina_core_fixture
    baseline_latency_ms = await measure_avg_token_latency(core, n=20)
    threshold_ms = baseline_latency_ms * 1.5   # v3 §5.2の相対式

    injected_at = time.monotonic()
    core.interrupts.inject("テスト割り込み")
    reflected_at = await wait_until_buffer_contains(core, "テスト割り込み", timeout_ms=threshold_ms * 3)

    actual_latency_ms = (reflected_at - injected_at) * 1000
    assert actual_latency_ms <= threshold_ms
```

- **共通fixture仕様（D）**: 校正テストは実モデル不要で高速に回す。`lucina_core_fixture`は`conftest.py`に定義し、意図した語彙へ確率を寄せるダミーlogits生成器を`InferenceEngine`に注入する。`write_calibration_report`は`scripts/calibrate_thresholds.py`に実装し、計測結果を`reports/calibration_*.json`へ出力してconfig更新のトリガーとする（テストからimportして使用）。

---

## 8. 監視・ロギング要件（実装者が見落としがちな点）

- **Drive時系列ログ**: `boredom/loneliness/fatigue`を毎ステップ構造化ログ（JSON Lines）に出力する。校正実験（§6タスク7）とデバッグの両方で必須。
- **ロジットバイアス適用前後の差分ログ**: `logit_bias_coefficient`のチューニング時に、バイアス適用前後で目的語彙の確率がどれだけ動いたかを可視化できるようにする。§5.4の語彙の先頭トークンへのバイアス適用ログも同様に残すこと。
- **メモリ圧縮イベントログ**: 圧縮が発火した時刻・トークン数・要約結果を記録し、v3 §5.3のメモリ健全性テストのデータソースにする。
- **語彙拡張結果ログ**: `DriveVocabExpander`が起動時に生成した語彙リストをINFOレベルで必ず出力（§5.2の運用フック）。
- **モデル選定比較ログ（v1.3新設）**: §1.2の3評価軸それぞれについて、候補モデルごとの計測結果を構造化ログとして残し、`reports/model_selection.md`の元データにする。

---

## 9. コーディング規約・レビュー基準

- 推論呼び出しを`await`で直接ブロッキングするコードはマージ禁止（§5.4参照）。
- `config/default.yaml`にないマジックナンバーをコード中にハードコードすることを禁止。数値はすべて設定ファイル経由にする。
- `thresholds`セクションの値を変更するPRには、必ず`calibrate_thresholds.py`の実行結果（レポートファイルへのリンクまたは添付）を紐付けること。
- 目的の異なる判定に同一閾値・設定値を流用しない（例: THINK判定とrelief発火判定）。校正の独立性を保つため、用途ごとに専用キーを設けること（②の設計判断を規約化）。
- Drive結合行列`A`に新しい非対角成分を追加する場合、v3の方針（観測データに基づく追加のみ）に従い、PR説明に根拠となる観測データまたは実験結果を記載すること。
- `model.path`を変更するPRは、§1のモデル選定フェーズを経た`reports/model_selection.md`の更新、または既存レポートへの追記が伴っていない場合マージ禁止とする（v1.3新設）。モデル変更後は`logit_bias_coefficient`等モデル固有の校正値を無条件で再校正すること。

---

## 10. マイルストーン

| マイルストーン | 内容 | 目安 |
|---|---|---|
| M0: モデル選定完了（v1.3新設） | §6タスク0完了。本採用モデルが確定し比較レポートが存在する | 実装着手の前提条件 |
| M1: コア動作確認 | タスク1〜2完了。ブロッキングなしで生成ループが回る | フェーズ1.5開始点 |
| M2: 自律行動の初観測 | タスク3〜6完了。Drive駆動での自発的行動選択をログで確認できる（v1.7達成: §5.6・`reports/autonomy.jsonl`） | MVP相当 |
| M3: 閾値校正完了 | タスク7完了。config内のPLACEHOLDERが全て実測値に | リリース前必須 |
| M4: 安定性確認 | タスク8完了。24時間稼働でオーバーフロー0件 | リリース判定基準 |

---

## 11. 未確定事項（実装開始前に決定すべき事項）

- 使用するLLMモデルの具体的な銘柄・サイズ・量子化レベル → **§1のモデル選定フェーズで決定する運用に変更（v1.3）**。特定モデルへの決め打ちは廃止。
- 要約専用軽量モデルと語彙拡張用埋め込みモデルの具体的な選定（日本語性能を要検証）。
- ChromaDBの永続化先（ローカルディスク／別プロセスサーバー）の運用方針。暫定案では`memory.persist_directory`によるローカル永続化を想定（§4参照）。
- `logit_bias_coefficient`の妥当性検証方法 → **解決（v1.6⑪）**: `scripts/calibrate_logit_bias.py`を新設（ΔP語彙確率シフト・アトラクタ収束・出力健全性の3指標を係数スイープで計測し、最大シフトの80%に達する最小係数を採用する最小十分原理）。Qwen3.5-9Bで実測し **3.5** を採用（§4反映済み・`reports/calibration_logit_bias_coefficient.json`）。モデル変更時は本スクリプトで無条件に再校正する。

**v1.1で暫定案を追記した項目（実装時・校正時に再検証必須）**: relief係数（§4 `drive.relief`）、サプライズ用途の閾値（§5.4）、シード語彙の中身（§5.2）、THINKトークンID（§4 `inference.think_token_ids`）。

**v1.2で追記・更新した項目（実装時・校正時に再検証必須）**: reliefの発火単位（§4 `drive.relief.unit`/`segment`・§5.1 ①）、`inference.surprise_relief_threshold`の新設（§4・§5.4 ②）、vocab_mapとrelief判定の一元化（§5.2 ③）。

**v1.3で追記・更新した項目（実装時・校正時に再検証必須）**: モデル選定フェーズ（§1、④）、複数トークン語彙へのバイアス適用方式（§5.4、⑤）。

**v1.4で追記・更新した項目（明確化・実装時の注意事項）**: モデル選定実験は実モデル接続で実行（§1.4・§7、⑥）、タスク0の仮実装は評価専用（§1.2・§6、⑦）、環境セットアップの先行完了（§1.4、⑧）、計測点・用語の明確化（§1.2・§8、⑨）。

**v1.5で追記・更新した項目（実装時バグ修正）**: `InterruptChannel`の初期化順序（§5.5、⑩）。`bind()`の新設と`core.run()`起動直後での呼び出し、`drain()`のループ内初期化フォールバック。

**v1.6で追記・更新した項目（校正完了）**: `logit_bias_coefficient`の校正手順定義と実測（§4・§11、⑪）。`scripts/calibrate_logit_bias.py`を新設し、Qwen3.5-9Bでスイープ校正の結果 **3.5** を採用。モデル固有値のため、モデル差し替え時は無条件で再校正。

**v1.7で追記・更新した項目（§0自律思考ループの実装）**: 発話スケジューリングと内言ループ（§5.6新設、⑫）。`drive.scheduling`（既定off・§4）・`config/demo_autonomous.yaml`・`run_agent.py --scheduled`。M2「Drive駆動での自発的行動選択をログで確認できる」を実測で達成（boredom閾値での自律的な speech_start/end を `reports/autonomy.jsonl` で確認）。**実測の発見（⑬）**: 実発話の平均サプライズは0.29〜0.51（校正スイープ実測）で `surprise_relief_threshold=0.7` に届かず、boredom reliefが実質発火しないことが判明。デモconfigでは0.3に下げてreliefをサイクルに参加させる。本番値の校正は§11の今後の課題。

**v1.8で追記・更新した項目（ネイティブThinking捕捉・⑭）**: `drive.scheduling.thinking_mode: "native"`（§5.6.1新設）。内言をモデル自身のネイティブ思考（`<think>`ブロック）として捕捉し、`</think>`以降を発話として扱う。`enable_thinking` テンプレート変数・`thinking_max_tokens`（思考上限・強制クローズ）・`system_prompt`（役割認識補助）を追加。シードのinternal化で「外部入力とエージェントの発話」の境界を明確化。境界タグ（`<think>`/`</think>`）の発話漏れ防止。実機検証: Qwen3.5-9Bで内言=ネイティブ推論・発話=回答の分離、自律サイクル、クリーン終了（`--seconds` 後の正常終了）を確認。内言は英語優位（本物のThinking特性。思考言語の強制日本語化は推論劣化のため非推奨）。

**v1.9で追記・更新した項目（モデル駆動スケジューリング・⑮）**: §5.6.2新設。制約付きデコード（`engine.generate_decision`）でモデル自身が「いつ話す・黙る・考える」を選ぶ3方式（A: 境界決断 / B: 待機中 introspection / C: 制御トークン）を実装し、実モデルで比較計測。**実測の結論: C（制御トークン）は出力0回で採用不可、AとBは採用**（A: セグメント境界で自ら黙る、B: 内言を自己選択。推奨はA+B併用）。`drive.scheduling` に `introspection_sec` / `decide_on_think_end` / `decide_on_segment_end` / `control_tokens` / `control_token_instruction` / `decision_max_rethink` を追加（全て既定off・後方互換）。`scripts/compare_decision_modes.py`（比較計測）・`tests/test_model_decisions.py`（10テスト）を新設。

**v1.10で追記・更新した項目（外部への働きかけの必要性・⑯）**: §5.6.3新設。「内部生成では絶対に解消できない欲求」を導入し、外部への働きかけを構造的に必然化した。①好奇心Drive（`drive.dynamics_matrix.curiosity`・`drive.relief.curiosity`・`curiosity_ask_threshold`・`idle_curiosity_rate`。reliefは外部入力のみ・閾値越えで質問を発する）②応答依存loneliness（`drive.relief.loneliness.speak_relief`。話すだけでは部分relief・応答でフル解消）③応答待ち状態（`await_timeout_sec`。質問後は生成を止めて応答を待ち、タイムアウトでgive-up）。`_drain_external`（外部入力で即時relief）・`_ask_question`（問いかけセッション。Aの決断で中断されない）を実装。実機で「好奇心→問いかけ→応答待ち」の全サイクルを確認（baseline: 問いかけ0 vs all: 1回）。`config/demo_external.yaml`・`scripts/compare_external_necessity.py`・`tests/test_external_necessity.py`（8テスト）を新設。

**v1.11で追記・更新した項目（記憶分類器・⑰）**: §5.3.1新設。`memory/classifier.py` の `RuleBasedMemoryClassifier`（日本語ルールベース・LLM呼び出しなし・決定的）を実装し、`MemoryKind`（EPISODIC/SEMANTIC/EMOTIONAL/PROCEDURAL）を実際に使い分けるようにした。分類器を全てのストア構築（`build_real_core` / `build_mock_core` / 比較スクリプト）に実配線し、`core._finalize_segment` はコミットの kind と重要度を `reports/memory.jsonl`（`memory_commit` イベント）に記録。Drive変化大の EMOTIONAL 強制ルールは分類器より優先を維持。`tests/test_memory_classifier.py`（10テスト）を新設。

**v1.12で追記・更新した項目（記憶の想起・⑱）**: §5.3.2新設。**「読む側」の記憶**（`retrieve`→文脈注入）を配線した。`memory.recall`（既定 enabled・`top_k`・`max_tokens`）を新設し、`core._recall_memories`（現在の文脈＝直近の発話＋Drive状態をクエリ文にして埋め込み、類似度上位の記憶をベクトルストアから引き出して internal 要素として注入）を実装。注入は内言（manual/native）・問いかけ（`_ask_question`）・連続生成の各生成前に実施し、1想起セッション1回のマーカー＋セッション終了（speech_end）でリセット。想起は internal のため発話表示・relief・記憶コミットに影響しない。`reports/memory.jsonl` の `recall` イベントで想起内容を観測可能に。実機検証: シード記憶3件を ChromaDB に投入し、内言・問いかけの前に「過去記憶3件を文脈に注入」を確認（想起→内言→think_end→問いかけ→await の全サイクル）。`tests/test_memory_recall.py`（7テスト）を新設。

**v1.13で追記・更新した項目（対話相手と実行エージェントの接続・⑲）**: §5.5.1（OutputChannel）・§5.5.2（ExecutorAdapter）新設。v1.10で作った「外部への働きかけ」の**応答相手**を接続し、外部ループを閉じた。①`io/output.py`（`OutputChannel`）: 発話・質問の外部配信キュー。`core._finalize_segment` がセグメント完了時に emit（問いかけ中は `question`）。②`scripts/run_agent.py --interact`: 人間コンソール対話。Lucina の発話・質問をリアルタイム表示し、stdin スレッドが人間の応答を `InterruptChannel.inject()` で注入。③`io/executor.py`（`ExecutorAdapter`）＋`--executor`: 質問を定型（時刻・URL＝自前サンドボックス）と複雑（調査・コード＝Opencode CLI）にルーティングし、結果を inject。④core の応答待ち（awaiting）は応答受信で**発話へ遷移**（質問→応答→Lucinaの応答の対話ループ）。実機検証: Qwen3.5-9B が自律的に調査質問「宇宙波が地上の新たな生命の誕生にどのような影響を及ぼしているか、深追調査を依頼したいです。」を発し、実行エージェント（Opencode + DeepSeek V4 Flash Free）が実調査して結果を注入、応答受信→発話開始まで確認。`tests/test_interact.py`（11テスト→13テスト）を新設。

**v1.13追補（実機検証のフィードバック反映）**: ①**Opencode モデルは無料の `opencode/deepseek-v4-flash-free`（DeepSeek V4 Flash 0731）を採用**（`executor.opencode_model`）。本マシンの既定モデル（qwen3-coder:30b）は未取得で失敗したため、利用可能モデルを明示指定する方式にした。②**セッション乱立対策**（`executor.opencode_reuse_session`）: `opencode run --format json` で sessionID を取得し、同一プロセス内では同じセッションを `--session` で継続（1プロセス=1セッション。実測でセッション数14→14の乱立ゼロ）、終了時に `opencode session delete` で削除。セッション継続により実行エージェントに調査の継続記憶も持たせられる（実測: 2回目の質問が前回の回答を記憶）。③**部分反唱のエコー抑制強化**: 想起メモリの完全ブロック一致に加え、**メモリ1行のみの部分反唱**も検出して発話・質問・記憶コミットから除外（`_strip_recall_echo`・`_recall_protected`）。実機で「質問が想起メモリの反唱になり、実行エージェントにルーティングされない・ジャンク記憶になる」問題を発見して修正。④`await_timeout_sec` をデモ設定で 120 秒に延長（実行エージェントの応答は数十秒かかるため）。

**v1.17（指示文リークの修正・㉓）**: 「こんばんはー」への応答が「【あなたの思考プロセスは明記しないでください。」になる問題を修正。実機調査で、**モデルが内部注入した【…】指示ブロック（返答指示・質問指示）を反唱・捏造し、それが発話配信・記憶コミット・想起の自己増幅ループになる**ことを特定。`core._strip_instruction_junk`（`【…】`ブロック＋「…」引用の除去）を新設し、`_finalize_segment` と内言生成に適用。**指示文のみのセグメントはコミット・配信せず破棄**（`instruction_junk_suppressed` ログ）。汚染記憶（ChromaDB 7件・memory.jsonl 20件）を除去。`tests/test_memory_recall.py`（+3テスト）・154テスト全通過。

**v1.17追補（思考リークの修正・対話性能）**: 「こんばんはー」への応答が「1. まず、ユーザーのメッセージをどのような文脈で受け取ろうかと考えます。」になる問題を修正。原因3点: ①シードプロンプトが内省・独白設計で対話を想定していない（`あなたは考える存在です。静かに今の気持ちを言葉にしてください。`）、②返答指示が思考を止めていない（ネイティブ思考モデルは `<think>` 閉鎖後も番号付き思考を続ける）、③思考リークが「まず」パターンで PROCEDURAL 分類され記憶化→想起で増幅。修正: ①返答指示を「思考の過程・分析・前置きは一切書かず、いきなり返答の言葉だけを」に強化、②`_strip_reasoning_junk`（番号付き思考行・思考見出し・メタ考察の除去。実際の返答が続く場合は返答だけ残す）を新設し `_finalize_segment`・内言生成に適用、③`drive.scheduling.seed_prompt` を config に追加（対話用シード。`run_agent.py` が `--prompt`→config→既定の順で読む）、④system_prompt に対話方針を追記。汚染記憶を除去。`tests/test_memory_recall.py`（+2テスト）・156テスト全通過。

**v1.16追補（バインド失敗の検出＋残存プロセスの自動キル）**: 「前回の `--web` 実行がポートを掴んだまま残り、新規起動で `[Errno 98] address already in use` が出る」問題に対応。①`start_server` がバインド結果（`server.started`）を最大5秒待って検出し、失敗時は `RuntimeError` → `run_agent.py` が「ポートが使用中・`--port` で別ポート」と明確に案内して終了コード1で終了する。②**同一プロセス（`run_agent.py`）がポートを掴んでいたら自動で終了して立て直す**（`kill_stale_web_process`）。`find_pid_on_port`（psutil 優先・`ss` フォールバック）で PID を特定し、cmdline に `run_agent.py` を含む場合のみキル（無関係なプロセスは殺さない）。実機で「1個目の --web 稼働中に2個目を起動 → `既存の Web プロセス (PID ...) を終了して再起動します` → 新規がポートを確保」を確認。`tests/test_web.py`（+5テスト）。

**v1.16で追記・更新した項目（ブラウザ自動オープン・㉒）**: §5.5.3に「ブラウザ自動オープン」を追記。`--web` 起動時に既定ブラウザで Web UI を自動で開く（`webbrowser` をバックグラウンドスレッドで実行し起動をブロックしない）。`src/lucina/web.py` に `open_browser()` を追加し、`scripts/run_agent.py` の `--web` 起動直後に呼ぶ。無効化は `--no-browser` フラグまたは `web.auto_open: false`（config に追加）。ヘッドレス環境で開けない場合はメッセージを出して継続。`tests/test_web.py`（+2テスト: バックグラウンド起動・ブラウザ無し時のフォールバック）を追加。モックで自動オープン/無効化の両パスを確認（136テスト全通過）。

**v1.15で追記・更新した項目（ロード進捗のスプラッシュ表示＋切断対策・㉑）**: §5.5.3に「ロード進捗のスプラッシュ表示」と「切断対策」を追記。モデルロード（数分）の間も Web UI を開けるように起動順序を変更した（Web サーバーを `build_real_core` より先に起動）。`build_real_core` / `build_vocab_map` に進捗コールバック（`on_progress`）を追加し、段階（モデルロード・embedder・語彙拡張 per-drive）を `status` イベントとして WebSocket で配信。HTML にスプラッシュ画面（進捗バー＋段階メッセージ）を追加し、完了（`stage: done`）でメイン UI へ遷移。ロード中に届いた `POST /send` はバッファして core 起動後に注入。`tests/test_web.py`（+2テスト: スプラッシュHTML・status配信）と `tests/test_vocab_expander.py`（+1テスト: per-drive進捗）を追加。実機検証: ロード中に HTTP 200・WS で status イベント受信 → モデルロード・語彙拡張完了 → 起動を確認。

**v1.14で追記・更新した項目（Web UI・⑳）**: §5.5.3新設。ブラウザから Lucina と対話・観察できる **Web UI** を実装した。①`src/lucina/web.py`: FastAPI アプリ（`GET /`＝HTML・`GET /ws`＝WebSocket・`POST /send`＝人間メッセージ注入）＋`WebBridge`（`tail_jsonl` で `reports/*.jsonl` を追尾し、chat / drives / autonomy / memory の4種イベントを WebSocket でブロードキャスト）。HTML はチャット・Driveゲージ（ライブ更新）・自律イベント・記憶の4ペイン。②`scripts/run_agent.py --web`: `web.host:port`（既定 127.0.0.1:8765）でサーバーを起動し、`_web_bridge_loop` がイベントを転送＋質問を実行エージェントへルーティング（対話モードと同等の処理）。`--web` 指定時は自動的にスケジューリングモードを有効化（人間が常駐しないため）。`POST /send` は `InterruptChannel.inject()` で注入され、**外部応答 relief・応答待ち解除→発話遷移**がブラウザ経由でも動作。`tests/test_web.py`（8テスト）を新設。実機検証: Qwen3.5-9B で `--web` を起動し、HTTP 200・Driveイベント11件/3秒の配信・`/send` 注入（「こんにちは、Lucina。今日はどんな気分？」）→ `response_received`（loneliness 0.22→0.0・curiosity 0.57→0.14 の relief）→ 発話 → 再問いかけ（await_start）まで確認。

※補足（実機検証中の観察）: 検証用 ChromaDB は過去の校正・デモ実行で蓄積したゴミ記憶（テンプレート断片・モックトークン等）が混在すると、想起結果にそれらが混入する。実運用では「何をコミットするか」の品質管理（セグメントの完成文のみコミット等）が想起品質に直結する。
