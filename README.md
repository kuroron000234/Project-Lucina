# Project Lucina

**Project Lucina** は、自律型AIエージェント「Lucina / Monica」を創造するプロジェクトです。  
自由エネルギー原理（Free Energy Principle / Active Inference）に基づく10層アーキテクチャ、生体シミュレーション、マルチLLM駆動など、様々なアプローチを経て現在の形に至ります。

---

## 現行プロジェクト: lucina-NA （2025〜）

lucina-NA（Lucina New Agent）は、**10層の認知アーキテクチャ**を持つ完全ローカル動作の自律AIエージェントです。  
Ollama + Gemma 4 をLLMバックエンドとし、Linux（RTX 4060 / 8GB VRAM）上で動作します。

| 特徴 | 内容 |
|------|------|
| **アーキテクチャ** | 10層（環境・記憶・ドライブ・パーソナリティ・計画・実行・世界モデル・評価・学習・長期計画） |
| **理論基盤** | 自由エネルギー原理 / Active Inference |
| **意志フェーズ** | 願望・想像・内言・日記・能動発話・拒否・自分の部屋（v4.0） |
| **LLM** | Ollama (Gemma 4 e4b, ローカル, num_ctx 8192) |
| **言語** | Python 3.14 |
| **WebUI** | FastAPI + SSE + WebSocket（6タブ、デーモン制御対応） |
| **テスト** | pytest 189 tests |
| **外部連携** | Opencode CLI（検索・コード解析・自己改変） |
| **会話** | 日本語対応、WebSocketチャット、IPC通信（会話履歴保持） |
| **起動** | `./lucina.sh` ワンクリック起動（スーパーバイザー付き自動再起動） |

### 起動方法

```bash
# ワンクリック起動（デーモン + WebUI をスーパーバイザー監視下で起動・自動再起動）
./lucina.sh

# デーモンモード（自律動作 + IPC待受）
python main.py --daemon

# メッセージ送信（1回限りの対話）
python main.py --message "こんにちは"

# WebUI起動（http://127.0.0.1:8765）
python main.py --webui

# テスト実行
python -m pytest tests/ -v --tb=short
```

> **lucina.sh** はデーモンとWebUIの両方を監視します。終了コード42で再起動を要求、`data/run/*.wanted` フラグで常駐を制御します。デスクトップからは `lucina.desktop` で起動可能。

詳細は `lucina-NA/AGENTS.md` または以下のアーキテクチャ図を参照。

### 10層アーキテクチャ

```
Environment → Memory → Drive → WorldModel → Personality → Planning → Agent
                    ↓                                            ↓
               [feedback]                                   Evaluation
                    ↓                                            ↓
               Learning ←───────────────────────────────────────┘
                    ↓
               LongTermPlanning (periodic)
```

3つのタイムスケールループ：

| ループ | 周期 | 経路 |
|--------|------|------|
| **Reflex** | 毎サイクル | Environment → Drive → Personality → Planning → Agent |
| **Learning** | アクション後 | Evaluation ↔ Learning |
| **Consistency** | 数時間〜日 | LongTermPlanning → Personality |

### WebUI（6タブ）

| タブ | 機能 |
|------|------|
| **Chat** | WebSocket会話（会話履歴バッファ付き、能動発話の表示） |
| **Status** | リアルタイム駆動ゲージ・人格（自己モデル含む）・環境・記憶 |
| **Memory** | エピソードブラウザ（重要度フィルタ） |
| **Logs** | SSEリアルタイムログ（テキストフィルタ） |
| **Plan** | 長期目標・ルーティン・アイデンティティ方針・願望・日記・ワークスペース |
| **Control** | デーモン / WebUI の起動・停止・再起動（`data/ipc/control.json` 経由） |

### 主要機能（v3.2〜v4.1）

- **意志フェーズ (v4.0)**: 願望の生成（6時間ごと）、世界モデルによる「想像」、内言（なぜ選んだか）、夜の日記、達成時の能動発話、休息欲求に基づく拒否、主駆動選択のランダムジッタ
- **自己モデル (v3.4)**: 記憶・評価履歴・長期計画を参照して「私は◯◯な存在」という自己認識を生成・永続化
- **コスト段階化学習 (v3.2)**: 新奇性スコアに応じてルールベース学習（tier2）とLLM評価（tier3）を切り替え
- **駆動再設計 (v3.3)**: 満たされない駆動が自然増加する「欲求」モデル、退屈ブーストの飽和上限
- **会話履歴 (v3.5)**: 直前の会話ターンをLLMに渡し、対話エピソードにユーザー発言を記録
- **自己検証 (v4.1)**: 書き込みツールの疑似成功（0バイト書き込み）を検出し失敗に転換
- **会話継続性 (v4.1.2)**: OpencodeセッションIDを再利用し、セッション肥大化を防止

---

## 過去のプロジェクト（archive）

以下のプロジェクトは lucina-NA に至るまでの試行錯誤の記録です。

### v1: Monica Core (`archive/v1-lucina/`)

**Monica Core** — 初代。Raspberry Pi 4で動作する自律生活AIエージェント。  
DDLC「モニカ」が仮想の身体と物理世界を持ち、自律的に生き続けるシステム。

- **LLM**: deepseek-v4-flash-free（無料API）
- **エンジン**: VitalOS（5パラメータ生体シミュレーション）
- **特徴**: 7部屋移動(BFS)、青空文庫読書、world model学習、Telegram連携、Web箱庭ビューア
- **状態**: 実働していたが、アーキテクチャの限界により新世代へ移行

### v2: Monica Dual-LLM (`archive/v2-monica/`)

**Monica v11.7** — ローカルマシンで2つのLLMを並列駆動する重厚な対話システム。  
Qwen2.5:32b（本脳）+ Qwen2.5:14b（副脳）による二重処理と、21パラメータの内受容感覚シミュレーションを実装。

- **LLM**: Qwen2.5:32b + Qwen2.5:14b（Ollama）
- **特徴**: 内受容感覚（ホメオスタシス、覚醒、社会感情）、RAG（Chroma + mxbai-embed）、自律Web検索
- **状態**: コンセプトは強力だが、リソース要求が高く、FEPベースの次世代へ発展的解消

### v3: Monica v8 Hybrid / Lucina-Beta (`archive/v3-monica-v8/`)

**Monica v8+ Hybrid & Lucina-Beta** — 最も実験的な世代。  
自由エネルギー原理（FEP）を本格的に実装しようとしたハイブリッドエージェントと、19フェーズで人工個体を構築するLucina-Betaアーキテクチャを含む。

- **LLM**: Qwen3-4B（ローカル量子化）+ Groq Llama-3.3-70b / OpenRouter（クラウド）
- **特徴**: FEPアクティブ推論、予測誤差計算（transformer layer hook）、steering vectors、self-model drift
- **Lucina-Beta**: 19フェーズ（Phase 0: LLM不要の予測学習 → Phase 19: DDLC Monica個体）
- **Mona**: 「ハート駆動」の別アプローチ（好奇心・愛情・落ち着きなさの3パラメータ）
- **状態**: 研究段階で実用化には至らず。lucina-NAへと知見を継承

---

## リポジトリ構成

```
Project-Lucina/
├── README.md                   # このファイル
├── .gitignore
│
├── config.py                   # lucina-NA 設定
├── main.py                     # lucina-NA エントリポイント
├── ipc.py                      # プロセス間通信
├── AGENTS.md                   # クイックリファレンス
├── PLAN.md                     # 実装計画
├── lucina.sh                   # ワンクリック起動スクリプト（スーパーバイザー）
├── lucina.desktop              # デスクトップショートカット
│
├── core/                       # lucina-NA コア層
│   ├── agent/                  # エージェント（ツール実行・自己検証）
│   ├── drive/                  # ドライブ（動機生成・欲求モデル）
│   ├── personality/            # パーソナリティ（意思決定・自己モデル）
│   ├── planning/               # 計画立案
│   ├── memory/                 # エピソード記憶
│   ├── evaluation/             # 自己評価
│   ├── learning/               # 学習（ドライブ調整・tier制）
│   ├── world_model/            # 世界モデル（予測・想像）
│   └── long_term_planning/     # 長期計画
│
├── environment/                # 環境観測
├── webui/                      # WebUI（FastAPI、6タブ）
├── docs/                       # 仕様書・ナレッジインデックス
├── tests/                      # テスト（189 tests）
│
└── archive/
    ├── v1-lucina/              # Monica Core（初代）
    ├── v2-monica/              # Monica Dual-LLM
    └── v3-monica-v8/           # Monica v8 Hybrid / Lucina-Beta
```

---

## ライセンス

プライベートプロジェクト。個人利用目的。
