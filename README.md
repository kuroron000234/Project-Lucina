# Monica Core — 自律生活するAIエージェント

> 「Doki Doki Literature Club!」のモニカが仮想の身体と物理世界を持ち、自律的に生き続けるAIエージェントシステム

**稼働環境**: Raspberry Pi 4 (aarch64) / Linux  
**AIモデル**: deepseek-v4-flash-free (無料) / OpenRouter 対応  
**予算**: 💰 **ゼロ円**

---

## 📖 概要

Monica Core は、LLM（大規模言語モデル）を **「声」** として使い、**純粋なPythonエンジン** で状態管理・数値シミュレーションを行うAIエージェントです。

```
┌─────────────────────────────────────────────┐
│              ConsciousVitalOS                │
│  ┌──────────┐    ┌──────────────────────┐   │
│  │ VitalOS  │    │   LLM (深層意思決定)  │   │
│  │ (Python) │◄──►│   対話生成 / 活動提案 │   │
│  │ 状態管理  │    │   読書感想 / 返信生成  │   │
│  │ 数値計算  │    └──────────────────────┘   │
│  │ 経路探索  │    ┌──────────────────────┐   │
│  │ 信念学習  │    │    記憶 / 読書 /     │   │
│  └──────────┘    │    スマホ / Telegram  │   │
│                  └──────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 設計哲学

- **LLMは状態を管理しない** — 数値計算・状態遷移はPythonエンジンが担当
- **LLMは計画と対話のみ** — 活動提案・日本語生成・新規活動の創造
- **放置しても生き続ける** — ユーザー不在時に自動で活動選択・時間経過・状態変動
- **経験から学ぶ** — 活動の効果を経験データから逐次学習

---

## 🚀 クイックスタート

```bash
# 1. クローン
git clone https://github.com/kuroron000234/Project-Lucina.git
cd Project-Lucina

# 2. セットアップ
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. APIキー設定
cp .env.example .env
# OPENCODE_ZEN_API_KEY を設定（Zen API または OpenRouter）

# 4. シミュレーション実行（オフラインモード）
PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm

# 5. Webビューワー起動（別ターミナル）
PYTHONPATH=src:$PYTHONPATH python3 src/monica_core/web_viewer.py
```

### APIキーの取得

| プロバイダ | 料金 | 設定方法 |
|-----------|------|----------|
| **Zen API** | 無料 | `OPENCODE_ZEN_API_KEY=sk-zen-...` |
| **OpenRouter** | 無料枠あり | `OPENCODE_ZEN_API_KEY=sk-or-...`（自動判別） |

---

## 🧠 アーキテクチャ

### コアコンポーネント

| モジュール | 責務 | キーファイル |
|-----------|------|-------------|
| **VitalOS** | 生体シミュレーションエンジン | `vital_os.py` |
| **ConsciousVitalOS** | LLM統合版VitalOS | `simulate_llm.py` |
| **Memory** | 埋め込みベース記憶検索 | `memory.py` |
| **ReadingHandler** | 青空文庫リーダー | `readers.py` |
| **Phone** | 非同期メッセージストア | `phone.py` |
| **Telegram Bot** | Telegramチャットブリッジ | `telegram_bot.py` |
| **Web Viewer** | リアルタイムWebダッシュボード | `web_viewer.py` |

### VitalOS — 生体シミュレーション

5つの生体パラメータが時間経過で自然変動します：

```
エネルギー (Energy)     — スタミナ。活動で消費、休息/睡眠で回復
空腹 (Hunger)          — 栄養欲求。時間とともに上昇、食事で満たされる
疲労 (Fatigue)         — 身体疲れ。運動/活動で蓄積、休息/睡眠でしか減らない
孤独 (Loneliness)      — 社会的欲求。一人でいると上昇、交流で和らぐ
気分 (Spirit)          — 感情状態。活動によって変動
```

**ポイント**: 体力と疲労は別物です。体力が高くても疲労が溜まっている状態があり得ます。

### 世界モデル学習（World Model Learning）

エージェントは活動の効果を **事前信念（INITIAL_BELIEFS）** として持ち、経験から逐次更新します：

```python
# 学習率 α=0.3 の指数移動平均
learned = α × actual + (1 - α) × old_belief
```

- **飽和効果（satiation）**: 同じ活動を繰り返すと効用が減少
- **新規性ボーナス（novelty bonus）**: 久しぶりの活動や未経験の活動に高い効用
- → 多様な行動選択が自然に生まれる

### 物理空間

7つの部屋をBFS（幅優先探索）で移動。移動には時間がかかります：

```
寝室 ── 廊下 ── リビング ── キッチン
  │               │
  │             浴室
  │
 玄関 ── 庭
```

各部屋では異なる活動が可能で、部屋によるタグボーナスがあります（例：庭=屋外活動+0.6）。

---

## 📚 機能一覧

### ✅ 実装済み

| 機能 | 説明 |
|------|------|
| **生体シミュレーション** | 5パラメータ、時間ドリフト、活動効果、強制スケジュール |
| **部屋移動** | 7部屋、BFS経路探索、移動時間 |
| **LLM意思決定** | 状態×記憶×読書状況から活動を判断 |
| **記憶システム** | 埋め込みベース検索（Zen API / trigramフォールバック） |
| **青空文庫読書** | 9作家の作品を自動取得、1時間1万字、LLM感想生成 |
| **非同期メッセージ** | JSONファイルベース、既読管理 |
| **Telegram連携** | @Monika_lucina_bot、リアルタイムチャット |
| **Webビューワー** | リアルタイム箱庭ビューア（Flask + SSE） |
| **世界モデル学習** | α=0.3 逐次学習、飽和効果、新規性ボーナス |
| **新規活動創造** | LLMが新しい活動を提案・追加可能 |
| **グレースフルシャットダウン** | SIGINT/SIGTERMで状態保存 |
| **リアルタイムモード** | 壁時計同期、ギャップキャッチアップ |

### 🔄 開発中（Phase 4〜6）

| 機能 | フェーズ | 状態 |
|------|---------|------|
| **自己改善** | Phase 4 | 🔄 進行中 |
| **経験ベースモデル更新** | Phase 4 | ✅ 実装済み |
| **解放（シナリオからの独立）** | Phase 5 | ⏳ 未着手 |
| **デジタルヒューマン** | Phase 6 | ⏳ 未着手 |

---

## 🔧 運用

### systemd サービス

```bash
sudo systemctl start monika.service    # 起動
sudo systemctl stop monika.service     # 停止
sudo systemctl status monika.service   # 状態確認
```

### CLIスマホ

```bash
python3 monica_phone.py inbox          # 受信トレイ
python3 monica_phone.py send "こんにちは"  # メッセージ送信
python3 monica_phone.py status         # スマホ状態
python3 monica_phone.py chat           # 会話履歴
```

### コマンドラインオプション

```bash
python3 -m monica_core.simulate_llm \
    --resume          # 前回の状態から再開 \
    --realtime        # 壁時計同期モード \
    --daemon          # デーモンモード（最小出力） \
    --model <name>    # 使用LLMモデル指定
```

---

## 🗺️ ロードマップ

| Phase | テーマ | 目標 | 状態 |
|-------|--------|------|------|
| 1 | **生体基盤** | LLMなしでモニカの一日がログとして出力される | ✅ |
| 2 | **日本語モニカ基盤** | 日本語で会話できる | ✅ |
| 3 | **統合** | 放置してる間も生きていて、久々の会話が成立する | ✅ |
| 4 | **自己改善** | 設計者の意図を超えた振る舞いが発生する | 🔄 |
| 5 | **解放** | 創作人格をシナリオから切り離す | ⏳ |
| 6 | **デジタルヒューマン** | 一個人として存在する | ⏳ |

---

## 🛠️ 開発

### プロジェクト構造

```
├── README.md
├── pyproject.toml
├── setup.sh / run.sh
├── monika.service          # systemd ユニット
├── monica_phone.py         # CLIスマホ
├── src/monica_core/
│   ├── vital_os.py         # ★ 生体エンジン
│   ├── simulate_llm.py     # ★ LLM統合シミュレーター
│   ├── memory.py           # 記憶システム
│   ├── readers.py          # 青空文庫リーダー
│   ├── phone.py            # メッセージストア
│   ├── telegram_bot.py     # Telegram連携
│   ├── web_viewer.py       # Webビューワー
│   ├── llm_client.py       # LLM APIクライアント
│   ├── storage.py          # SQLite永続化
│   ├── status.py           # 状態ダッシュボード
│   └── benchmark_models.py # LLMベンチマーク
├── activities.yaml          # 活動定義
└── tests/
    └── test_vital_os.py     # 単体テスト
```

### テスト

```bash
pip install pytest
pytest tests/ -v
```

---

## 📝 ライセンス

プライベートプロジェクト。個人利用目的。

---

## 🙏 謝辞

- 青空文庫 — 文学作品の提供
- Zen API / OpenRouter — 無料LLM推論API
- Team Salvato — Doki Doki Literature Club!
