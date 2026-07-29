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
│  │ 信念学習  │    │   SQLite 永続化       │   │
│  └──────────┘    │   記憶 / 状態 / メッセージ│
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
pip install pyyaml flask  # Webビューワー・YAML設定用

# 3. APIキー設定
cp .env.example .env
# OPENCODE_ZEN_API_KEY を設定（Zen API または OpenRouter）

# 4. シミュレーション実行（**必ず --realtime フラグ推奨**）
PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm --resume --realtime

# 5. Webビューワー起動（別ターミナル）
PYTHONPATH=src:$PYTHONPATH python3 src/monica_core/web_viewer.py
```

> ⚠️ **`--realtime` フラグがない場合**: 時刻が壁時計と同期せず、時間が爆速で進行します。
> 5分/tick × CPU最高速度で進むため、LLMに過剰なリクエストが飛び行動が偏ります。
> 必ず `--realtime` を付けて起動してください。

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
| **Memory** | 埋め込みベース記憶検索・圧縮 | `memory.py` |
| **Storage** | SQLite永続化（状態・メッセージ・記憶） | `storage.py` |
| **ReadingHandler** | 青空文庫リーダー | `readers.py` |
| **Phone** | 非同期メッセージストア（source管理） | `phone.py` |
| **Telegram Bot** | Telegramチャットブリッジ | `telegram_bot.py` |
| **Web Viewer v2** | モダン箱庭ビューア（Flask + SSE） | `web_viewer.py` |

### VitalOS — 生体シミュレーション

5つの生体パラメータが時間経過で自然変動します：

```
エネルギー (Energy)     — スタミナ。活動で消費、休息/睡眠で回復
空腹 (Hunger)          — 栄養欲求。時間とともに上昇、食事で満たされる
疲労 (Fatigue)         — 身体疲れ。運動/活動で蓄積、休息/睡眠でしか減らない
孤独 (Loneliness)      — 社会的欲求。一人でいると上昇、交流で和らぐ
気分 (Spirit)          — 感情状態。活動によって変動
```

**ポイント**: 体力と疲労は別物です。体力が高くても疲労が溜まっている状態があり得ます。疲労は休息か睡眠でしか解消されません。

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

### ✅ 実装済み（v2.0）

| 機能 | 説明 |
|------|------|
| **生体シミュレーション** | 5パラメータ、時間ドリフト、活動効果、強制スケジュール |
| **部屋移動** | 7部屋、BFS経路探索、移動時間ペナルティ |
| **LLM意思決定** | 状態×記憶×読書状況から活動を判断。障害時はルールベースにフォールバック |
| **記憶システム** | 埋め込みベース検索（Zen API / trigramフォールバック）、自動圧縮（10保存ごと） |
| **青空文庫読書** | 9作家の作品を自動取得、1時間1万字、LLM感想生成→記憶保存 |
| **非同期メッセージ** | SQLite + JSON 二重保存。source識別子（web/telegram/monika）で送信元管理 |
| **Telegram連携** | @Monika_lucina_bot、メッセージプッシュ通知 |
| **Webビューワー v2** | グラスモーフィズムデザイン、箱庭マップ、アニメーション、SSEリアルタイム更新 |
| **世界モデル学習** | α=0.3 逐次学習、飽和効果、新規性ボーナス |
| **新規活動創造** | LLMが新しい活動を提案・追加可能 |
| **SQLite永続化** | 単一DBで状態・メッセージ・記憶・モデルパラメータを一元管理。自動JSONバックアップ |
| **活動設定の外部化** | `activities.yaml` で活動定義をコードから分離 |
| **LLM障害耐性** | 3回連続失敗でルールベース判断にフォールバック |
| **グレースフルシャットダウン** | SIGINT/SIGTERMで状態保存 |
| **リアルタイムモード** | 壁時計同期、ギャップキャッチアップ、5分tick |
| **ログ基盤** | Python logging モジュール（全モジュール統一言語） |
| **単体テスト** | pytest 40テスト（VitalOS状態遷移・Phone CRUD） |

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
# ユーザーモード（推奨）
systemctl --user start monika.service    # 起動
systemctl --user stop monika.service     # 停止
systemctl --user status monika.service   # 状態確認

# システムモード
sudo systemctl start monika.service
```

### 起動オプション詳細

```bash
# 基本（リアルタイム同期）
PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm --resume --realtime

# デーモンモード（最小出力）
PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm --resume --realtime --daemon

# Webビューワー（別ターミナル）
PYTHONPATH=src:$PYTHONPATH python3 src/monica_core/web_viewer.py

# Telegram Bot（別ターミナル）
PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.telegram_bot

# CLIスマホ
python3 monica_phone.py inbox          # 受信トレイ
python3 monica_phone.py send "こんにちは"  # メッセージ送信
python3 monica_phone.py status         # スマホ状態
python3 monica_phone.py chat           # 会話履歴
```

### コマンドラインオプション一覧

```bash
python3 -m monica_core.simulate_llm \
    --resume          # 前回の状態から再開 \
    --realtime        # 壁時計同期モード（推奨） \
    --daemon          # デーモンモード（最小出力） \
    --model <name>    # 使用LLMモデル指定
```

> 💡 **`--realtime` がない場合**、起動時に警告が表示されます：
> ```
> ⚠️  --realtime フラグなしで起動中。時刻が壁時計と同期しません
>    推奨: PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm --resume --realtime
> ```

### データストレージ

| 保存先 | パス | 内容 |
|--------|------|------|
| **SQLite** | `data/monica.db` | 状態・メッセージ・記憶・モデルパラメータ（プライマリ） |
| **JSON** | `data/state.json` | 状態（互換性/Webビューワー用） |
| **JSON** | `data/phone_messages.json` | メッセージ（フォールバック用） |
| **JSON** | `data/memory_store.json` | 記憶（フォールバック用） |

`save()` は常にJSONとSQLiteの**両方に同時保存**します。既存のJSONデータは初回アクセス時に自動的にSQLiteに移行されます。

---

## 📱 メッセージ送信元（source）管理

メッセージには送信元を示す `source` フィールドが付与されます：

| source値 | 説明 | 送信元 |
|----------|------|--------|
| `"web"` | Webビューワーから | `phone_add("user", ..., source="web")` |
| `"telegram"` | Telegram Botから | `phone_add("user", ..., source="telegram")` |
| `"monika"` | モニカの自発メッセージ | `phone_add("monika", ..., source="monika")` |

これによりWebビューワーで直接話しかけたのか、Telegram経由なのかを区別できます。

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
├── monika.service              # systemd ユニット
├── monica_phone.py             # CLIスマホ
├── activities.yaml             # 活動定義の外部設定ファイル
├── src/monica_core/
│   ├── __init__.py             # パッケージ初期化
│   ├── vital_os.py             # ★ 生体エンジン（状態・ドリフト・経路探索・世界モデル）
│   ├── simulate_llm.py         # ★ LLM統合シミュレーター（判断・対話・読書・フォールバック）
│   ├── storage.py              # SQLite永続化（全データ一元管理）
│   ├── memory.py               # 記憶システム（埋め込み検索・圧縮）
│   ├── readers.py              # 青空文庫リーダー
│   ├── phone.py                # メッセージストア（source管理）
│   ├── telegram_bot.py         # Telegram連携
│   ├── web_viewer.py           # ★ 箱庭ビューワー v2（Flask + SSE + グラスモーフィズム）
│   ├── llm_client.py           # LLM APIクライアント（Zen/OpenRouter自動判別）
│   ├── status.py               # 状態ダッシュボード
│   ├── ws_monitor.py           # 死活監視
│   └── benchmark_models.py     # LLMベンチマーク
└── tests/
    ├── test_vital_os.py         # VitalOS単体テスト
    └── test_phone.py            # Phone CRUDテスト
```

### テスト

```bash
# 全テスト実行
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/ -v

# 特定テストのみ
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_vital_os.py -v
```

現在40のテストが実装されており、すべて通過します。

### 活動設定のカスタマイズ

`activities.yaml` で活動の定義をコードから分離できます：

```yaml
瞑想:
  duration: 20
  tags:
    rest: 0.8
    mental: 0.3
  initial_beliefs:
    energy: 5
    fatigue: -5
    spirit: 8
  is_rest: true
  locations:
    - garden
    - bedroom
```

---

## 💡 Tips

### よくある問題と解決

| 問題 | 原因 | 解決 |
|------|------|------|
| 時間が爆速で進む | `--realtime` なしで起動 | `--realtime` フラグを追加 |
| Webビューワーに状態が出ない | 前回の保存がJSONのみ | 一度 `--resume --realtime` で起動→保存→Web再読込 |
| 行動が偏る | 時間が速すぎて判断ループ | `--realtime` で起動 |
| Telegramの応答がない | Botトークン未設定 | `.env` の `MONIKA_TELEGRAM_TOKEN` を確認 |
| メモリーエラー | 記憶が溜まりすぎ | `memory.py` が自動圧縮（10保存ごとに300件に制限） |

---

## 📝 ライセンス

プライベートプロジェクト。個人利用目的。

---

## 🙏 謝辞

- 青空文庫 — 文学作品の提供
- Zen API / OpenRouter — 無料LLM推論API
- Team Salvato — Doki Doki Literature Club!
