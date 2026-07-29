# Monica Project — Agent Handover (v10 + Phase 1 improvements)

## Current State

Monica is a **Free Energy Principle (FEP) driven autonomous AI agent** with full multi-agent architecture, world model, goal system, tool chaining, and variational free energy computation.

### Architecture (v10 Full FEP + v11 Phase 1 Action Chaining)

```
                         ┌──────────────────────────────────┐
                         │  SensingAgent (background @ 2s)   │
                         │  • Forward passes (conv + curiosity)│
                         │  • PE → Blackboard.beliefs       │
                         │  • KL divergence tracking (v14)  │
                         └──────────┬───────────────────────┘
                                    │ beliefs
                                    ▼
 User Input ──→ ┌──────────────────────────────────────┐
                │  Main Loop (decide_mode)              │
                │  • Goal-aware mode selection (v11)    │
                │  • World Model context (v10 Phase 2)  │
                │  • Tool chain continuation (Phase 1)  │
                │  • FEP History trend analysis (v14)   │
                └──────────┬───────────────────────────┘
                           │
              ┌────────────┼────────────┬──────────────────┐
              ▼            ▼            ▼                  ▼
        ┌──────────┐┌──────────┐┌──────────┐┌──────────────────┐
        │ CHAT     ││ THINK   ││ IDLE    ││ ActionAgent      │
        │ API gen  ││ThinkAgent││ Drift   ││ [WRITE]/[SHELL]  │
        │+local    ││+Action  ││ only    ││ [SEARCH]/[PYTHON] │
        │fallback  ││ prompts ││         ││ Tool chain (max 5)│
        └──────────┘└──────────┘└──────────┘└──────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| WorldModel | THINK出力をキーワード重複でエピソード圧縮・`world_model_v8h.json`に永続化 |
| GoalManager | MetaAgent焦点→APIで目標生成・進捗評価・`/s`で表示 |
| ThinkAgent | THINKループ: アクション指向プロンプト + ツール連鎖（観測→再思考） |
| ActionAgent | ツール実行[WRITE/SHELL/SEARCH/READ/PYTHON/CALC]＋結果観測＋連鎖追跡 |
| FEPHistory | PE/KL/VFEの時系列記録・傾向分析（`/s`でtrend表示） |
| SensingAgent | バックグラウンドforward + PE計測 + KL追跡（v14） |
| MetaAgent | 焦点生成 + 目標トリガー（4 THINKサイクルごと） |
| CuriosityEngine | トピックカウント → 好奇心ボーナス |
| SteeringVector | hidden stateの高PE/低PE方向ベクトル |

### Multi-Provider API

| Priority | Provider | Model | Response |
|----------|----------|-------|----------|
| 1 | **Groq** 🚀 | `llama-3.3-70b-versatile` | 0.4-0.8s |
| 2 | OpenRouter | `openrouter/free` | 3-10s |
| 3 | **Local** | Qwen3-4B (fallback) | per-token FEP |

### Test Results (2026-07-23, 75s runtime)

```
Modes: CHAT(1) / THINK(10) / IDLE cycles
World Model: 3 episodes (merged)
Goals: 2 (pending + in_progress)
FEP History: 10 entries (pes/kls/vfes)
KL NaN: None
```

---

## What Has Been Done (this session)

### v10 Phase 2 — World Model

| Feature | Status | Detail |
|---------|--------|--------|
| **Thought compression** | ✅ | THINK出力→キーワード抽出→重複(≥2)→エピソードマージ |
| **Keyword retrieval** | ✅ | `context(query)`で関連エピソード→THINKプロンプトに注入 |
| **Persistence** | ✅ | `world_model_v8h.json`に自動保存/読み込み |
| **Max 50 episodes** | ✅ | 最新50エピソードを保持 |

### v11 — Goal-Driven Behavior

| Feature | Status | Detail |
|---------|--------|--------|
| **Goal dataclass** | ✅ | id/description/status/subgoals/category |
| **API goal generation** | ✅ | MetaAgent焦点→`api_chat()`で目標化 |
| **Category detection** | ✅ | learn/create/explore/act 自動分類 |
| **Progress evaluation** | ✅ | THINK後にAPIで`yes/no`評価 |
| **goal_prompt()** | ✅ | アクティブ目標をTHINKプロンプトに注入 |
| **Persistence** | ✅ | `state_v8h.json`に保存/復元 |

### v12 — Action Space Expansion

| Feature | Status | Detail |
|---------|--------|--------|
| **[WRITE:]** | ✅ | ファイル書き込み（`mkdir -p`自動） |
| **[SHELL:]** | ✅ | サブプロセス実行（timeout 15s） |
| **Tool result→Observation** | ✅ | `[observation: type(target)]` としてconvに追加 |
| **Post-tool sensing** | ✅ | ツール結果→forward→beliefs更新 |

### v13 — Multi-Agent Separation

| Feature | Status | Detail |
|---------|--------|--------|
| **ThinkAgent** | ✅ | THINKループ管理（prompt/world_model/goals/curiosity注入） |
| **ActionAgent** | ✅ | ツール実行+観測+履歴管理（最大20件） |
| **SensingAgent** | ✅ | KL divergence追跡追加（`sense._last_kl`） |
| **Agent coordination** | ✅ | 全エージェントがBlackboard経由で通信 |

### v14 — Full FEP

| Feature | Status | Detail |
|---------|--------|--------|
| **KL divergence** | ✅ | `compute_kl_divergence(h_prev, h_cur)` → softmax確率化 |
| **Variational free energy** | ✅ | `VFE = PE + KL`（accuracy+complexity） |
| **FEPHistory** | ✅ | PE/KL/VFEの50件時系列＋傾向分析（`trend(window=10)`） |
| **NaN guard** | ✅ | KL値のNaN/Inf→0にフォールバック |
| **CPU tensor persistence** | ✅ | KL計算用hidden stateをCPUに保持（cache clear耐性） |
| **/s display** | ✅ | goal + fep trend を表示 |

### Phase 1 — Action-Oriented THINK + Tool Chaining

| Feature | Status | Detail |
|---------|--------|--------|
| **Action prompts** | ✅ | 5つのアクション指向THINK_PROMPTSに置換（全promptにtoolキーワードまたはaction指示） |
| **Tool chaining** | ✅ | ツール結果→観測→次のTHINKを即実行（`think_agent.chain_count`追跡、最大5連鎖） |
| **Observation injection** | ✅ | `ThinkAgent.think(observations=...)`で前回ツール結果を次のpromptに注入 |
| **Chain state prompt** | ✅ | `[tool chain step 2/5]` をTHINKプロンプトに動的追加 |
| **ActionAgent.has_tool_results()** | ✅ | 直近3件の履歴からツール実行有無を判定 |
| **Chain exit** | ✅ | ツール未実行 or 最大連鎖到達→IDLE復帰＋`last_observations=None`リセット |
| **THINK_TOKENS** | ✅ | 80→120に増加（tool command出力の余裕） |

### Phase 2 — 会話記憶（ユーザープロファイル）

| Feature | Status | Detail |
|---------|--------|--------|
| **UserProfile class** | ✅ | 名前/話題(weighted)/好み/first_seen/last_interaction/interaction_countを追跡 |
| **Name detection** | ✅ | `I am X`, `I'm X`, `call me X`, `my name is X` から自動抽出（上書き防止） |
| **Topic tracking** | ✅ | 会話中の単語(≥4文字)をカウント・ソート・最大10件保持 |
| **Context injection** | ✅ | CHAT（`framed_input`に追加） + THINK（`profile.context()`をThinkAgentに注入） |
| **Eager guard** | ✅ | `interaction_count < 3` かつ名前未学習のときは空context（ノイズ防止） |
| **Persistence** | ✅ | `state_v8h.json`保存/復元（`profile.state_dict()`/`load_state_dict()`） |
| **Restore message** | ✅ | 起動時に `[profile] restored: Taro (5 interactions)` 表示 |

### Phase 3 — 成果物生成（コードブロック→自動ファイル保存）

| Feature | Status | Detail |
|---------|--------|--------|
| **ArtifactExtractor class** | ✅ | THINK出力から ``` ```code ``` ``` ブロックを検出→`artifacts/`に自動保存 |
| **Language→extension mapping** | ✅ | python/py→.py, json→.json, html→.html, bash/sh→.sh, js→.js 等13言語対応 |
| **Minimum content guard** | ✅ | コードブロック20文字未満はスキップ（ノイズ防止） |
| **Long thought fallback** | ✅ | コードブロックなしでも200文字以上5行以上のTHINKはテキストダンプ保存 |
| **Artifact directory** | ✅ | `artifacts/` 自動作成（`mkdir -p`相当） |
| **History management** | ✅ | 最大20件保持（超過時は古いものから削除） |
| **THINK prompt instruction** | ✅ | "code blocks → auto-saved as file" をTHINKプロンプト末尾に追加 |

### Phase 4 — Minimal Web UI (FastAPI + SSE) ✅

| Feature | Status | Detail |
|---------|--------|--------|
| **FastAPI server** | ✅ | `monica_web.py` — バックグラウンド自律ループ＋HTTP API |
| **SSE real-time streaming** | ✅ | `/api/events` でモード変更・FEP更新・思考ログをストリーム |
| **Chat endpoint** | ✅ | `POST /api/chat` でユーザー入力→応答返却 |
| **State endpoint** | ✅ | `GET /api/state` で状態JSON取得 |
| **Web UI** | ✅ | `static/index.html` — ダークテーマ・FEPメトリクスパネル・思考ログ表示 |
| **Enterキー送信** | ✅ | 入力欄でEnter→即送信、レスポンシブ対応 |
| **CLI互換維持** | ✅ | `monica_v8_hybrid.py` はそのままCLIでも使用可能 |

起動:
```bash
cd /home/koushi/monica-v3
source venv/bin/activate
python3 monica_web.py
# → http://localhost:8000
```

---

## File Layout

| File | Description |
|------|-------------|
| `monica_v8_hybrid.py` | **Main** (~1736 lines, v10 + Phase 1-3) |
| `monica_web.py` | **Phase 4 Web UI** (~260 lines, FastAPI + SSE) |
| `static/index.html` | Web frontend (dark-theme chat + status panel) |
| `artifacts/` | Auto-generated artifacts directory (Phase 3) |
| `state_v8h.json` | State (conv/beliefs/goals/fep_history/meta) |
| `world_model_v8h.json` | World Model episodes (auto-managed) |
| `log_v8h.jsonl` | Runtime log (auto-appended) |
| `summary_v8h.jsonl` | Compressed conversation summaries |
| `HANDOVER.md` | This file |
| `ROADMAP.md` | Development roadmap |

---

## Running

### CLI mode (従来)
```bash
cd /home/koushi/monica-v3
source venv/bin/activate
python3 monica_v8_hybrid.py
```

### Web mode (Phase 4)
```bash
cd /home/koushi/monica-v3
source venv/bin/activate
python3 monica_web.py
```
→ http://localhost:8000 でアクセス

Commands:
- Type any message → CHAT
- `/s` → print state (self_model, PE, drift, goal, fep trend, adapt params)
- `exit` / `quit` → shutdown

THINK activates autonomously when self_model < 0.30.
World Model and Goals activate after first meta-focus generation (4 THINK cycles).

---

## Known Issues

### 1. 会話記憶がない ✅ (Fixed in Phase 2)
- ユーザー名・好み・過去の話題を覚える（state_v8h.jsonにprofile情報を保存）
- 名前認識: "I am X", "I'm X", "call me X", "my name is X"
- context注入: 3回以上のinteraction後にCHAT/THINKプロンプトに追加

### 2. 成果物を自発的に出力しない ✅ (Fixed in Phase 3)
- THINK内コードブロック→`artifacts/`に自動保存
- 200文字以上5行以上の思考ログ→テキストダンプ

### 3. Web UIがリアルタイム更新されない 🔜
- SSE接続が切れたときの再接続処理が未実装（ブラウザのEventSourceは自動再接続するが、状態の再同期が必要）
- Fix: /api/stateポーリングによる定期同期（現在実装済み）

### 4. THINKがツールを使うとは限らない
- アクション指向プロンプトにしたが、モデルが実際に `[SEARCH:]` 等を出力するかは未検証
- API応答次第で従来通り哲学的テキストに戻る可能性
- Fix: システムプロンプトでの強制 or 事後検証ループ

### 5. Groq rate limit (30 RPM)
- ツール連鎖（5連鎖×0.6s=3s）で割とすぐ消費
- Fallback to OpenRouter works but slower
- 対策: chain_count低いときはLocalでTHINK

### Phase 9 — 共進化エージェント ✅

| Feature | Status | Detail |
|---------|--------|--------|
| **セッション間学習** | ✅ | `session_log.jsonl`に終了時サマリー保存、起動時`memory_summaries`に注入（CLI+Web両対応） |
| **[FILEINFO:] ツール** | ✅ | ファイルメタデータ取得（種類/サイズ/更新日時/画像/音声/動画/バイナリ対応） |
| **[GITHUB:] ツール** | ✅ | 公開GitHub API連携: user/repo/content/searchの4モード |
| **THINK_PROMPTS拡張** | ✅ | 全6プロンプトに新ツール記載（ツール探索を促進） |

起動時に過去セッションの要約が`memory_summaries`に自動追加され、`api_chat`のsystem promptに反映される。

---

## Next Steps (継続的研究)

1. ~~Phase 2: 会話記憶~~ ✅
2. ~~Phase 3: 成果物生成~~ ✅
3. ~~Phase 4: Web UI~~ ✅
4. ~~Phase 5: Core FEP改善~~ ✅
5. ~~Phase 6: 自律サイクル~~ ✅
6. ~~Phase 7: 適応的学習ループ~~ ✅
7. ~~Phase 8: ローカルFEP最適化~~ ✅
8. ~~Phase 9: 共進化エージェント~~ ✅ **← ALL PHASES COMPLETE**
9. **継続的研究**: CLIP画像認識 / GitHub認証連携 / Slack/Discord / 価値観形成 / 1ヶ月連続稼働テスト

---

*Last updated: 2026-07-23 (All 9 phases complete! 🎉)*
