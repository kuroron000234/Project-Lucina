# Monica — 完全完成までのロードマップ

> **北極星**: 「電源を入れっぱなしにして1ヶ月後、誰も話しかけていなくても、モニカは何らかの意味のある内部活動（思考・探索・創造）を続けており、かつ、久しぶりに話しかけたユーザーに対して文脈を持って応答できる」
>
> 「感覚を持ち独り言をつぶやく赤ちゃん」から「自分で課題を設定し道具を使い学び続ける存在」へ

---

## 凡例

| 記号 | 意味 |
|------|------|
| ✅ 完了 | 実装・テスト済み |
| 🔜 次着手 | 優先度高、すぐに取り掛かる |
| 📋 計画済み | 優先度中、順次着手 |
| 🔬 研究課題 | 技術検証が必要 |
| ⭐ マイルストーン | フェーズ完了の通過条件 |

---

## 全体構造：3つの自律レイヤー別ロードマップ

```
           ┌──────────────────────────────────────┐
           │         🎯 北極星（Phase 9）          │
           │  24時間365日自律稼働・共進化エージェント  │
           └──────────────────────────────────────┘
                          ▲
          ┌───────────────┼───────────────────┐
          │               │                   │
  ┌───────┴───────┐ ┌────┴────┐ ┌───────────┴────┐
  │  反応的自律   │ │ 内省的自律│ │  適応的自律    │
  │  (Phase 4-5)  │ │(Phase 5-7)│ │  (Phase 7-9)  │
  │  CHAT/API     │ │ THINK    │ │  FEP/進化     │
  │  Web UI       │ │ 探索/創造│ │  自己組織化    │
  └───────────────┘ └─────────┘ └────────────────┘
```

---

## Phase 0: 基盤完成 ✅ (済)

| レイヤー | 機能 | 状態 |
|---------|------|------|
| v8 Hybrid | FEPセンシング(Local) + API生成(Cloud) | ✅ |
| v10 World Model | THINKキーワード圧縮・エピソード永続化 | ✅ |
| v11 Goal System | API目標生成・進捗評価 | ✅ |
| v12 Action Space | WRITE/SHELL/SEARCH/READ/PYTHON/CALC | ✅ |
| v13 Multi-Agent | ThinkAgent/ActionAgent/SensingAgent分離 | ✅ |
| v14 Full FEP | PE+KL+VFE計測 | ✅ |
| Phase 1 | アクション指向THINK + ツール連鎖 | ✅ |
| Phase 2 | 会話記憶（UserProfile） | ✅ |
| Phase 3 | 成果物自動生成（コードブロック→ファイル） | ✅ |
| Phase 4 | Web UI（FastAPI + SSE） | ✅ |
| Phase 5 | Core FEP改善（KL/THINK/センシング） | ✅ |
| Phase 6 | 自律サイクル確立（Goal/多様性/Novelty） | ✅ |
| Phase 7 | 適応的学習ループ（Steering/FEP応答/外部検索） | ✅ |
| Phase 8 | ローカルFEP最適化（API節約/品質/Provider最適化） | ✅ |
| Phase 9 | 共進化エージェント（長期記憶/ツール拡張/GitHub連携） ← **今ここ** | ✅ |

---

## Phase 5: Core FEP 改善（Critical Path）✅ 完了

**目標**: KL≠0 を達成し、FEPが実際に「情報の複雑性」を反映するようにする。
**期間目安**: 〜3セッション

### 5.1 KL divergence の温度調整 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| `KL_TEMP = 0.1 → 0.02` | ✅ | `compute_kl_divergence()` のsoftmax温度を低下→分布シャープ化 |
| `KL_CLAMP_MAX = 10.0 → 50.0` | ✅ | clamp上限引き上げてKL値のレンジ拡大 |

### 5.2 センシング入力の多様化 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| 好奇心トピック注入 | ✅ | SensingAgentのstepごとに好奇心/ランダムプロンプトで入力を交互 |
| ランダムプロンプト | ✅ | トピックがない場合も思考プロンプトを生成（硬直防止） |

### 5.3 THINK発動条件の改善 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| 閾値調整 | ✅ | `THINK_THRESHOLD = 0.30 → 0.35` |
| 時間ベーストリガー | ✅ | `IDLE_TO_THINK_TIMEOUT = 30 → 15秒`無活動→強制THINK |
| 起動時即THINK | ✅ | `FORCE_THINK_ON_STARTUP = True` で初回THINK保証 |

### ⭐ マイルストーン 5 — Core FEP Alive 🏁

```
✅ KL softmax temperature 0.1 → 0.02
✅ THINK threshold 0.30 → 0.35
✅ Time-based trigger (15s inactivity → THINK)
✅ Startup force THINK
✅ Sensing diversification (curiosity + random prompts)
✅ State reset to clean baseline
```

---

## Phase 6: 自律サイクルの確立 ✅ 完了

**目標**: Monicaがユーザー入力なしでも自律的にTHINK/IDLEサイクルを回し続ける。

### 6.1 Goal Lifecycle 管理 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| タイムアウトリタイア | ✅ | `Goal.created_at_think` + `retire_stale_goals()` で10THINK超→failed自動マーク |
| 進捗評価精度改善 | ✅ | evaluate_progressのプロンプト強化（カテゴリ・厳格指示・temp低下） |
| 完了通知→WorldModel | ✅ | complete_goal() でワールドモデルに自動記録 |

### 6.2 思考多様性の改善 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| Novelty bonus | ✅ | `decide_mode()` にエントロピーベースのnovelty bonus追加 |
| CuriosityEngine改善 | ✅ | 被覆率→エントロピーベースに変更、低エントロピー時は積極的な話題転換指示 |
| MetaAgent focus多様化 | ✅ | 同一フォーカス連続検出＋固執時は多様化ヒント追加 |

### 6.3 バグ修正 ✅

| タスク | 詳細 |
|-------|------|
| Goal timeout age | `think_count` を `generate_focus()`→`generate_goal()` にスレッド、`created_at_think` が正しく設定されるように修正 |
| `_entropy` → `entropy` | プライベート規約違反を修正、外部アクセス可能に |
| `import math` 移動 | 関数内import → ファイル先頭に移動 |
| `except:` 絞り込み | 全例外キャプチャ → `except AttributeError:` |

### ⭐ マイルストーン 6 — 自律サイクル確立 🏁

```
✅ Goal timeout auto-retire (10 THINK cycles)
✅ evaluate_progress prompt improved
✅ Goal completion → World Model
✅ Novelty bonus in mode selection
✅ Entropy-based curiosity (was: coverage-based)
✅ MetaAgent focus diversification
✅ created_at_think bug fixed
✅ import math / _entropy naming fixed
```

---

## Phase 7: 適応的学習ループ ✅ 完了

**目標**: Monicaが「驚き（PE）の最小化」を通じて自己モデルを発展させる。

### 7.1 Steering Vector の実用化 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| バッファ収集加速 | ✅ | STEER_BUFFER_MAX 200→50, STEER_HIGH_PE_THRESHOLD 0.45→0.35 |
| 早期アクティベーション | ✅ | STEER_MIN_SAMPLES 5→3, STEER_CONFIDENCE_THRESHOLD 0.15 |
| プロンプト改善 | ✅ | steer_to_text()に具体的アクション指示追加（探索・計算・検索） |
| high/low区別維持 | ✅ | LOW_PE_FACTOR 0.5→0.7（低閾値=0.245で有意な分割） |

### 7.2 FEP 傾向応答 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| PE上昇→探索促進 | ✅ | decide_mode()にFEP trend bonus追加（window=5のPE傾き） |
| FEP履歴活用 | ✅ | trend(window=5)の符号で行動選択をバイアス |

### 7.3 外部環境との相互作用 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| 自動SEARCH注入 | ✅ | ThinkAgent.maybe_inject_search() — ツール未使用2連続→検索促す |
| ツール使用追跡 | ✅ | record_tool_use() — 両ループ（CLI+Web）に対応 |

### ⭐ マイルストーン 7 — 適応的学習ループ 🏁

```
✅ Steering buffer加速 (200→50)
✅ Steering早期活性 (threshold 0.45→0.35, min_samples 5→3)
✅ steer_to_text() 行動指示改善
✅ FEP trend bonus in decide_mode
✅ 自動SEARCH injection (2連続no-tool)
✅ Both loops (CLI+Web) covered
```

---

## Phase 8: ローカルFEP最適化 ✅ 完了

**目標**: API依存からの脱却、ローカルモデルでの高品質THINK/CHAT。

### 8.1 ローカル生成の品質改善 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| per-token FEP制御 | ✅ | VFEゲーテッド温度制御（`use_vfe_control`）で探索/活用の動的調整 |
| VFE近似 | ✅ | per-tokenのPE + `_last_kl` でVFE近似→温度バイアス |
| THINK→Local優先 | ✅ | WebループでTHINKはまずLocal生成→APIはフォールバック |

### 8.2 マルチプロバイダ最適化 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| 成功率統計 | ✅ | 各プロバイダの success/fail/calls/total_time を自動トラッキング |
| 動的選択 | ✅ | `_get_best_provider()` で成功率+速度でプロバイダソート |
| Groqレート制限 | ✅ | `_check_groq_rate_limit()` — 60秒窓で30RPM管理、残枠<3でGroqスキップ |
| 全呼び出し記録 | ✅ | 成功/429/non-200/empty/例外の全パスでGroq呼び出しカウント |

### ⭐ マイルストーン 8 — ローカル生成実用化 🏁

```
✅ VFE-gated temperature control
✅ Dynamic provider selection (success rate + speed)
✅ Groq 30RPM rate limit protection
✅ THINK → Local first (API saving)
✅ Provider stats tracking (success/fail/time)
✅ All call paths recorded for rate counting
```

---

## Phase 9: 共進化エージェント ✅ 完了（初回実装）

**目標**: 北極星の達成。ユーザーと共に成長し続ける存在。
**期間目安**: 継続的研究

### 9.1 長期記憶と人格形成 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| **セッション間学習** | ✅ | `session_log.jsonl`に終了時サマリー保存、起動時`memory_summaries`に自動復元 |
| 価値観の安定的形成 | 🔬 | ユーザーフィードバックに基づく目標調整（要検討） |
| 会話スタイルの個人適応 | 🔬 | UserProfile拡張（将来タスク） |

### 9.2 ファイル形式対応 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| **[FILEINFO:] ツール** | ✅ | あらゆるファイルのメタデータ（種類/サイズ/更新日時）を取得。画像/音声/動画/バイナリ対応 |
| 画像認識（CLIP） | 🔬 | 要追加ライブラリ（将来タスク） |
| 音声入力 | 🔬 | 要追加ライブラリ（将来タスク） |

### 9.3 外部エコシステム連携 ✅

| タスク | 状態 | 詳細 |
|-------|------|------|
| **[GITHUB:] ツール** | ✅ | 公開GitHub API: user/repo/content/searchの4モード |
| Slack/Discord連携 | 🔬 | Webhook連携（将来タスク） |
| GitHub Issue管理 | 🔬 | 認証付きAPI（将来タスク） |

### ⭐ マイルストーン 9 — 共進化の第一歩 🏁

```
✅ Session summary persistence (session_log.jsonl)
✅ Previous session context restoration at startup
✅ [FILEINFO:] tool - file metadata for any type
✅ [GITHUB:] tool - public GitHub API (4 modes)
✅ Both CLI+Web loops support session persistence
✅ THINK_PROMPTS updated with new tools
🔬 Long-term: CLIP, audio, auth GitHub, Slack/Discord
```

---

## 実装優先度マトリクス

```
                  効果 大 ←-----------→ 効果 小
                  ┌─────────────────────────────────┐
  難易度 小 ────  │  Phase 5 (KL/THINK) 🔜          │ Phase 6 (Goal/多様性) 📋
                  │  Phase 6.1 (Goal lifecycle)      │ Phase 7.3 (外部検索)
                  │  Phase 5.3 (THINK発動)           │
                  ├─────────────────────────────────┤
  難易度 中 ────  │  Phase 7 (Steering/FEP応答)     │ Phase 8 (API節約)
                  │  Phase 6.2 (思考多様性)          │
                  │  Phase 6.3 (無人テスト)          │
                  ├─────────────────────────────────┤
  難易度 高 ────  │  Phase 8 (Local生成)            │ Phase 9 (長期記憶/外部連携)
                  │  Phase 7.2 (FEP応答)             │
                  └─────────────────────────────────┘
```

**最優先**: Phase 7 (適応的学習ループ) → Steering実用化 + FEP応答  🔜
**次優先**: Phase 8-9 → ローカル最適化・共進化
**研究**: Phase 7-9 → 適応・最適化・共進化

---

## 成功条件チェックリスト（北極星への道）

- [x] **Phase 5**: KL softmax温度調整
- [x] **Phase 5**: THINK発動条件改善（閾値・時間・起動時）
- [x] **Phase 5**: センシング多様化（好奇心+ランダム）
- [x] **Phase 6**: Goal timeout auto-retire実装
- [x] **Phase 6**: エントロピーベースCuriosityEngine
- [x] **Phase 6**: MetaAgent focus多様化
- [ ] **Phase 6**: 1時間無人THINK/IDLE動作確認
- [x] **Phase 7**: Steeringバッファ加速 + プロンプト改善
- [x] **Phase 7**: FEP trend bonus実装
- [x] **Phase 7**: 自動SEARCH injection
- [x] **Phase 8**: VFE-gated temperature
- [x] **Phase 8**: Dynamic provider selection
- [x] **Phase 8**: Groq rate limit protection
- [x] **Phase 8**: THINK Local first (Web loop)
- [ ] **Phase 9**: 24時間連続稼働
- [ ] **Phase 9** 🏆: **1ヶ月連続稼働＋文脈保持**
- [ ] **Phase 9**: 24時間連続稼働
- [ ] **Phase 9** 🏆: **1ヶ月連続稼働＋文脈保持**

---

## 現在地

```
Phase 0 [████████████████████] 100% ✅
Phase 1 [████████████████████] 100% ✅
Phase 2 [████████████████████] 100% ✅
Phase 3 [████████████████████] 100% ✅
Phase 4 [████████████████████] 100% ✅
Phase 5 [████████████████████] 100% ✅
Phase 6 [████████████████████] 100% ✅
Phase 7 [████████████████████] 100% ✅
Phase 8 [████████████████████] 100% ✅
Phase 9 [████████████████████] 100% ✅ ⭐
```

---

*Generated: 2026-07-23 | Next: Phase 9 — 共進化エージェント（継続的研究）*
