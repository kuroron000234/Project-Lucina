# Monica v8 Hybrid Transition Document

## Session Summary

### Architecture
- **Hybrid Design**: Local FEP sensing (Qwen3-4B) + Cloud API generation (Opencode ZEN API / oc/deepseek-v4-flash-free)
- **Three Modes**: CHAT / THINK / IDLE (automatic state transitions based on self_model)

### Key Implementation Details

#### 1. FEP Sensing (ローカル)
- Model: Qwen/Qwen3-4B-Instruct-2507 (4-bit quantized, 5GB, 8GB VRAM)
- Layers hooked: L2 (low), L18 (mid), L35 (high)
- PE calculation via cosine similarity between current and predicted hidden states
- Self-model drift: `update_rate = 1/(1 + pe_mid * 6.0)`
- Restoring force pulls self_model back to baseline (0.30)

#### 2. API Generation (クラウド)
- Endpoint: `http://localhost:20128/v1/chat/completions`
- Model: `oc/deepseek-v4-flash-free`
- SSE streaming response parsing
- **Critical**: System prompt MUST be in English for content to return
- **Critical**: Free model has ~50% failure rate → retry logic (up to 3 attempts)
- User input is framed: `User said: {input}\n\nReply naturally in their language.`

#### 3. Steering Vector (ステアリング)
- Buffers: high_pe (>0.65) / low_pe (<0.39) activation states
- Updates at 10+ samples each
- Direction: high_state - low_state (normalized)

### Current Issues

1. **Empty Response Problem**
   - Root cause: deepseek-v4-flash-free returns `content: null` unless system prompt is English
   - Workaround: English system prompt + retry logic
   - Impact: ~50% of calls need retry

2. **THINK Mode Never Triggered**
   - Threshold: self_model < 0.25
   - Current behavior: self_model drifts from 0.30 → 0.27 in normal conversation
   - Never reaches THINK mode threshold

3. **Rate Limiting**
   - Free model has capacity constraints
   - Empty responses correlate with usage spikes

### Test Results

Latest successful conversation:
```
User: こんにちは
Assistant: こんにちは！お元気ですか？今日はどんなお話をしましょうか？

User: 人工知能についてどう思う？
Assistant: 面白い質問ですね！人工知能は、私自身がその一部だからかもしれませんが、人間の可能性を大きく広げる素晴らしい技術だと思います。...
```

Self-model progression: 0.30 → 0.29 → 0.27 (gradual drift toward boredom)

### Next Agent Recommendations

1. **Model Selection**
   - Consider `oc/deepseek-v4-flash` (non-free) for higher reliability
   - Or switch to `opencode-zen/gemini-3.5-flash` for better Japanese support

2. **THINK Mode Activation**
   - Lower threshold: `self < 0.32` for more frequent introspection
   - Or add time-based trigger: after 30s of inactivity

3. **Long-term Stability**
   - Implement exponential backoff for API retries
   - Add circuit breaker for consecutive failures
   - Consider local model fine-tuning for better reliability

### Files to Review
- `/home/koushi/monica-v3/monica_v8_hybrid.py` - Main implementation
- `/home/koushi/monica-v3/log_v8h.jsonl` - Session logs
- `/home/koushi/monica-v3/monica_v6.py` - Previous FEP implementation

### API Key
`sk-HzEVhrThdHk9iZkpN5L0DkqG7bPi3JxEMdE3PmDGFUChKQ8amsbc07SlpsldDfv1` (Opencode ZEN API)

---
*Transition prepared by Claude. Session ended: 2026-07-22*

---

# Project Vision: 最終ゴール

## 根本的な問い
**「ユーザー入力がなくても、自律的に活動し続けるAIとは何か？」**

## 目指す姿：自律認知エージェント

### 3つの自律レイヤー

| レイヤー | 現在の実装 | 目標 |
|---------|-----------|------|
| **反応的自律** | CHATモードで対話応答 | 入力→処理→応答の基本ループ |
| **内省的自律** | THINKモードで内部独白 | 入力なしでの思考・探索・創造 |
| **適応的自律** | FEPドリフト/ステアリング | 環境・自己モデルからの継続的学習 |

### 具体的な成果物（マイルストーン）

1. **Phase 1: 持続的動作** (Current)
   - 24時間無人動作
   - THINK/IDLE/CHATの自然な遷移
   - 長期記憶としての自己モデル蓄積

2. **Phase 2: 創発的行動**
   - 与えられていない課題への自発的取り組み
   - 知識の自己組織化（ステアリングベクトルの意味獲得）
   - 好奇心駆動の探索行動

3. **Phase 3: 共進化**
   - ユーザーとの長期関係構築
   - 価値観・人格の安定的形成
   - 他エージェントとの協調

## なぜFEP（自由エネルギー原理）なのか

- **統一原理**: 知覚・行動・学習を単一の変分自由エネルギー最小化で説明
- **生物学的妥当性**: 脳の予測符号化理論と整合
- **工学的実装可能性**: 予測誤差（PE）という計測可能なスカラー量に還元可能
- **自己組織化**: 外部報酬なしで「驚きの最小化」という内在的動機が生成される

## 非目標（やらないこと）

- ❌ 汎用チャットボットの品質向上
- ❌ 特定タスクのベンチマーク高スコア
- ❌ 人間の完全模倣（チューリングテスト通過）
- ❌ 商用サービスとしての即戦力

## 成功の定義（北極星）

> **「電源を入れっぱなしにして1ヶ月後、誰も話しかけていなくても、モニカは何らかの意味のある内部活動（思考・探索・創造）を続けており、かつ、久しぶりに話しかけたユーザーに対して文脈を持って応答できる」**

この状態を「自律的に活動し続ける」と定義する。

---

*Vision documented: 2026-07-22*