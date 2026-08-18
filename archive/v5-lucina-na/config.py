# LLM
LLM_CONFIG = {
    "model": "gemma4:e4b",
    "temperature": 0.7,
    # v3.5.2: 1024 → 3072。長い返答が finish_reason=length で
    # 文途中に切断される問題を修正。日本語は 1トークン≈1.5文字のため
    # 2048 トークン ≈ 3100文字 で十分でなく、3072 トークン ≈ 4600文字 まで
    # 生成可能（num_ctx=8192 と併用、llm.py がプロンプト長に応じて
    # 動的クランプするため窓オーバーフローしない）
    "max_tokens": 3072,
    "api_key": "",
    "base_url": "http://localhost:11434/v1",
    # v3.5.1: 2048 → 4096。会話履歴追加でプロンプトが肥大化し、
    # 生成がコンテキスト窓で途中切断される問題を修正（RTX 4060 8GB で実測 OK）
    # v3.5.2: 4096 → 8192。max_tokens=3072 と併用できるよう窓を拡大。
    # プロンプト ~1285トークン + 生成3072トークン = 4357 < 8192 で余裕
    "num_ctx": 8192,
}

# メインループ
LOOP_CONFIG = {
    "phase": 1,
    "interval_seconds": 10,
    "max_iterations": 0,
    "async_learning": False,
    "forget_interval_iterations": 100,
    "max_episodes": 10000,
    "change_threshold": 0.05,
    "llm_skip_max_iterations": 10,
}

# 記憶
MEMORY_CONFIG = {
    "storage_path": "data/episodes/",
    "max_episodes": 10000,
    "search_top_k": 5,
    "auto_summarize_threshold": 100,
    "forget_threshold": 0.15,
    # v5.0: Phase 3 — ハイブリッド検索（キーワード + 文字n-gram類似度）
    # 記憶保持ベンチマークで実測された弱点（言い換え・表記揺れで Recall@k=0.0）を
    # 改善する。日本語は形態素解析なしで動く文字n-gramコサイン類似度を使用。
    "hybrid": {
        "enabled": True,
        "n_gram_size": 2,          # 文字バイグラム
        "similarity_weight": 1.0,  # 類似度スコアの重み（キーワードヒット=1.0に加算）
        "min_similarity": 0.25,    # 類似度のみヒットの最低閾値（cosine）
        "max_episodes_full_scan": 5000,  # 全走査する上限（超えたらキーワードのみ）
    },
}

# 人格（v3.4: 自己モデル — 自身の記憶等を参照して自己認識を保持・永続化）
PERSONALITY_CONFIG = {
    "state_path": "data/personality_state.json",
    "self_model_interval": 20,  # 自己モデル再生成の間隔（サイクル数）
}

# 駆動
DRIVE_CONFIG = {
    "base_values": {
        "exploration": 0.35,
        "social": 0.35,
        "achievement": 0.35,
        "rest": 0.35,
        "maintenance": 0.35,
    },
    "decay_rate": 0.01,
    "learning_rate": 0.1,
    "min_baseline": 0.15,
    "boredom_threshold": 5,
    "boredom_boost": 0.2,
    # v3.2 修正: 退屈ブーストの累積上限（飽和して exploration が 1.0 に張り付くのを防ぐ）
    "max_boredom_boost": 0.3,
    # v3.3: 駆動再設計 — 欲求の自然増加（満たされない駆動が時間経過で上昇）
    "drive_urge": {
        "satisfied_decay": 0.005,   # primary に選ばれた駆動が減少する量/cycle
        "unsatisfied_growth": 0.002, # 選ばれなかった駆動が増加する量/cycle
        "max_base": 0.8,             # 自然増加の上限（学習・ブーストは別枠）
    },
    # v3.3: 学習ゲートを緩和（自律行動の報酬分散が小さいため）
    "learning_gate": {
        "variance_threshold": 0.005,  # 報酬分散がこの値以上なら調整実行
        "min_history": 1,             # 最低履歴数
    },
}

# 長期計画
LONG_TERM_CONFIG = {
    "review_interval_hours": 24,
    "routine_check_interval_minutes": 60,
    "max_goals": 5,
}

# v4.0: 意志フェーズ (Will Phase)
# 「自分の意志でなんでもできる」SF感を出すための設定群。
# - 願望層: 人格がメニューから選ぶのをやめ、自分でやりたいことを生成する
# - 想像: 世界モデルが未来候補を生成し、好みとの一致度で選ぶ（アクティブ推論）
# - 内面: 内言 (inner_monologue) と日記で「なぜ」を可視化する
# - 能動発話: 達成時・思いつき時に自分からユーザーへ話しかける
# - 拒否: 休息欲求・不機嫌時に理由付きで先延ばしを提案できる
# - 揮発性: 主駆動選択にランダムジッタを加え、リセットしても同じ行動にならない
WILL_CONFIG = {
    "workspace_dir": "data/workspace",   # 自分の部屋（自由にファイルを作れる場所）
    "aspiration_count": 3,               # 保持する願望の数
    "aspiration_interval_hours": 6,      # 願望の再生成間隔（時間）
    "diary_dir": "data/diary",           # 日記の保存先
    "diary_hour": 22,                    # 日記を書く時間（22時以降）
    "proactive_cooldown_minutes": 60,    # 能動的発話のクールダウン（分）
    "proactive_probability": 0.3,        # 自律サイクル後の能動発話確率
    "volatility": 0.1,                   # 主駆動選択のランダムジッタ幅
    "imagination_count": 3,              # 世界モデルが想像する未来候補数
}

# 評価
EVALUATION_CONFIG = {
    "history_size": 100,
    "weights": {
        "goal_achievement": 0.4,
        "efficiency": 0.2,
        "correctness": 0.2,
        "novelty": 0.2,
    },
    # 評価履歴の永続化先（v3.2: 再起動しても学習がリセットされない）
    "storage_path": "data/evaluation_history.json",
}

# 学習ループ (v3.2: 単一パイプライン + コスト段階化)
LEARNING_CONFIG = {
    # コスト段階（tier）の閾値・間隔
    "tier": {
        "novelty_tier2": 0.25,   # 新奇性 >= 0.25 → tier2（ルールベース学習）
        "novelty_tier3": 0.35,   # 新奇性 >= 0.35 → tier3（LLM評価）
        "interval_tier2": 5,     # 5サイクル毎に tier2
        "interval_tier3": 20,    # 20サイクル毎に tier3
        "cooldown_tier3": 10,    # tier3 のクールダウン（10サイクル）
    },
    # 重要度の連続値計算（LLM不要）
    "importance": {
        "base": 0.15,
        "w_success": 0.30,
        "w_efficiency": 0.20,
        "w_correctness": 0.10,
        "w_novelty": 0.10,
        "dialog_bonus": 0.15,    # 対話エピソードのボーナス
        "rep_penalty": 0.08,     # 類似エピソード繰り返しの減点
        "rep_max": 3,            # 減点の累積上限
        "dialog_rule_squash": 0.5,  # 対話+ルール評価時の重要度圧縮係数
    },
    # 学習ゲート
    "variance_gate": 0.02,       # 報酬分散 < 0.02 なら駆動調整をスキップ
    "history_min_same_type": 3,  # 同一 eval_type の履歴が3件未満ならスキップ
    # 記憶検索キャッシュ
    "memory_cache_seconds": 60,
}

# v5.0: Phase 3 — サプライズ層（本物のFEPコンポーネント）
# 予測誤差（サプライズ）を実測し、駆動の新奇性・学習率・行動選択に反映する。
# 外部レビュー対応: FEPのうち数学的に実装するのはこの層のみ（他は inspired-by）。
SURPRISE_CONFIG = {
    "sigma_floor": 0.05,          # 分散 σ の下限（0除算防止）
    "default_sigma": 0.3,         # 予測が不確実性を持たない場合のデフォルト σ
    # 新奇性スコアへのサプライズ反映率（0.0=不使用）。
    # 注意: surprise=0.0（予測が正確）時もブレンド式 (1-blend)*heuristic + blend*s
    # が適用され、新奇性が (1-blend) 倍に減衰する（正確 = エピステミック価値低）。
    # このため tier2/3 判定（novelty_tier2=0.25 等）の頻度が低下し得る。
    "novelty_blend": 0.4,
    "learning_modulation": 1.5,   # 学習率の増幅係数（surprise=1.0 で +150%）
    "learning_modulation_cap": 2.0,  # 変調後の学習率上限（base の何倍まで）
}

# v5.0: Phase 3 — ベンチマーク（検証と本物化、M15-M17）
BENCHMARK_CONFIG = {
    "report_dir": "data/benchmarks",
}
