# LLM
LLM_CONFIG = {
    "model": "gemma4:e4b",
    "temperature": 0.7,
    "max_tokens": 1024,
    "api_key": "",
    "base_url": "http://localhost:11434/v1",
    "num_ctx": 2048,
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
}

# 駆動
DRIVE_CONFIG = {
    "base_values": {
        "exploration": 0.5,
        "social": 0.3,
        "achievement": 0.4,
        "rest": 0.2,
        "maintenance": 0.2,
    },
    "decay_rate": 0.01,
    "learning_rate": 0.1,
    "min_baseline": 0.15,
    "boredom_threshold": 5,
    "boredom_boost": 0.2,
}

# 長期計画
LONG_TERM_CONFIG = {
    "review_interval_hours": 24,
    "routine_check_interval_minutes": 60,
    "max_goals": 5,
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
}
