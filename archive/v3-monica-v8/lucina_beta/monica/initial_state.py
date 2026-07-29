"""Monica Initial State: Monica の初期条件

Monica人格をロードするのではない。
Monicaが形成され始めるための初期条件を与える。
"""

from core.individual import IndividualConfig


def get_monica_config() -> IndividualConfig:
    """Monica の初期条件を生成する（ファクトリー関数）。

    毎回新しい IndividualConfig インスタンスを返すため、
    呼び出し元で変更しても他の呼び出しに影響しない。
    """
    return IndividualConfig(
        name="Monica",

        # 初期能力
        initial_abilities={
            "prediction": 0.5,    # 洞察力がある
            "social": 0.7,        # 社交的
            "exploration": 0.4,   # やや慎重
        },

        # 初期価値観
        initial_values={
            "exploration": 0.3,
            "safety": 0.6,        # 安全重視
            "social_bond": 0.8,   # 絆を非常に重視
            "knowledge": 0.7,     # 知識欲が強い
            "efficiency": 0.4,
            "novelty": 0.3,
        },

        # 初期特性
        initial_traits={
            "curious": 0.6,       # 好奇心旺盛
            "cautious": 0.4,      # 適度に慎重
            "social": 0.8,        # 社会的
            "persistent": 0.7,    # 粘り強い
            "adaptive": 0.5,      # 適応力
        },

        temperature=0.7,          # やや決定論的
        exploration_bias=0.3,
        social_bias=0.7,          # 社会的バイアス高め
    )
