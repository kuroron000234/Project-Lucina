"""Project Lucina-Next — 常時稼働型の自律思考ループ。

内部Drive状態（boredom / loneliness / fatigue）がロジット分布に連続的に
影響することで、AIエージェントが外部キックなしに自発的に行動を選び取る。
仕様書: Lucina-nna_実装者向け仕様書_v1.4.md
"""

__version__ = "0.1.0"

from .config import load_config  # noqa: F401

__all__ = ["__version__", "load_config"]
