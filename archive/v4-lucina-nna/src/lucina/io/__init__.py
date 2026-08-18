"""I/O 系: 外部刺激の注入口（interrupts）と構造化ログ（logging）。"""

from .interrupts import InterruptChannel  # noqa: F401
from .logging import StructuredLogger, setup_console_logging  # noqa: F401

__all__ = ["InterruptChannel", "StructuredLogger", "setup_console_logging"]
