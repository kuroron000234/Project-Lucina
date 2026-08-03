"""ベンチマーク共通ヘルパー（フェイク環境・レポート保存）。"""

import json
import os
from datetime import datetime

import config
from environment.interface import EnvironmentOutput, NetworkState, SystemState


def make_env(cpu: float = 30.0, memory: float = 50.0,
             user_input: str | None = None, files: int = 3) -> EnvironmentOutput:
    """決定論的なフェイク環境を構築する（実環境・実LLMを使わない）。

    drive 層は files の len しか参照しないため、ダミー要素で十分。
    """
    return EnvironmentOutput(
        timestamp=datetime.now(),
        user_input=user_input,
        system_state=SystemState(
            cpu_percent=cpu,
            memory_percent=memory,
            active_window=None,
            uptime=3600.0,
            current_directory="/home/koushi/lucina-NA",
        ),
        files=[None] * files if files else [],
        network=NetworkState(is_connected=True, ip_address=None, signal_strength=None),
    )


def save_report(report, report_dir: str | None = None) -> str:
    """レポートを JSON として data/benchmarks/ に保存し、パスを返す。"""
    report_dir = report_dir or config.BENCHMARK_CONFIG.get(
        "report_dir", "data/benchmarks"
    )
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"{report.name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return path
