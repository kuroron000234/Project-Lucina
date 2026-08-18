"""pytest 共通フィクスチャ（仕様書 v1.4 §7 共通fixture仕様 D）。

lucina_core_fixture:
    実モデル不要で高速に回す。意図した語彙へ確率を寄せるダミーlogits生成器
    （MockTokenBackend）を InferenceEngine に注入した LucinaCore を返す。
    校正テスト（test_attractor_survival / test_interrupt_latency）が使用する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest_asyncio

# src と scripts を import パスに追加（editable install 無しでも動作）
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lucina.testing import build_mock_core, make_test_config  # noqa: E402


@pytest_asyncio.fixture
async def lucina_core_fixture(tmp_path):
    """校正テスト用コア（token_delay=0ms: 高速）。close() で logger と executor を閉じる。"""
    config = make_test_config(log_dir=str(tmp_path / "logs"))
    core = build_mock_core(config, token_delay_ms=0.0, log_dir=str(tmp_path / "logs"))
    yield core
    core.close()


@pytest_asyncio.fixture
async def slow_core_fixture(tmp_path):
    """割り込みレイテンシ計測用コア（token_delay=5ms: タイミングが安定）。"""
    config = make_test_config(log_dir=str(tmp_path / "logs"))
    core = build_mock_core(config, token_delay_ms=5.0, log_dir=str(tmp_path / "logs"))
    yield core
    core.close()
