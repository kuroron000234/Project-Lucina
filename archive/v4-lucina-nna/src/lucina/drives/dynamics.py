"""DriveDynamics — 結合行列 A による Drive 力学系（仕様書 v1.4 §5.1）。

更新式（B1）:
    x_next = clip(x + dt * (A * x) - relief_delta, 0.0, 1.0)
    A[i][j] は「Drive j が Drive i を加速する係数」（行=対象、列=源）。

契約:
    - 戻り値の各Drive値は必ず [0.0, 1.0] にクリップされていること。
    - 対角成分が全て正のため relief がなければ全Driveは1.0へ飽和する。
      この「飽和→reliefによる解消」サイクルが行動変動の源泉であり意図された挙動（C3）。
"""

from __future__ import annotations

from typing import Any


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


class DriveDynamics:
    def __init__(self, matrix_config: dict, initial_state: dict[str, float]):
        """matrix_config: {drive: {source_drive: coef}}。Aの非対角成分は根拠のあるもののみ。"""
        # A[i][j]: 行=対象Drive i、列=源Drive j
        self.matrix: dict[str, dict[str, float]] = {
            row: {col: float(coef) for col, coef in row_cfg.items()}
            for row, row_cfg in matrix_config.items()
        }
        self.drives: list[str] = sorted(set(initial_state) | set(matrix_config))
        # 起動時に初期状態をクリップ
        self.state: dict[str, float] = {k: _clip(float(initial_state.get(k, 0.0))) for k in self.drives}

    def step(self, dt: float, relief: dict[str, float] | None = None) -> dict[str, float]:
        """1ステップ分Drive状態を更新して返す。副作用として self.state も更新する（in-place）。

        relief: {drive: 解消量}（decay.py の ReliefController が計算した relief_delta）
        """
        dt = max(0.0, float(dt))
        relief = relief or {}
        for i in self.drives:
            growth = sum(
                self.matrix.get(i, {}).get(j, 0.0) * self.state.get(j, 0.0)
                for j in self.drives
            )
            x = self.state.get(i, 0.0) + dt * growth - float(relief.get(i, 0.0))
            self.state[i] = _clip(x)
        return dict(self.state)

    def set_state(self, updates: dict[str, float]) -> None:
        """テスト・校正実験用: Drive値を直接設定する（クリップ付き）。"""
        for k, v in updates.items():
            self.state[k] = _clip(float(v))
