"""relief（解消）ロジック（仕様書 v1.4 §5.1・§4 drive.relief）。

- decay_rate: 毎ステップの緩やかな減衰（比例減衰: relief += decay_rate * state * dt）
- per_action: 発話セグメント単位で最大1回適用（①）。セグメント完了時に
  apply_per_action() でキューされ、次のステップで1回だけ消費される。
"""

from __future__ import annotations

from typing import Any


class ReliefController:
    def __init__(self, relief_config: dict[str, Any]):
        """relief_config: {drive: {enabled, per_action, decay_rate}}

        設定には drive.relief 配下の運用キー（unit / segment）も混在するため、
        per_action を持つ Drive エントリのみを対象とする。
        """
        self._config: dict[str, dict[str, Any]] = {
            drive: dict(cfg)
            for drive, cfg in relief_config.items()
            if isinstance(cfg, dict) and "per_action" in cfg
        }
        self._pending: dict[str, float] = {}

    def apply_per_action(self, drive: str) -> None:
        """セグメント完了時に呼ぶ。該当Driveの per_action 解消量を1回分キューする。"""
        cfg = self._config.get(drive)
        if cfg is None or not cfg.get("enabled", True):
            return
        self._pending[drive] = self._pending.get(drive, 0.0) + float(cfg["per_action"])

    def apply_amount(self, drive: str, amount: float) -> None:
        """任意の解消量をキューする（v1.10・②部分relief用）。

        セグメントの語彙一致で「話すだけ」の部分 relief（speak_relief）を適用し、
        応答（外部入力）でのフル relief（per_action）と区別するために使う。
        """
        cfg = self._config.get(drive)
        if cfg is None or not cfg.get("enabled", True):
            return
        amount = max(0.0, float(amount))
        if amount > 0.0:
            self._pending[drive] = self._pending.get(drive, 0.0) + amount

    def step(self, dt: float, states: dict[str, float]) -> dict[str, float]:
        """1ステップ分の relief_delta を返す（decay + 消費される per_action）。"""
        out: dict[str, float] = {}
        for drive, cfg in self._config.items():
            if not cfg.get("enabled", True):
                continue
            delta = 0.0
            decay = float(cfg.get("decay_rate", 0.0))
            if decay > 0.0:
                delta += decay * float(states.get(drive, 0.0)) * max(0.0, float(dt))
            delta += self._pending.pop(drive, 0.0)
            if delta > 0.0:
                out[drive] = delta
        return out

    def per_action_of(self, drive: str) -> float:
        """該当Driveの per_action 解消量を返す（v1.10・外部イベント駆動の即時 relief 用）。

        セグメント単位（pending キュー）ではなく、外部応答のように即時反映が必要な
        イベントで使う。無効・未設定なら 0.0。
        """
        cfg = self._config.get(drive)
        if cfg is None or not cfg.get("enabled", True):
            return 0.0
        return float(cfg.get("per_action", 0.0))

    @property
    def pending(self) -> dict[str, float]:
        return dict(self._pending)
