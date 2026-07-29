"""
駆動層 (Drive)

責務: 生物的な欲求（探索・休息・社会・達成）の優先度を生成する。
FEPにおける「予測誤差」に相当する信号を出力する。

Phase 1: ルールベース + 環境・記憶からの減衰・ブースト
"""

import logging
import math
import random
import time as time_module

import config
from core.drive.interface import (
    DRIVE_DEFINITIONS,
    DriveInput,
    DriveOutput,
)

logger = logging.getLogger("Drive")


class Drive:
    """
    駆動層: 環境状態と記憶要約から5つの基本駆動の強度を計算する。

    エッジケース:
    - 全駆動が低い: デフォルトで exploration を primary に
    - 駆動が拮抗: ランダム要素を加えてバランスを崩す
    - 外部から強制駆動: adjustments で特定駆動を強制上昇
    """

    def __init__(self):
        base_values = config.DRIVE_CONFIG["base_values"]
        self.min_baseline = config.DRIVE_CONFIG["min_baseline"]
        self.boredom_threshold = config.DRIVE_CONFIG["boredom_threshold"]
        self.boredom_boost = config.DRIVE_CONFIG["boredom_boost"]
        self.params: dict[str, dict] = {}
        for name in DRIVE_DEFINITIONS:
            self.params[name] = {
                "base": max(base_values.get(name, 0.3), self.min_baseline),
                "decay_per_hour": config.DRIVE_CONFIG["decay_rate"],
                "boost": 0.0,
            }
        self.learning_rate = config.DRIVE_CONFIG["learning_rate"]
        self._last_update = time_module.time()
        self._stagnation_counter = 0

    def generate(self, input: DriveInput) -> DriveOutput:
        """
        現在の内部・外部状態から駆動状態を生成する。

        各駆動は以下の影響を受ける:
        - ベース値: 初期設定
        - ブースト値: 学習層からの調整
        - 環境要因: CPU/メモリ負荷、時間帯、ユーザー入力
        - 記憶要因: 最近のエピソード内容
        """
        drives: dict[str, float] = {}

        for name, param in self.params.items():
            val = param["base"] + param["boost"]

            # 環境要因の適用
            val = self._apply_environment_factors(name, val, input.environment)

            # 記憶要因の適用
            val = self._apply_memory_factors(name, val, input.memory_summary)

            # 外部調整値の適用
            if input.adjustments and name in input.adjustments:
                val += input.adjustments[name]

            # 0.0〜1.0 にクリッピング
            val = max(0.0, min(1.0, val))
            drives[name] = val

        # 環境変化に応じて停滞カウンタを更新
        self._update_stagnation(input.environment)

        # 停滞（退屈）ブーストを適用
        self._apply_boredom_boost()

        # 経過時間による自然減衰（min_baseline を下限に）
        self._apply_decay()

        # プライマリ駆動の決定
        primary = self._select_primary(drives)

        # 駆動テンション（標準偏差）
        drive_values = list(drives.values())
        mean = sum(drive_values) / len(drive_values)
        variance = sum((v - mean) ** 2 for v in drive_values) / len(drive_values)
        drive_tension = math.sqrt(variance)

        # 新奇性スコア
        novelty_score = self._compute_novelty(input.environment)

        return DriveOutput(
            drives=drives,
            primary_drive=primary,
            drive_tension=drive_tension,
            novelty_score=novelty_score,
        )

    def update_parameters(self, adjustments: dict[str, float]):
        """
        学習層からのフィードバックで駆動パラメータを調整する。

        adjustments = {"exploration": 0.1, "rest": -0.05, ...}
        正の値 = その駆動を強化、負の値 = 抑制
        """
        for drive_name, delta in adjustments.items():
            if drive_name in self.params:
                # 最大調整量でクリッピング
                clipped_delta = max(-0.2, min(0.2, delta))
                self.params[drive_name]["base"] = max(
                    0.0, min(1.0, self.params[drive_name]["base"] + clipped_delta * self.learning_rate)
                )
                logger.debug(
                    f"Drive parameter adjusted: {drive_name} -> {self.params[drive_name]['base']:.3f}"
                )

    def get_drive_profile(self) -> dict:
        """現在の駆動プロファイルを返す（デバッグ用）。"""
        return {
            name: {
                "base": param["base"],
                "boost": param["boost"],
                "decay_per_hour": param["decay_per_hour"],
                "effective": param["base"] + param["boost"],
            }
            for name, param in self.params.items()
        }

    def _apply_environment_factors(self, drive_name: str, val: float, env) -> float:
        """環境要因による駆動値の調整。"""
        # CPU負荷が高い → 休息欲求上昇
        if drive_name == "rest":
            cpu = env.system_state.cpu_percent
            val += cpu / 100.0 * 0.3

        # メモリ使用率が高い → メンテナンス欲求上昇
        if drive_name == "maintenance":
            mem = env.system_state.memory_percent
            val += mem / 100.0 * 0.2

        # ユーザー入力がある → 社会欲求上昇
        if drive_name == "social" and env.user_input:
            val += 0.2

        # ファイル数が多い → 探索欲求上昇
        if drive_name == "exploration":
            files_count = len(env.files) if env.files else 0
            val += min(files_count / 100.0, 0.2)

        return val

    def _apply_memory_factors(self, drive_name: str, val: float, memory_summary: str) -> float:
        """記憶要因による駆動値の調整。"""
        if not memory_summary or memory_summary == "まだ記憶がありません":
            # 記憶がない → 探索欲求上昇
            if drive_name == "exploration":
                val += 0.1
            return val

        summary_lower = memory_summary.lower()

        # エラーに関する記憶が多い → 休息欲求上昇
        if drive_name == "rest" and "error" in summary_lower:
            val += 0.15

        # 達成に関する記憶 → さらなる達成欲求
        if drive_name == "achievement" and ("成功" in summary_lower or "完了" in summary_lower):
            val += 0.1

        # 社会的な記憶 → 社会欲求が満たされる
        if drive_name == "social" and "会話" in summary_lower:
            val -= 0.1  # 満たされたので減少

        return val

    def _update_stagnation(self, env) -> None:
        """
        環境の変化を検出して停滞カウンタを更新する。
        変化があればリセット、なければインクリメント。
        """
        fingerprint = (
            env.system_state.cpu_percent // 10,
            env.system_state.memory_percent // 10,
            env.user_input is not None,
            len(env.files) if env.files else 0,
        )
        if not hasattr(self, '_prev_fingerprint'):
            self._prev_fingerprint = fingerprint
            self._stagnation_counter = 0
        elif self._prev_fingerprint != fingerprint:
            self._prev_fingerprint = fingerprint
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

    def _apply_boredom_boost(self) -> None:
        """
        停滞（同じ状態が続く）が閾値を超えたら探索欲求をブーストする。
        「退屈」を模倣し、単調なループから抜け出す動機を与える。
        """
        if self._stagnation_counter >= self.boredom_threshold:
            boost = min(
                self.boredom_boost * (self._stagnation_counter - self.boredom_threshold + 1),
                0.5,
            )
            if "exploration" in self.params:
                self.params["exploration"]["boost"] += boost
                logger.debug(f"Boredom boost applied: exploration += {boost:.2f}")

    def _apply_decay(self):
        """
        全駆動のベース値を時間経過に基づいて自然減衰させる。
        呼び出し回数ではなく、実際の経過時間に比例して減衰。
        下限は min_baseline でクリップ。
        """
        now = time_module.time()
        elapsed_hours = (now - self._last_update) / 3600.0
        self._last_update = now

        for name, param in self.params.items():
            decay_amount = param["decay_per_hour"] * elapsed_hours
            param["base"] = max(self.min_baseline, param["base"] - decay_amount)

    def _select_primary(self, drives: dict[str, float]) -> str:
        """
        最も強い駆動を選択する。

        駆動が拮抗している場合（差が0.1未満）はランダム要素を加える。
        全駆動が低い場合はデフォルトで exploration。
        """
        max_val = max(drives.values())
        candidates = [k for k, v in drives.items() if v >= max_val - 0.1]

        if max_val < 0.2:
            return "exploration"

        if len(candidates) > 1:
            # 拮抗時はランダム
            return random.choice(candidates)

        return candidates[0]

    def _compute_novelty(self, env) -> float:
        """
        環境の新奇性スコアを計算する。
        新しいファイル、高いCPU変動、未知のユーザー入力など。
        """
        novelty = 0.0

        # ユーザー入力があれば新奇性上昇
        if env.user_input:
            novelty += 0.3

        # CPU変動が大きいと新奇性上昇
        cpu = env.system_state.cpu_percent
        if cpu > 80:
            novelty += 0.2
        elif cpu > 50:
            novelty += 0.1

        # ファイル数が多いと新奇性上昇
        files_count = len(env.files) if env.files else 0
        novelty += min(files_count / 200.0, 0.2)

        return min(novelty, 1.0)
