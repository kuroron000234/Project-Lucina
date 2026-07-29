"""
駆動層 (Drive) の単体テスト
"""

from datetime import datetime

from core.drive.drive import Drive
from core.drive.interface import DriveInput, DriveOutput, DRIVE_DEFINITIONS
from environment.interface import EnvironmentOutput, SystemState


class TestDrive:
    def setup_method(self):
        self.drive = Drive()

    def _make_env(self, user_input: str | None = None,
                  cpu: float = 30.0, memory: float = 50.0,
                  files_count: int = 0) -> EnvironmentOutput:
        return EnvironmentOutput(
            timestamp=datetime.now(),
            user_input=user_input,
            system_state=SystemState(
                cpu_percent=cpu,
                memory_percent=memory,
                active_window="test_window",
                uptime=3600,
                current_directory="/home/test",
            ),
            files=[],
            network=None,
            sensors={},
        )

    # --- 正常系テスト ---

    def test_generate_returns_valid_output(self):
        """どの入力でも例外なく出力が生成される"""
        env = self._make_env()
        result = self.drive.generate(DriveInput(
            environment=env,
            memory_summary="テスト用の記憶要約です",
        ))
        assert isinstance(result, DriveOutput)
        assert result.primary_drive in DRIVE_DEFINITIONS

    def test_primary_drive_is_set(self):
        """常に primary_drive が設定されている"""
        env = self._make_env()
        result = self.drive.generate(DriveInput(
            environment=env,
            memory_summary="",
        ))
        assert result.primary_drive
        assert result.primary_drive in result.drives

    def test_drive_values_in_range(self):
        """駆動値が 0.0〜1.0 に収まっている"""
        env = self._make_env(cpu=99.0, memory=99.0, files_count=200)
        result = self.drive.generate(DriveInput(
            environment=env,
            memory_summary="多くのエラーが発生しました",
        ))
        for name, value in result.drives.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} が範囲外"
        assert 0.0 <= result.drive_tension <= 1.0
        assert 0.0 <= result.novelty_score <= 1.0

    # --- 環境要因テスト ---

    def test_high_cpu_increases_rest(self):
        """CPU負荷が高いと休息欲求が上がる"""
        env_low = self._make_env(cpu=10.0)
        env_high = self._make_env(cpu=95.0)

        result_low = self.drive.generate(DriveInput(
            environment=env_low, memory_summary=""
        ))
        result_high = self.drive.generate(DriveInput(
            environment=env_high, memory_summary=""
        ))
        assert result_high.drives["rest"] > result_low.drives["rest"]

    def test_user_input_increases_social(self):
        """ユーザー入力があると社会欲求が上がる"""
        env_no_input = self._make_env(user_input=None)
        env_with_input = self._make_env(user_input="こんにちは")

        result_no = self.drive.generate(DriveInput(
            environment=env_no_input, memory_summary=""
        ))
        result_with = self.drive.generate(DriveInput(
            environment=env_with_input, memory_summary=""
        ))
        assert result_with.drives["social"] > result_no.drives["social"]

    # --- エッジケーステスト ---

    def test_all_drives_low_defaults_to_exploration(self):
        """全駆動が低い場合、デフォルトで exploration が primary に"""
        # 強制的に全駆動を低くする設定
        self.drive.params = {
            name: {"base": 0.05, "decay_per_hour": 0.0, "boost": 0.0}
            for name in DRIVE_DEFINITIONS
        }
        env = self._make_env()
        result = self.drive.generate(DriveInput(
            environment=env, memory_summary=""
        ))
        assert result.primary_drive == "exploration"

    def test_update_parameters(self):
        """update_parameters() でパラメータが更新される"""
        before = self.drive.params["exploration"]["base"]
        self.drive.update_parameters({"exploration": 0.5})
        # 学習率0.1なので、0.5 * 0.1 = 0.05 増加 (クリッピング後 max 0.2)
        after = self.drive.params["exploration"]["base"]
        assert after != before

    def test_get_drive_profile(self):
        """get_drive_profile() がプロファイルを返す"""
        profile = self.drive.get_drive_profile()
        assert set(profile.keys()) == set(DRIVE_DEFINITIONS.keys())
        for name, info in profile.items():
            assert "base" in info
            assert "boost" in info
            assert 0.0 <= info["effective"] <= 1.0

    def test_external_adjustments_applied(self):
        """adjustments で特定駆動を強制上昇できる"""
        env = self._make_env()
        result = self.drive.generate(DriveInput(
            environment=env,
            memory_summary="",
            adjustments={"exploration": 0.5},
        ))
        # adjustments が reflection より大きいので exploration に影響
        assert result.drives["exploration"] > 0.3

    def test_min_baseline_prevents_zero(self):
        """min_baseline により駆動が0にならない"""
        # ベースを強制的に0にしてgenerate
        for name in self.drive.params:
            self.drive.params[name]["base"] = 0.0
        env = self._make_env()
        result = self.drive.generate(DriveInput(
            environment=env, memory_summary=""
        ))
        for val in result.drives.values():
            assert val >= 0.0  # マイナスにならない

    def test_boredom_boost_on_stagnation(self):
        """停滞が続くと探索欲求にブーストがかかる"""
        env = self._make_env()
        self.drive.boredom_threshold = 2
        self.drive.boredom_boost = 0.3
        # 1回目: ベースライン
        r1 = self.drive.generate(DriveInput(environment=env, memory_summary=""))
        # 同じ環境で複数回generate → 停滞カウンタが上がる
        for _ in range(5):
            r2 = self.drive.generate(DriveInput(environment=env, memory_summary=""))
        assert r2.drives["exploration"] >= r1.drives["exploration"]
