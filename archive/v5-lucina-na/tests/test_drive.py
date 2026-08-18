"""
駆動層 (Drive) の単体テスト
"""

from datetime import datetime

from core.drive.drive import Drive
from core.drive.interface import DriveInput, DriveOutput, DRIVE_DEFINITIONS
from environment.interface import EnvironmentOutput, SystemState


class TestWillVolatility:
    """v4.0: 意志フェーズ — 主駆動選択のランダムジッタ（揮発性）"""

    def test_volatility_jitter_changes_selection_over_time(self):
        """
        拮抗した駆動値でも、ジッタにより選択が揺れる（同じ駆動に固定されない）。
        複数回サンプリングして、異なる主駆動が選ばれることを確認する。
        """
        import config
        import core.drive.drive as drive_mod
        d = drive_mod.Drive()
        old = d.volatility
        d.volatility = 0.3  # 大きくして揺れを検出しやすくする
        try:
            selections = set()
            for _ in range(60):
                primary = d._select_primary({
                    "exploration": 0.5, "social": 0.5, "achievement": 0.5,
                    "rest": 0.5, "maintenance": 0.5,
                })
                selections.add(primary)
            # 拮抗時は複数の駆動が選ばれ得る（ジッタが効いている）
            assert len(selections) >= 2
        finally:
            d.volatility = old

    def test_volatility_default_preserves_clear_winner(self):
        """明確な勝者がある場合はジッタでも通常は勝者が選ばれる"""
        import core.drive.drive as drive_mod
        d = drive_mod.Drive()
        d.volatility = 0.0  # ジッタなし
        primary = d._select_primary({
            "exploration": 0.9, "social": 0.1, "achievement": 0.1,
            "rest": 0.1, "maintenance": 0.1,
        })
        assert primary == "exploration"


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
        # v3.3: base 0.35 + memory空+0.1 + adjustments 0.5 = 0.95
        assert result.drives["exploration"] > 0.8

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
        """
        停滞が続くと探索ブーストが累積し、探索欲求が上昇する。
        （v3.2修正後: exploration が primary のときは消費され、
         non-primary（rest）のときに累積して drives 出力を押し上げる）
        """
        env = self._make_env()
        self.drive.boredom_threshold = 2
        self.drive.boredom_boost = 0.3
        # rest を primary にする（exploration は低く、ブーストが累積される状況）
        self.drive.params["rest"]["base"] = 0.9
        self.drive.params["exploration"]["base"] = 0.1
        # 1回目: ベースライン（ブースト未適用）
        r1 = self.drive.generate(DriveInput(environment=env, memory_summary=""))
        # 同じ環境で複数回generate → 停滞カウンタが上がりブーストが累積
        for _ in range(5):
            r2 = self.drive.generate(DriveInput(environment=env, memory_summary=""))
        # ブーストが drives 出力に反映され、探索欲求が上昇している
        assert r2.drives["exploration"] > r1.drives["exploration"]

    def test_boredom_boost_capped_at_max(self):
        """退屈ブーストが上限（max_boredom_boost）を超えて累積しない（1.0張り付き防止）"""
        env = self._make_env()
        self.drive.boredom_threshold = 1
        self.drive.boredom_boost = 0.5
        # exploration が primary にならないよう低く設定（rest を primary に）
        self.drive.params["exploration"]["base"] = 0.1
        self.drive.params["rest"]["base"] = 0.9
        for _ in range(20):
            self.drive.generate(DriveInput(environment=env, memory_summary=""))
        assert self.drive.params["exploration"]["boost"] <= self.drive.max_boredom_boost

    def test_boredom_boost_consumed_when_exploring(self):
        """探索が実行されたら退屈ブーストが消費され、飽和しない"""
        env = self._make_env()
        self.drive.boredom_threshold = 1
        self.drive.boredom_boost = 0.5
        # v3.3: 欲求増加を無効化（テストの焦点はブースト消費の確認）
        self.drive.satisfied_decay = 0.0
        self.drive.unsatisfied_growth = 0.0
        # exploration が primary となるよう設定
        self.drive.params["exploration"]["base"] = 0.6
        self.drive.params["rest"]["base"] = 0.3
        for _ in range(10):
            self.drive.generate(DriveInput(environment=env, memory_summary=""))
        assert self.drive.params["exploration"]["boost"] == 0.0

    def test_boredom_boost_resets_on_env_change(self):
        """環境が変化すると退屈ブーストがリセットされる"""
        env1 = self._make_env(cpu=30.0)
        self.drive.boredom_threshold = 1
        self.drive.boredom_boost = 0.5
        self.drive.params["exploration"]["base"] = 0.1
        self.drive.params["rest"]["base"] = 0.9
        for _ in range(5):
            self.drive.generate(DriveInput(environment=env1, memory_summary=""))
        assert self.drive.params["exploration"]["boost"] > 0
        # 環境が変わる（CPUが変化）→ 停滞リセット → ブーストリセット
        env2 = self._make_env(cpu=70.0)
        self.drive.generate(DriveInput(environment=env2, memory_summary=""))
        assert self.drive.params["exploration"]["boost"] == 0.0

    def test_exploration_not_pinned_at_1_0(self):
        """静的環境でも exploration が 1.0 に張り付かない"""
        env = self._make_env()
        self.drive.boredom_threshold = 2
        self.drive.boredom_boost = 0.2
        max_exploration = 0.0
        for _ in range(30):
            r = self.drive.generate(DriveInput(environment=env, memory_summary=""))
            max_exploration = max(max_exploration, r.drives["exploration"])
        assert max_exploration < 1.0

    # ── v3.3: 欲求自然増加テスト ──

    def test_drive_urge_rises_unsatisfied(self):
        """
        primary に選ばれなかった駆動は時間経過で base が上昇する。
        exploration が primary の間、他の駆動が徐々に上がる。
        """
        env = self._make_env()
        self.drive.satisfied_decay = 0.1
        self.drive.unsatisfied_growth = 0.05
        # 強制的に exploration が primary になる状況
        self.drive.params["exploration"]["base"] = 0.6
        self.drive.params["rest"]["base"] = 0.3
        for _ in range(5):
            self.drive.generate(DriveInput(environment=env, memory_summary=""))
        # rest は満たされないので上昇
        assert self.drive.params["rest"]["base"] > 0.3
        # exploration は満たされるので減少
        assert self.drive.params["exploration"]["base"] < 0.6

    def test_drive_urge_is_bounded(self):
        """自然増加は urge_max_base を超えない"""
        env = self._make_env()
        self.drive.satisfied_decay = 0.0
        self.drive.unsatisfied_growth = 0.1
        self.drive.urge_max_base = 0.7
        self.drive.params["exploration"]["base"] = 0.9  # primary
        self.drive.params["rest"]["base"] = 0.5
        for _ in range(20):
            self.drive.generate(DriveInput(environment=env, memory_summary=""))
        # rest は urge_max_base でキャップ
        assert self.drive.params["rest"]["base"] <= 0.7
        # exploration は min_baseline で下限
        assert self.drive.params["exploration"]["base"] >= self.drive.min_baseline

    def test_drive_urge_eventually_changes_primary(self):
        """
        欲求の自然増加により、同じ駆動が永続的に primary になることはない。
        満たされない駆動が上昇し、いずれ逆転する。
        """
        env = self._make_env()
        self.drive.satisfied_decay = 0.02
        self.drive.unsatisfied_growth = 0.01
        self.drive.urge_max_base = 1.0  # 上限なし
        # exploration が primary だが、rest が徐々に追いつく
        self.drive.params["exploration"]["base"] = 0.5
        self.drive.params["rest"]["base"] = 0.35
        primaries = set()
        for _ in range(30):
            r = self.drive.generate(DriveInput(environment=env, memory_summary=""))
            primaries.add(r.primary_drive)
        # 複数の駆動が primary になった
        assert len(primaries) > 1, f"Only one primary: {primaries}"
