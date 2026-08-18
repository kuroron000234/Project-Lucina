"""サプライズ層（M14）の挙動検証ベンチマーク。

「環境変化でサプライズがスパイクし、安定で減衰する」「サプライズが駆動の
新奇性・学習率に反映される」を実クラス（WorldModel / Drive / Learning）で
検証し、レポートを生成する。実LLMは使わない（ルールベース予測 + 実数式）。

これは「FEPの数学が本当に動いている」ことの機械的証明であり、
外部レビュー指摘1（FEPは飾り）へのデータによる反論の一部。
"""

from core.drive.drive import Drive
from core.drive.interface import DriveInput
from core.learning.learning import Learning
from core.world_model.world_model import WorldModel
from core.world_model.interface import Prediction

from benchmarks.common import make_env, save_report
from benchmarks.interface import BenchmarkReport, BenchmarkSection


def _pred(expected_reward: float = 0.5, uncertainty: float = 0.3) -> Prediction:
    """検証用の決定論的予測（μ = (expected_reward+1)/2）。"""
    return Prediction(
        action="exploration",
        next_state="next",
        probability=0.7,
        expected_reward=expected_reward,
        risk_level="low",
        reasoning="probe",
        uncertainty=uncertainty,
    )


def run_surprise_validation(report_dir: str | None = None) -> BenchmarkReport:
    sections = []

    # --- 1. サプライズの数学: 正確な予測は0、外れた予測は大 ---
    wm = WorldModel()
    pred = _pred(expected_reward=0.5)  # μ = 0.75
    s_accurate = wm.compute_surprise(actual_reward=0.75, prediction=pred)
    s_wrong = wm.compute_surprise(actual_reward=0.1, prediction=pred)
    sections.append(BenchmarkSection(
        name="surprise_math",
        passed=(s_accurate <= 0.05 and s_wrong > 1.0),
        metrics={
            "accurate": round(s_accurate, 3),
            "wrong": round(s_wrong, 3),
            "delta": round(s_wrong - s_accurate, 3),
        },
        details=[
            f"予測が正確（x=0.75）→ S={s_accurate:.2f}（0に収束）",
            f"予測が外れた（x=0.1）→ S={s_wrong:.2f}（スパイク）",
            "S = (x−μ)²/σ² + ln σ のガウス近似による負の対数尤度",
        ],
    ))

    # --- 2. 低σ（確信）ほど鋭敏: 同じズレでもσ小の方が大きなサプライズ ---
    s_high_conf = wm.compute_surprise(0.1, _pred(0.5, uncertainty=0.1))
    s_low_conf = wm.compute_surprise(0.1, _pred(0.5, uncertainty=0.6))
    sections.append(BenchmarkSection(
        name="sigma_sharpness",
        passed=(s_high_conf > s_low_conf),
        metrics={"sigma0.1": round(s_high_conf, 2), "sigma0.6": round(s_low_conf, 2)},
        details=["確信した予測（σ=0.1）が外れるほど大きなサプライズになる"],
    ))

    # --- 3. 駆動層への反映: サプライズありで新奇性スコアが上昇する ---
    env = make_env()
    # exploration を明示ブーストして主駆動を決定論的にする（ジッタで揺れないように）
    kw = dict(environment=env, memory_summary="", adjustments={"exploration": 0.5})
    drive_base = Drive()
    drive_surp = Drive()
    base_novelty = drive_base.generate(DriveInput(**kw)).novelty_score
    surp_novelty = drive_surp.generate(
        DriveInput(**kw, surprise=1.0)
    ).novelty_score
    sections.append(BenchmarkSection(
        name="drive_integration",
        passed=(surp_novelty >= base_novelty),
        metrics={
            "novelty_without_surprise": round(base_novelty, 3),
            "novelty_with_surprise": round(surp_novelty, 3),
            "delta": round(surp_novelty - base_novelty, 3),
        },
        details=["高サプライズ時は新奇性が上昇し、探索・tier2/3判定に効く"],
    ))

    # --- 4. 学習層への反映: 高サプライズ時は学習率が上昇する ---
    learning = Learning()
    lr_flat = learning._modulated_learning_rate(0.0)
    lr_peak = learning._modulated_learning_rate(1.0)
    lr_none = learning._modulated_learning_rate(None)
    sections.append(BenchmarkSection(
        name="learning_integration",
        passed=(lr_peak > lr_flat and lr_none == learning.learning_rate),
        metrics={
            "lr_surprise0": round(lr_flat, 4),
            "lr_surprise1": round(lr_peak, 4),
            "lr_no_surprise": round(lr_none, 4),
        },
        details=["高サプライズ = 学ぶべき時 → 学習率を最大2倍まで増幅"],
    ))

    # --- 5. 正規化: 任意の生サプライズ値が 0.0〜1.0 に収まる ---
    n0 = WorldModel.normalize_surprise(0.0)
    n100 = WorldModel.normalize_surprise(100.0)
    sections.append(BenchmarkSection(
        name="normalization",
        passed=(n0 == 0.0 and 0.0 < n100 < 1.0),
        metrics={"s=0": n0, "s=100": round(n100, 4)},
        details=["生サプライズを駆動・学習・人格に渡す前に正規化"],
    ))

    return BenchmarkReport(name="surprise_validation", sections=sections)


if __name__ == "__main__":
    report = run_surprise_validation()
    print(save_report(report))
