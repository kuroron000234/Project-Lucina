"""
Phase 3 (M14): サプライズ層のテスト

- サプライズの数学: S = (x−μ)²/σ² + ln σ（正確な予測は0、外れは大）
- 駆動層: サプライズが新奇性スコアにブレンドされる
- 学習層: サプライズが学習率を変調する
- 人格層: プロンプトにサプライズ文脈が注入される
"""

from core.drive.drive import Drive
from core.drive.interface import DriveInput
from core.learning.learning import Learning
from core.memory.interface import MemoryOutput
from core.personality.personality import Personality
from core.personality.interface import PersonalityInput
from core.world_model.world_model import WorldModel
from core.world_model.interface import Prediction

from benchmarks.common import make_env


def _pred(expected_reward: float = 0.5, uncertainty: float = 0.3) -> Prediction:
    return Prediction(
        action="exploration",
        next_state="next",
        probability=0.7,
        expected_reward=expected_reward,
        risk_level="low",
        reasoning="probe",
        uncertainty=uncertainty,
    )


# --- サプライズの数学 ---


def test_accurate_prediction_yields_zero_surprise():
    """予測が正確（x == μ）ならサプライズは 0 に収束する。"""
    wm = WorldModel()
    p = _pred(expected_reward=0.5)  # μ = (0.5+1)/2 = 0.75
    assert wm.compute_surprise(actual_reward=0.75, prediction=p) == 0.0


def test_wrong_prediction_yields_large_surprise():
    """予測が外れるとサプライズが大きくなる（スパイク）。"""
    wm = WorldModel()
    p = _pred(expected_reward=0.5)
    s = wm.compute_surprise(actual_reward=0.1, prediction=p)
    assert s > 1.0


def test_low_sigma_sharpens_surprise():
    """同じズレでも確信（σ小）の方が大きなサプライズになる。"""
    wm = WorldModel()
    s_hi_conf = wm.compute_surprise(0.1, _pred(0.5, uncertainty=0.1))
    s_low_conf = wm.compute_surprise(0.1, _pred(0.5, uncertainty=0.6))
    assert s_hi_conf > s_low_conf


def test_surprise_respects_reward_scale():
    """期待報酬（-1..1）が 0..1 に変換されて比較される。"""
    wm = WorldModel()
    p_neg = _pred(expected_reward=-0.8)  # μ = 0.1
    s = wm.compute_surprise(actual_reward=0.1, prediction=p_neg)
    assert s == 0.0  # 正確な予測


def test_normalize_surprise_bounds():
    """正規化は 0.0〜1.0 に収まる単調写像。"""
    assert WorldModel.normalize_surprise(0.0) == 0.0
    assert 0.0 < WorldModel.normalize_surprise(100.0) < 1.0
    assert WorldModel.normalize_surprise(100.0) > WorldModel.normalize_surprise(1.0)


def test_prediction_default_uncertainty():
    """Prediction の不確実性はデフォルト 0.3。"""
    p = _pred()
    assert p.uncertainty == 0.3


def test_rule_based_predictions_carry_uncertainty():
    """ルールベース予測も不確実性を持つ（サプライズ計算に使用可能）。"""
    wm = WorldModel()
    from core.world_model.interface import WorldModelInput
    from core.drive.drive import Drive
    env = make_env()
    ds = Drive().generate(DriveInput(environment=env, memory_summary=""))
    out = wm.predict(WorldModelInput(
        environment=env, drive=ds, active_goal="", use_llm=False,
    ))
    assert out.predictions
    assert all(getattr(p, "uncertainty", None) is not None for p in out.predictions)


# --- 駆動層への反映 ---


def test_drive_novelty_blends_surprise():
    """サプライズが与えられると新奇性スコアが上昇する。"""
    env = make_env()
    # exploration を明示的にブーストして主駆動を決定論的にする
    # （プライマリ選択のランダムジッタで deficit_cycles が揺れないように）
    kw = dict(environment=env, memory_summary="", adjustments={"exploration": 0.5})
    drive_base = Drive()
    drive_surp = Drive()
    base = drive_base.generate(DriveInput(**kw)).novelty_score
    with_surp = drive_surp.generate(DriveInput(**kw, surprise=1.0)).novelty_score
    assert with_surp >= base


def test_drive_surprise_none_keeps_behavior():
    """surprise=None なら従来通りの新奇性計算（後方互換）。"""
    env = make_env()
    kw = dict(environment=env, memory_summary="", adjustments={"exploration": 0.5})
    d1 = Drive()
    d2 = Drive()
    a = d1.generate(DriveInput(**kw)).novelty_score
    b = d2.generate(DriveInput(**kw, surprise=None)).novelty_score
    assert a == b


# --- 学習層への反映 ---


def test_learning_rate_modulated_by_surprise():
    """高サプライズで学習率が上昇し、None なら不変。"""
    lr = Learning()
    assert lr._modulated_learning_rate(None) == lr.learning_rate
    assert lr._modulated_learning_rate(1.0) > lr._modulated_learning_rate(0.0)
    assert lr._modulated_learning_rate(1.0) <= lr.learning_rate * 2.0  # cap


def test_learning_drive_adjustment_uses_modulated_lr():
    """サプライズが学習率を介して駆動調整量に影響する。"""
    from core.learning.interface import LearningInput
    from core.evaluation.interface import EvaluationOutput, EvaluationScore
    from core.drive.drive import Drive
    from core.drive.interface import DriveOutput

    env = make_env()
    drive = Drive()
    ds = drive.generate(DriveInput(environment=env, memory_summary=""))

    def score(overall):
        return EvaluationOutput(
            score=EvaluationScore(goal_achievement=overall, efficiency=0.7,
                                  correctness=0.8, novelty=0.4, overall=overall,
                                  eval_type="rule", source="autonomous"),
            discrepancy="", improvement_suggestion="",
        )

    def run_once(surprise_val):
        l = Learning()
        history = [score(0.8).score, score(0.4).score, score(0.9).score,
                   score(0.3).score, score(0.7).score]
        out = l.learn(LearningInput(
            evaluation=score(0.8),
            evaluation_history=history,
            drive_snapshot=ds,
            episode_id="ep",
            driving_drive="exploration",
            source="autonomous",
            surprise=surprise_val,
        ))
        return out.drive_adjustments.get("exploration", 0.0)

    delta_flat = run_once(0.0)
    delta_surp = run_once(1.0)
    assert delta_surp > delta_flat


# --- 人格層への反映 ---


def test_personality_prompt_includes_surprise():
    """decide プロンプトにサプライズ文脈が注入される。"""
    p = Personality()
    env = make_env()
    ds = Drive().generate(DriveInput(environment=env, memory_summary=""))
    pin = PersonalityInput(
        drive=ds,
        memory=MemoryOutput(episodes=[], summary="まだ記憶がありません", total_count=0),
        surprise=0.9,
    )
    prompt = p._build_decision_prompt(pin)
    assert "Recent Prediction Surprise" in prompt
    assert "Surprise: 0.90" in prompt


def test_personality_prompt_without_surprise():
    """surprise=None ならプロンプトにセクションが無い（後方互換）。"""
    p = Personality()
    env = make_env()
    ds = Drive().generate(DriveInput(environment=env, memory_summary=""))
    pin = PersonalityInput(
        drive=ds,
        memory=MemoryOutput(episodes=[], summary="まだ記憶がありません", total_count=0),
    )
    prompt = p._build_decision_prompt(pin)
    assert "Recent Prediction Surprise" not in prompt
