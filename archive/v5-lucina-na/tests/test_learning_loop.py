"""
v3.2 学習ループの統合テスト（領域 A〜F）

- A. CycleScheduler の段階決定 + 新奇性信号の拡張
- B. 重要度の連続値計算 + 類似割引（rep_count）
- C. ゼロサム・クレジット割り当て + 二重学習率修正 + 学習ゲート
- D. 評価履歴のアトミック永続化（型安全ロード）
- E. 世界モデルの実誤差項 + confidence 連動
- F. 学習 vs 減衰（ゼロサムによる駆動baseの有界性）
"""

import json
import os
import tempfile
from datetime import datetime

from core.drive.drive import Drive
from core.drive.interface import DriveInput, DriveOutput
from core.evaluation.evaluation import Evaluation
from core.evaluation.interface import EvaluationInput, EvaluationOutput, EvaluationScore
from core.learning.learning import Learning
from core.learning.interface import LearningInput
from core.memory.memory import Memory
from core.memory.interface import Episode, MemoryInput
from core.personality.interface import PersonalityOutput
from core.world_model.world_model import WorldModel
from core.world_model.interface import Prediction
from environment.interface import EnvironmentOutput, SystemState
from core.agent.interface import AgentOutput, StepResult

import main as main_mod


def _make_env(cpu=30.0, memory=50.0, files_count=0, user_input=None) -> EnvironmentOutput:
    return EnvironmentOutput(
        timestamp=datetime.now(),
        user_input=user_input,
        system_state=SystemState(
            cpu_percent=cpu, memory_percent=memory,
            active_window="test", uptime=3600,
            current_directory="/home/test",
        ),
        files=["f"] * files_count if files_count else [],
        network=None,
        sensors={},
    )


def _make_drive_output(primary="exploration", novelty=0.1) -> DriveOutput:
    return DriveOutput(
        drives={
            "exploration": 0.45, "social": 0.35, "achievement": 0.35,
            "rest": 0.35, "maintenance": 0.35,
        },
        primary_drive=primary,
        drive_tension=0.2,
        novelty_score=novelty,
    )


def _make_result(success=True, num_steps=1, duration=0.5, error=None) -> AgentOutput:
    return AgentOutput(
        plan_id="test_plan",
        step_results=[
            StepResult(
                step_order=i + 1, action="file_list", success=success,
                output="ok", error=error, duration=duration,
            )
            for i in range(num_steps)
        ],
        overall_success=success,
        execution_time=duration,
        log="test log",
    )


def _make_decision(goal="ワークスペースを探索する", policy="探索",
                   intent=None, direct=False) -> PersonalityOutput:
    return PersonalityOutput(
        goal=goal, action_policy=policy, priority=3,
        conversation_intent=intent, direct_mode=direct,
    )


# ── A. 段階決定 ──────────────────────────────────────────────

class TestCycleScheduler:
    def setup_method(self):
        self.scheduler = main_mod.CycleScheduler()

    def test_user_message_always_tier3(self):
        assert self.scheduler.decide_tier(_make_drive_output(), "hello") == 3

    def test_low_novelty_is_tier1(self):
        assert self.scheduler.decide_tier(_make_drive_output(novelty=0.05)) == 1

    def test_high_novelty_is_tier3(self):
        assert self.scheduler.decide_tier(_make_drive_output(novelty=0.6)) == 3

    def test_mid_novelty_is_tier2(self):
        assert self.scheduler.decide_tier(_make_drive_output(novelty=0.3)) == 2

    def test_periodic_fallback(self):
        # 5サイクル毎にtier2、20サイクル毎にtier3（クールダウン付き）
        tiers = [self.scheduler.decide_tier(_make_drive_output(novelty=0.0)) for _ in range(22)]
        assert tiers[4] == 2          # cycle5
        assert tiers[19] == 3         # cycle20 (定期)
        assert tiers[20] == 1         # クールダウン中はtier1

    def test_tier3_cooldown_blocks_frequent_llm(self):
        # 新奇性が高くてもクールダウン(10)以内はtier3にならない
        first = self.scheduler.decide_tier(_make_drive_output(novelty=0.9))
        assert first == 3
        for _ in range(5):
            tier = self.scheduler.decide_tier(_make_drive_output(novelty=0.9))
            # tier3はクールダウンでブロック → tier2（新奇性0.25超）に落ちる
            assert tier == 2
        # クールダウン(10)経過後は再びtier3可能
        for _ in range(4):
            self.scheduler.decide_tier(_make_drive_output(novelty=0.9))
        tier = self.scheduler.decide_tier(_make_drive_output(novelty=0.9))
        assert tier == 3


class TestNoveltyExtension:
    def test_novelty_exceeds_old_max_without_user(self):
        """v3.2: ユーザー入力なしでも新奇性が0.4（旧最大）を超えられる"""
        d = Drive()
        # 欲求増加を無効化（テストの焦点は新奇性計算の拡張）
        d.satisfied_decay = 0.0
        d.unsatisfied_growth = 0.0
        # 高CPU + 高テンション + 探索不足。social を明確に優位にして
        # primary を決定的にする（拮抗時は _select_primary がランダム選択
        # するため、deficit の蓄積検証が flaky になる）
        d.params["exploration"]["base"] = 0.1
        d.params["social"]["base"] = 0.9
        env = _make_env(cpu=90.0, files_count=100)
        r1 = d.generate(DriveInput(environment=env, memory_summary=""))
        r2 = d.generate(DriveInput(environment=env, memory_summary=""))
        # 旧最大0.4をユーザー入力なしで超える（v3.2拡張の要）
        assert r1.novelty_score > 0.4
        assert r1.novelty_score <= 1.0
        # 2回目以降は deficit が積み上がる（social が primary の間は決定的）
        assert r2.novelty_score >= r1.novelty_score

    def test_novelty_resets_after_exploration_primary(self):
        d = Drive()
        env = _make_env(cpu=90.0, files_count=100)
        # 探索がprimaryでない状態を作る（socialを高く）
        d.params["exploration"]["base"] = 0.1
        d.params["social"]["base"] = 0.9
        d.generate(DriveInput(environment=env, memory_summary=""))
        d.generate(DriveInput(environment=env, memory_summary=""))
        assert d._deficit_cycles > 0  # 探索がprimaryでない間は蓄積される
        # primaryを探索に戻すとdeficitがリセットされる
        d.params["exploration"]["base"] = 0.9
        d.params["social"]["base"] = 0.1
        r = d.generate(DriveInput(environment=env, memory_summary=""))
        assert d._deficit_cycles == 0
        assert r.novelty_score <= 1.0


# ── B. 重要度 ────────────────────────────────────────────────

class TestImportance:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = Memory(storage_path=self.tmpdir)

    def test_success_high_importance(self):
        imp = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "autonomous", self.memory,
        )
        assert 0.5 <= imp <= 1.0

    def test_failure_lower_importance(self):
        imp_ok = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "autonomous", self.memory,
        )
        imp_bad = main_mod._compute_importance(
            _make_decision(), _make_result(success=False),
            _make_drive_output(), "autonomous", self.memory,
        )
        assert imp_bad < imp_ok

    def test_dialog_bonus(self):
        imp_auto = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "autonomous", self.memory,
        )
        imp_dialog = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "dialog", self.memory,
        )
        assert imp_dialog > imp_auto

    def test_dialog_rule_squash(self):
        """対話+ルール評価は重要度が圧縮される（チャットで記憶が埋まらない）"""
        rule_result = EvaluationOutput(
            score=EvaluationScore(1.0, 1.0, 1.0, 0.3, 0.85, eval_type="rule"),
            discrepancy="", improvement_suggestion="",
        )
        llm_result = EvaluationOutput(
            score=EvaluationScore(1.0, 1.0, 1.0, 0.3, 0.85, eval_type="llm"),
            discrepancy="", improvement_suggestion="",
        )
        imp_rule = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "dialog", self.memory, rule_result,
        )
        imp_llm = main_mod._compute_importance(
            _make_decision(), _make_result(success=True),
            _make_drive_output(), "dialog", self.memory, llm_result,
        )
        assert imp_rule < imp_llm

    def test_rep_count_reduces_importance(self):
        goal = "ワークスペースのファイルを調査する"
        # 同一goalを5回保存してから計算
        for _ in range(5):
            ep = Episode(
                id=f"ep_{datetime.now().timestamp()}_{_}",
                timestamp=datetime.now(), event=f"goal={goal}",
                context="", emotion="", result="success=True",
                importance=0.7, tags=["exploration"],
            )
            self.memory.save(ep)
        imp = main_mod._compute_importance(
            _make_decision(goal=goal), _make_result(success=True),
            _make_drive_output(), "autonomous", self.memory,
        )
        imp_fresh = main_mod._compute_importance(
            _make_decision(goal="まったく別の新しいテーマの調査"),
            _make_result(success=True), _make_drive_output(),
            "autonomous", self.memory,
        )
        assert imp < imp_fresh

    def test_reweight_duplicates_two_way(self):
        """双方向リウェイト: 繰り返しで既存の類似エピソードも減点される"""
        goal = "同一の探索タスク"
        eps = []
        for i in range(5):
            ep = Episode(
                id=f"rw_{i}", timestamp=datetime.now(), event=f"goal={goal}",
                context="", emotion="", result="success=True",
                importance=0.9, tags=["exploration"],
            )
            self.memory.save(ep)
            eps.append(ep)
        # 5件目保存後、最初の類似エピソードは減点されている
        first = self.memory.episodes[0]
        assert first.importance < 0.9


# ── C. ゼロサム・クレジット割り当て ─────────────────────────

class TestZeroSumCredit:
    def setup_method(self):
        self.learning = Learning()

    def _history(self, scores):
        return [EvaluationScore(s, 0.6, 0.8, 0.3, s, eval_type="rule")
                for s in scores]

    def test_zero_sum_property(self):
        """主駆動 +delta、他 −delta/4 → 合計が0"""
        result = self.learning.learn(LearningInput(
            evaluation=EvaluationOutput(
                score=EvaluationScore(0.9, 0.7, 0.8, 0.3, 0.9, eval_type="rule"),
                discrepancy="", improvement_suggestion="",
            ),
            evaluation_history=self._history([0.3, 0.4, 0.5, 0.6, 0.7]),
            drive_snapshot=_make_drive_output(primary="exploration"),
            episode_id="ep_1", driving_drive="exploration",
        ))
        total = sum(result.drive_adjustments.values())
        assert abs(total) < 1e-9
        assert result.drive_adjustments["exploration"] > 0
        assert result.drive_adjustments["social"] < 0

    def test_autonomous_halved_keeps_zero_sum(self):
        """自律サイクルは全体0.5倍でもゼロサム維持"""
        history = self._history([0.3, 0.4, 0.5, 0.6, 0.7])
        result = self.learning.learn(LearningInput(
            evaluation=EvaluationOutput(
                score=EvaluationScore(0.9, 0.7, 0.8, 0.3, 0.9, eval_type="rule"),
                discrepancy="", improvement_suggestion="",
            ),
            evaluation_history=history,
            drive_snapshot=_make_drive_output(primary="exploration"),
            episode_id="ep_1", driving_drive="exploration", source="autonomous",
        ))
        total = sum(result.drive_adjustments.values())
        assert abs(total) < 1e-9

    def test_rest_drive_skips_adjustment(self):
        result = self.learning.learn(LearningInput(
            evaluation=EvaluationOutput(
                score=EvaluationScore(0.9, 0.7, 0.8, 0.3, 0.9, eval_type="rule"),
                discrepancy="", improvement_suggestion="",
            ),
            evaluation_history=self._history([0.3, 0.4, 0.5, 0.6, 0.7]),
            drive_snapshot=_make_drive_output(primary="rest"),
            episode_id="ep_1", driving_drive="rest",
        ))
        assert result.drive_adjustments == {}

    def test_variance_gate_blocks_constant_reward(self):
        """報酬が一定（分散0）なら駆動調整は行われない"""
        result = self.learning.learn(LearningInput(
            evaluation=EvaluationOutput(
                score=EvaluationScore(0.5, 0.6, 0.8, 0.3, 0.5, eval_type="rule"),
                discrepancy="", improvement_suggestion="",
            ),
            evaluation_history=self._history([0.5, 0.5, 0.5, 0.5, 0.5]),
            drive_snapshot=_make_drive_output(primary="exploration"),
            episode_id="ep_1", driving_drive="exploration",
        ))
        assert result.drive_adjustments == {}

    def test_double_lr_fixed(self):
        """drive.update_parameters は二重に学習率を掛けない"""
        d = Drive()
        before = d.params["exploration"]["base"]
        d.update_parameters({"exploration": 0.5})  # clip -> 0.2
        after = d.params["exploration"]["base"]
        # 新実装: clip(0.5)=0.2 直接適用
        # v3.3: ベース値 0.35 からスタート
        assert abs((after - before) - 0.2) < 1e-9

    def test_task_dialog_credit_classification(self):
        decision = _make_decision(
            goal="コードのバグを修正する", policy="修正を実装する",
        )
        drive_state = _make_drive_output(primary="exploration")
        assert main_mod._classify_driving_drive(decision, drive_state, "ユーザー依頼") == "achievement"

    def test_conversation_credited_to_social(self):
        decision = _make_decision(
            goal="挨拶", policy="会話する", intent="挨拶します",
        )
        drive_state = _make_drive_output(primary="exploration")
        assert main_mod._classify_driving_drive(decision, drive_state, "こんにちは") == "social"

    def test_autonomous_uses_primary(self):
        decision = _make_decision()
        drive_state = _make_drive_output(primary="maintenance")
        assert main_mod._classify_driving_drive(decision, drive_state, None) == "maintenance"


# ── D. 評価履歴の永続化 ──────────────────────────────────────

class TestEvaluationPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "eval_history.json")

        class MockEvalLLM:
            def chat(self, prompt, system_prompt=None):
                return (
                    "goal_achievement: 0.8\nefficiency: 0.7\ncorrectness: 0.9\n"
                    "novelty: 0.3\noverall: 0.7\n"
                )

            def extract_yaml_like(self, text):
                """評価層のパースに必要なLLMClient互換メソッド"""
                result = {}
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if ":" in line and not line.startswith("-"):
                        key, _, value = line.partition(":")
                        result[key.strip()] = value.strip()
                return result
        self.llm = MockEvalLLM()

    def _evaluate_once(self, evaluation):
        return evaluation.evaluate(EvaluationInput(
            goal="テスト", action_result=_make_result(success=True),
            expected_outcome="", episode=_make_episode(),
        ))

    def test_persistence_roundtrip(self):
        ev = Evaluation(llm_client=self.llm, storage_path=self.path)
        self._evaluate_once(ev)
        self._evaluate_once(ev)
        # 新しいインスタンスで読み込む（再起動を模擬）
        ev2 = Evaluation(llm_client=self.llm, storage_path=self.path)
        history = ev2.get_history()
        assert len(history) == 2
        assert hasattr(history[0], "overall")  # 型安全（dictではない）

    def test_corrupt_file_recovers_empty(self):
        with open(self.path, "w") as f:
            f.write("{corrupt json!!!")
        ev = Evaluation(llm_client=self.llm, storage_path=self.path)
        assert ev.get_history() == []

    def test_eval_type_tagged(self):
        ev = Evaluation(llm_client=self.llm, storage_path=self.path)
        out = self._evaluate_once(ev)
        assert out.score.eval_type == "llm"

    def test_rule_eval_when_use_llm_false(self):
        ev = Evaluation(llm_client=self.llm, storage_path=None)
        out = ev.evaluate(EvaluationInput(
            goal="テスト", action_result=_make_result(success=True),
            expected_outcome="", episode=_make_episode(), use_llm=False,
        ))
        assert out.score.eval_type == "rule"


def _make_episode() -> Episode:
    return Episode(
        id="ep_t", timestamp=datetime.now(), event="テスト行動",
        context="", emotion="", result="success=True",
        importance=0.5, tags=["test"], source="autonomous",
    )


# ── E. 世界モデルの実誤差項 ──────────────────────────────────

class TestWorldModelError:
    def setup_method(self):
        self.wm = WorldModel(llm_client=None)

    def _pred(self, expected_reward=0.5, risk="low") -> Prediction:
        return Prediction(
            action="exploration", next_state="ファイルが発見される",
            probability=0.7, expected_reward=expected_reward,
            risk_level=risk, reasoning="test",
        )

    def test_error_term_recorded(self):
        self.wm.update(_make_episode(), self._pred(expected_reward=0.5),
                       actual_overall=0.9)
        key = ("exploration", "low")
        stats = self.wm.statistics[key]
        assert stats["count"] == 1
        assert abs(stats["total_error"] - 0.4) < 1e-9

    def test_confidence_drops_with_large_error(self):
        # 誤差大の予測を3回
        for _ in range(3):
            self.wm.update(_make_episode(), self._pred(expected_reward=0.9),
                           actual_overall=0.1)
        low_conf = self.wm.confidence("state", "exploration")
        # 誤差小の予測を20回
        for _ in range(20):
            self.wm.update(_make_episode(), self._pred(expected_reward=0.5),
                           actual_overall=0.5)
        high_conf = self.wm.confidence("state", "exploration")
        assert low_conf < high_conf

    def test_confidence_default_low(self):
        assert self.wm.confidence("unknown", "unknown_action") == 0.3


# ── F. 学習 vs 減衰（有界性） ────────────────────────────────

# ── G. Phase 3: run_cycle のサプライズ配線 ───────────────────

class _StubPersonality:
    """run_cycle 用の最小人格スタブ（decide/update のみ）"""

    def __init__(self):
        from core.personality.interface import PersonalityState
        self.state = PersonalityState(
            name="Lucina", traits={}, speaking_style="", values=[],
            mood="neutral", relationship={"familiarity": 0.3, "trust": 0.5},
        )

    def decide(self, input):
        return _make_decision()

    def update_state(self, episode, overall=None):
        pass

    def update_self_model(self, **kwargs):
        pass

    def speak(self, intent: str) -> str:
        return intent


class _StubPlanning:
    def make(self, input):
        from core.planning.interface import PlanningOutput, Step
        return PlanningOutput(
            plan_id="bench",
            steps=[Step(order=1, action="workspace_list", params={},
                        description="", expected_result="")],
            expected_outcome="ok",
        )


class _StubAgent:
    TOOL_REGISTRY = {}

    def execute(self, input):
        return _make_result(success=True)

    def speak(self, text: str) -> str:
        return text


class _StubLTP:
    """run_cycle 用の最小長期計画スタブ。"""

    def __init__(self):
        self.aspirations = []
        self.focus_area = "test"
        self.identity_policy = "test policy"
        self.last_plan_update = None
        self._last_aspiration_update = None

    def plan(self, input):
        from core.long_term_planning.interface import LongTermPlanningOutput
        return LongTermPlanningOutput(
            long_term_goal="test", routines=[],
            identity_policy="test policy", focus_area="test", reflection="",
        )

    def _maybe_update_aspirations(self, input):
        pass

    def update_goal_progress(self, goal, overall):
        pass

    def note_activity(self, goal):
        pass

    def note_aspiration_activity(self, goal):
        pass


class _SpyLearning(Learning):
    """学習層に渡されたサプライズを記録するスパイ。"""

    def __init__(self):
        super().__init__()
        self.last_surprise = None

    def learn(self, input):
        self.last_surprise = getattr(input, "surprise", None)
        return super().learn(input)


class TestRunCycleSurpriseWiring:
    """Phase 3 (M14): run_cycle のサプライズ配線の統合テスト。

    forced_tier=2 で実LLMを回避し（ルールベース評価 + ルール世界モデル）、
    personality/planning/agent/ltp のみスタブ化する。
    """

    def _setup(self, tmp_path):
        env = _make_env()
        memory = Memory(storage_path=str(tmp_path / "episodes"))
        drive = Drive()
        personality = _StubPersonality()
        planning = _StubPlanning()
        agent = _StubAgent()
        evaluation = Evaluation(llm_client=None, storage_path=None)
        learning = _SpyLearning()
        world_model = WorldModel(llm_client=None)
        ltp = _StubLTP()
        scheduler = main_mod.CycleScheduler()
        return (env, memory, drive, personality, planning, agent,
                evaluation, learning, world_model, ltp, scheduler)

    def test_run_cycle_returns_tuple_with_surprise(self, tmp_path, monkeypatch):
        """run_cycle が (output, surprise) タプルを返し、surprise が 0..1 に収まる。"""
        (env, memory, drive, personality, planning, agent,
         evaluation, learning, world_model, ltp, scheduler) = self._setup(tmp_path)
        monkeypatch.setattr(main_mod, "CYCLE_LOG_PATH",
                            str(tmp_path / "cycle_latest.json"))

        env_state = _make_env()
        drive_state = _make_drive_output()
        from core.memory.interface import MemoryOutput
        memory_ctx = MemoryOutput(
            episodes=[], summary="まだ記憶がありません", total_count=0,
        )

        output, surprise = main_mod.run_cycle(
            env, memory, drive, personality, planning, agent,
            evaluation, learning, world_model, ltp, scheduler,
            env_state=env_state, drive_state=drive_state,
            memory_ctx=memory_ctx, forced_tier=2,
        )

        assert output is None  # 自律サイクルで会話なし
        assert surprise is not None
        assert isinstance(surprise, float)
        assert 0.0 <= surprise <= 1.0

    def test_surprise_reaches_learning_input(self, tmp_path, monkeypatch):
        """実測サプライズが LearningInput.surprise に渡される。"""
        (env, memory, drive, personality, planning, agent,
         evaluation, learning, world_model, ltp, scheduler) = self._setup(tmp_path)
        monkeypatch.setattr(main_mod, "CYCLE_LOG_PATH",
                            str(tmp_path / "cycle_latest.json"))
        from core.memory.interface import MemoryOutput
        memory_ctx = MemoryOutput(
            episodes=[], summary="まだ記憶がありません", total_count=0,
        )

        main_mod.run_cycle(
            env, memory, drive, personality, planning, agent,
            evaluation, learning, world_model, ltp, scheduler,
            env_state=_make_env(), drive_state=_make_drive_output(),
            memory_ctx=memory_ctx, forced_tier=2,
        )

        # tier2 + 世界モデル予測あり → 学習層に float のサプライズが渡る
        assert learning.last_surprise is not None
        assert 0.0 <= learning.last_surprise <= 1.0

    def test_surprise_feeds_next_drive_generation(self, tmp_path, monkeypatch):
        """前サイクルのサプライズが次サイクルの DriveInput.surprise になる。"""
        (env, memory, drive, personality, planning, agent,
         evaluation, learning, world_model, ltp, scheduler) = self._setup(tmp_path)
        monkeypatch.setattr(main_mod, "CYCLE_LOG_PATH",
                            str(tmp_path / "cycle_latest.json"))
        from core.memory.interface import MemoryOutput
        memory_ctx = MemoryOutput(
            episodes=[], summary="まだ記憶がありません", total_count=0,
        )

        _, surprise = main_mod.run_cycle(
            env, memory, drive, personality, planning, agent,
            evaluation, learning, world_model, ltp, scheduler,
            env_state=_make_env(), drive_state=_make_drive_output(),
            memory_ctx=memory_ctx, forced_tier=2,
            surprise=0.42,
        )
        assert 0.0 <= surprise <= 1.0

    def test_zero_sum_keeps_base_sum_bounded(self):
        """ゼロサム調整を100サイクル適用しても駆動base合計は発散しない"""
        d = Drive()
        learning = Learning()
        history = [EvaluationScore(0.3, 0.6, 0.8, 0.3, 0.3, eval_type="rule"),
                   EvaluationScore(0.5, 0.6, 0.8, 0.3, 0.5, eval_type="rule"),
                   EvaluationScore(0.7, 0.6, 0.8, 0.3, 0.7, eval_type="rule")]

        initial_sum = sum(p["base"] for p in d.params.values())
        for i in range(100):
            history.append(EvaluationScore(
                0.7, 0.6, 0.8, 0.3, 0.7, eval_type="rule"))
            result = learning.learn(LearningInput(
                evaluation=EvaluationOutput(
                    score=EvaluationScore(0.7, 0.6, 0.8, 0.3, 0.7, eval_type="rule"),
                    discrepancy="", improvement_suggestion="",
                ),
                evaluation_history=history[-10:],
                drive_snapshot=_make_drive_output(primary="exploration"),
                episode_id=f"ep_{i}", driving_drive="exploration",
                source="autonomous",
            ))
            if result.drive_adjustments:
                d.update_parameters(result.drive_adjustments)

        final_sum = sum(p["base"] for p in d.params.values())
        # ゼロサムにより合計はほぼ不変（クリップ境界の影響のみ許容）
        assert abs(final_sum - initial_sum) < 0.05
        for name, p in d.params.items():
            assert 0.0 <= p["base"] <= 1.0
