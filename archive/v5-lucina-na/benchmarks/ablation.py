"""アブレーション検証（M15）: 層の貢献をON/OFF比較で数字で証明する。

「10層は飾り」という外部レビュー批判へのデータによる反論。
実LLMは使わず、決定論的な合成サイクルで比較する。

検証項目:
1. 学習層 ON/OFF → 駆動パラメータ（base値）の軌跡が ON 時のみ有意に動く
2. 記憶層 ON/OFF → 繰り返し検出（repetition_count）が ON 時のみ機能する
3. 評価層 LLM/ルール → 両モードのスコア整合性（乖離率）
"""

import os
import tempfile
from datetime import datetime

from core.agent.interface import AgentOutput, StepResult
from core.drive.drive import Drive
from core.drive.interface import DriveInput
from core.evaluation.evaluation import Evaluation
from core.evaluation.interface import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationScore,
)
from core.learning.learning import Learning
from core.learning.interface import LearningInput
from core.llm import LLMClient
from core.memory.memory import Memory
from core.memory.interface import Episode, MemoryInput

from benchmarks.common import make_env, save_report
from benchmarks.interface import BenchmarkReport, BenchmarkSection


class _MockEvalLLM(LLMClient):
    """LLM評価モードのフェイク（YAML風スコアを返す）。"""

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "goal_achievement: 0.9\n"
            "efficiency: 0.8\n"
            "correctness: 0.7\n"
            "novelty: 0.6\n"
            "overall: 0.85\n"
        )


def _synthetic_evaluation(overall: float) -> EvaluationOutput:
    """指定の overall を持つ評価結果を合成する（ルールレジーム相当）。"""
    return EvaluationOutput(
        score=EvaluationScore(
            goal_achievement=overall,
            efficiency=0.7,
            correctness=0.8,
            novelty=0.4,
            overall=overall,
            eval_type="rule",
            source="autonomous",
        ),
        discrepancy="",
        improvement_suggestion="",
    )


def _variation(trace: list[float]) -> float:
    """軌跡の総変動（隣接差の絶対値和）。「層が何かを変えている」指標。"""
    return sum(abs(b - a) for a, b in zip(trace, trace[1:]))


def _learning_trajectory(with_learning: bool, cycles: int = 24) -> list[float]:
    """学習ON/OFFそれぞれで駆動 base 値の軌跡を返す（同一の合成報酬系列）。"""
    drive = Drive()
    learning = Learning()
    history: list[EvaluationScore] = []
    env = make_env(cpu=40.0, memory=50.0)
    rewards = [0.8, 0.4, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1]
    trace = []
    for i in range(cycles):
        ds = drive.generate(DriveInput(environment=env, memory_summary=""))
        trace.append(drive.params["exploration"]["base"])
        score = _synthetic_evaluation(rewards[i % len(rewards)])
        history.append(score.score)
        if with_learning:
            out = learning.learn(LearningInput(
                evaluation=score,
                evaluation_history=history,
                drive_snapshot=ds,
                episode_id=f"ep_{i}",
                driving_drive="exploration",
                source="autonomous",
            ))
            if out.drive_adjustments:
                drive.update_parameters(out.drive_adjustments)
    return trace


def _eval_modes_comparison() -> dict:
    """評価層 LLM/ルール の両モードで同じ入力を評価し整合性を測る。"""
    episode = Episode(
        id="ep_eval", timestamp=datetime.now(), event="テスト行動",
        context="", emotion="", result="success", importance=0.5,
    )
    agent_out = AgentOutput(
        plan_id="p", overall_success=True, execution_time=0.5,
        step_results=[StepResult(step_order=1, action="file_list",
                                 success=True, output="ok")],
    )
    common = dict(goal="ファイルを調査する", action_result=agent_out,
                  expected_outcome="一覧表示", episode=episode)

    eval_llm = Evaluation(llm_client=_MockEvalLLM(), storage_path=None)
    r_llm = eval_llm.evaluate(EvaluationInput(use_llm=True, **common))
    eval_rule = Evaluation(llm_client=_MockEvalLLM(), storage_path=None)
    r_rule = eval_rule.evaluate(EvaluationInput(use_llm=False, **common))
    return {
        "llm_overall": r_llm.score.overall,
        "rule_overall": r_rule.score.overall,
        "llm_type": r_llm.score.eval_type,
        "rule_type": r_rule.score.eval_type,
        "divergence": abs(r_llm.score.overall - r_rule.score.overall),
    }


def run_ablation_validation(report_dir: str | None = None) -> BenchmarkReport:
    sections = []

    # --- 1. 学習層 ON/OFF: 駆動base値の軌跡の総変動を比較 ---
    trace_on = _learning_trajectory(with_learning=True)
    trace_off = _learning_trajectory(with_learning=False)
    var_on = _variation(trace_on)
    var_off = _variation(trace_off)
    sections.append(BenchmarkSection(
        name="learning_on_off",
        passed=(var_on > var_off + 0.02),
        metrics={
            "variation_learning_ON": round(var_on, 4),
            "variation_learning_OFF": round(var_off, 4),
            "delta": round(var_on - var_off, 4),
        },
        details=[
            "学習ON時のみ駆動base値がゼロサム調整で有意に動く",
            "OFF時は欲求の自然増加（urge）のみなので軌跡はほぼ平坦",
        ],
    ))

    # --- 2. 記憶層 ON/OFF: 繰り返し検出と想起が ON 時のみ機能 ---
    # メモリの一時ディレクトリはレポート保存先を汚さないよう常に tempfile を使う
    base = tempfile.mkdtemp(prefix="lucina_bench_ablation_")
    mem_on_dir = os.path.join(base, "mem_on")
    mem_off_dir = os.path.join(base, "mem_off")
    os.makedirs(mem_on_dir, exist_ok=True)
    os.makedirs(mem_off_dir, exist_ok=True)

    mem_on = Memory(storage_path=mem_on_dir)
    for i in range(5):
        mem_on.save(Episode(
            id=f"rep_{i}", timestamp=datetime.now(),
            event="goal=同じ実験を繰り返す", context="",
            emotion="", result="", importance=0.5, tags=["repeat"],
        ))
    rep_on = mem_on.repetition_count("同じ実験を繰り返す")
    recall_on = len(mem_on.search(
        MemoryInput(query="実験", top_k=5)
    ).episodes)

    mem_off = Memory(storage_path=mem_off_dir)
    rep_off = mem_off.repetition_count("同じ実験を繰り返す")
    recall_off = len(mem_off.search(
        MemoryInput(query="実験", top_k=5)
    ).episodes)

    sections.append(BenchmarkSection(
        name="memory_on_off",
        passed=(rep_on > 0 and rep_off == 0 and recall_on > recall_off),
        metrics={
            "repetition_count_ON": rep_on,
            "repetition_count_OFF": rep_off,
            "recall_hits_ON": recall_on,
            "recall_hits_OFF": recall_off,
        },
        details=[
            "記憶ON時のみ繰り返し検出（多様性圧力）が機能する",
            "記憶OFF時は過去を参照できず、同じ目標を繰り返す",
        ],
    ))

    # --- 3. 評価層 LLM/ルール: 両モードの整合性 ---
    comp = _eval_modes_comparison()
    sections.append(BenchmarkSection(
        name="evaluation_modes",
        passed=(comp["llm_type"] == "llm" and comp["rule_type"] == "rule"
                and comp["divergence"] <= 0.5),
        metrics={
            "llm_overall": comp["llm_overall"],
            "rule_overall": comp["rule_overall"],
            "divergence": round(comp["divergence"], 3),
        },
        details=[
            "LLM評価とルール評価は同一入力を別レジームでスコア化",
            "乖離率が小さいほど両モードの整合性が高い",
        ],
    ))

    return BenchmarkReport(name="ablation", sections=sections)


if __name__ == "__main__":
    report = run_ablation_validation()
    print(save_report(report))
