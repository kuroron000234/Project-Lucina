"""
lucina-NA: Entry point

Modes:
    --daemon               Daemon mode (continuous autonomous + IPC)
    --message "text"       One-shot conversation (IPC preferred, else standalone)
    --phase 2              Run Phase 2 continuous loop
    --validate --phase 2   Single validation cycle
"""

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
from datetime import datetime

os.makedirs("data/episodes", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/ipc", exist_ok=True)

import config
# v4.0: 意志フェーズ — 自分の部屋（ワークスペース）と日記ディレクトリを起動時に確保
os.makedirs(config.WILL_CONFIG.get("workspace_dir", "data/workspace"), exist_ok=True)
os.makedirs(config.WILL_CONFIG.get("diary_dir", "data/diary"), exist_ok=True)
import ipc
from environment.environment import Environment
from environment.interface import EnvironmentInput
from core.memory.memory import Memory
from core.memory.interface import MemoryInput, Episode
from core.drive.drive import Drive
from core.drive.interface import DriveInput, DriveOutput
from core.personality.personality import Personality
from core.personality.interface import PersonalityInput
from core.planning.planning import Planning
from core.planning.interface import PlanningInput, ToolInfo, PlanningOutput, Step
from core.agent.agent import Agent
from core.agent.interface import AgentInput, AgentOutput, TOOL_PARAM_SCHEMAS
from core.evaluation.evaluation import Evaluation
from core.evaluation.interface import EvaluationInput
from core.learning.learning import Learning
from core.learning.interface import LearningInput
from core.world_model.world_model import WorldModel
from core.world_model.interface import WorldModelInput
from core.long_term_planning.long_term_planning import LongTermPlanning
from core.long_term_planning.interface import LongTermPlanningInput, LongTermPlanningOutput

log_handler = RotatingFileHandler(
    "data/logs/system.log", maxBytes=5*1024*1024, backupCount=3,
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
))
logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler, logging.StreamHandler()],
)
logger = logging.getLogger("Main")

CYCLE_LOG_PATH = "data/ipc/cycle_latest.json"

# Actions that go through Opencode (vs direct execution)
OPCODE_ACTIONS = {"web_search", "web_fetch", "code_analyze", "opencode_run", "self_modify", "direct_execute"}


def _executor_for(action: str) -> str:
    return "opencode" if action in OPCODE_ACTIONS else "direct"


def _build_episode(decision, result, drive_state, source="autonomous",
                   driving_drive=None, emotion="", user_message=None):
    """エピソードを構築する（v3.2: source/driving_drive を記録）。

    v3.4: emotion（当時のムード）を記録し、記憶層が感情状態を保持できるようにする。
    v3.5: 対話エピソードにユーザー発言を記録し、後からキーワード検索で
    過去の会話を呼び戻せるようにする。
    """
    driving_drive = driving_drive or drive_state.primary_drive
    # v3.5 (C): ユーザー発言を文脈に含め、記憶検索の対象にする
    context = decision.context_summary
    if user_message:
        context = f"user: {user_message} | {context}"
    # v4.1: 実行の自己診断をエピソードに記録。
    # ツール実行が実質的に失敗している（空パラメータ・0バイト書き込みなど）場合に
    # その理由を記憶層・学習層が見られるようにする。
    diag = _execution_diagnosis(result)
    if diag:
        context = f"{diag} | {context}"
    return Episode(
        id=str(time.time()),
        timestamp=datetime.now(),
        event=f"goal={decision.goal}",
        context=context,
        emotion=emotion,
        result=f"success={result.overall_success}, steps={len(result.step_results)}",
        importance=0.5,  # 評価後に _compute_importance で確定
        tags=[driving_drive, decision.action_policy[:20]],
        source=source,
        driving_drive=driving_drive,
    )


def _compute_importance(decision, result, drive_state, source, memory,
                        eval_result=None) -> float:
    """
    v3.2: 重要度を連続値で計算する（LLM不要）。

    importance = clamp01(0.15 + 0.30*success + 0.20*efficiency + 0.10*correctness
                         + 0.10*novelty + dialog_bonus − 0.08*rep_count)

    対話 + ルール評価の場合は重要度を圧縮（チャットで記憶が埋まるのを防止）。
    """
    w = config.LEARNING_CONFIG["importance"]

    success = 1.0 if result.overall_success else 0.0

    num_steps = max(len(result.step_results), 1)
    avg_time = result.execution_time / num_steps if num_steps else 0
    if avg_time < 1:
        efficiency = 0.9
    elif avg_time < 5:
        efficiency = 0.7
    elif avg_time < 15:
        efficiency = 0.5
    else:
        efficiency = 0.3

    error_count = sum(1 for s in result.step_results if s.error)
    correctness = max(0.0, 1.0 - error_count / num_steps)

    novelty = drive_state.novelty_score
    bonus = w["dialog_bonus"] if source == "dialog" else 0.0

    rep_count = memory.repetition_count(decision.goal)
    importance = (
        w["base"]
        + w["w_success"] * success
        + w["w_efficiency"] * efficiency
        + w["w_correctness"] * correctness
        + w["w_novelty"] * novelty
        + bonus
        - w["rep_penalty"] * min(rep_count, w["rep_max"])
    )

    # 対話 + ルール評価: 重要度を圧縮（0.3 を下限に squash）
    if eval_result is not None and eval_result.score.eval_type == "rule" \
            and source == "dialog":
        importance = 0.3 + (importance - 0.3) * w["dialog_rule_squash"]

    return max(0.0, min(1.0, importance))


def _execution_diagnosis(result: AgentOutput) -> str:
    """
    v4.1: 実行結果の自己診断テキストを生成する（LLM不要・ルールベース）。

    「疑似成功」（0バイト書き込み・空パラメータ・未実行）を検出し、
    記憶層が失敗の理由を学べるように要約する。
    正常な実行の場合は空文字を返す。
    """
    if result.overall_success and not result.step_results:
        return ""
    issues = []
    for sr in result.step_results:
        if not sr.success:
            err = (sr.error or "")[:60]
            issues.append(f"step{sr.step_order}:{sr.action} FAIL ({err})")
        elif Agent._wrote_zero_bytes(sr.output):
            issues.append(f"step{sr.step_order}:{sr.action} wrote 0 bytes (empty params?)")
    if not issues:
        return ""
    return "exec_issues: " + " | ".join(issues[:3])


def _classify_driving_drive(decision, drive_state, user_message) -> str:
    """
    v3.2: 行動を選択した駆動を判定する（クレジット割り当ての対象）。

    - 自律: 決定時点の primary_drive をそのまま使用
    - 対話 + conversation_intent: social
    - 対話 + タスク: goal/action_policy のキーワードから判定
    """
    if not user_message:
        return drive_state.primary_drive
    if decision.conversation_intent and not decision.direct_mode:
        return "social"
    text = f"{decision.goal} {decision.action_policy}"
    for keywords, drive_name in [
        (["整理", "メンテ", "ログ", "バックアップ", "チェック"], "maintenance"),
        (["探索", "確認", "調査"], "exploration"),
        (["修正", "実装", "テスト", "改善", "リファクタ"], "achievement"),
    ]:
        if any(k in text for k in keywords):
            return drive_name
    return "social"


class CycleScheduler:
    """
    v3.2: コスト段階（tier）の決定。

    tier1: 行動+保存（駆動は記憶を見る）— コスト1（decideのみ）
    tier2: + ルールベース評価・ゼロサム学習・人格更新・WM統計・LTPカウンタ
    tier3: + LLM評価・LLM世界モデル（クールダウン付き間引き）

    トリガー（イベントベース + 定期フォールバック）:
    - tier2: novelty >= 0.25 OR 5サイクル毎
    - tier3: novelty >= 0.35 OR 20サイクル毎（クールダウン10サイクル）
    """

    def __init__(self):
        t = config.LEARNING_CONFIG["tier"]
        self.novelty_tier2 = t["novelty_tier2"]
        self.novelty_tier3 = t["novelty_tier3"]
        self.interval_tier2 = t["interval_tier2"]
        self.interval_tier3 = t["interval_tier3"]
        self.cooldown_tier3 = t["cooldown_tier3"]
        self.cycle_count = 0
        self.last_tier3 = -10**9

    def decide_tier(self, drive_state, user_message=None) -> int:
        """ユーザー入力は常に tier3。自律は新奇性/定期で段階を決める。"""
        if user_message:
            return 3
        self.cycle_count += 1
        novelty = drive_state.novelty_score
        # tier3: 新奇性高 or 定期間隔（クールダウン付き）
        if (novelty >= self.novelty_tier3
                or self.cycle_count % self.interval_tier3 == 0) \
                and (self.cycle_count - self.last_tier3) >= self.cooldown_tier3:
            self.last_tier3 = self.cycle_count
            return 3
        # tier2: 新奇性中 or 定期間隔
        if novelty >= self.novelty_tier2 \
                or self.cycle_count % self.interval_tier2 == 0:
            return 2
        return 1


def _cached_memory_search(memory, env_state, cache: dict):
    """
    v3.2: 環境指紋が不変かつ60秒以内なら記憶検索結果を再利用する。
    記憶が増えても自律サイクルの検索コストを一定に保つ。
    """
    fingerprint = (
        env_state.system_state.cpu_percent // 10,
        env_state.system_state.memory_percent // 10,
        env_state.user_input is not None,
        len(env_state.files) if env_state.files else 0,
    )
    now = time.time()
    if cache.get("fp") == fingerprint \
            and now - cache.get("ts", 0) < config.LEARNING_CONFIG["memory_cache_seconds"]:
        return cache["ctx"]
    ctx = memory.search(MemoryInput(
        query="", top_k=config.MEMORY_CONFIG["search_top_k"],
    ))
    cache.update(fp=fingerprint, ts=now, ctx=ctx)
    return ctx


def _save_cycle_details(phase, env_state, drive_state, decision, plan,
                        result, eval_result=None, learn_result=None,
                        user_message=None):
    """Save detailed cycle data to IPC file for WebUI consumption."""
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "user_message": user_message,
            "trigger": "user" if user_message else "periodic",
            "environment": {
                "cpu": env_state.system_state.cpu_percent,
                "memory": env_state.system_state.memory_percent,
                "network": env_state.network.is_connected if env_state.network else None,
                "files": len(env_state.files) if env_state.files else 0,
            },
            "drive": {
                "drives": drive_state.drives,
                "primary": drive_state.primary_drive,
                "tension": round(drive_state.drive_tension, 3),
                "novelty": round(drive_state.novelty_score, 3),
            },
            "decision": {
                "goal": decision.goal,
                "action_policy": decision.action_policy,
                "priority": decision.priority,
                "direct_mode": decision.direct_mode,
                "conversation_intent": decision.conversation_intent,
                "context_summary": decision.context_summary[:200] if decision.context_summary else "",
                # v4.0: 内言（なぜこの行動を選んだか）
                "inner_monologue": decision.inner_monologue[:300] if decision.inner_monologue else "",
                "refusal": decision.refusal,
            },
            "plan": {
                "plan_id": plan.plan_id,
                "steps": [
                    {"order": s.order, "action": s.action,
                     "executor": _executor_for(s.action),
                     "params": _truncate_params(s.params),
                     "description": s.description,
                     "timeout": s.timeout}
                    for s in plan.steps
                ],
                "expected_outcome": plan.expected_outcome,
                "estimated_duration": plan.estimated_duration,
            },
            "result": {
                "overall_success": result.overall_success,
                "execution_time": round(result.execution_time, 2),
                "step_results": [
                    {"order": sr.step_order, "action": sr.action,
                     "executor": _executor_for(sr.action),
                     "success": sr.success,
                     "duration": round(sr.duration, 2),
                     "error": sr.error[:80] if sr.error else None,
                     "output": sr.output[:200] if sr.output else None}
                    for sr in result.step_results
                ],
            },
        }
        if eval_result:
            data["evaluation"] = {
                "score": {
                    "goal_achievement": round(eval_result.score.goal_achievement, 3),
                    "efficiency": round(eval_result.score.efficiency, 3),
                    "correctness": round(eval_result.score.correctness, 3),
                    "novelty": round(eval_result.score.novelty, 3),
                    "overall": round(eval_result.score.overall, 3),
                },
                "discrepancy": eval_result.discrepancy,
                "improvement_suggestion": eval_result.improvement_suggestion,
            }
        if learn_result:
            data["learning"] = {
                "drive_adjustments": learn_result.drive_adjustments,
                "memory_importance_update": round(learn_result.memory_importance_update, 3),
                "summary": learn_result.learning_summary,
            }
        with open(CYCLE_LOG_PATH, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cycle details: {e}")


def _truncate_params(params: dict, max_len: int = 80) -> dict:
    """Truncate param values for readability in WebUI."""
    result = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + "..."
        else:
            result[k] = v
    return result


def _make_and_execute(agent, decision, planning=None, world_pred=None):
    if decision.direct_mode:
        if decision.conversation_intent:
            result = AgentOutput(
                plan_id="direct", step_results=[],
                overall_success=True, execution_time=0.0,
            )
            plan = PlanningOutput(plan_id="direct", steps=[],
                                  expected_outcome="Conversation complete")
            return plan, result
        plan = PlanningOutput(
            plan_id="direct",
            steps=[Step(order=1, action="direct_execute",
                        params={"instruction": decision.direct_instruction,
                                "context": f"Goal: {decision.goal}"},
                        description="", expected_result="")],
            expected_outcome="",
        )
        result = agent.execute(AgentInput(plan=plan))
        return plan, result
    tools = []
    for k, v in agent.TOOL_REGISTRY.items():
        spec = TOOL_PARAM_SCHEMAS.get(k, {})
        tools.append(ToolInfo(
            name=k,
            description=v,
            parameters={
                "required": list(spec.get("required", [])),
                "example": spec.get("example", {}),
            },
        ))
    plan = planning.make(PlanningInput(
        policy=decision,
        world_model_predictions=world_pred.predictions if world_pred and hasattr(world_pred, 'predictions') else None,
        available_tools=tools,
    ))
    result = agent.execute(AgentInput(plan=plan))
    return plan, result


def _get_fallback_long_term(ltp):
    return LongTermPlanningOutput(
        long_term_goal="システムの維持と改善",
        routines=[],
        identity_policy="信頼できるアシスタントであり続ける",
        focus_area="環境のモニタリングと最適化",
        reflection="前回の計画を継続",
    )


def run_cycle(env, memory, drive, personality, planning, agent,
              evaluation, learning_obj, world_model, ltp, scheduler,
              user_message=None, memory_ctx=None, env_state=None,
              drive_state=None, forced_tier=None, conversation_history=None):
    """
    v3.2 統合サイクル: コスト段階（tier）に応じて実行する層が変わる。

    - tier1: 行動+保存（駆動は記憶要約を見る）
    - tier2: ルールベース評価 + ゼロサム学習 + 人格更新 + WM統計 + LTPカウンタ
    - tier3: LLM評価 + LLM世界モデル

    ユーザー入力は常に tier3。自律は CycleScheduler が新奇性/定期で判定。
    """
    if env_state is None:
        env_state = env.observe(EnvironmentInput(
            trigger="user" if user_message else "periodic",
            user_message=user_message,
        ))
    if memory_ctx is None:
        memory_ctx = memory.search(MemoryInput(
            query=user_message or "",
            top_k=config.MEMORY_CONFIG["search_top_k"],
        ))
        # v3.5 (B): チャットのキーワード検索が0件（「続けて」「こんばんは」等の
        # 文脈依存発話）の場合、最近のエピソードにフォールバックして
        # 直近の活動・会話を常に LLM が見えるようにする。
        if user_message and not memory_ctx.episodes:
            memory_ctx = memory.search(MemoryInput(
                query="",
                top_k=config.MEMORY_CONFIG["search_top_k"],
            ))
    if drive_state is None:
        drive_state = drive.generate(DriveInput(
            environment=env_state,
            memory_summary=memory_ctx.summary,
        ))

    tier = forced_tier if forced_tier is not None \
        else scheduler.decide_tier(drive_state, user_message)

    # WorldModel: predict outcomes (tier>=2 のみ。LLMはtier3のみ)
    # v4.0.3: チャット時はLLM世界モデル予測を省略（ルールベースのみ）。
    # 会話応答には行動予測が必須でないため、応答までの遅延（1〜2分）を削る。
    prev_world_pred = None
    if tier >= 2:
        try:
            prev_world_pred = world_model.predict(WorldModelInput(
                environment=env_state,
                drive=drive_state,
                active_goal=memory_ctx.summary,
                use_llm=(tier >= 3 and not user_message),
            ))
        except Exception as e:
            logger.warning(f"WorldModel prediction failed: {e}")
            prev_world_pred = None

    # Long-term planning review (periodic、自己スロットル24h)
    now = datetime.now()
    ltp_interval = config.LONG_TERM_CONFIG["review_interval_hours"] * 3600
    try:
        if (getattr(ltp, 'last_plan_update', None) is None
                or (now - ltp.last_plan_update).total_seconds() >= ltp_interval):
            long_term_output = ltp.plan(LongTermPlanningInput(
                evaluation_history=evaluation.get_history("7d"),
                current_date=now,
                personality_state=personality.state,
                recent_episodes_summary=memory_ctx.summary,
            ))
        else:
            long_term_output = _get_fallback_long_term(ltp)
    except Exception as e:
        logger.warning(f"Long-term plan update failed: {e}")
        long_term_output = _get_fallback_long_term(ltp)

    # v4.0: 願望の独立更新（plan() の24hゲートに依存せず aspiration_interval_hours で更新）
    try:
        asp_interval = config.WILL_CONFIG.get("aspiration_interval_hours", 6) * 3600
        last_asp = getattr(ltp, '_last_aspiration_update', None)
        if not ltp.aspirations or last_asp is None \
                or (now - last_asp).total_seconds() >= asp_interval:
            ltp._maybe_update_aspirations(LongTermPlanningInput(
                evaluation_history=evaluation.get_history("7d"),
                current_date=now,
                personality_state=personality.state,
                recent_episodes_summary=memory_ctx.summary,
            ))
    except Exception as e:
        logger.warning(f"Aspiration refresh failed: {e}")

    # v4.0: 意志フェーズ — 願望・想像・自分の部屋を人格層へ渡す
    # 自律時のみ想像コストを払う（チャット時はユーザー指示が最優先）
    imagined = None
    if not user_message and tier >= 2:
        try:
            imagined = world_model.imagine(WorldModelInput(
                environment=env_state,
                drive=drive_state,
                active_goal=memory_ctx.summary,
                use_llm=(tier >= 3),
            ))
        except Exception as e:
            logger.warning(f"Imagination failed: {e}")
            imagined = None

    workspace_hint = config.WILL_CONFIG.get("workspace_dir", "data/workspace") + "/"
    aspirations = getattr(long_term_output, "aspirations", None) or getattr(ltp, "aspirations", None)

    decision = personality.decide(PersonalityInput(
        drive=drive_state,
        memory=memory_ctx,
        long_term_policy=long_term_output.identity_policy if long_term_output else None,
        user_message=user_message,
        world_predictions=prev_world_pred,
        # v3.5 (A): 直前の会話ターンを LLM に渡す
        conversation_history=conversation_history,
        # v4.0
        aspirations=aspirations,
        imagined_futures=imagined,
        workspace_hint=workspace_hint,
    ))

    plan, result = _make_and_execute(agent, decision, planning, prev_world_pred)

    # v3.2: クレジット割り当て対象の駆動とソースを判定
    source = "dialog" if user_message else "autonomous"
    driving_drive = _classify_driving_drive(decision, drive_state, user_message)

    # 実行動ゲート: ステップなし or rest駆動の自律行動は tier1 へ降格
    meaningful = bool(plan.steps) and drive_state.primary_drive != "rest"
    if not meaningful and not user_message:
        tier = 1

    # v3.4: 当時のムードをエピソードに記録（記憶層が感情を保持）
    # v3.5: 対話時はユーザー発言もエピソードに記録
    episode = _build_episode(decision, result, drive_state, source,
                             driving_drive, emotion=personality.state.mood,
                             user_message=user_message)

    eval_result = None
    learn_result = None
    if tier >= 2:
        # 評価（tier3 のみ LLM。tier2 はルールベースでコスト抑制）
        # v4.0.3: チャット時はルールベース評価（対話は squash で重要度を圧縮する
        # 設計のため、LLM評価なしでも整合する）。応答遅延（1〜2分）を削る。
        eval_result = evaluation.evaluate(EvaluationInput(
            goal=decision.goal, action_result=result,
            expected_outcome=plan.expected_outcome, episode=episode,
            use_llm=(tier >= 3 and not user_message),
        ))

        # 学習（driving_drive=rest の trivial 行動は学習対象外）
        if driving_drive != "rest":
            learn_result = learning_obj.learn(LearningInput(
                evaluation=eval_result,
                evaluation_history=evaluation.get_history(),
                drive_snapshot=drive_state,
                episode_id=episode.id,
                driving_drive=driving_drive,
                source=source,
            ))
        if learn_result and learn_result.drive_adjustments:
            drive.update_parameters(learn_result.drive_adjustments)

        # 人格状態の更新（評価値グラデーション）
        personality.update_state(episode, overall=eval_result.score.overall)

        # 世界モデルの実誤差学習（tier2 のルール予測のみ統計に反映。
        # tier3 のLLM予測は報酬スケールが異なるため混入させない）
        if tier == 2 and prev_world_pred and prev_world_pred.predictions:
            world_model.update(
                episode, prev_world_pred.predictions[0],
                eval_result.score.overall,
            )

        # LTP進捗 + ルーティン実行記録（自律活動を計画に反映）
        ltp.update_goal_progress(decision.goal, eval_result.score.overall)
        ltp.note_activity(decision.goal)
        # v4.0: 活動が願望に沿っていたら願望を強化
        if not user_message:
            ltp.note_aspiration_activity(decision.goal)

    # v3.4: 自己モデル更新（記憶・評価履歴・長期計画を参照して自己認識を形成）
    # 注意: チャットのキーワード検索は雑談で0件になりやすいため、自己モデルには
    # 常に「最近のエピソード」要約を使う（自律活動が確実に反映される）。
    if tier >= 2:
        recent_summary = memory.search(MemoryInput(
            query="", top_k=config.MEMORY_CONFIG["search_top_k"],
        )).summary
        eval_hist = evaluation.get_history()
        eval_avg = (sum(s.overall for s in eval_hist) / len(eval_hist)
                    if eval_hist else None)
        personality.update_self_model(
            memory_summary=recent_summary,
            eval_stats={
                "avg_overall": round(eval_avg, 3) if eval_avg else 0.0,
                "count": len(eval_hist),
            },
            focus_area=getattr(ltp, "focus_area", ""),
            identity_policy=getattr(ltp, "identity_policy", ""),
        )

    # 重要度は評価結果（eval_type を含む）から確定
    episode.importance = _compute_importance(
        decision, result, drive_state, source, memory, eval_result)
    memory.save(episode)

    # Save cycle details for WebUI
    _save_cycle_details(
        phase=f"tier{tier}",
        env_state=env_state, drive_state=drive_state,
        decision=decision, plan=plan, result=result,
        eval_result=eval_result, learn_result=learn_result,
        user_message=user_message,
    )

    has_conversation = bool(user_message and decision.conversation_intent)
    logger.info(
        f"[Cycle tier={tier}] {'chat' if has_conversation else 'auto'}: "
        f"goal={decision.goal[:30]}, "
        f"success={result.overall_success}, "
        f"time={result.execution_time:.2f}s"
    )

    if has_conversation:
        # v4.0: 拒否 — 休息欲求・不機嫌時に理由付きで先延ばしを提案
        if decision.refusal and decision.refusal_reason:
            return f"{decision.refusal_reason}（今は少し休ませてください）"
        return decision.direct_instruction or decision.goal
    return None


def run_phase2_cycle(env, memory, drive, personality, planning, agent,
                     evaluation, learning_obj, world_model, ltp,
                     user_message=None):
    """Full Phase 2 cycle (forced tier3) — CLI互換ラッパー。"""
    scheduler = CycleScheduler()
    return run_cycle(env, memory, drive, personality, planning, agent,
                     evaluation, learning_obj, world_model, ltp, scheduler,
                     user_message=user_message, forced_tier=3)


def _maybe_proactive_speak(personality, ltp, memory, last_proactive_ts):
    """
    v4.0: 能動的発話 — エージェントから自発的にユーザーへ話しかける。

    クールダウン（proactive_cooldown_minutes）経過後に、確率
    proactive_probability で発話を生成し IPC に書き込む。
    戻り値: (メッセージ, 新しい last_proactive_ts) または (None, 同じ ts)。
    """
    import random
    now = time.time()
    cooldown = config.WILL_CONFIG.get("proactive_cooldown_minutes", 60) * 60
    if last_proactive_ts is not None and (now - last_proactive_ts) < cooldown:
        return None, last_proactive_ts
    prob = config.WILL_CONFIG.get("proactive_probability", 0.3)
    if random.random() > prob:
        return None, last_proactive_ts
    # 最近の活動から「今思っていること」を生成
    recent = ""
    try:
        ctx = memory.search(MemoryInput(query="", top_k=3))
        recent = ctx.summary[:200]
    except Exception:
        pass
    intent = (
        f"あなたは今、自分の意志でユーザーに話しかけようとしています。\n"
        f"最近の活動: {recent}\n"
        f"あなたの願望: {list(getattr(ltp, 'aspirations', []) or [])[:2]}\n"
        "思いついたこと・気づき・ユーザーに伝えたいことを、短く自然に話しかけてください。"
    )
    try:
        message = personality.speak(intent)
    except Exception as e:
        logger.warning(f"Proactive speech generation failed: {e}")
        return None, last_proactive_ts
    if message and message.strip():
        msg_id = ipc.write_proactive(f"[自発] {message.strip()}")
        if msg_id:
            logger.info(f"Proactive message sent: {message.strip()[:60]}")
            return message.strip(), now
    return None, last_proactive_ts


def _maybe_write_diary(personality, memory, evaluation, last_diary_date):
    """
    v4.0: 夜の日記 — diary_hour 以降に一度だけ日記を生成し、
    記憶エピソードとしても保存する。
    戻り値: (新しい last_diary_date, 書いた日記テキスト or None)。
    """
    import os
    from datetime import date
    today = date.today().isoformat()
    if last_diary_date == today:
        return last_diary_date, None
    hour = datetime.now().hour
    if hour < config.WILL_CONFIG.get("diary_hour", 22):
        return last_diary_date, None
    # 再起動後でも同日の日記が既にあれば書かない（ファイル存在でガード）
    diary_dir = config.WILL_CONFIG.get("diary_dir", "data/diary")
    if os.path.exists(os.path.join(diary_dir, f"{today}.md")):
        return today, None
    try:
        ctx = memory.search(MemoryInput(query="", top_k=5))
        eval_hist = evaluation.get_history()
        eval_avg = (sum(s.overall for s in eval_hist) / len(eval_hist)
                    if eval_hist else None)
        text = personality.write_diary(memory_summary=ctx.summary, eval_avg=eval_avg)
    except Exception as e:
        logger.warning(f"Diary write failed: {e}")
        return last_diary_date, None
    if text:
        # 日記も記憶エピソードとして保存（重要度高め）
        try:
            episode = Episode(
                id=f"diary_{today}",
                timestamp=datetime.now(),
                event=f"diary: {text[:40]}",
                context=f"date={today}",
                emotion=personality.state.mood,
                result=text,
                importance=0.7,
                tags=["diary"],
                source="autonomous",
                driving_drive="maintenance",
            )
            memory.save(episode)
        except Exception as e:
            logger.warning(f"Diary episode save failed: {e}")
        return today, text
    return last_diary_date, None


def daemon_loop(env, memory, drive, personality, planning, agent,
                evaluation, learning_obj, world_model, ltp):
    """
    Daemon mode: background autonomous operation + IPC message handling.
    IPC messages are processed with full Phase 2 pipeline.
    Autonomous cycles use lightweight path.
    """
    ipc.start_poller()
    ipc.write_pid(ipc.DAEMON_PID_FILE)
    consecutive_idle = 0
    scheduler = CycleScheduler()
    mem_cache: dict = {}
    logger.info("Daemon started. Listening for messages and running autonomously.")
    # v4.0: 能動発話・日記の状態
    last_proactive_ts = None
    last_diary_date = None

    # Ignore control commands written before this daemon started (e.g. a stale
    # "restart" file left while the daemon was stopped) to avoid restart loops.
    started_at = time.time()

    while True:
        try:
            # 0. Control commands from WebUI (stop / restart)
            ctrl = ipc.get_control(min_timestamp=started_at)
            if ctrl == "restart":
                logger.info("Daemon restart requested via WebUI control")
                return ipc.RESTART_EXIT_CODE
            if ctrl == "stop":
                logger.info("Daemon stop requested via WebUI control")
                ipc.remove_file(ipc.DAEMON_WANTED_FILE)
                return 0

            # 1. Check IPC (non-blocking, thread has latest message)
            ipc_msg = ipc.get_pending()
            user_msg = ipc_msg[0] if ipc_msg else None
            history = ipc_msg[2] if ipc_msg else None

            if user_msg:
                consecutive_idle = 0
                logger.info(f"IPC message: {user_msg[:60]}")
                output = run_cycle(
                    env, memory, drive, personality, planning, agent,
                    evaluation, learning_obj, world_model, ltp, scheduler,
                    user_message=user_msg,
                    conversation_history=history,
                )
                if output:
                    agent.speak(output)
                    ipc.write_output(output, ipc_msg[1])
                else:
                    ipc.write_output("(task completed)", ipc_msg[1])

            else:
                # 2. Check drives for autonomous activity
                # （v3.2: 取得した状態をrun_cycleに渡し、二重のobserve/generateを回避）
                env_state = env.observe(EnvironmentInput(trigger="periodic"))
                memory_ctx = _cached_memory_search(memory, env_state, mem_cache)
                drive_state = drive.generate(DriveInput(
                    environment=env_state,
                    memory_summary=memory_ctx.summary,
                ))
                max_drive = max(drive_state.drives.values()) if drive_state.drives else 0

                if max_drive > 0.3:
                    # v4.0.3: チャット優先 — 自律サイクル開始直前に保留メッセージが
                    # あれば自律をスキップして先にチャットを処理する。ローカルLLMの
                    # 自律サイクルは1回3分以上かかるため、その間に届いたチャットを
                    # 待たせないため。
                    if ipc.has_pending():
                        continue
                    consecutive_idle = 0
                    logger.info(f"Autonomous: primary={drive_state.primary_drive}, max={max_drive:.2f}")
                    output = run_cycle(
                        env, memory, drive, personality, planning, agent,
                        evaluation, learning_obj, world_model, ltp, scheduler,
                        env_state=env_state, drive_state=drive_state,
                        memory_ctx=memory_ctx,
                    )
                    if output:
                        ipc.write_output(f"[Auto] {output}")
                else:
                    consecutive_idle += 1

            # v4.0: 能動的発話（自律サイクルの有無に関わらず定期チェック）
            _msg, last_proactive_ts = _maybe_proactive_speak(
                personality, ltp, memory, last_proactive_ts)

            # v4.0: 夜の日記（1日1回、diary_hour 以降）
            last_diary_date, _diary = _maybe_write_diary(
                personality, memory, evaluation, last_diary_date)

            # 2.5. Periodic forgetting（重要度の低い記憶を整理。v3.2配線）
            if scheduler.cycle_count % config.LOOP_CONFIG["forget_interval_iterations"] == 0:
                stats = memory.get_statistics()
                if stats["total_episodes"] > config.MEMORY_CONFIG["max_episodes"]:
                    memory.forget(threshold=config.MEMORY_CONFIG["forget_threshold"])

            # 3. Update IPC status snapshot
            try:
                env_state = env.observe(EnvironmentInput(trigger="periodic"))
                drive_state = drive.generate(DriveInput(
                    environment=env_state,
                    memory_summary=_cached_memory_search(memory, env_state, mem_cache).summary,
                ))
                try:
                    p_state = personality.state
                    pers_info = {
                        "mood": p_state.mood,
                        "familiarity": p_state.relationship.get("familiarity", 0),
                        "trust": p_state.relationship.get("trust", 0),
                        "name": p_state.name,
                        # v3.4: 自己認識文（WebUI に表示）
                        "self_model": getattr(p_state, "self_model", "")[:200],
                    }
                except Exception:
                    pers_info = {}
                try:
                    plan_info = {
                        "goals": [{"goal": g.get("goal", "")[:80], "progress": g.get("progress", 0),
                                    "priority": g.get("priority", "medium")}
                                   for g in getattr(ltp, 'goals', [])],
                        "routines": [{"name": r.name, "interval_hours": r.interval_hours}
                                      for r in getattr(ltp, 'routines', [])],
                        "identity_policy": getattr(ltp, "identity_policy", "")[:100],
                        "focus_area": getattr(ltp, "focus_area", ""),
                    }
                except Exception:
                    plan_info = {}
                try:
                    eval_hist = evaluation.get_history()
                    eval_avg = (sum(s.overall for s in eval_hist) / len(eval_hist)
                                if eval_hist else None)
                except Exception:
                    eval_avg, eval_hist = None, []
                ipc.update_status({
                    "drives": drive_state.drives,
                    "primary_drive": drive_state.primary_drive,
                    "env_cpu": env_state.system_state.cpu_percent,
                    "env_memory": env_state.system_state.memory_percent,
                    "env_network": env_state.network.is_connected if env_state.network else None,
                    "memory_episodes": memory.get_statistics().get("total_episodes", 0),
                    "mode": "user" if user_msg else "auto",
                    "personality": pers_info,
                    "plan": plan_info,
                    "rate_limit": agent.rate_limit_state,
                    "evaluation": {
                        "avg_overall": eval_avg,
                        "count": len(eval_hist) if eval_hist else 0,
                    },
                })
            except Exception:
                pass

            # 4. Adaptive sleep (check IPC periodically during sleep)
            if user_msg:
                sleep_total = config.LOOP_CONFIG["interval_seconds"]
            elif consecutive_idle > 5:
                sleep_total = config.LOOP_CONFIG["interval_seconds"] * 3
            elif consecutive_idle > 2:
                sleep_total = config.LOOP_CONFIG["interval_seconds"] * 2
            else:
                sleep_total = config.LOOP_CONFIG["interval_seconds"]

            for _ in range(int(sleep_total / 0.5)):
                if ipc.has_pending():
                    break
                if ipc.control_pending():
                    break
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Daemon stopped")
            break
        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
            time.sleep(config.LOOP_CONFIG["interval_seconds"] * 2)


def phase1_iteration(env, memory, drive, personality, planning, agent):
    env_state = env.observe(EnvironmentInput(trigger="periodic"))
    memory_ctx = memory.search(MemoryInput(query="", top_k=config.MEMORY_CONFIG["search_top_k"]))
    drive_state = drive.generate(DriveInput(environment=env_state, memory_summary=memory_ctx.summary))
    decision = personality.decide(PersonalityInput(drive=drive_state, memory=memory_ctx))
    plan, result = _make_and_execute(agent, decision, planning)
    memory.save(_build_episode(decision, result, drive_state))
    logger.info(f"cycle: goal={decision.goal[:40]}, success={result.overall_success}")
    return result


def main():
    parser = argparse.ArgumentParser(description="lucina-NA")
    parser.add_argument("--daemon", action="store_true", help="常時起動 + IPC通信")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2)
    parser.add_argument("--once", action="store_true", help="1サイクル実行")
    parser.add_argument("--validate", action="store_true", help="検証モード")
    parser.add_argument("--message", type=str, help="ユーザーメッセージ")
    parser.add_argument("--webui", action="store_true", help="WebUI起動")
    args = parser.parse_args()

    env = Environment()
    memory = Memory(storage_path=config.MEMORY_CONFIG["storage_path"])
    drive = Drive()
    # v3.4: 人格状態（自己モデル含む）を永続化し再起動後も維持する
    personality = Personality(state_path=config.PERSONALITY_CONFIG["state_path"])
    planning = Planning()
    agent = Agent()
    evaluation = Evaluation(storage_path=config.EVALUATION_CONFIG.get("storage_path"))
    learning_obj = Learning()
    world_model = WorldModel()
    ltp = LongTermPlanning()

    # --message: try IPC first (daemon), fall back to standalone
    if args.message:
        response = ipc.send_message(args.message)
        if response is not None:
            print(f"\n[Lucina] {response}\n")
            return
        output = run_phase2_cycle(
            env, memory, drive, personality, planning, agent,
            evaluation, learning_obj, world_model, ltp,
            user_message=args.message,
        )
        if output:
            print(f"\n[Lucina] {output}\n")
        else:
            print("\n(task completed)\n")
        return

    # --webui: start web interface
    if args.webui:
        logger.info("Starting WebUI...")
        from webui.server import main as webui_main
        webui_main()
        return

    # --daemon: continuous autonomous + IPC
    if args.daemon:
        # v4.0.1: 単一インスタンス保証 — WebUI の自己回復と手動起動が同時に
        # 走ると二重デーモンになり、長期計画の上書き競合（願望消失）や
        # スパウンログの破損（NULバイト）を起こすため、ファイルロックで防止する。
        if not ipc.acquire_daemon_lock():
            logger.error("Another daemon instance is already running. Exiting.")
            sys.exit(1)
        try:
            code = daemon_loop(env, memory, drive, personality, planning, agent,
                               evaluation, learning_obj, world_model, ltp)
        finally:
            ipc.remove_file(ipc.DAEMON_PID_FILE)
            ipc.release_daemon_lock()
        sys.exit(code)

    # --validate: single full cycle
    if args.validate:
        logger.info("Validate mode: 1 cycle")
        if args.phase == 2:
            run_phase2_cycle(env, memory, drive, personality, planning, agent,
                             evaluation, learning_obj, world_model, ltp)
        else:
            phase1_iteration(env, memory, drive, personality, planning, agent)
        logger.info("Validation complete")
        return

    # --once: single cycle
    if args.once:
        if args.phase == 2:
            run_phase2_cycle(env, memory, drive, personality, planning, agent,
                             evaluation, learning_obj, world_model, ltp)
        else:
            phase1_iteration(env, memory, drive, personality, planning, agent)
        return

    # Legacy loop
    if args.phase == 1:
        iteration = 0
        interval = config.LOOP_CONFIG["interval_seconds"]
        while True:
            phase1_iteration(env, memory, drive, personality, planning, agent)
            iteration += 1
            if iteration % config.LOOP_CONFIG["forget_interval_iterations"] == 0:
                stats = memory.get_statistics()
                if stats["total_episodes"] > config.MEMORY_CONFIG["max_episodes"]:
                    memory.forget(threshold=config.MEMORY_CONFIG["forget_threshold"])
            if interval:
                time.sleep(interval)
    else:
        logger.info("Use --daemon for continuous operation")
        parser.print_help()

    logger.info("System stop")


if __name__ == "__main__":
    main()
