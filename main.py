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
import time
from datetime import datetime

os.makedirs("data/episodes", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/ipc", exist_ok=True)

import config
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
from core.agent.interface import AgentInput, AgentOutput
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


def _build_episode(decision, result, drive_state):
    return Episode(
        id=str(time.time()),
        timestamp=datetime.now(),
        event=f"goal={decision.goal}",
        context=decision.context_summary,
        emotion="",
        result=f"success={result.overall_success}, steps={len(result.step_results)}",
        importance=0.5,
        tags=[drive_state.primary_drive, decision.action_policy[:20]],
    )


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
    tools = [ToolInfo(name=k, description=v, parameters={}) for k, v in agent.TOOL_REGISTRY.items()]
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


def run_phase2_cycle(env, memory, drive, personality, planning, agent,
                     evaluation, learning_obj, world_model, ltp,
                     user_message=None):
    """Full Phase 2 cycle with all layers. Returns output text or None."""
    env_state = env.observe(EnvironmentInput(
        trigger="user" if user_message else "periodic",
        user_message=user_message,
    ))
    memory_ctx = memory.search(MemoryInput(
        query=user_message or "",
        top_k=config.MEMORY_CONFIG["search_top_k"],
    ))
    drive_state = drive.generate(DriveInput(
        environment=env_state,
        memory_summary=memory_ctx.summary,
    ))

    # WorldModel: predict outcomes (may fail, continue anyway)
    try:
        prev_world_pred = world_model.predict(WorldModelInput(
            environment=env_state,
            drive=drive_state,
            active_goal=memory_ctx.summary,
        ))
    except Exception as e:
        logger.warning(f"WorldModel prediction failed: {e}")
        prev_world_pred = None

    # Long-term planning review (periodic)
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

    decision = personality.decide(PersonalityInput(
        drive=drive_state,
        memory=memory_ctx,
        long_term_policy=long_term_output.identity_policy if long_term_output else None,
        user_message=user_message,
        world_predictions=prev_world_pred,
    ))

    plan, result = _make_and_execute(agent, decision, planning, prev_world_pred)

    episode = _build_episode(decision, result, drive_state)

    # Full pipeline: Evaluation + Learning (both chat & autonomous)
    eval_result = evaluation.evaluate(EvaluationInput(
        goal=decision.goal, action_result=result,
        expected_outcome=plan.expected_outcome, episode=episode,
    ))
    learn_result = learning_obj.learn(LearningInput(
        evaluation=eval_result,
        evaluation_history=evaluation.get_history(),
        drive_snapshot=drive_state,
        episode_id=episode.id,
    ))

    if learn_result.drive_adjustments:
        drive.update_parameters(learn_result.drive_adjustments)
    if learn_result.memory_importance_update:
        memory.update_importance(episode.id, learn_result.memory_importance_update)
    ltp.update_goal_progress(decision.goal, eval_result.score.overall)

    memory.save(episode)

    # Save cycle details for WebUI
    _save_cycle_details(
        phase="phase2",
        env_state=env_state, drive_state=drive_state,
        decision=decision, plan=plan, result=result,
        eval_result=eval_result, learn_result=learn_result,
        user_message=user_message,
    )

    has_conversation = bool(user_message and decision.conversation_intent)
    logger.info(
        f"[Phase2] {'chat' if has_conversation else 'cycle'}: "
        f"goal={decision.goal[:30]}, "
        f"success={result.overall_success}, "
        f"overall={eval_result.score.overall:.2f}, "
        f"time={result.execution_time:.2f}s"
    )

    if has_conversation:
        return decision.direct_instruction or decision.goal
    return None


def run_lightweight_cycle(env, memory, drive, personality, planning, agent):
    """Lightweight autonomous cycle (no evaluation/learning/world_model)."""
    env_state = env.observe(EnvironmentInput(trigger="periodic"))
    drive_state = drive.generate(DriveInput(
        environment=env_state,
        memory_summary="",
    ))
    decision = personality.decide(PersonalityInput(
        drive=drive_state,
        memory=memory.search(MemoryInput(query="", top_k=3)),
        user_message=None,
    ))
    plan, result = _make_and_execute(agent, decision, planning)
    memory.save(_build_episode(decision, result, drive_state))

    # Save cycle details for WebUI
    _save_cycle_details(
        phase="lightweight", env_state=env_state, drive_state=drive_state,
        decision=decision, plan=plan, result=result,
    )

    logger.info(
        f"[Auto] {decision.goal[:40]} | "
        f"success={result.overall_success}, time={result.execution_time:.2f}s"
    )
    if decision.conversation_intent:
        return decision.direct_instruction or decision.goal
    return None


def daemon_loop(env, memory, drive, personality, planning, agent,
                evaluation, learning_obj, world_model, ltp):
    """
    Daemon mode: background autonomous operation + IPC message handling.
    IPC messages are processed with full Phase 2 pipeline.
    Autonomous cycles use lightweight path.
    """
    ipc.start_poller()
    consecutive_idle = 0
    logger.info("Daemon started. Listening for messages and running autonomously.")

    while True:
        try:
            # 1. Check IPC (non-blocking, thread has latest message)
            ipc_msg = ipc.get_pending()
            user_msg = ipc_msg[0] if ipc_msg else None

            if user_msg:
                consecutive_idle = 0
                logger.info(f"IPC message: {user_msg[:60]}")
                output = run_phase2_cycle(
                    env, memory, drive, personality, planning, agent,
                    evaluation, learning_obj, world_model, ltp,
                    user_message=user_msg,
                )
                if output:
                    agent.speak(output)
                    ipc.write_output(output, ipc_msg[1])
                else:
                    ipc.write_output("(task completed)", ipc_msg[1])

            else:
                # 2. Check drives for autonomous activity
                env_state = env.observe(EnvironmentInput(trigger="periodic"))
                drive_state = drive.generate(DriveInput(
                    environment=env_state,
                    memory_summary="",
                ))
                max_drive = max(drive_state.drives.values()) if drive_state.drives else 0

                if max_drive > 0.3:
                    consecutive_idle = 0
                    logger.info(f"Autonomous: primary={drive_state.primary_drive}, max={max_drive:.2f}")
                    output = run_lightweight_cycle(env, memory, drive, personality, planning, agent)
                    if output:
                        ipc.write_output(f"[Auto] {output}")
                else:
                    consecutive_idle += 1

            # 3. Update IPC status snapshot
            try:
                env_state = env.observe(EnvironmentInput(trigger="periodic"))
                drive_state = drive.generate(DriveInput(
                    environment=env_state, memory_summary="",
                ))
                try:
                    p_state = personality.state
                    pers_info = {
                        "mood": p_state.mood,
                        "familiarity": p_state.relationship.get("familiarity", 0),
                        "trust": p_state.relationship.get("trust", 0),
                        "name": p_state.name,
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
    personality = Personality()
    planning = Planning()
    agent = Agent()
    evaluation = Evaluation()
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
        daemon_loop(env, memory, drive, personality, planning, agent,
                    evaluation, learning_obj, world_model, ltp)
        return

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
