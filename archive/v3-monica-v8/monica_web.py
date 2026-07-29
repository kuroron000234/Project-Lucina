"""
Monica v10 — Phase 4: Minimal Web UI (FastAPI + SSE)
====================================================
MonicaコアエンジンをCLIから切り離し、FastAPI + SSEで
リアルタイムWeb対話を実現する。

起動:
  cd /home/koushi/monica-v3
  source venv/bin/activate
  python3 monica_web.py

→ http://localhost:8000 でアクセス
"""
import json, sys, time, threading, queue, asyncio, os, re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import uvicorn

# ── Monica コアをインポート（ここでモデル読み込みが発生） ──
import monica_v8_hybrid as monica

# ── asyncioイベントキュー（SSE用、非ブロッキング） ──
sse_queue = asyncio.Queue(maxsize=100)

# ── FastAPI アプリ ──
app = FastAPI(title="Monica Web", version="v10-phase4")

# ── 静的ファイル ──
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── グローバル状態 ──
monica_bb = monica.Blackboard()
user_input_queue = queue.Queue()
response_queue = queue.Queue()

think_agent = monica.ThinkAgent(
    monica_bb, monica.world_model, monica.goal_manager,
    monica.meta, monica.notes, monica.curiosity
)
action_agent = monica.ActionAgent(monica_bb)

# 永続状態の読み込み
loaded = monica.load_state(monica_bb.conv, monica_bb.beliefs, monica.steer)
think_count = loaded.get("think_count", 0) if loaded else 0
think_prompt_idx = loaded.get("think_prompt_idx", 0) if loaded else 0

# ── 状態管理 ──
mode = "IDLE"
last_activity = time.time()
idle_cycles = 0
think_cooldown = 0
idle_lock = 0
consecutive_api_failures = 0
last_observations = None
pending_response = {"text": None}

# ── イベントSSEジェネレータ ──
async def event_generator():
    while monica_bb.running:
        try:
            event = await asyncio.wait_for(sse_queue.get(), timeout=2.0)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (asyncio.TimeoutError, asyncio.CancelledError):
            yield ": heartbeat\n\n"


def push_event(event_type, data):
    """SSEイベントを非同期キューに投入"""
    event_data = {"type": event_type, **data, "ts": datetime.now().strftime("%H:%M:%S")}
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(sse_queue.put(event_data), loop)
    except RuntimeError:
        pass


# ── APIエンドポイント ──

@app.get("/")
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse("<h1>Monica Web</h1><p>index.html not found</p>")


@app.get("/api/state")
async def get_state():
    beliefs, conv, internal_log, recent_outputs, _ = monica_bb.read()
    g = monica.goal_manager.active_goal()
    pe_s, kl_s, vfe_s = monica.fep_history.trend()
    profile_data = {}
    if monica.profile.name:
        profile_data["name"] = monica.profile.name
        profile_data["interaction_count"] = monica.profile.interaction_count
        profile_data["topics"] = monica.profile.topics[:5]
    state = {
        "mode": mode,
        "self_model": round(beliefs.self_model, 3),
        "self_drift": round(beliefs.self_drift, 3),
        "pe_low": round(beliefs.low.running_avg, 3),
        "pe_mid": round(beliefs.mid.running_avg, 3),
        "pe_high": round(beliefs.high.running_avg, 3),
        "restoring": round(monica.DRIFT_RESTORING_COEFF, 4),
        "noise_std": round(monica.DRIFT_NOISE_STD, 4),
        "steer_ready": monica.steer.vector is not None,
        "steer_samples": len(monica.steer.high_buffer) + len(monica.steer.low_buffer),
        "fep_trend": {"pe": round(pe_s, 3), "kl": round(kl_s, 3), "vfe": round(vfe_s, 3)},
        "goal": {"id": g.id, "description": g.description[:80], "status": g.status} if g else None,
        "meta_focus": monica.meta.focus[:60] if monica.meta.focus else "",
        "profile": profile_data,
        "think_count": think_count,
        "idle_cycles": idle_cycles,
        "conv_count": len(conv),
        "fep_history_len": len(monica.fep_history.pes),
        "timestamp": datetime.now().isoformat(),
    }
    return JSONResponse(state)


@app.get("/api/events")
async def sse_events(request: Request):
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


class ChatMessage(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(msg: ChatMessage):
    global pending_response
    user_text = msg.message.strip()
    if not user_text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if user_text.lower() in ("exit", "quit"):
        def _shutdown():
            time.sleep(0.5)
            monica_bb.running = False
        threading.Thread(target=_shutdown, daemon=True).start()
        return JSONResponse({"response": "Goodbye."})

    user_input_queue.put(user_text)
    pending_response = {"text": None}
    try:
        resp = response_queue.get(timeout=60)
        return JSONResponse({"response": resp})
    except queue.Empty:
        return JSONResponse({"error": "timeout"}, status_code=504)


# ── バックグラウンド自律ループ ──

def autonomous_loop():
    """Monica自律ループ（バックグラウンドスレッド、例外は全て捕捉）"""
    import traceback

    global mode, idle_cycles, think_cooldown, idle_lock
    global think_count, think_prompt_idx, consecutive_api_failures
    global last_observations, pending_response

    # Phase 9.1: 前回セッションのコンテキストを復元
    monica.load_previous_session_context()

    sensing_agent = monica.SensingAgent(monica_bb)
    sensing_agent.start()
    push_event("system", {"message": "Monica v10 Web started"})

    # Phase 5: 起動時強制THINK（初回のみ）
    startup_think_done = False
    last_activity = time.time()

    while monica_bb.running:
        try:
            beliefs, conv, internal_log, recent_outputs, _ = monica_bb.read()
            sm = beliefs.self_model
            user_active = not user_input_queue.empty()
            recent_div = 1.0
            if len(recent_outputs) >= 3:
                recent_div = monica.diversity_score(recent_outputs[-5:])

            intended_mode = monica.decide_mode(beliefs, recent_div, user_active, idle_cycles)
            if intended_mode == "THINK":
                if think_cooldown > 0 or idle_lock > 0:
                    intended_mode = "IDLE"

            # Phase 5: 時間ベースTHINKトリガー（IDLE_TO_THINK_TIMEOUT秒無活動）
            if (not user_active and mode == "IDLE" and idle_cycles >= 5
                and time.time() - last_activity > monica.IDLE_TO_THINK_TIMEOUT
                and think_cooldown == 0 and idle_lock == 0):
                intended_mode = "THINK"
                push_event("system", {"message": f"Time trigger: {monica.IDLE_TO_THINK_TIMEOUT}s inactivity → THINK"})

            # Phase 5: 起動時強制THINK
            if not startup_think_done and monica.FORCE_THINK_ON_STARTUP and not user_active:
                if idle_cycles >= 2:  # 起動後少し待ってから
                    intended_mode = "THINK"
                    startup_think_done = True
                    push_event("system", {"message": "Startup force THINK"})

            if intended_mode != mode:
                old_mode = mode
                if intended_mode in ("CHAT", "THINK") and mode == "IDLE":
                    beliefs.reset_drift()
                mode = intended_mode
                monica_bb.set_mode(mode)
                idle_cycles = 0
                if mode == "CHAT" or mode == "THINK":
                    last_activity = time.time()
                push_event("mode", {"from": old_mode, "to": mode, "self": round(sm, 3)})

            # ─── CHAT ───
            if mode == "CHAT" and user_active:
                idle_cycles = 0
                last_activity = time.time()
                try:
                    u = user_input_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                if u.startswith("/"):
                    if u == "/s":
                        push_event("status", beliefs.state())
                    continue

                conv_msgs = list(conv)
                inner_ctx = monica.notes.context()
                profile_ctx = monica.profile.context()
                framed_input = f"User said: {u}\n\nReply naturally in their language."
                if profile_ctx:
                    framed_input = profile_ctx + "\n" + framed_input
                if inner_ctx:
                    framed_input = inner_ctx + "\n" + framed_input

                conv_msgs.append({"role": "user", "content": framed_input})
                monica.profile.update_from_message(u)
                monica.curiosity.add_interest(u)
                monica.curiosity.extract_topics(u)
                push_event("user", {"text": u})

                full_text = monica.tokenizer.apply_chat_template(
                    conv_msgs, tokenize=False, add_generation_prompt=True
                )
                beliefs = monica.sense(full_text, beliefs)
                t0 = time.time()
                resp = monica.api_chat(conv_msgs, beliefs=beliefs, max_tokens=200)
                source = "api"
                n_tok = 0

                if resp is None or not resp:
                    consecutive_api_failures += 1
                    push_event("system", {"message": f"API fail x{consecutive_api_failures}"})
                    local_resp, token_log, beliefs = monica.generate_chat_local(
                        conv_msgs, beliefs, max_new=monica.FALLBACK_MAX_NEW, temp=0.8
                    )
                    resp = local_resp
                    source = "local_fallback"
                    if token_log:
                        n_tok = len(token_log)
                else:
                    consecutive_api_failures = 0

                dt = time.time() - t0
                if resp is None:
                    resp = "…"
                monica_bb.write_conv({"role": "user", "content": framed_input})
                monica_bb.write_conv({"role": "assistant", "content": resp})
                monica_bb.write_output(resp)
                monica_bb.set_beliefs(beliefs)
                pending_response["text"] = resp
                response_queue.put(resp)
                push_event("assistant", {"text": resp[:500], "source": source,
                                          "time": round(dt, 1),
                                          "self": round(beliefs.self_model, 3)})
                monica.log_state(beliefs, "chat", u, resp, n_tok, source=source)

            # ─── THINK ───
            elif mode == "THINK":
                think_count += 1
                if think_cooldown > 0:
                    think_cooldown -= 1
                if idle_lock > 0:
                    idle_lock -= 1

                if monica.meta.should_refocus(think_count):
                    monica.meta.generate_focus(beliefs, monica.curiosity, recent_outputs, think_count)

                think_msgs = [{"role": "system", "content": "You are Monica, an introspective AI."}]
                if len(conv) > 2:
                    think_msgs.extend(conv[-3:])

                # Phase 7: ツール未使用連続→自動SEARCH注入（別メッセージとして追加）
                search_hint = think_agent.maybe_inject_search(think_count)
                if search_hint:
                    think_msgs.append({"role": "user", "content": search_hint})

                # Phase 8: API節約 — THINKはまずローカル生成を試す
                think_msgs_for_local = list(think_msgs)
                local_thought, token_log, beliefs_local = monica.generate_chat_local(
                    think_msgs_for_local, beliefs, max_new=monica.THINK_TOKENS, temp=0.9
                )
                if local_thought and len(local_thought) > 10:
                    thought = local_thought
                    push_event("system", {"message": f"Local THINK ({len(local_thought)} chars)"})
                else:
                    # ローカル失敗→API
                    thought = think_agent.think(
                        think_msgs, think_count, think_prompt_idx,
                        observations=last_observations
                    ) or ""
                think_prompt_idx += 1

                if not thought:
                    mode = "IDLE"
                    monica_bb.set_mode("IDLE")
                    time.sleep(1)
                    continue
                push_event("think", {"text": thought[:300], "count": think_count})
                monica.world_model.add_thought(thought)
                monica.notes.add(thought)

                observations = action_agent.execute_and_observe(thought, think_msgs)
                # Phase 7: ツール使用状況を記録
                think_agent.record_tool_use(len(observations) > 0)

                # 目標連動: ツール未使用かつ目標あり→自動SEARCH実行
                if not observations:
                    search_query = monica.auto_search_from_goal(monica.goal_manager, observations, thought)
                    if search_query:
                        push_event("system", {"message": f"Auto SEARCH from goal: {search_query[:60]}"})
                        print(f"  [auto-search] from goal: {search_query[:60]}")
                        # SEARCHを直接実行（execute_and_observe経由）
                        fake_thought = f"[SEARCH: {search_query}]"
                        search_obs = action_agent.execute_and_observe(fake_thought, think_msgs)
                        if search_obs:
                            observations = search_obs
                            # 検索結果で再センシング
                            full_text = monica.tokenizer.apply_chat_template(
                                think_msgs, tokenize=False, add_generation_prompt=True
                            )
                            beliefs = monica.sense(full_text, beliefs)
                            push_event("tool", {"results": [
                                {"type": "auto_search", "target": search_query[:60],
                                 "result": search_obs[0]["result"][:100] if search_obs else ""}
                            ]})

                if observations:
                    push_event("tool", {"results": [
                        {"type": o["type"], "target": o["target"], "result": o["result"][:100]}
                        for o in observations
                    ]})
                    last_observations = observations
                else:
                    last_observations = None

                created = monica.artifact_extractor.extract(thought)
                if created:
                    push_event("artifact", {"files": [
                        {"name": Path(c["path"]).name, "lang": c["lang"], "size": c["size"]}
                        for c in created
                    ]})

                if think_count % 3 == 0:
                    monica.goal_manager.evaluate_progress(beliefs, recent_outputs)
                    monica.goal_manager.retire_stale_goals(think_count)  # Phase 6
                    g = monica.goal_manager.active_goal()
                    if g:
                        push_event("goal", {"id": g.id, "desc": g.description[:60],
                                              "status": g.status})

                kl_val = getattr(monica.sense, "_last_kl", 0.0)
                if kl_val is None or (isinstance(kl_val, float) and kl_val != kl_val):
                    kl_val = 0.0
                vfe_val, acc, comp = monica.compute_vfe(
                    beliefs.mid.running_avg, kl_val, kl_val, beliefs.self_model
                )
                monica.fep_history.add(beliefs.mid.running_avg, kl_val, vfe_val)
                push_event("fep", {"pe": round(beliefs.mid.running_avg, 3),
                                    "kl": round(kl_val, 3), "vfe": round(vfe_val, 3),
                                    "self": round(beliefs.self_model, 3)})
                monica.adapt_fep.update(beliefs.self_model)

                if not observations or think_agent.chain_count >= think_agent.max_chain:
                    if last_observations is not None:
                        think_agent.chain_count = 0
                        last_observations = None
                    mode = "IDLE"
                    monica_bb.set_mode("IDLE")

                if think_count % 10 == 0:
                    monica.save_state(monica_bb.conv, beliefs, monica.steer,
                                      think_count, think_prompt_idx, monica_bb.internal_log)
                time.sleep(0.3)

            # ─── IDLE ───
            else:
                idle_cycles += 1
                if think_cooldown > 0:
                    think_cooldown -= 1
                if idle_lock > 0:
                    idle_lock -= 1
                beliefs.drift_update(
                    beliefs.mid.running_avg,
                    noise_mult=monica.IDLE_DRIFT_NOISE_MULT,
                    inv_temp=monica.DRIFT_INV_TEMP_COEFF,
                    noise_std=monica.DRIFT_NOISE_STD,
                )
                monica_bb.set_beliefs(beliefs)
                if idle_cycles % 20 == 0:
                    monica.save_state(monica_bb.conv, beliefs, monica.steer,
                                      think_count, think_prompt_idx, monica_bb.internal_log)
                    push_event("fep", {"pe": round(beliefs.mid.running_avg, 3),
                                        "self": round(beliefs.self_model, 3),
                                        "drift": round(beliefs.self_drift, 3)})
                time.sleep(0.5)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(f"  [loop error] {error_msg}")
            traceback.print_exc()
            push_event("system", {"message": f"Loop error: {error_msg}"})
            time.sleep(1.0)

    # Phase 9.1: セッションサマリーを保存
    mode_count = {"CHAT": len(monica_bb.conv) // 2}  # user+assistantペア
    monica.save_session_summary(monica_bb.beliefs, think_count, mode_count)
    monica.save_state(monica_bb.conv, monica_bb.beliefs, monica.steer,
                      think_count, think_prompt_idx, monica_bb.internal_log)
    push_event("system", {"message": "Monica shutting down..."})


# ── 起動 ──

@app.on_event("startup")
async def startup():
    thread = threading.Thread(target=autonomous_loop, daemon=True)
    thread.start()


def main():
    print(f"""
  Monica v10 — Phase 4: Web UI (FastAPI + SSE)
  → http://localhost:8000
    """)
    for p in monica.API_PROVIDERS:
        print(f"    API: {p['name']} → {p['model']}")
    print(f"    World Model: {len(monica.world_model.episodes)} episodes")
    print(f"    Goals: {len(monica.goal_manager.goals)} total")
    print(f"    Profile: {monica.profile.name or '(unknown)'}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
