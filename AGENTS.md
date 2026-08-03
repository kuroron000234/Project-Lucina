# lucina-NA

10-layer autonomous AI agent inspired by the Free Energy Principle (FEP) / active inference.
Runs fully locally with Ollama + Gemma 4 on RTX 4060 (8GB VRAM).

Note on FEP terminology: the architecture is **inspired by** FEP / active inference
(see docs/SPECIFICATION.md 1.2). The only mathematically genuine FEP component is the
surprise layer (v5.0): measured prediction error S = (x-mu)^2/sigma^2 + ln(sigma)
computed from world-model predictions vs. actual evaluation outcomes, fed back into
drive novelty, learning rate, and action selection. The remaining layers use FEP as a
design metaphor, not as variational inference.

## Architecture

Each layer has `interface.py` (dataclass IO) and `layer.py` (implementation):

| Layer | File | Responsibility |
|-------|------|----------------|
| Environment | `environment/environment.py` | Observe system state (CPU, memory, files, user input) |
| Drive | `core/drive/drive.py` | Generate drive values (exploration, social, achievement, rest, maintenance) |
| Personality | `core/personality/personality.py` | Decide what to do based on drives + memory + user input |
| Planning | `core/planning/planning.py` | Break goal into concrete steps |
| Agent | `core/agent/agent.py` | Execute plan steps via tools |
| World Model | `core/world_model/world_model.py` | Predict outcomes of actions |
| Memory | `core/memory/memory.py` | Store/retrieve episodes via keyword search |
| Evaluation | `core/evaluation/evaluation.py` | Score goal achievement |
| Learning | `core/learning/learning.py` | Adjust drives/parameters from evaluation |
| Long-term Planning | `core/long_term_planning/long_term_planning.py` | Update long-term goals and routines |

## Commands

- `./lucina.sh` — one-click launcher: daemon + WebUI (supervised, auto-restart)
- `python main.py --message "hello"` — one-shot user interaction
- `python main.py --daemon` — background autonomous + IPC mode
- `python main.py --webui` — start WebUI (FastAPI on port 8765)
- `python main.py --validate --phase 2` — run one full Phase 2 cycle
- `python main.py --phase 1` — start Phase 1 loop
- `.venv/bin/python webui/server.py` — start WebUI directly
- `python -m pytest tests/ -v --tb=short` — run all 97 tests

## One-click launch & process control

- `./lucina.sh` supervises both daemon and WebUI. Processes exit with code 42 to request a restart; the supervisor restarts them automatically.
- Wanted flags (`data/run/daemon.wanted`, `data/run/webui.wanted`) decide whether the supervisor keeps a process alive. Removing one stops the process; touching it starts it.
- WebUI コントロールタブ: デーモン/UI の起動・停止・再起動が可能。制御ファイルは `data/ipc/control.json`（stop/restart）、PID は `data/run/*.pid`。
- `lucina.desktop` — desktop shortcut; install with `cp lucina.desktop ~/.local/share/applications/`

## WebUI

FastAPI-based web interface with 5 tabs:
- **Chat** — WebSocket conversation with Lucina (full Phase 2 pipeline)
- **Status** — real-time drive gauges, personality stats, environment, memory
- **Memory** — episode browser with importance filter
- **Logs** — real-time SSE log stream with text filter
- **Plan** — long-term goals, routines, identity policy

`python main.py --webui` then open http://127.0.0.1:8765

## Model

- Model: `gemma4:e4b` (Gemma 4 MoE, 4.5B active, 9.6B total)
- VRAM: ~4.3 GB
- Ollama endpoint: `http://localhost:11434/v1`
- `think: False` is NOT needed (Gemma 4 uses opt-in `<|think|>` tokens)

## Conventions

- Always write English comments and code (user communicates in Japanese, but code/doc is English)
- No emoji in code output
- `interface.py` defines dataclasses; `layer.py` contains logic
- Personality outputs `direct_mode: true/false` and `conversation_intent` for conversational routing
- When `direct_mode=true` and `conversation_intent` exists, the response is spoken to user (not executed via Opencode)

## File structure

lucina-NA/
  config.py
  main.py
  AGENTS.md
  .opencode/opencode.json
  core/
    agent/agent.py, interface.py, opencode_bridge.py
    drive/drive.py, interface.py
    personality/personality.py, interface.py
    planning/planning.py, interface.py
    world_model/world_model.py, interface.py
    memory/memory.py, interface.py
    evaluation/evaluation.py, interface.py
    learning/learning.py, interface.py
    long_term_planning/long_term_planning.py, interface.py
  environment/environment.py, interface.py
  tests/
  data/episodes/
  docs/
