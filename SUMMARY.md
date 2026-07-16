## Objective
- Build a physically grounded virtual world (VitalOS) where an LLM agent (Monika) inhabits a body with realistic metabolism, fatigue, spatial environment, and asynchronous messaging, communicating via OpenCode Zen API (free), now with memory, real reading from Aozora Bunko, and Telegram bridge for real-time chat.

## Important Details
- LLM must NOT handle state or numerical simulation; it plans, proposes activities via tags, and generates Japanese dialogue
- All inference via OpenCode Zen API (OpenAI-compatible), default model = `deepseek-v4-flash-free`
- No LoRA training (budget $0)
- GitHub: `kuroron000234/Project-Lucina`, pushed with latest changes
- Japanese SYSTEM_PROMPT defines Monika personality (casual, friendly tone)
- Phone message store (`data/phone_messages.json`) shared between simulation and all clients
- Memory store at `data/memory_store.json` (Zen API embeddings fallback to trigram hash)
- Reading state at `data/reading_state.json` (Aozora Bunko book progress)
- Telegram bot token and user ID configured on RPi (new bot: @Monika_lucina_bot)
- smolagents `CodeAgent` works with `flatten_messages_as_text=True` (Zen API does NOT support tool_calls/function calling natively, so `ToolCallingAgent` blocked)
- RPi at 192.168.1.33, SSH key auth set up, systemd service `monika.service` enabled
- `.env` on RPi has all keys: OPENCODE_ZEN_API_KEY, MONIKA_TELEGRAM_TOKEN, MONIKA_TELEGRAM_USERS
- Relative imports in simulate_llm.py require running as module: `PYTHONPATH=src:$PYTHONPATH python3 -m monica_core.simulate_llm`
- `data/` is gitignored (runtime state not in repo)
- `.env.example` exists for fresh clones

## Work State
### Completed
- **Prompt bias fix**: SYSTEM_PROMPT removed specific examples (朝の光, コーヒー, ショパン); SEND_PROMPT removed "詩的で" instruction, added "等身大の口調"; _generate_reply prompt changed to "堅苦しくなく、友達に話すように"
- **Graceful shutdown**: SIGINT/SIGTERM signal handlers save state before exit
- **Idle spiral fix**: `idle` removed from `_simple_choices_str()` (LLM can't choose it, only fallback on parse failure)
- **Realtime mode**: `--realtime` flag adds `time.sleep()` between ticks to sync simulation to wall clock; 15-min tick = 15 real minutes
- **Memory system** (`memory.py`): Memory class with Zen API embeddings (fallback to trigram hash), cosine similarity search, JSON persistence; auto-saves on every `add()`; context injected into CHOICE_PROMPT, SEND_PROMPT, MONOLOGUE_PROMPT, and _generate_reply
- **Aozora Bunko reader** (`readers.py`): Fetches works from 9 authors (漱石/太宰/鷗外/賢治/芥川/谷崎/乱歩/中島/梶井) via card pages → cp932 text → clean HTML; ReadingHandler tracks state (book, position, total_chars) with `reading_state.json`; picks longest available work >2000 chars; CHARS_PER_HOUR = 10000
- **Reading integration**: `start_activity` override calls `_do_reading()` → fetches chunk → LLM generates reaction → saved to memory; progress persists across sessions
- **smolagents integration**: CodeAgent with `flatten_messages_as_text=True` works with Zen API; `_get_tools()` defines `current_status`, `memory_search`, `reading_progress`; agent infrastructure ready but `_llm_decide` currently uses plain `_call_llm` (tools not needed for decisions)
- **Telegram bot** (`telegram_bot.py`): No-command chat mode — any text message auto-sends to Monika via `phone_messages.json`; `/status` for vitals; `/start` for help; Monika's outgoing messages pushed to Telegram every 5 seconds via `job_queue.run_repeating`; read receipts sync on user message; `.env` loading for token/user config
- **Raspberry Pi setup**: Pi 4 aarch64, Python 3.13.5; repo cloned; pip/smolagents/python-telegram-bot[job-queue] installed; `.env` configured with API key + Telegram token + user ID
- **Systemd service**: `monika.service` running simulation with `--resume --realtime --daemon`; PYTHONPATH set via Environment=; auto-starts on boot
- **Git push**: All changes committed and pushed (`d2663b4`)
- **`time` import fix**: `import time as _time` in `simulate_living()` for `_time.monotonic()` and `_time.sleep()`
- **Duplicate `start_activity` removed**: Two versions existed (one with reading hook at 346, one bare at 590); latter removed

### Active (Running on RPi)
- **Simulation** (PID 3019, systemd): `python3 -m monica_core.simulate_llm --resume --realtime --daemon`
- **Telegram bot** (PID 3434, nohup): `python3 -u -m monica_core.telegram_bot` → log at `bot.log`

### Blocked
- **ToolCallingAgent**: Zen API / deepseek-v4-flash-free does not support native `tool_calls` (function calling); only CodeAgent with flattened text works
- **Running as script fails**: relative imports (`from .vital_os import`) require running as module, not direct `python3 src/...py`
- **No GitHub token/SSH on Pi**: requires manual `scp` for updates until SSH key is added to GitHub

## Next Move
1. Send a test message to @Monika_lucina_bot on Telegram → verify `phone_messages.json` created and Monika replies
2. Check bot.log after test message
3. If all works, add Telegram bot to systemd service or create a second unit for it
4. Optionally: add SSH deploy key for `git push`-based updates to Pi

## Relevant Files
- `/home/koushi/lucina/src/monica_core/simulate_llm.py`: All prompt templates, ConsciousVitalOS with memory/reader/agent integration, graceful shutdown, realtime mode
- `/home/koushi/lucina/src/monica_core/memory.py`: Memory class (Zen API embeddings, cosine search, JSON persistence)
- `/home/koushi/lucina/src/monica_core/readers.py`: ReadingHandler (Aozora Bunko fetcher, progress tracking)
- `/home/koushi/lucina/src/monica_core/telegram_bot.py`: Telegram chat bridge (polling, no-command, push notifications)
- `/home/koushi/lucina/src/monica_core/phone.py`: Shared message store (data/phone_messages.json)
- `/home/koushi/lucina/src/monica_core/vital_os.py`: Core simulation (VitalState, TAG_EFFECTS, DRIFT_RATES, LOCATIONS with BFS)
- `/home/koushi/lucina/.env`: API keys + Telegram config (gitignored)
- `/home/koushi/lucina/.env.example`: Template for fresh clones
- `/home/koushi/lucina/run.sh`: Launch script for simulation + bot (needs PYTHONPATH fix)
- `/etc/systemd/system/monika.service`: Systemd unit for auto-start on RPi
