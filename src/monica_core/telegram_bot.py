"""Telegram bridge for Monika — chat-style, no commands needed."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from .phone import PhoneMessage, add as phone_add, load as phone_load, mark_read as phone_mark_read

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATE_PATH = DATA_DIR / "state.json"

BOT_TOKEN = os.environ.get("MONIKA_TELEGRAM_TOKEN", "")
ALLOWED_USERS = {int(u) for u in os.environ.get("MONIKA_TELEGRAM_USERS", "").split(",") if u}

_last_monika_ts: dict[int, str] = {}


def _load_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


async def _check_auth(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    if not ALLOWED_USERS or uid in ALLOWED_USERS:
        return True
    await update.message.reply_text("許可されていません")
    return False


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_auth(update):
        return
    await update.message.reply_text(
        "モニカと繋がったよ 📱\n\n"
        "そのままメッセージを送るとモニカに届くよ。\n"
        "モニカからの返信は自動で通知される。\n\n"
        "コマンド:\n"
        "/status — モニカの状態を確認"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_auth(update):
        return
    state = _load_state()
    if not state:
        await update.message.reply_text("まだシミュレーションが起動してないみたい")
        return
    s = state.get("state", {})
    room = state.get("current_room", "?")
    activity = state.get("current_activity") or "何もしてない"
    sim_time = state.get("time", "?")[11:16] if len(state.get("time", "")) > 16 else "?"
    history = state.get("history", [])
    last_acts = history[-5:] if history else []
    memory = _load_json(MSG_PATH.parent / "memory_store.json")
    mem_count = len(memory) if isinstance(memory, list) else len(memory.get("entries", [])) if isinstance(memory, dict) else 0
    lines = [
        f"🕐 {sim_time}",
        f"📍 {room}",
        f"📖 {activity}",
        f"⚡ {s.get('energy', '?'):.0f} 🍽️ {s.get('hunger', '?'):.0f} 😴 {s.get('fatigue', '?'):.0f}",
        f"💔 {s.get('loneliness', '?'):.0f} 😊 {s.get('spirit', '?'):.0f}",
        f"🧠 {mem_count}件の記憶",
    ]
    if last_acts:
        lines.append("")
        lines.append("直近の行動:")
        for e in last_acts:
            lines.append(f"  [{e.get('time','?')}] {e.get('activity','?')}")
    await update.message.reply_text("📊 モニカの状態\n\n" + "\n".join(lines))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_auth(update):
        return
    text = update.message.text.strip()
    if not text:
        return

    now = datetime.now().isoformat()

    # モニカからのメッセージを既読に
    phone_mark_read("monika")

    # phone.add 経由でユーザーメッセージを追加
    phone_add("user", text, now)


async def push_monika_messages(context: ContextTypes.DEFAULT_TYPE):
    """Periodically push new Monika messages to Telegram."""
    for chat_id in ALLOWED_USERS:
        try:
            msgs = phone_load()
            monika_msgs = [m for m in msgs if m.sender == "monika"]
            if not monika_msgs:
                continue
            last_ts = monika_msgs[-1].timestamp
            seen = _last_monika_ts.get(chat_id, "")
            if last_ts > seen:
                for m in monika_msgs:
                    if m.timestamp > seen:
                        await context.bot.send_message(chat_id=chat_id, text=m.text[:500])
                _last_monika_ts[chat_id] = last_ts
        except Exception:
            pass


def run_bot():
    if not BOT_TOKEN:
        print("MONIKA_TELEGRAM_TOKEN が設定されていません")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if ALLOWED_USERS:
        app.job_queue.run_repeating(push_monika_messages, interval=5, first=3)

    print("🤖 Telegram bot 起動（チャットモード）")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
