#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Monika VitalOS ==="
echo "起動: $(date)"

# Cleanup on exit
cleanup() {
    echo "終了処理中..."
    kill $SIM_PID 2>/dev/null
    kill $BOT_PID 2>/dev/null
    wait
    echo "終了: $(date)"
}
trap cleanup EXIT INT TERM

# Start simulation (realtime + daemon mode)
python3 src/monica_core/simulate_llm.py --resume --realtime --daemon &
SIM_PID=$!
echo "📟 シミュレーション PID: $SIM_PID"

# Wait for simulation to initialize
sleep 3

# Start Telegram bot
python3 src/monica_core/telegram_bot.py &
BOT_PID=$!
echo "🤖 Telegram bot PID: $BOT_PID"

wait
