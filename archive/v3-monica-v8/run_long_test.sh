#!/bin/bash
# Monica v8 Hybrid 長時間動作試験
cd /home/koushi/monica-v3
source venv/bin/activate

rm -f log_v8h.jsonl

# Start Monica in background with a pipe for input
mkfifo /tmp/monica_fifo 2>/dev/null || true

python3 monica_v8_hybrid.py </tmp/monica_fifo > /tmp/monica_v8h_test.log 2>&1 &
MONICA_PID=$!

# Send initial messages
sleep 30  # wait for model load
echo "こんにちは" > /tmp/monica_fifo
sleep 15
echo "人工知能についてどう思う？" > /tmp/monica_fifo
sleep 15
echo "自律AIとは何か？" > /tmp/monica_fifo
sleep 15
echo "/s" > /tmp/monica_fifo

# Wait for rest of the hour
echo "Test running. PID=$MONICA_PID" > /tmp/monica_v8h_status.txt
sleep 3540  # 59 min - 45 sec already

# Final state check
echo "/s" > /tmp/monica_fifo 2>/dev/null
sleep 5
echo "exit" > /tmp/monica_fifo 2>/dev/null
sleep 5

# Kill if still alive
kill $MONICA_PID 2>/dev/null || true

rm -f /tmp/monica_fifo
echo "Done at $(date)" >> /tmp/monica_v8h_status.txt
