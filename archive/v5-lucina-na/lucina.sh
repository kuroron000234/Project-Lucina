#!/usr/bin/env bash
# lucina.sh — one-click launcher for lucina-NA (daemon + WebUI)
#
# Starts both the daemon (`main.py --daemon`) and the WebUI (`main.py --webui`)
# and supervises them: a process that exits with code 42 is restarted
# immediately, otherwise it is only kept alive while its "wanted" flag exists
# (data/run/<name>.wanted). The WebUI control tab uses this to start/stop/
# restart the daemon and the UI without touching the terminal.
#
# Double-click this file (or use lucina.desktop) to launch.
set -u

cd "$(dirname "$0")" || exit 1

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

RUN_DIR="data/run"
LOCK_FILE="$RUN_DIR/launcher.lock"
WEBUI_URL="http://127.0.0.1:8765"
mkdir -p "$RUN_DIR"

# Wait until the WebUI responds, then open it in a new browser tab.
# Runs under nohup so the browser still opens even when the launching
# terminal closes right away (e.g. the "launcher already running" branch).
open_webui() {
  WEBUI_URL="$WEBUI_URL" nohup bash -c '
    ready=0
    if command -v curl >/dev/null 2>&1; then
      for _ in $(seq 1 60); do
        if curl -s -o /dev/null --max-time 1 "$WEBUI_URL"; then
          ready=1
          break
        fi
        sleep 1
      done
    else
      sleep 5
      ready=1
    fi
    if [ "$ready" -eq 0 ]; then
      echo "[lucina] WebUI did not respond yet - open it manually: $WEBUI_URL" >&2
      exit 0
    fi
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$WEBUI_URL" >/dev/null 2>&1 || echo "[lucina] WebUI is ready: $WEBUI_URL (open it in your browser)" >&2
    else
      echo "[lucina] WebUI is ready: $WEBUI_URL (open it in your browser)" >&2
    fi
  ' >/dev/null 2>&1 &
  disown 2>/dev/null || true
}

# If a launcher is already supervising, re-arm the wanted flags, open the
# WebUI, and keep this terminal as a live log monitor. Ctrl+C only closes
# this monitor window - the background system keeps running because the
# cleanup trap is not installed on this branch.
if [ -f "$LOCK_FILE" ] && kill -0 "$(cat "$LOCK_FILE")" 2>/dev/null; then
  echo "[lucina] Launcher already running (PID $(cat "$LOCK_FILE"))."
  echo "[lucina] Re-arming wanted flags and opening WebUI: $WEBUI_URL"
  touch "$RUN_DIR/daemon.wanted" "$RUN_DIR/webui.wanted"
  open_webui
  echo "[lucina] Live log stream (Ctrl+C closes this window only - system keeps running)"
  # -F follows by name so the monitor survives log rotation (RotatingFileHandler).
  tail -F data/logs/system.log 2>/dev/null || sleep 5
  exit 0
fi
echo $$ > "$LOCK_FILE"

# Arm both processes (the supervisors below only start them while these exist).
touch "$RUN_DIR/daemon.wanted" "$RUN_DIR/webui.wanted"

cleanup() {
  echo ""
  echo "[lucina] Stopping daemon + WebUI..."
  rm -f "$RUN_DIR/daemon.wanted" "$RUN_DIR/webui.wanted" "$LOCK_FILE"
  for f in daemon.pid webui.pid; do
    if [ -f "$RUN_DIR/$f" ]; then
      kill "$(cat "$RUN_DIR/$f")" 2>/dev/null || true
    fi
  done
  kill ${SUP_DAEMON:-} ${SUP_WEBUI:-} ${BROWSER_PID:-} 2>/dev/null || true
  echo "[lucina] All stopped. Goodbye."
}
trap cleanup INT TERM HUP EXIT

echo "[lucina] lucina-NA launcher"
echo "[lucina] WebUI: $WEBUI_URL"
echo "[lucina] Opening WebUI in your browser when ready..."
echo "[lucina] Ctrl+C to stop everything"

# Supervisor for the daemon
(
  while true; do
    if [ -f "$RUN_DIR/daemon.wanted" ]; then
      "$PYTHON" main.py --daemon
      code=$?
      if [ $code -eq 42 ]; then
        echo "[lucina] Daemon restart requested -> restarting"
        continue
      fi
      echo "[lucina] Daemon exited (code=$code)"
    fi
    sleep 1
  done
) &
SUP_DAEMON=$!

# Supervisor for the WebUI
(
  while true; do
    if [ -f "$RUN_DIR/webui.wanted" ]; then
      "$PYTHON" main.py --webui
      code=$?
      if [ $code -eq 42 ]; then
        echo "[lucina] WebUI restart requested -> restarting"
        continue
      fi
      echo "[lucina] WebUI exited (code=$code)"
    fi
    sleep 1
  done
) &
SUP_WEBUI=$!

open_webui

wait $SUP_DAEMON $SUP_WEBUI
