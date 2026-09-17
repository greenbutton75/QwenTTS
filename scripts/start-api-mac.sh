#!/bin/bash
# Mac-native launcher for the qwentts FastAPI server. Adapted from the
# original scripts/run_api.sh (Linux/Vast.ai): that version's port-check
# guard used `ss`, which doesn't exist on macOS, and it hardcoded
# /workspace/QwenTTS paths and /venv/main/bin/uvicorn. This version uses
# lsof for the port guard and this repo's own venv, everything else the
# same (no `set -e`, so a crashed uvicorn loops and restarts here --
# launchd's KeepAlive handles the outer restart-on-exit).
set -uo pipefail

QWENTTS_HOME="/Users/rixtrema/qwentts"
cd "$QWENTTS_HOME"

ENV_FILE="$QWENTTS_HOME/config/qwentts.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

LOG_DIR="${LOG_DIR:-$QWENTTS_HOME/logs}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8000}"
mkdir -p "$LOG_DIR" "$QWENTTS_HOME/data" "$QWENTTS_HOME/tmp"

while true; do
  if /usr/sbin/lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    sleep 10
    continue
  fi
  echo "$(date -Iseconds) start-api-mac: no listener on ${PORT}, starting uvicorn" >> "$LOG_DIR/uvicorn.supervisor.log"
  "$QWENTTS_HOME/.venv-qwentts/bin/uvicorn" server.app:app --host "$HOST" --port "$PORT" >> "$LOG_DIR/uvicorn.out.log" 2>&1
  echo "$(date -Iseconds) start-api-mac: uvicorn exited, restart in 5s" >> "$LOG_DIR/uvicorn.supervisor.log"
  sleep 5
done
