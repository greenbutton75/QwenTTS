#!/bin/bash
# Supervisor for the QwenTTS API. Intentionally NOT `set -e`: a non-zero uvicorn
# exit (crash / SIGKILL) must restart the loop, not kill the supervisor.
# Port-check guard: if something is already listening on the API port, do not
# launch a second uvicorn (lets this script adopt an already-running instance).

LOG_DIR="${LOG_DIR:-/workspace/QwenTTS/logs}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8000}"

mkdir -p "$LOG_DIR"

while true; do
  if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    sleep 10
    continue
  fi
  echo "$(date -Is) run_api: no listener on ${PORT}, starting uvicorn" >> "$LOG_DIR/uvicorn.supervisor.log"
  /venv/main/bin/uvicorn server.app:app --host "$HOST" --port "$PORT" >> "$LOG_DIR/uvicorn.out" 2>&1 || true
  echo "$(date -Is) run_api: uvicorn exited, restart in 5s" >> "$LOG_DIR/uvicorn.supervisor.log"
  sleep 5
done
