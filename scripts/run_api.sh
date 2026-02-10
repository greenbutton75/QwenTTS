#!/bin/bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/QwenTTS/logs}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8000}"

mkdir -p "$LOG_DIR"

while true; do
  /venv/main/bin/uvicorn server.app:app --host "$HOST" --port "$PORT"
  code=$?
  echo "$(date -Is) uvicorn exited with code $code, restarting in 5s" >> "$LOG_DIR/uvicorn.supervisor.log"
  sleep 5
done
