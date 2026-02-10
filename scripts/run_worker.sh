#!/bin/bash
set -euo pipefail

LOG_DIR="${TASK_WORKER_LOG_DIR:-/workspace/QwenTTS/logs}"

mkdir -p "$LOG_DIR"

while true; do
  /venv/main/bin/python -m task_worker.main
  code=$?
  echo "$(date -Is) task_worker exited with code $code, restarting in 5s" >> "$LOG_DIR/task_worker.supervisor.log"
  sleep 5
done
