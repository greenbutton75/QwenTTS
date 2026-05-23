#!/bin/bash
# Supervisor for the QwenTTS task worker. Intentionally NOT `set -e`: a non-zero
# exit (crash / SIGTERM) must restart the loop, not kill the supervisor.
# Guard: do not start a second worker if one is already running.

LOG_DIR="${TASK_WORKER_LOG_DIR:-/workspace/QwenTTS/logs}"

mkdir -p "$LOG_DIR"

while true; do
  if pgrep -f 'python -m task_worker.main' >/dev/null 2>&1; then
    sleep 10
    continue
  fi
  echo "$(date -Is) run_worker: starting task_worker" >> "$LOG_DIR/task_worker.supervisor.log"
  /venv/main/bin/python -m task_worker.main >> "$LOG_DIR/task_worker.out" 2>&1 || true
  echo "$(date -Is) run_worker: task_worker exited, restart in 5s" >> "$LOG_DIR/task_worker.supervisor.log"
  sleep 5
done
