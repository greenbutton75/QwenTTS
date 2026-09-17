#!/bin/bash
# Mac-native launcher for the qwentts task_worker. Adapted from the
# original scripts/run_worker.sh the same way as start-api-mac.sh.
#
# CRITICAL: exactly one task_worker may run anywhere against this shared
# queue at a time (no server-side dedup by fingerprint). This script does
# not check that for you -- it is your job to confirm nothing else
# (the DGX or otherwise) is polling before this ever runs at boot.
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

LOG_DIR="${TASK_WORKER_LOG_DIR:-$QWENTTS_HOME/logs}"
HEALTH_PORT="${TASK_WORKER_HEALTH_PORT:-8010}"
mkdir -p "$LOG_DIR"

while true; do
  if /usr/sbin/lsof -iTCP:"${HEALTH_PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    sleep 10
    continue
  fi
  echo "$(date -Iseconds) start-worker-mac: no listener on ${HEALTH_PORT}, starting task_worker" >> "$LOG_DIR/task_worker.supervisor.log"
  "$QWENTTS_HOME/.venv-qwentts/bin/python3" -u -m task_worker.main >> "$LOG_DIR/task_worker.out.log" 2>&1
  echo "$(date -Iseconds) start-worker-mac: task_worker exited, restart in 5s" >> "$LOG_DIR/task_worker.supervisor.log"
  sleep 5
done
