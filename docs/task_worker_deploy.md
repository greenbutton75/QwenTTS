# Task Worker Deployment (GPU instance)

This guide describes how to deploy `task_worker` on the same GPU instance as QwenTTS API.

## Environment Variables

If your env exceeds Vast limits, store it in S3 and download in Onstart:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts
TASK_ENV_S3_KEY=secrets/qwentts.env
```

The `qwentts.env` file should contain:

```
TASK_BASE_URL=https://rixtrema.net/api/async_task_manager
USER_TOKEN=...
SYSTEM_TOKEN=...
FINGERPRINT=...

QWEN_TTS_BASE_URL=http://127.0.0.1:8000

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts

TASK_WORKER_HEALTH_PORT=8010
TASK_WORKER_LOG_DIR=/workspace/QwenTTS/logs
TASK_WORKER_LOG_MAX_BYTES=10485760
TASK_WORKER_LOG_BACKUPS=7
```

## Install

```
/venv/main/bin/pip install -r task_worker/requirements.txt
```

## Run

```
/venv/main/bin/python -m task_worker.main
```

## Health

```
http://localhost:8010/health
```

## systemd (auto-restart)

```
sudo cp /workspace/QwenTTS/docs/systemd/task_worker.service /etc/systemd/system/task_worker.service
sudo systemctl daemon-reload
sudo systemctl enable task_worker
sudo systemctl start task_worker
sudo systemctl status task_worker
```

## Notes

- The worker polls `QWEN_TTS_CREATE_PROFILE` and `QWEN_TTS_PHRASE`.
- Profiles and phrases are checked in S3, so the instance can be interrupted safely.
- Batch size for phrase tasks: 50 (no new fetch until batch finishes).

### Phrase splice optimization (production)

`QWEN_TTS_PHRASE` now has an internal optimization path:

- Worker groups similar phrase tasks inside one poll batch.
- If multiple tasks share the same normalized body, worker calls server splice endpoint.
- Server generates greeting + cached body and merges with content-aware splice.
- If task is unique or split fails, worker falls back to the old full-phrase path.

No mandatory new env variables are required.

Optional flag:

```
ENABLE_PHRASE_SPLICE_GROUPING=true   # default true
```

## Token File Format

The S3 env file must be **LF** and **no spaces around `=`** (e.g. `USER_TOKEN=...`).
If it was edited on Windows, run: `sed -i 's/\r$//' /etc/qwentts.tokens.env` and `sed -i 's/ *= */=/' /etc/qwentts.tokens.env`.

## Auto-restart (no systemd)

Use the watchdog scripts:

```
chmod +x /workspace/QwenTTS/scripts/run_api.sh /workspace/QwenTTS/scripts/run_worker.sh
nohup /workspace/QwenTTS/scripts/run_api.sh > /workspace/QwenTTS/logs/uvicorn.out 2>&1 &
nohup /workspace/QwenTTS/scripts/run_worker.sh > /workspace/QwenTTS/logs/task_worker.out 2>&1 &
```

Important:

- these scripts do not load `.env` or `/etc/qwentts.env` on their own
- before manual restart in SSH, load env into the current shell first

Minimal manual restart sequence:

```bash
set -a
source /etc/qwentts.env
set +a

pkill -f "scripts/run_api.sh" || true
pkill -f "uvicorn server.app:app" || true
pkill -f "scripts/run_worker.sh" || true
pkill -f "python -m task_worker.main" || true

nohup /workspace/QwenTTS/scripts/run_api.sh > /workspace/QwenTTS/logs/uvicorn.out 2>&1 &
nohup /workspace/QwenTTS/scripts/run_worker.sh > /workspace/QwenTTS/logs/task_worker.out 2>&1 &
```

If `/etc/qwentts.env` is missing or stale, restore it first from `/workspace/QwenTTS/qwentts.env` or rebuild it from S3 as shown in `README.md` / `docs/vastai.md`.

## Audio cleanup update (2026-04-04)

The server-side output cleanup was extended to handle real production failures:

- removes long silence/noise at the start of generated audio
- removes trailing silence more aggressively
- compacts long internal silent spans in generated phrases
- affects both full-phrase and splice generation paths because the cleanup lives in the shared server audio postprocess

New relevant env:

```
OUTPUT_AUDIO_TRIM_MAX_LEADING_MS=15000
OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS=2000
OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS=600
OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION=rms_flatness_pause_compact_v3
```

## Resilience

The worker loop catches unexpected errors (e.g., DNS failures) and retries with exponential backoff (up to 5 minutes).


Note: task listing uses SYSTEM_TOKEN + FINGERPRINT so it can see tasks created by other users.
