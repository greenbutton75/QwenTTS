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

## Token File Format

The S3 env file must be **LF** and **no spaces around `=`** (e.g. `USER_TOKEN=...`).
If it was edited on Windows, run: `sed -i 's/\r$//' /etc/qwentts.tokens.env` and `sed -i 's/ *= */=/' /etc/qwentts.tokens.env`.
