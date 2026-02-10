# Task Worker Deployment (GPU instance)

This guide describes how to deploy `task_worker` on the same GPU instance as QwenTTS API.

## Environment Variables

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
```

## Install

```
pip install -r task_worker/requirements.txt
```

## Run

```
python -m task_worker.main
```

## Notes

- The worker polls `QWEN_TTS_CREATE_PROFILE` and `QWEN_TTS_PHRASE`.
- Profiles and phrases are checked in S3, so the instance can be interrupted safely.
- Batch size for phrase tasks: 50 (no new fetch until batch finishes).
