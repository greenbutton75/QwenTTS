# Vast.ai Deployment (VVK API)

This guide documents the production deployment of the VVK FastAPI service on Vast.ai.
Target: interruptible V100 32GB, direct IP:PORT (no tunnels).

## 1) Create a Custom Template

Base image: **PyTorch (Vast)**.

### Docker options (ports)

```
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 8000:8000
```

### Environment Variables (UI)

Note: Vast UI limits the number of env vars (around 32). To avoid hitting the limit, we hardcode the S3 env file path: `s3://$S3_BUCKET_NAME/secrets/qwentts.env`.


Vast has a 4096‑char limit for env. Store long tokens in S3:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts
```

The `qwentts.env` (stored in S3) contains the full set:

```
TASK_BASE_URL=https://rixtrema.net/api/async_task_manager
USER_TOKEN=...
SYSTEM_TOKEN=...
FINGERPRINT=...

QWEN_TTS_BASE_URL=http://127.0.0.1:8000

S3_PREFIX=support
ADMIN_USER=admin
ADMIN_PASSWORD=YOUR_PASSWORD

MODEL_SIZE=1.7B
LANGUAGE=English

SQLITE_PATH=/workspace/QwenTTS/data/queue.db
LOG_DIR=/workspace/QwenTTS/logs
LOG_MAX_BYTES=10485760
LOG_BACKUPS=7

MAX_RETRIES=3
RETRY_BASE_SECONDS=30

TASK_WORKER_HEALTH_PORT=8010
TASK_WORKER_LOG_DIR=/workspace/QwenTTS/logs
TASK_WORKER_LOG_MAX_BYTES=10485760
TASK_WORKER_LOG_BACKUPS=7

OPEN_BUTTON_PORT=8000
```

### Onstart Script

```
#!/bin/bash
set -e

mkdir -p /workspace

apt-get update
apt-get install -y ffmpeg libsndfile1 sox
# fix occasional DNS issues on Vast host
printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf
/venv/main/bin/pip install awscli

# 1) Слить env из Template
env | grep -E 'FINGERPRINT|AWS_|S3_|ADMIN_|MODEL_|LANGUAGE|SQLITE_PATH|LOG_|MAX_RETRIES|RETRY_BASE_SECONDS|S3_PREFIX|OPEN_BUTTON_PORT|TASK_WORKER_|QWEN_TTS_BASE_URL|TASK_BASE_URL|QWEN_TTS_HOST|QWEN_TTS_PORT' > /etc/qwentts.env

# 2) Подхватить его, чтобы S3 vars были в shell
set -a
source /etc/qwentts.env
set +a

# 3) Скачать токены и добавить
aws s3 cp s3://$S3_BUCKET_NAME/secrets/qwentts.env /etc/qwentts.tokens.env
sed -i 's/
$//' /etc/qwentts.tokens.env
sed -i 's/ *= */=/' /etc/qwentts.tokens.env
cat /etc/qwentts.tokens.env >> /etc/qwentts.env

# 4) Перезагрузить env уже с токенами
set -a
source /etc/qwentts.env
set +a

cd /workspace
rm -rf QwenTTS
git clone https://github.com/greenbutton75/QwenTTS.git
cd QwenTTS

/venv/main/bin/pip install -r server/requirements.txt
/venv/main/bin/pip uninstall -y numpy
/venv/main/bin/pip install numpy==1.26.4

/venv/main/bin/pip uninstall -y torch torchvision torchaudio transformers accelerate
/venv/main/bin/pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
/venv/main/bin/pip install -U huggingface_hub==0.34.0 safetensors==0.4.3 tokenizers==0.22.2 regex==2024.11.6
/venv/main/bin/pip install transformers==4.57.3 accelerate==1.12.0

/venv/main/bin/pip install -r task_worker/requirements.txt

/venv/main/bin/pip install --no-cache-dir setuptools==75.8.0 

mkdir -p logs data tmp
chmod +x /workspace/QwenTTS/scripts/run_api.sh /workspace/QwenTTS/scripts/run_worker.sh

nohup /workspace/QwenTTS/scripts/run_api.sh > logs/uvicorn.out 2>&1 &
nohup /workspace/QwenTTS/scripts/run_worker.sh > logs/task_worker.out 2>&1 &
```

## 2a) Manual SSH Restart With Env Reload

Important:

- `scripts/run_api.sh` and `scripts/run_worker.sh` do not source env files.
- They only inherit variables from the current shell.
- In a fresh SSH session you must reload env first, otherwise API/worker may start without tokens or S3 settings.

Quick restart when `/etc/qwentts.env` already exists and is valid:

```bash
cd /workspace/QwenTTS
git pull origin main

set -a
source /etc/qwentts.env
set +a

mkdir -p /workspace/QwenTTS/logs /workspace/QwenTTS/data /workspace/QwenTTS/tmp
chmod +x /workspace/QwenTTS/scripts/run_api.sh /workspace/QwenTTS/scripts/run_worker.sh

pkill -f "scripts/run_api.sh" || true
pkill -f "uvicorn server.app:app" || true
pkill -f "scripts/run_worker.sh" || true
pkill -f "python -m task_worker.main" || true

nohup /workspace/QwenTTS/scripts/run_api.sh > /workspace/QwenTTS/logs/uvicorn.out 2>&1 &
nohup /workspace/QwenTTS/scripts/run_worker.sh > /workspace/QwenTTS/logs/task_worker.out 2>&1 &

sleep 8
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8010/health
```

If `/etc/qwentts.env` needs to be restored quickly:

```bash
cp /workspace/QwenTTS/qwentts.env /etc/qwentts.env
set -a
source /etc/qwentts.env
set +a
```

If you need to refresh env from S3 and `awscli` is failing, use boto3 instead:

```bash
cd /workspace/QwenTTS
set -a
source /workspace/QwenTTS/qwentts.env
set +a

python - <<'PY'
import os
import boto3

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
)
body = s3.get_object(
    Bucket=os.environ["S3_BUCKET_NAME"],
    Key="secrets/qwentts.env",
)["Body"].read().decode("utf-8").replace("\r\n", "\n")
open("/etc/qwentts.env", "w", encoding="utf-8").write(body if body.endswith("\n") else body + "\n")
PY

set -a
source /etc/qwentts.env
set +a
```

Then run the restart block above.



## 2) Start Instance Using the Template

After the instance starts, check **IP & Port Info**:

```
PublicIP:PORT -> 8000/tcp
```

Use:

```
http://PublicIP:PORT/health
http://PublicIP:PORT/admin
```
Note: if `/admin` does not show a login prompt, open in another browser/incognito (HTTP Basic cache) and verify `ADMIN_USER/ADMIN_PASSWORD` in env.


## 3) Health Check (CLI)

```
curl http://PublicIP:PORT/health
```

## 4) Task Worker (same GPU instance)

Install and run the queue worker:

```
pip install -r task_worker/requirements.txt
python -m task_worker.main
```

The worker reads tasks from async_task_manager and calls the Qwen API on localhost.

## 5) End-to-End Check

1. Upload sample and create profile task (local machine):

```
pip install -r task_submitter/requirements.txt
python -m task_submitter.main create-profile --support-id user1_46847 --voice-id voice_21327ef670 --voice-name "Name of Voice 2" --ref-text "Each book in the series was originally published in hardcover format with a number of full-color illustrations spread throughout." --sample "D:\\Work\\ML\\Voice\\Sveta.m4a"
```

2. Create phrase task:

```
python -m task_submitter.main create-phrase --support-id user1_46847 --voice-id voice_21327ef670 --phrase-id phrase_006 --text "Protect and distribute your assets according to your wishes with our comprehensive trust administration services."
```

3. Verify task completion in async_task_manager (status SUCCESS / data contains presigned URL).

## 6) Notes

- Tunnels are optional; direct IP:PORT is preferred for production.
- If the instance is outbid, just create a new one from the template.
- All outputs and statuses are in S3, not on the instance disk.

## 7) Phrase splice optimization (2026-03-13)

Production now includes greeting/body splice optimization for phrase tasks:

- Worker may use splice path for repeated scripts in the same batch.
- Content-aware audio splice is used by default.
- Fallback to full-phrase generation remains in place for safety.

No additional required env vars for this feature.

Optional kill switch:

```
ENABLE_PHRASE_SPLICE_GROUPING=false
```

Verification:

- API admin page (`/admin`) now shows:
  - `phrase_grouped`
  - `phrase_splice_path`
  - `phrase_fallback_full`
  - `splice_failures`

## 7a) Audio Cleanup Update (2026-04-04)

Production audio postprocess was strengthened after real cases with multi-second silence/noise:

- leading output trim is now much more permissive for bad starts:
  - `OUTPUT_AUDIO_TRIM_MAX_LEADING_MS=15000`
- trailing trim limit increased:
  - `OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS=2000`
- long internal quiet spans are compacted:
  - `OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS=600`
- greeting/start quality is now protected for phrases beginning with `Hi` / `Hello`:
  - short voiced garbage at the very beginning is rejected
  - long low-energy pre-roll before the first strong speech segment is rejected
  - bad candidates are retried instead of being hard-cut blindly
- fatal CUDA generation failures are now handled more defensively:
  - access to the shared GPU model is serialized with a global lock
  - fatal CUDA errors such as `device-side assert triggered` cause the API process to exit
  - `scripts/run_api.sh` then restarts uvicorn automatically
  - stale local API queue items with status `running` are moved back to `queued` on startup
  - `task_worker` treats retryable Qwen API / CUDA failures as retryable instead of final task failures
- cleanup now applies to:
  - full phrase generation,
  - splice greeting,
  - cached/generated body,
  - final merged phrase
- trim/cache fingerprint version was bumped:
  - `OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION=rms_flatness_pause_compact_v3`
  - new code will not reuse old body cache objects generated with older trim behavior

Relevant greeting-start env:

```bash
GREETING_ONSET_ARTIFACT_CHECK=true
GREETING_ONSET_ARTIFACT_REQUIRE_PASS=true
```

Operational note:

- old `phrases/*.wav` already stored in S3 remain unchanged and must be regenerated
- old `splice_cache/` can be left in place, but a profile refresh or new cache key will bypass it automatically
- if all greeting attempts still produce a bad start, the service now fails the candidate instead of returning a broken WAV
- after a fatal CUDA crash, the recommended path is to let `run_api.sh` restart the API instead of trying to reuse the poisoned CUDA context
- remote tasks already marked as `failed` in async_task_manager still need to be recreated manually

## 8) Windows Watchdog

For interruptible production use, run the local watchdog from a Windows host outside Vast. It uses the Vast.ai CLI and `vast_api_key` to recreate the instance when it is outbid, becomes unhealthy, or disappears.

Files:

- `scripts/vast-qwentts-common.ps1`
- `scripts/start-qwentts.ps1`
- `scripts/monitor-qwentts.ps1`
- `docs/watchdog.md`

One-shot start or restore:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-qwentts.ps1
```

Continuous watchdog:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\monitor-qwentts.ps1
```

Default behavior:

- watches the instance by label
- verifies public `GET /health`
- removes dead or duplicate instances
- treats repeated health failures as a recycle condition
- searches new offers on 1x A100 80GB class GPUs
- applies an aggressive bid ladder above `min_bid`
- waits for a stability window before accepting a replacement as healthy

Full details: `docs/watchdog.md`
