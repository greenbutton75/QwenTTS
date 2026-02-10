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

mkdir -p logs data tmp
chmod +x /workspace/QwenTTS/scripts/run_api.sh /workspace/QwenTTS/scripts/run_worker.sh

nohup /workspace/QwenTTS/scripts/run_api.sh > logs/uvicorn.out 2>&1 &
nohup /workspace/QwenTTS/scripts/run_worker.sh > logs/task_worker.out 2>&1 &
```



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
