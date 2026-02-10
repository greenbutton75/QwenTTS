# Qwen3-TTS Portable PRO (локальный UI)

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
</p>

**Портативная русскоязычная версия** мощной системы синтеза речи Qwen3-TTS с поддержкой:
- 🎙️ **Клонирование голоса** — создание копии голоса из короткого аудио
- 🎨 **Дизайн голоса** — генерация голоса по текстовому описанию
- 👥 **Multi-speaker режим** — создание диалогов с несколькими дикторами
- 🌍 **Мультиязычность** — поддержка 10+ языков включая русский

## Авторы

**Собрал [Nerual Dreaming](https://t.me/nerual_dreming)** — основатель [ArtGeneration.me](https://artgeneration.me/), техноблогер и нейро-евангелист.

**[Нейро-Софт](https://t.me/neuroport)** — репаки и портативки полезных нейросетей

## Быстрый старт (Windows, локальный UI)

1. Установите зависимости: `portable/install.bat`
2. Запустите UI: `portable/run.bat`
3. Откройте в браузере: `http://127.0.0.1:7860`

> При первом запуске модели будут скачаны автоматически и займут место на диске.

## Системные требования (рекомендуемые)

- **ОС:** Windows 10/11
- **GPU:** NVIDIA с поддержкой CUDA (минимум 8GB VRAM)
- **RAM:** 16GB+
- **Интернет:** Требуется при первом запуске для загрузки моделей

## Что входит в репозиторий

- Локальный Gradio UI: `portable/app.py`
- Скрипты запуска: `portable/install.bat`, `portable/run.bat`
- Python пакет `qwen_tts` (исходники)

## Что НЕ хранится в репозитории

- Скачанные модели, кэши, временные файлы, результаты генерации
- Портативный Python и сторонние бинарники (SoX)

Все такие папки добавлены в `.gitignore`.

## VVK API (cloud-ready)

Для прод-развертывания без UI используется FastAPI-сервис из папки `server/`.
Он поддерживает только VVK-сценарии:

- Создание профиля голоса (reference + ref_text)
- Очередь генерации фраз
- Polling статуса по phrase_id
- Админка (basic auth) с очередью и статистикой

### Переменные окружения

Смотрите шаблон `.env.example`.
Ключевые переменные:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME`
- `ADMIN_USER`, `ADMIN_PASSWORD`
- `MODEL_SIZE` (по умолчанию `1.7B`)
- `SQLITE_PATH`, `LOG_DIR`

### S3 структура

```
s3://rixtrema-qwentts/support/{support_id}/
  voices/{voice_id}/sample.wav
  voices/{voice_id}/reference.wav
  voices/{voice_id}/voice.json
  voices/{voice_id}/prompt.pt
  phrases/{phrase_id}.wav
  phrases/{phrase_id}.json
```

### API

`POST /profiles` (multipart/form-data)
- `support_id`, `voice_id`, `voice_name`, `ref_text` (opt), `xvector_only` (opt)
- sample берётся из S3: `support/{support_id}/voices/{voice_id}/sample.wav`

`POST /phrases` (json)
```
{
  "support_id": "user1_46847",
  "voice_id": "voice_7f1c2b3e",
  "phrase_id": "phrase_91a0c",
  "text": "Want to sell you"
}
```

`GET /phrases/{phrase_id}?support_id=...` (ответ содержит `public_url` на 60 дней)

`GET /profiles/{voice_id}?support_id=...`

`GET /admin` (basic auth)

### Docker

```
docker build -t qwentts .
docker run --gpus all --env-file .env -p 8000:8000 qwentts
```

## Vast.ai (interruptible, рекомендованный шаблон)

Для стабильного доступа используйте **прямой IP:порт**, а не туннели.
Порт 8000 нужно пробросить при создании инстанса.

### Docker options (ports)

```
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 8000:8000
```

### Environment Variables (UI)

Чтобы не упереться в лимит 4096 символов, длинные токены храним в S3.
В Template оставляем **минимум**:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts
TASK_ENV_S3_KEY=secrets/qwentts.env
```

Файл `qwentts.env` (лежит в S3) содержит все остальные переменные:

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

### Onstart Script (Vast.ai)

```
#!/bin/bash
set -e

mkdir -p /workspace

apt-get update
apt-get install -y ffmpeg libsndfile1 sox
/venv/main/bin/pip install awscli

# build env file from template + tokens from S3
env | grep -E 'FINGERPRINT|AWS_|S3_|ADMIN_|MODEL_|LANGUAGE|SQLITE_PATH|LOG_|MAX_RETRIES|RETRY_BASE_SECONDS|S3_PREFIX|OPEN_BUTTON_PORT|TASK_WORKER_|QWEN_TTS_BASE_URL|TASK_BASE_URL|TASK_ENV_S3_KEY|QWEN_TTS_HOST|QWEN_TTS_PORT' > /etc/qwentts.env
set -a
source /etc/qwentts.env
set +a
aws s3 cp s3://$S3_BUCKET_NAME/$TASK_ENV_S3_KEY /etc/qwentts.tokens.env
sed -i 's/$//' /etc/qwentts.tokens.env
sed -i 's/ *= */=/' /etc/qwentts.tokens.env
cat /etc/qwentts.tokens.env >> /etc/qwentts.env
set -a
source /etc/qwentts.env
set +a

cd /workspace
rm -rf QwenTTS
git clone https://github.com/greenbutton75/QwenTTS.git
cd QwenTTS

/venv/main/bin/pip install -r server/requirements.txt --no-deps
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



### Проверка

После старта в UI будет строка вида:
`PublicIP:PORT -> 8000/tcp`.

Доступ:
```
http://PublicIP:PORT/health
http://PublicIP:PORT/admin
```

Если `/admin` не показывает форму логина, откройте ссылку в другом браузере/инкогнито
(HTTP Basic может быть закеширован) и убедитесь, что `ADMIN_USER/ADMIN_PASSWORD`
заданы в env.

## Task Worker (GPU queue service)

Сервис `task_worker` обрабатывает задачи из очереди:

- `QWEN_TTS_CREATE_PROFILE`
- `QWEN_TTS_PHRASE`

И создает downstream задачи:

- `QWEN_TTS_READY_PROFILE`
- `QWEN_TTS_READY_PHRASE`

### Переменные окружения

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

### Запуск (локально на GPU-инстансе)

```
pip install -r task_worker/requirements.txt
python -m task_worker.main
```

Health:
`http://localhost:8010/health`

## Task Submitter (local)

Локальный скрипт для:
- загрузки sample в S3
- создания задач в очереди

### Переменные окружения

```
TASK_BASE_URL=https://rixtrema.net/api/async_task_manager
USER_TOKEN=...

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts
```

### Создать профиль

```
pip install -r task_submitter/requirements.txt
python -m task_submitter.main create-profile ^
  --support-id user1_46847 ^
  --voice-id voice_21327ef670 ^
  --voice-name "Name of Voice 2" ^
  --ref-text "Each book in the series was originally published in hardcover format with a number of full-color illustrations spread throughout." ^
  --sample "D:\\Work\\ML\\Voice\\Sveta.m4a"
```

### Создать фразу

```
python -m task_submitter.main create-phrase ^
  --support-id user1_46847 ^
  --voice-id voice_21327ef670 ^
  --phrase-id phrase_006 ^
  --text "Protect and distribute your assets according to your wishes with our comprehensive trust administration services."
```

## Docs

- `docs/vastai.md` — Vast.ai deployment + e2e check
- `docs/task_worker_deploy.md` — task_worker deployment notes

## Возможности

### Пресеты голосов
Использование встроенных голосовых пресетов (Aiden, Dylan, Eric, Serena и др.)

### Клонирование голоса
Загрузите короткое аудио (5-30 сек) и получите синтез речи этим голосом.

### Multi-speaker
Создавайте диалоги с несколькими дикторами в формате:
```
Speaker 0: Привет! Как дела?
Speaker 1: Отлично, спасибо!
```

### Дизайн голоса
Опишите желаемый голос текстом на английском языке:
```
Young female voice, warm and friendly, speaking with enthusiasm
```

## Голосовые пакеты

При установке автоматически загружается голосовой пакет с русскими голосами.
Дополнительные голоса можно загрузить из облака прямо в приложении.

## Лицензия

Модель Qwen3-TTS распространяется под лицензией [Qwen License](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT).

## Оригинальный проект

- 🤗 [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts)
- 📑 [Blog](https://qwen.ai/blog?id=qwen3tts-0115)
- 📑 [Paper](https://arxiv.org/abs/2601.15621)
