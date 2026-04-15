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
# fix occasional DNS issues on Vast host
printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf
/venv/main/bin/pip install awscli

# build env file from template + tokens from S3
env | grep -E 'FINGERPRINT|AWS_|S3_|ADMIN_|MODEL_|LANGUAGE|SQLITE_PATH|LOG_|MAX_RETRIES|RETRY_BASE_SECONDS|S3_PREFIX|OPEN_BUTTON_PORT|TASK_WORKER_|QWEN_TTS_BASE_URL|TASK_BASE_URL|QWEN_TTS_HOST|QWEN_TTS_PORT' > /etc/qwentts.env
set -a
source /etc/qwentts.env
set +a
aws s3 cp s3://$S3_BUCKET_NAME/secrets/qwentts.env /etc/qwentts.tokens.env
sed -i 's/
$//' /etc/qwentts.tokens.env
sed -i 's/ *= */=/' /etc/qwentts.tokens.env
cat /etc/qwentts.tokens.env >> /etc/qwentts.env
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

### Manual SSH Restart (important)

If you restart the service manually in an SSH session, note:

- `scripts/run_api.sh` and `scripts/run_worker.sh` do **not** load env files by themselves.
- They use only the environment already present in the current shell.
- So before restart you must `source` the same env that Onstart assembled into `/etc/qwentts.env`.

Quick path if `/etc/qwentts.env` is already valid:

```bash
set -a
source /etc/qwentts.env
set +a
```

Full restart flow from SSH:

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

If `/etc/qwentts.env` was lost or is incomplete, rebuild it before restart.

Fastest fallback from the local checked-out file:

```bash
cp /workspace/QwenTTS/qwentts.env /etc/qwentts.env
set -a
source /etc/qwentts.env
set +a
```

If you want to refresh `/etc/qwentts.env` from S3 but `awscli` is broken, use boto3 instead:

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

After that, run the restart block above.



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


## Vast.ai Watchdog

For interruptible Vast.ai production use, the repository now includes a local Windows watchdog that recreates the QwenTTS instance when it is outbid or stops answering on `/health`.

Files:

- `scripts/vast-qwentts-common.ps1`
- `scripts/start-qwentts.ps1`
- `scripts/monitor-qwentts.ps1`
- `docs/watchdog.md`

Quick start:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-qwentts.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\monitor-qwentts.ps1
```

The watchdog uses the local Vast.ai CLI plus `vast_api_key`, monitors the public API health endpoint, cleans up dead or duplicate instances, and recreates the service from the configured Vast template on A100 80GB offers.

See `docs/watchdog.md` for configuration, recovery logic, and required env overrides.
## Docs

- `docs/vastai.md` — Vast.ai deployment + e2e check
- `docs/task_worker_deploy.md` — task_worker deployment notes
- `docs/splice_strategy_notes.md` — greeting/body splice strategy, cache, rollout
- `docs/voice_recovery_runbook.md` — safe recovery steps for one bad `voice_id` in production

## Update (2026-03-13)

В прод-пайплайн добавлена оптимизация для `QWEN_TTS_PHRASE`:

- Новый серверный endpoint: `POST /phrases/splice-prod`
- Worker-группировка фраз внутри текущего батча по `(support_id, voice_id, normalized_body)`
- Для групп `>=2`: синтез `greeting + cached body` с `content_aware` склейкой
- Для одиночных/сомнительных случаев: fallback на старый full-phrase путь (`POST /phrases`)
- S3 cache body: `support/{support_id}/voices/{voice_id}/splice_cache/body_<hash>.wav`
- Метрики в worker health: `phrase_grouped`, `phrase_splice_path`, `phrase_fallback_full`, `splice_failures`
- Эти метрики отображаются в `/admin` (порт API)

Важно:

- Дополнительные env для этой функции не обязательны.
- `ENABLE_PHRASE_SPLICE_GROUPING` по умолчанию `true`.
- Для принудительного отключения: `ENABLE_PHRASE_SPLICE_GROUPING=false`.

## Update (2026-04-04)

Сегодня доработан общий постпроцесс аудио в прод-контуре:

- Усилен trim начала/конца выходного WAV:
  - лучше режет стартовую тишину;
  - лучше режет короткие стартовые шумы/артефакты;
  - `OUTPUT_AUDIO_TRIM_MAX_LEADING_MS` поднят до `15000`;
  - `OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS` поднят до `2000`.
- Добавлено схлопывание чрезмерно длинных тихих участков внутри фразы:
  - новый env: `OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS` (default `600`);
  - длинные паузы внутри generated audio теперь сокращаются до разумного размера.
- Добавлена защита от дефектных стартов greeting/full-phrase для фраз, начинающихся с `Hi`/`Hello`:
  - детектируется короткий `voiced`-мусор в самом начале;
  - детектируется длинный тихий pre-roll перед первой нормальной сильной речью;
  - такие кандидаты не режутся вслепую, а отбраковываются и уходят на retry;
  - это снижает риск отрезать живое начало настоящей речи.
- Очистка применяется к:
  - full-phrase output,
  - `greeting`,
  - cached/generated `body`,
  - final merged splice output.
- Для greeting/start quality теперь сохраняются дополнительные признаки в результатах:
  - `greeting_onset_checked`, `greeting_onset_passed`, `greeting_onset_artifact`;
  - `greeting_ending_checked`, `greeting_ending_passed`, `greeting_ending_artifact`;
  - `greeting_preroll_checked`, `greeting_preroll_passed`, `greeting_preroll_artifact`;
  - `greeting_start_passed`, `greeting_passed`.
- Обновлена версия алгоритма output trim/cache fingerprint:
  - старый `splice_cache/body_*.wav` не используется новым кодом;
  - уже сгенерированные плохие `phrases/*.wav` в S3 нужно перегенерировать отдельно.
- Добавлены регрессионные тесты на:
  - длинную тишину в начале,
  - длинную тишину в середине фразы,
  - короткий мусорный `voiced`-старт,
  - длинный тихий pre-roll перед реальной речью,
  - recovery зависших локальных `running`-задач после рестарта API,
  - retryable обработку фатальных ошибок Qwen API / CUDA,
  - смену версии trim-алгоритма в cache key.
- Усилена устойчивость к фатальным CUDA-сбоям при генерации:
  - доступ к shared GPU-модели теперь сериализован через глобальный lock;
  - `device-side assert`, `illegal memory access` и похожие ошибки считаются фатальными для процесса API;
  - при таком сбое API-процесс завершается сам, а `scripts/run_api.sh` поднимает его заново;
  - локальная sqlite-очередь API на старте переводит зависшие `running` обратно в `queued`;
  - `task_worker` не помечает remote task как окончательно failed при retryable Qwen API / CUDA-сбоях, а даёт системе шанс на повтор после рестарта API.
- Исправлена регрессия с обрезанием начала greeting после вчерашних изменений:
  - причина была в двойном leading trim для `Hi` / `Hello` фраз;
  - `generate_voice_with_similarity_retry()` больше не делает pre-trim кандидата;
  - для greeting/full-phrase с `Hi` / `Hello` используется более щадящий стартовый pad;
  - финальная cleanup после splice больше не режет leading edge повторно.
- Добавлена дополнительная boundary-cleanup защита против длинных шумовых блоков на краях:
  - сначала убираются длинные low-energy pre-roll / post-roll артефакты;
  - затем применяется chunk-based clarity trim, который ищет главный блок нормальной речи и отрезает длинные шумовые блоки по краям;
  - после этого выполняется локальный boundary refinement, чтобы снять остаточное жужжание на первых/последних секундах уже очищенного WAV;
  - это исправляет реальные кейсы с файлами вида `*_StartNoise.wav`, `*_EndNoise.wav`, `*_StartAndEndNoise.wav`, где шум был не тишиной, а речеподобным монотонным мусором.
- Для голосов с нестабильным длинным ICL reference добавлен подтверждённый рабочий fallback:
  - если profile в режиме `xvector_only=false` начинает протаскивать хвост prompt-а в начало генерации (`Me ...`, `ME I hope ...`), такой voice можно безопасно пересчитать в `xvector_only=true`;
  - production splice pipeline при этом не меняется: используется тот же `splice-prod`, тот же `content_aware=true`, меняется только voice profile;
  - `ref_text` нельзя менять отдельно от sample, если voice остаётся в `xvector_only=false`.
- Исправлены дефекты короткого greeting `Hi/Hello Name`:
  - добавлен reject кандидатов с обрубленным концом (`Hi D..`, `Hi De..`);
  - для короткого greeting cleanup больше не режет trailing edge, чтобы не съедать конец имени;
  - ручная проверка на production-подобном профиле показала, что `content_aware=true` нужно оставлять включённым.
- Ускорен `task_worker` без изменения качества аудио:
  - `PHRASE_POLL_INTERVAL` уменьшен с `15` до `5` секунд;
  - вызовы `task_worker -> API` получили отдельные timeout env;
  - default timeout для `POST /phrases` и `POST /phrases/splice-prod` уменьшен до `150` секунд;
  - worker теперь быстрее замечает готовые фразы и меньше времени теряет на подвисших splice/full запросах.
- `splice` теперь стал основным путём для любой фразы, которую удалось разбить на `greeting + body`:
  - grouped tasks по-прежнему дают дополнительную выгоду через shared-body reuse;
  - singleton split-able tasks больше не уходят сразу в full phrase;
  - full phrase остаётся только для non-splittable текстов или как настоящий fallback.
- Подтверждён важный production-урок про cache reuse:
  - если в конце body/signoff повторяется имя лида (`Again, Audrey...`, `Again, Marty...`), то для каждого лида получается разный `body`;
  - splice всё ещё работает, но shared `body` cache reuse почти пропадает;
  - best practice: персонализацию держать в `greeting`, а общий `body` оставлять идентичным.
- Добавлен быстрый вариант B для ограничения цены неудачного splice:
  - greeting retry budget теперь разделён для splice и full phrase;
  - `GREETING_SPLICE_MAX_ATTEMPTS` default `3`;
  - `GREETING_FULL_PHRASE_MAX_ATTEMPTS` default `5`;
  - это удешевляет неудачный splice, не трогая более редкий full fallback слишком жёстко.
- Расширено timing-логирование для расследования idle / slow batch windows:
  - в `task_worker_timing.log` теперь логируются не только вызовы к локальному API, но и обращения к production task API (`Tasks/List`, `ChangeTaskProgress`, `CompleteTask`);
  - добавлен отдельный span подготовки batch: `task_worker.process_phrases.prepare`;
  - теперь можно отличить GPU-idle из-за real backoff/recovery от времени, ушедшего в list/parsing/profile-ready checks/grouping.

Новые/важные env для контроля поведения:

```
OUTPUT_AUDIO_TRIM_ENABLED=true
OUTPUT_AUDIO_TRIM_PAD_MS=30
OUTPUT_AUDIO_TRIM_MAX_LEADING_MS=15000
OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS=2000
OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS=600
OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION=rms_flatness_pause_compact_v4
GREETING_OUTPUT_TRIM_PAD_MS=160
GREETING_ONSET_ARTIFACT_CHECK=true
GREETING_ONSET_ARTIFACT_REQUIRE_PASS=true
PHRASE_POLL_INTERVAL=5
QWEN_TTS_HEALTH_TIMEOUT_SECONDS=5
QWEN_TTS_STATUS_TIMEOUT_SECONDS=30
QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS=180
QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS=150
QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS=150
GREETING_SPLICE_MAX_ATTEMPTS=3
GREETING_FULL_PHRASE_MAX_ATTEMPTS=5
```

Операционно это значит:

- старые плохие `phrases/*.wav` в S3 не исправятся сами, их нужно перегенерировать;
- если все попытки greeting дают дефектный старт, сервис теперь лучше вернёт ошибку/ретрай, чем отдаст пользователю бракованный WAV;
- новый код не должен повторно использовать старый `body` cache, если trim fingerprint изменился.
- при `CUDA error: device-side assert triggered` теперь нормальная стратегия не full fallback в том же умирающем процессе, а авто-рестарт API;
- после такого рестарта зависшие локальные задачи API должны сами вернуться в `queued`;
- уже завершённые как `failed` remote tasks в async_task_manager автоматически не оживают, их нужно пересоздать отдельно.
- после выката фикса `Preserve greeting starts during output cleanup` нужно перегенерировать фразы из окна регрессии, где начало greeting уже было обрезано.
- если `splice_failures` растёт одновременно с `phrase_fallback_full`, это означает, что дорогой full-phrase путь используется как следствие неуспешного splice;
- если в логах при этом доминируют `Read timed out`, проблема в зависающем API/splice path, а не в слишком маленьком количестве greeting retries.
- если текущий `xvector_only=true` profile и его `body` cache уже звучат хорошо, не нужно заново refresh-ить profile перед проверкой новых серверных фиксов: сначала выкатывайте код и проверяйте тот же `voice_id`.
- если `splice` уже основной путь, а latency всё ещё высокая, смотрите в `server_timing.log` поля `greeting_attempts` и `api.splice_synthesize.total`;
- после разделения retry budgets у splice не должно быть `greeting_attempts > GREETING_SPLICE_MAX_ATTEMPTS`; если и с этим голос остаётся медленным, проблема уже в самом voice/profile, а не в cache или merge.

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

