# Splice Strategy Notes (QwenTTS)

## Context

В задачах outbound-озвучки часто меняется только обращение (`"Hi Lloyd,"`, `"Hi Emma,"`), а основной текст (`body`) остается одинаковым.  
Цель: сократить время генерации и стоимость, сохранив натуральное звучание.

---

## Идеи, которые обсуждали

### 1) Plain audio splice (WAV + WAV)

- Сгенерировать отдельно `greeting` и `body`
- Склеить на аудио-уровне
- Плюс: просто внедрить в текущую архитектуру
- Минус: при грубой склейке слышен шов

### 2) Token-level concat (лучший теоретически)

- Кэшировать/переиспользовать акустические коды (codec/talker codes), а не WAV
- Склеивать коды и декодировать одним проходом
- Плюс: максимально бесшовно
- Минус: требует отдельной доработки API/пайплайна, сейчас не было внедрено

### 3) Prefix/KV cache continuation

- Переиспользовать префиксные состояния генерации
- Плюс: потенциально быстрый и элегантный путь
- Минус: в текущей серверной обвязке не реализован

---

## Что реально поддерживает текущий сервис

- Сервис сейчас ориентирован на `text -> wav` через `generate_voice_clone(...)`
- В текущем API уже есть клонирование профиля и генерация фраз
- Для быстрого теста внедрен путь через **audio-level splice**
- Для ускорения добавлен **S3 body cache**

---

## Что реализовано в коде

## 1) Тестовый endpoint для склейки

- Endpoint: `POST /phrases/splice-test`
- Request:
  - `support_id`
  - `voice_id`
  - `greeting`
  - `body`
  - `pause_ms`
  - `crossfade_ms`
- Response:
  - `audio/wav` (готовый merged audio)
  - служебные headers:
    - `X-Body-Cache: hit|miss`
    - `X-Body-Cache-Key: <sha256>`

### Основная логика endpoint

1. Загружает `prompt.pt` профиля из S3  
2. Генерирует `greeting` всегда  
3. Для `body` вычисляет хэш кэша  
4. Пытается взять `body.wav` из S3 кэша  
5. Если cache miss — генерирует `body`, сохраняет в S3  
6. Склеивает `greeting + pause + body` с crossfade  
7. Отдает итоговый WAV

---

## 2) Алгоритм склейки (audio-level)

Реализован простым и стабильным способом:

- Нормализация mono float32
- Добавление паузы (`pause_ms`) между `greeting` и `body`
- Overlap на участке `crossfade_ms`:
  - `fade_out` на хвосте `greeting + silence`
  - `fade_in` на начале `body`
- Конкатенация частей
- Защитная peak-нормализация при клиппинге

Практически рабочий набор параметров для старта:

- `pause_ms=120`
- `crossfade_ms=10`

---

## 3) S3 cache для body

Кэш-ключ строится как SHA-256 от:

- `support_id`
- `voice_id`
- `body` (trimmed)
- `model_params`:
  - `MODEL_SIZE`
  - `LANGUAGE`
  - engine marker
- `prompt_meta`:
  - `x_vector_only_mode`
  - `icl_mode`
  - `ref_text`

S3 paths:

- `support/{support_id}/voices/{voice_id}/splice_cache/body_<hash>.wav`
- `support/{support_id}/voices/{voice_id}/splice_cache/body_<hash>.json`

---

## 4) Batch-скрипт для тестов

Добавлен клиентский скрипт:

- `task_submitter/splice_batch_generate.py`

Назначение:

- Генерировать серию фраз с **одним body** и **множеством greeting** (по умолчанию 10)
- Сохранять WAV локально
- Печатать `cache hit/miss` из response headers

Пример запуска:

```bash
python task_submitter/splice_batch_generate.py \
  --base-url "http://<host>:<port>" \
  --support-id "63180" \
  --voice-id "ab196dbb-3ef0-4ee4-ae15-1c4701b203c4" \
  --body "This is Alex, Director of Customer Support from Rixtrema." \
  --pause-ms 120 \
  --crossfade-ms 10 \
  --out-dir "splice-tests"
```

---

## Наблюдения и практические выводы

- Audio splice в текущей реализации — самый быстрый путь проверки гипотезы
- Качество склейки сильно зависит от `pause_ms/crossfade_ms`
- Для прод-оптимизации уже полезен `body` cache в S3 (снижает повторную генерацию)
- Потенциальный следующий уровень качества — переход к token-level concat

---

## Files changed for current approach

- `server/models.py`
- `server/tts.py`
- `server/app.py`
- `task_submitter/splice_batch_generate.py`

---

## Production rollout design

Реализован безопасный прод-путь на базе той же идеи:

- Новый endpoint: `POST /phrases/splice-prod`
  - Принимает: `support_id`, `voice_id`, `phrase_id`, `greeting`, `body`
  - Генерирует greeting
  - Загружает/генерирует body из S3 cache
  - Склеивает через `content_aware` по умолчанию
  - Пишет результат в обычный путь phrase (`support/{support_id}/phrases/{phrase_id}.wav`)
  - Обновляет `phrase_json` со статусом `done/failed`

- Worker-side группировка задач:
  - Включается env-флагом `ENABLE_PHRASE_SPLICE_GROUPING`
  - Опциональный allowlist: `PHRASE_SPLICE_SUPPORT_IDS=63180,....`
  - Группировка только внутри текущего батча `QWEN_TTS_PHRASE`
  - Ключ группы: `(support_id, voice_id, normalized_body)`
  - `split` консервативный: фразы, начинающиеся с `Hi|Hello <Name>...`
  - Для групп `>=2`: вызов `/phrases/splice-prod`
  - Иначе: старый full-phrase путь (`/phrases`)

- Fallback:
  - Любая ошибка splice-пути приводит к попытке full-phrase генерации
  - Если fallback тоже неуспешен, задача фейлится как и раньше

- Наблюдаемость (`task_worker` health):
  - `phrase_grouped`
  - `phrase_splice_path`
  - `phrase_fallback_full`
  - `splice_failures`
