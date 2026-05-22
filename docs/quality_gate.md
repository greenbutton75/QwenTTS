# QwenTTS Quality Gate — стабилизация генерации

Документ описывает, что было сделано для стабилизации генерации голоса
(борьба с артефактами, потерей приветствия, гарблингом тела, переростом длины).
Парный документ — [generation_stability.md](generation_stability.md) (постановка
проблемы) и [../IMPLEMENTATION_NOTES.md](../IMPLEMENTATION_NOTES.md) (журнал внедрения).

## 1. Корневая причина

Доминирующий дефект — **runaway по `max_new_tokens`**. Авторегрессионная модель
иногда не выдаёт EOS и генерит до лимита токенов:
- greeting: `GREETING_SPLICE_MAX_NEW_TOKENS=256` ≈ **20.5 с** мусора (12.5 Гц);
- body / full render: `VOICE_CLONE_MAX_NEW_TOKENS=2048` ≈ до **163 с**.

Результат — длинная тишина/шорохи в начале и потеря/гарблинг текста. Дефект
**стохастический**: один и тот же голос+текст даёт то чистый результат, то брак
(подтверждено: голос 82938 «Geoff Roush», фраза Carter — 103 с / WER 0.97 против
того же тела 31.9 с / WER 0.08 на другом прогоне). Старые эвристики
(speaker similarity, onset/preroll) это пропускали: similarity оставался 0.88–0.96.

## 2. Решение — quality gate

Идея: не пытаться «починить» один прогон, а **сгенерировать несколько кандидатов,
оценить каждого независимым контентным сигналом (ASR) и выбрать лучшего**. Плюс
контентная проверка через распознавание (catch потери/гарблинга текста).

### 2.1. ASR-слой (`server/asr.py`)
- Ленивый singleton `faster-whisper large-v3` (≈3 ГБ VRAM, ~0.4 с/клип на 4090).
- `transcribe_audio()` + чистая `analyze_transcript()`: WER/CER, лишний префикс,
  обрезанный суффикс, протечка reference. Логика отделена от модели → юнит-тесты
  без GPU. WER/CER — самодостаточный edit-distance (без jiwer).

### 2.2. Единая оценка и composite score (`server/quality.py`)
`evaluate_candidate()` объединяет существующие эвристики `tts` (onset/preroll/
ending/duration для greeting; boundary для body) + speaker similarity + ASR в
`QualityReport`. `composite_score` (выше = лучше):

```
score =  W_SIM*similarity + W_ASR*(1-WER)
       - штрафы(prefix/suffix/refleak/onset/preroll/ending/duration/body-артефакты)
```
Веса в `config.score_weights()` (env-настраиваемы; калибровка — TODO после baseline).
`duration_artifact` — только штраф, не hard-fail (сохраняет урок «длительность сама
по себе не режет»).

### 2.3. Best-of-N для greeting (`server/candidate_pool.py`)
4 кандидата с эскалирующей температурой (это и есть «temperature schedule»
ансамблем, без модификации generation loop):
`greedy (do_sample=False)` → `sample_lo 0.25` → `sample_hi 0.35` → `sample_var 0.45`.
`select_best`: сначала прошедшие все проверки, затем по composite score. Runaway-
кандидат (20 с, WER 1.0) получает ~0.08 против ~2.45 у чистого → отбраковывается.
Флаг `GREETING_BEST_OF_N_ENABLED`.

### 2.4. Адаптивный best-of-N для body / длинного рендера
`generate_body_candidates`: сначала greedy; если он хороший (similarity ок, нет
boundary-артефакта, WER ≤ `BODY_ASR_MAX_WER=0.35`) — оставляем (1 генерация,
без потери скорости). Иначе генерим до `BODY_BEST_OF_N_MAX_COUNT=3` сэмплированных
и берём лучшего. Порог WER для body мягче greeting — произнесённые по цифрам
телефоны завышают WER на легитимном аудио. Флаг `BODY_BEST_OF_N_ENABLED`.

### 2.5. Trim хвостовой тишины (`tts.trim_trailing_silence`)
Тело генерится с `preserve_tail` (чтобы не срезать конец слова), поэтому хвостовая
тишина/шорохи доживали до финала. VAD находит последнюю реальную речь и срезает
только не-речевой хвост за паддингом 500 мс (последнее слово срезать нельзя —
оно выше порога речи). Срабатывает только если хвост > 700 мс. Применяется в
`clean_output_audio_for_spliced_phrase`. Флаг `SPLICE_TRAILING_SILENCE_TRIM_ENABLED`
(default on).

### 2.6. Роутинг приветствий в кавычках
Текст вида `"Hi Riley, …` (в т.ч. «ёлочки») не матчил regex разбиения и уходил на
**незащищённый** full-render. Теперь ведущие кавычки срезаются в обоих
`_split_greeting_body` (роутинг `task_worker` + проба `server.worker`) → такие
фразы идут в защищённый splice-путь.

### 2.7. ASR-диагностика (Phase 1)
`ASR_DIAGNOSTIC_MODE` / `BODY_ASR_DIAGNOSTIC_MODE` — пишут ASR-метрики в
`phrase_json` (`schema_version=2`), без отбраковки. Для сбора baseline и калибровки.

## 3. Env-флаги

| Флаг | Default | Назначение |
|---|---|---|
| `GREETING_BEST_OF_N_ENABLED` | **true** | best-of-N для greeting (раскатано) |
| `GREETING_BEST_OF_N_COUNT` | 4 | число кандидатов greeting |
| `BODY_BEST_OF_N_ENABLED` | **true** | адаптивный best-of-N для body (раскатано) |
| `BODY_BEST_OF_N_MAX_COUNT` | 3 | макс. кандидатов body на плохом прогоне |
| `BODY_ASR_MAX_WER` | 0.35 | порог WER приёмки greedy body |
| `SPLICE_TRAILING_SILENCE_TRIM_ENABLED` | true | trim хвостовой тишины |
| `GREETING_ASR_CHECK` | true | считать ASR при оценке greeting (только скоринг) |
| `GREETING_ASR_REQUIRE_PASS` | false | hard-fail greeting по ASR (выкл.) |
| `ASR_DIAGNOSTIC_MODE` | false | писать ASR-поля в phrase_json |

**Раскатка:** best-of-N включены по умолчанию в коде на `main`. Это сознательно:
`onstart.sh` при старте инстанса делает `git clone` (main) и пересобирает
`/etc/qwentts.env` из template — поэтому правки только в env не переживают
пересоздание инстанса (watchdog), а код-дефолт переживает. Отключить при
необходимости: выставить `=false` в env (template или `/etc/qwentts.env`).
Quote-strip — всегда активный код (без флага).

## 4. Результаты валидации (RTX 4090, реальные прод-фразы)

| Фраза (голос) | До | После |
|---|---|---|
| Carter (82938, пожилой М) | 103 с, WER 0.97 «Thank you for watching» | 29.9 с, WER 0.0 |
| Jodi (тот же голос) | мусорное тело | 29.8 с, WER 0.0 |
| Floyd | WER 0.79 | WER 0.034 |
| Jacque | ~4.8 с хвостовой тишины | хвост срезан (≈4799 мс) |
| Riley (в кавычках) | full-render, без защиты | splice, WER 0.0 |
| Penny / H. | ок | ок |

Greeting-best-of-N до этого исправил 20-с runaway-приветствия (Aleksandra 25.2→5.1 с,
Dennis 9.97→4.9 с, Mike 14.4→4.8 с), все WER 0.

## 5. Стоимость и эксплуатация

- Хороший body = 1 генерация (скорость не меняется). Плохой = до 3 → выше латентность,
  но качество приоритетно. Body кэшируется в S3 — общий body платит за регенерацию один раз.
- VRAM: Whisper large-v3 ≈3 ГБ поверх Qwen; на 24 ГБ 4090 запас большой, OOM не было.

## 6. Остаточный пробел (задокументирован, не реализован)

Путь **full-render** (`server/worker.py::_handle_phrase`, только для текста без
распознаваемого «Hi/Hello Name») использует старые защиты (greeting-probe +
body-quality-retry), а не новый ASR best-of-N. После фикса кавычек этот путь для
outbound-приветствий почти не задействуется. Follow-up: при необходимости
обобщить best-of-N и trailing-trim на этот путь.

## 7. Сознательно НЕ внедрено

- **UTMOS** в composite score — независимый сигнал «натуральности» (ловит
  «робоголос» при верном тембре и тексте). Реальная ценность, но дефект не
  наблюдался, а это +модель/+VRAM/+риск. Отложено.
- **Безусловная обрезка первого фрейма** — рискует регрессией «обрезанный старт
  greeting»; onset уже покрыт отбором в best-of-N, не обрезкой.
- **Per-voice similarity threshold**, **батчинг кандидатов** — отложены.
