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
- rejects bad greeting starts for phrases beginning with `Hi` / `Hello`:
  - short voiced garbage onset
  - long low-energy pre-roll before the first strong speech segment
  - clipped short greeting endings such as `Hi D..` / `Hi De..`
- bad greeting/start candidates are retried instead of being hard-cut blindly
- fatal CUDA generation failures are now handled explicitly:
  - shared GPU inference is serialized with a global lock
  - fatal CUDA errors kill the API process so `run_api.sh` can restart it cleanly
  - local API queue items stuck in `running` are recovered back to `queued` on startup
  - retryable Qwen API / CUDA failures are bubbled up for retry instead of immediately failing the remote task
- the greeting over-trim regression was fixed:
  - root cause was double leading trim on `Hi` / `Hello` outputs
  - accepted retry candidates are no longer pre-trimmed before final cleanup
  - greeting/full-phrase paths now preserve more leading context with a larger pad
  - final merged splice cleanup no longer trims the start a second time
  - short greeting cleanup now preserves the tail of the name as well, so the end of `Hi, Dennis.` is not removed
- boundary cleanup was extended to remove long noisy edge blocks that are not true silence:
  - low-energy boundary spans are trimmed after the ordinary edge pass
  - a chunk-based clarity trim keeps the main speech block and removes long noisy prefixes/suffixes
  - a local clarity refinement removes residual buzzing on the first/last seconds after the coarse cut
- for problematic long ICL references, an operational fallback is to rebuild the profile with `xvector_only=true`
- when staying in `xvector_only=false`, `ref_text` must remain aligned with the actual sample audio
- affects both full-phrase and splice generation paths because the cleanup lives in the shared server audio postprocess
- `task_worker` latency was reduced without touching audio quality:
  - `PHRASE_POLL_INTERVAL` default changed from `15` to `5`
  - `POST /phrases` and `POST /phrases/splice-prod` submit timeouts default to `150s`
  - health/status timeouts are configurable separately
- `splice` is now attempted for every split-able phrase, not only for grouped phrases:
  - if `_split_greeting_body(...)` succeeds, worker goes to splice first even for singleton tasks
  - grouping still matters for shared-body batching and cache reuse, but no longer decides whether splice is allowed at all
  - full phrase is now reserved for non-splittable texts or real splice fallback
- production showed that repeated lead names inside the body/signoff destroy body-cache reuse:
  - examples like `Again, Audrey...` / `Again, Marty...` create different `body` text for every lead
  - worker may still group less often and the server gets more unique body cache keys
  - best practice is to keep names in greeting only and leave the reusable body identical
- a fast mitigation was added for costly failed splice attempts:
  - `GREETING_SPLICE_MAX_ATTEMPTS=3`
  - `GREETING_FULL_PHRASE_MAX_ATTEMPTS=2`
  - `GREETING_SPLICE_MAX_NEW_TOKENS=256`
  - splice now fails over sooner, and failed splice greeting generations are cheaper
- greeting duration is still logged, but no longer rejected by itself:
  - `duration_artifact` remains diagnostic metadata only
  - hard-fail remains for bad start / preroll / clipped ending
- full fallback was redesigned to avoid multiple full-length renders just to validate greeting:
  - server now runs a short `greeting probe` first
  - only if probe passes does it perform one normal full render of the long phrase
- timing coverage was extended for investigation of long non-GPU pauses:
  - `task_worker_timing.log` now records production task API calls such as `list_tasks`, `update_progress`, and task completion/failure updates
  - `task_worker.process_phrases.prepare` measures the batch-preparation window before the first splice/full submission
  - this helps distinguish real idle/recovery windows from time spent in external task API, parsing, S3 profile checks, and grouping

New relevant env:

```
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
GREETING_FULL_PHRASE_MAX_ATTEMPTS=2
GREETING_SPLICE_MAX_NEW_TOKENS=256
```

Operational note:

- previously generated bad `phrases/*.wav` in S3 must be regenerated
- if all greeting attempts still have a bad start, the task will now fail/retry instead of publishing a broken file
- phrase metadata now includes greeting quality fields such as `greeting_onset_*`, `greeting_ending_*`, `greeting_preroll_*`, `greeting_start_passed`, and `greeting_passed`
- after `CUDA error: device-side assert triggered`, do not trust same-process full fallback; the intended recovery path is API restart
- tasks already completed as remote `failed` in async_task_manager must still be recreated manually
- phrases generated during the short greeting over-trim regression window must be regenerated because the saved audio is already clipped
- if `splice_failures` grows together with `phrase_fallback_full`, the full-phrase path is being used as the expensive recovery path after failed splice attempts
- if splice failures are dominated by `Read timed out`, increasing greeting retries is not the right fix; address API hangs/timeouts first
- if a current `xvector_only=true` profile and its `body` cache already sound good, keep that profile and validate only the new server code before reprocessing the voice again
- if `body_cache_hit` is already high but phrase latency is still large, inspect `greeting_attempts` in `server_timing.log`
- after the retry-budget split, `splice` should never exceed `GREETING_SPLICE_MAX_ATTEMPTS`; long latency above that point means the voice/profile itself is unstable on short greetings
- if `splice` still fails for one voice, do not blindly disable splice for the whole voice in production: successful splice is often still cheaper than forcing all remaining phrases through full fallback
- for current diagnostics, compare:
  - `api.splice_synthesize.greeting_attempt`
  - `api.splice_synthesize.total`
  - `api.worker.phrase.greeting_attempt`
  - `api.worker.phrase.generate`

## Health counter notes

`task_worker` health counters are cumulative since worker start and are not final task statuses.

- `splice_failures` means the splice path failed for a phrase attempt before fallback
- `phrase_fallback_full` means worker switched that phrase to the full-phrase path
- `phrase_failed` increases only when the final result for the phrase is actually failed

So a state like `splice_failures > 0`, `phrase_fallback_full > 0`, `phrase_failed = 0` is valid and means fallback recovered successfully.

## Resilience

The worker loop catches unexpected errors (e.g., DNS failures) and retries with exponential backoff (up to 5 minutes).


Note: task listing uses SYSTEM_TOKEN + FINGERPRINT so it can see tasks created by other users.

See also: `docs/voice_recovery_runbook.md`
