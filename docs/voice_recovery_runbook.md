# Voice Recovery Runbook

This runbook is for production incidents where one `voice_id` starts producing bad audio while the rest of the service may still be busy.

Use the narrowest recovery path that solves the actual problem.

## 1a. One more common cause of "slow splice"

Before touching profile or cache, check whether the body text is really shared.

If the script repeats the lead's name inside the trailing body/signoff, for example:

- `Again, Audrey...`
- `Again, Marty...`
- `Again, Ivan...`

then every lead gets a different `body`, so shared `splice_cache` reuse collapses.

In that case:

- profile recovery is not the right first step;
- the better fix is to normalize task text so the name stays only in `greeting`;
- after the text change, rerun phrase tasks and let a new shared body cache build naturally.

## 1. Choose the right path

### Text-only rerun

Use this when:

- the current profile already sounds good;
- the current `body` cache sounds good;
- the defect is in task text or task content itself;
- example: `I am would like` must become `I would like`.

In this path:

- do **not** rebuild the profile;
- do **not** delete `prompt.pt`;
- do **not** delete `reference.wav`;
- do **not** delete `splice_cache/`.

### Full voice recovery

Use this when:

- the voice itself is bad;
- you see leakage like `Me ...`, `ME I hope ...`;
- generated `body` is bad even after `splice_cache/` was cleared;
- profile must be rebuilt in `xvector_only=true`.

In this path:

- keep only `sample.wav`;
- delete `voice.json`, `reference.wav`, `prompt.pt`, and `splice_cache/`;
- recreate the profile manually through `:8000`;
- smoke-test the voice before returning prod tasks to the queue.

## 2. Safety rules

- Stop `task_worker` before recovery so it does not keep pulling new prod tasks.
- If you need a clean manual test window, keep only API running.
- Do not delete `sample.wav`.
- Do not manually delete `support/{support_id}/phrases/*.wav` unless you really need to.
  Re-running the same `phrase_id` overwrites the old `.wav` and `.json`.
- If a current `xvector_only=true` profile and its `body` cache already sound good, keep that profile and test only the new server logic first.
- If the defect is only "slow splice" and the body differs by lead name, fix task text first and do not rebuild the profile yet.

## 3. Common variables

```bash
export SUPPORT_ID=82938
export VOICE_ID=953fe137-c12c-4495-9b09-fd9d8a2a8762
export VOICE_NAME='Geoff Roush'
```

## 4. Stop intake from prod

```bash
set -a
source /etc/qwentts.env
set +a

pkill -f "scripts/run_worker.sh" || true
pkill -f "python -m task_worker.main" || true
```

## 5. Inspect local API queue for one voice

Run this before touching profile or local sqlite tasks.

```bash
python - <<'PY'
import os, sqlite3, json
db = os.environ["SQLITE_PATH"]
voice_id = os.environ["VOICE_ID"]
conn = sqlite3.connect(db)
rows = conn.execute("SELECT id, task_type, status, updated_at, payload_json FROM tasks ORDER BY updated_at DESC").fetchall()
hits = []
for row in rows:
    payload = json.loads(row[4])
    if payload.get("voice_id") == voice_id:
        hits.append({
            "id": row[0],
            "task_type": row[1],
            "status": row[2],
            "updated_at": row[3],
            "payload": payload,
        })
print(hits)
PY
```

Interpretation:

- if there are `running` tasks for this `voice_id`, the safest path is to let them finish;
- if you must cut over immediately, stop API too and delete local sqlite tasks for this voice before restarting API.

## 6. Delete local API sqlite tasks for one voice

Do this only if you want a clean local state for that `voice_id`.

```bash
python - <<'PY'
import os, sqlite3, json
db = os.environ["SQLITE_PATH"]
voice_id = os.environ["VOICE_ID"]
conn = sqlite3.connect(db)
rows = conn.execute("SELECT id, payload_json, status FROM tasks").fetchall()
ids = []
for row in rows:
    payload = json.loads(row[1])
    if payload.get("voice_id") == voice_id:
        ids.append((row[0],))
print({"delete_ids": [x[0] for x in ids]})
if ids:
    conn.executemany("DELETE FROM tasks WHERE id=?", ids)
    conn.commit()
conn.close()
PY
```

## 7. Full voice recovery

### 7.1 Stop API too

```bash
pkill -f "scripts/run_api.sh" || true
pkill -f "uvicorn server.app:app" || true
```

### 7.2 Delete only voice-specific artifacts in S3

This keeps `sample.wav`.

```bash
python - <<'PY'
import boto3, os

bucket = os.environ["S3_BUCKET_NAME"]
support_id = os.environ["SUPPORT_ID"]
voice_id = os.environ["VOICE_ID"]
base = f"support/{support_id}/voices/{voice_id}"

keys = [
    f"{base}/voice.json",
    f"{base}/reference.wav",
    f"{base}/prompt.pt",
]

prefix = f"{base}/splice_cache/"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
)

for key in keys:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        print({"deleted": key})
    except Exception as exc:
        print({"delete_failed": key, "error": str(exc)})

token = None
deleted = 0
while True:
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
    if token:
        kwargs["ContinuationToken"] = token
    resp = s3.list_objects_v2(**kwargs)
    items = resp.get("Contents", [])
    if items:
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": x["Key"]} for x in items], "Quiet": True},
        )
        deleted += len(items)
    if not resp.get("IsTruncated"):
        break
    token = resp.get("NextContinuationToken")

print({"splice_cache_deleted": deleted, "prefix": prefix})
PY
```

### 7.3 Start only API

```bash
nohup /workspace/QwenTTS/scripts/run_api.sh > /workspace/QwenTTS/logs/uvicorn.out 2>&1 &

sleep 10
curl --max-time 5 http://127.0.0.1:8000/health && echo
tail -n 40 /workspace/QwenTTS/logs/uvicorn.out
```

### 7.4 Rebuild profile in `xvector_only=true`

```bash
curl -X POST http://127.0.0.1:8000/profiles \
  -F "support_id=$SUPPORT_ID" \
  -F "voice_id=$VOICE_ID" \
  -F "voice_name=$VOICE_NAME" \
  -F "ref_text=" \
  -F "xvector_only=true"
```

Wait for `done`:

```bash
while true; do
  curl "http://127.0.0.1:8000/profiles/$VOICE_ID?support_id=$SUPPORT_ID"
  echo
  sleep 3
done
```

### 7.5 Smoke-test the voice before returning prod tasks

```bash
curl -X POST http://127.0.0.1:8000/phrases/splice-prod \
  -H "Content-Type: application/json" \
  -d "{
    \"support_id\": \"$SUPPORT_ID\",
    \"voice_id\": \"$VOICE_ID\",
    \"phrase_id\": \"recovery-check-001\",
    \"greeting\": \"Hi, Dennis.\",
    \"body\": \"I hope your day is going well. I would like to get a few minutes of your time to learn more about your business and your retirement plan. Do you have 15-20 minutes over the next week or two?\",
    \"pause_ms\": 220,
    \"crossfade_ms\": 10,
    \"content_aware\": true,
    \"target_lufs\": -16.0
  }"
```

```bash
curl -X POST http://127.0.0.1:8000/phrases/splice-prod \
  -H "Content-Type: application/json" \
  -d "{
    \"support_id\": \"$SUPPORT_ID\",
    \"voice_id\": \"$VOICE_ID\",
    \"phrase_id\": \"recovery-check-002\",
    \"greeting\": \"Hi, Dennis.\",
    \"body\": \"I hope you are doing well today. This is Geoff from RiXtrema calling with a quick follow up.\",
    \"pause_ms\": 220,
    \"crossfade_ms\": 10,
    \"content_aware\": true,
    \"target_lufs\": -16.0
  }"
```

## 8. Text-only rerun

Use this when the profile is already good and only the task text must be corrected.

### 8.1 Stop both worker and API for a clean cutover

```bash
pkill -f "scripts/run_worker.sh" || true
pkill -f "python -m task_worker.main" || true
pkill -f "scripts/run_api.sh" || true
pkill -f "uvicorn server.app:app" || true
```

### 8.2 Fix text in prod DB and reset phrase tasks

Example for one voice:

```sql
update TasksManager..tasks
set
  TaskParameters = replace(TaskParameters, 'I am would like', 'I would like'),
  Status = 'NEW',
  Progress = 0,
  Error = null,
  Stage = 'START',
  data = null
where type like 'QWEN_TTS_PHRASE%'
  and TaskParameters like '%953fe137-c12c-4495%';
```

Notes:

- this path does **not** rebuild the profile;
- this path does **not** clear `splice_cache/`;
- the same `phrase_id` will overwrite the old phrase `.wav` and `.json` in S3.
- this is also the correct path when you want to remove a second personalized name from the body/signoff to improve shared body-cache reuse.

### 8.3 Delete local sqlite tasks for that voice

Use the command from section 6.

### 8.4 Start only API

```bash
nohup /workspace/QwenTTS/scripts/run_api.sh > /workspace/QwenTTS/logs/uvicorn.out 2>&1 &

sleep 10
curl --max-time 5 http://127.0.0.1:8000/health && echo
tail -n 40 /workspace/QwenTTS/logs/uvicorn.out
```

### 8.5 Manual smoke test on the corrected text

```bash
curl -X POST http://127.0.0.1:8000/phrases/splice-prod \
  -H "Content-Type: application/json" \
  -d "{
    \"support_id\": \"$SUPPORT_ID\",
    \"voice_id\": \"$VOICE_ID\",
    \"phrase_id\": \"text-fix-check-001\",
    \"greeting\": \"Hi, Dennis.\",
    \"body\": \"I hope your day is going well. I would like to get a few minutes of your time to learn more about your business and your retirement plan. Do you have 15-20 minutes over the next week or two?\",
    \"pause_ms\": 220,
    \"crossfade_ms\": 10,
    \"content_aware\": true,
    \"target_lufs\": -16.0
  }"
```

If the result is good, continue.

### 8.6 Start `task_worker` again

```bash
nohup /workspace/QwenTTS/scripts/run_worker.sh > /workspace/QwenTTS/logs/task_worker.out 2>&1 &

sleep 5
curl --max-time 5 http://127.0.0.1:8010/health && echo
tail -n 40 /workspace/QwenTTS/logs/task_worker.out
```

## 9. Current known-good state

For the tested voice:

- current `xvector_only=true` profile is good;
- current `body` cache is good;
- `content_aware=true` should stay enabled;
- short greeting cleanup fix is required so `Hi, Dennis.` does not become `Hi De..`.

## 10. Fast splice cost limiter

The server now uses separate greeting retry budgets:

- `GREETING_SPLICE_MAX_ATTEMPTS=3`
- `GREETING_FULL_PHRASE_MAX_ATTEMPTS=2`
- `GREETING_SPLICE_MAX_NEW_TOKENS=256`

Operational meaning:

- expensive failed splice attempts should now fail over sooner;
- full fallback keeps only a small retry budget;
- before a full-length render, server now runs a short `greeting probe`;
- `duration_artifact` no longer hard-fails greeting by itself, it stays only in diagnostics;
- if latency stays high even after this change, inspect `greeting_attempts` in `server_timing.log` before rebuilding the profile.
