# QwenTTS on Mac Studio (native macOS/MPS) — Deployment Notes

Migrated from the DGX Spark's Docker/CUDA deployment on 2026-09-17. This
document records what's different here, every code change made to get it
running correctly on Apple Silicon, and the operational procedures learned
along the way. Read this before touching the deployment again.

## Why this exists

The original `server/` and `task_worker/` code was written and validated
only against NVIDIA/CUDA hardware (Vast.ai, then a DGX Spark). Nothing in
it was broken on that hardware — everything found here is a real gap
between "runs on CUDA" and "runs correctly on Apple Silicon's MPS backend",
plus two genuine pre-existing bugs the switch in hardware happened to
expose.

## Environment

- Dedicated venv: `~/qwentts/.venv-qwentts` (Python 3.11, isolated from the
  screenplay LLM's venv on this same Mac).
- `torch`/`torchvision`/`torchaudio` 2.2.2 from plain PyPI — no special
  index URL needed (unlike the DGX's `+cu121` build). MPS is built into the
  standard macOS wheel.
- `torchaudio` was added explicitly even though it's absent from
  `server/requirements.txt`. The DGX's custom NGC/CUDA-alpha torch build
  couldn't get a working torchaudio wheel, so the code carries a
  `kaldi_native_fbank` fallback for fbank extraction
  (`qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`). On this Mac, standard
  torchaudio installs fine, so the primary (faster) path is used and
  `kaldi_native_fbank` just sits there unused (it does have real macOS
  arm64 wheels, so it still installs without issue via
  `server/requirements.txt` — just never exercised).
- `brew install sox` — the Python `sox` package needs the actual CLI
  binary; wasn't present on this Mac. `ffmpeg` was already installed
  (needed by `pydub`).
- Model pre-fetched once via `huggingface-cli download
  Qwen/Qwen3-TTS-12Hz-1.7B-Base` (no gating, unlike some other HF repos
  used elsewhere on this machine) before setting `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1`.
- `ASR_DEVICE=cpu` — ctranslate2 (faster-whisper's backend) has no Metal
  support. This required zero code changes: `server/asr.py` already reads
  `ASR_DEVICE` before ever touching `torch.cuda.is_available()`.

## Code changes

All in `~/qwentts` (a clone of `github.com/greenbutton75/QwenTTS`, not the
upstream repo — commit these changes there if you want them preserved
across a fresh clone).

### 1. `server/tts.py::_get_device()` — the actual CUDA→MPS patch

Only place that needed a real device-selection change:

```python
def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
```

Everything else in the 42 "cuda" references found across the codebase
already degraded safely without changes: dtype falls back to `float32`
for any non-`"cuda"` device, `flash_attn` import is wrapped in a caught
`ImportError` (prints a warning, falls back to plain PyTorch attention),
and `torch.cuda.manual_seed_all()` calls are already guarded behind
`torch.cuda.is_available()`.

### 2. `server/s3_store.py::download_torch()` — bfloat16 crash on MPS

Every voice's cached `prompt.pt` (speaker embedding etc.) was produced on
the CUDA box in bfloat16. Moving a bfloat16 tensor onto an MPS device
raises `TypeError: BFloat16 is not supported on MPS` in torch 2.2.2 —
before any dtype conversion even runs. Fixed once, centrally, at the one
place all cached tensors load from S3:

```python
def _to_inference_dtype(obj):
    """Downcast bfloat16 -> float32 on non-CUDA hosts, recursively."""
    if torch.cuda.is_available():
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.float32) if obj.dtype == torch.bfloat16 else obj
    if isinstance(obj, dict):
        return {k: _to_inference_dtype(v) for k, v in obj.items()}
    ...

def download_torch(key: str) -> Any:
    ...
    return _to_inference_dtype(torch.load(bio, map_location="cpu"))
```

CUDA path is untouched.

### 3. `server/candidate_pool.py::_seed()` — MPS seeding (minor)

Added `torch.mps.manual_seed(seed)` alongside the existing
`torch.cuda.manual_seed_all()` guard, for reproducibility parity. Not a
bug fix, just completeness.

### 4. `server/tts.py::clean_output_audio_for_greeting()` — trailing-clip on non-Hi/Hello greetings

**Pre-existing bug, not MPS-specific.** This function only protected
greeting endings from trailing-trim clipping for text matching
`_SHORT_GREETING_RE` (literally "Hi \<Name\>." / "Hello \<Name\>." only).
Any other phrasing (e.g. "Good afternoon, Alex.") fell through to the
generic trim path, which clipped the final consonant off the name
("Alex" → "Alek"). Found via real testing. Fix: removed the
regex-gated special case, applied the same protection (`max_trailing_ms=0`,
`preserve_tail=True`) to every greeting unconditionally — a greeting is a
short opener regardless of wording, there's no reason only two specific
templates need the protection.

### 5. `server/tts.py::detect_body_excessive_duration_artifact()` — new function, body-side runaway detection

**Pre-existing gap, not MPS-specific**, though the unaccelerated hardware
here makes it easier to hit. `detect_greeting_excessive_duration_artifact`
only ever applied to greetings; body generation of *any* length had zero
duration sanity checking. Found two real cases in testing: a cached body
with ~4-5s of stray noise before the actual words, and a fresh render that
came out at 11s for text that should take ~6-7s. Both passed every
existing check (similarity, ASR-WER) because Whisper doesn't transcribe
non-lexical noise as extra words — nothing flagged either one.

New function estimates expected duration from word count
(~2.3 words/sec) and flags a large excess:

```python
def detect_body_excessive_duration_artifact(text, wav, sr):
    words = len((text or "").split())
    expected_ms = int((words / 2.3) * 1000.0)
    expected_max_ms = max(int(expected_ms * 1.8) + 1500, 2500)
    excess_ms = duration_ms - expected_max_ms
    artifact = int(duration_ms > expected_max_ms and excess_ms >= 2000)
    ...
```

Wired into `server/quality.py::evaluate_candidate()`'s body branch (sets
`duration_artifact` for body candidates, previously always `False`), and
into `server/candidate_pool.py::_body_is_good()`'s reject conditions —
which feeds the *already-existing* `BODY_QUALITY_REQUIRE_PASS` hard-reject
path, so no new env flag was needed for this one. Verified: re-running the
same 11s case after the fix, the system correctly rejected the bad greedy
candidate, tried 2 more, and landed on a clean 6.8s render.

### 6. `server/config.py` / `server/app.py` — new `GREETING_DURATION_ARTIFACT_REQUIRE_PASS` flag

**Pre-existing gap.** `duration_artifact` for greetings was detected
(scored into `composite_score`) but never had a hard-reject path — unlike
onset/preroll/ending, which each have their own `*_REQUIRE_PASS` flag.
Found via testing: a "Hi Alex." greeting kept winning best-of-4 despite an
audible trailing artifact, because onset/preroll/ending all scored clean
(false negative for this artifact shape) and only `duration_artifact`
caught it — as a score penalty only. Added the missing flag, same pattern
as the existing ones:

```python
GREETING_DURATION_ARTIFACT_REQUIRE_PASS = _get_env_bool(
    "GREETING_DURATION_ARTIFACT_REQUIRE_PASS", False
)
```

Set to `true` in this Mac's `config/qwentts.env` (default stays `False` in
code, matching every other `*_REQUIRE_PASS` flag's default).

Same fallback-to-full-phrase safety net as the other `REQUIRE_PASS` gates
applies on reject (see `task_worker/worker.py`'s splice→full fallback
logic) — enabling this is not a "fail with nothing" risk.

## Known remaining issue (not fixed, understood)

**Voice profiles built with `x_vector_only=false` (full in-context-learning
mode) can leak the tail of their own reference recording into generated
speech** — audible as "Me...", "ME I hope...", or a subtler "eh"/"em" onset
sound. This is a **documented, pre-existing issue** (see
`docs/voice_recovery_runbook.md`, "Full voice recovery" section), not
something introduced by this migration. Two voices were found and fixed by
this exact runbook procedure during testing (both used the same reference
script, ending in "...I want my voice to sound like me."):

- `support/55379/voices/49a88fde-...` ("TEST")
- `support/85159/voices/b716c84c-...` ("Vince's Voice")

**If you notice a similar leak on any other voice**: check
`x_vector_only_mode` in that voice's `prompt.pt` (or `x_vector_only` in its
`voice.json`) — if `false`, follow `docs/voice_recovery_runbook.md` section
7 (Full voice recovery): delete `voice.json`/`reference.wav`/`prompt.pt`
and `splice_cache/` for that one voice (keep `sample.wav`), rebuild via
`POST /profiles` with `xvector_only=true`, smoke-test before returning it
to production traffic.

## Latency / availability finding

**Confirmed real incident during testing**: a body generation on this
unaccelerated MPS/float32 setup that fails to find EOS and runs to the
token ceiling can take 30+ minutes for a single candidate — one such
request was killed after 27 minutes with zero progress (had to `kill -9`;
plain `SIGTERM` doesn't interrupt an in-flight generation, per this
codebase's own documented incident lesson in `IMPLEMENTATION_NOTES.md`).
Since this is a single-model-instance server, one pathological input can
block the entire queue behind it.

Mitigation applied: `VOICE_CLONE_MAX_NEW_TOKENS` lowered from the default
2048 to **1024** in `config/qwentts.env`. This roughly halves worst-case
single-candidate latency. No production body text tested so far (up to
~20 words, ~6-7s of audio) came anywhere close to needing 1024 tokens, let
alone 2048 — raise it back only if legitimate long bodies start getting
cut short (watch for `output_trim`/duration anomalies on genuinely long
scripts).

## Config file

`config/qwentts.env` (chmod 600) — merged from:
- AWS credentials (`~/.aws/credentials`, profile `qwentts`)
- `secrets/qwentts.env` pulled from `s3://rixtrema-qwentts/secrets/` (the
  real `FINGERPRINT`/`USER_TOKEN`/`SYSTEM_TOKEN` — **must be identical to
  whatever the DGX used**; this is a shared-identity system, not
  per-machine credentials, see `IMPLEMENTATION_NOTES.md` / the original
  migration spec)
- Mac-specific overrides: `ASR_DEVICE=cpu`, ports, paths under `~/qwentts`,
  the two quality-gate additions above, the token ceiling.

## Resilience (launchd)

Two services, mirroring the pattern already used for the other AI
services on this Mac (`~/Library/LaunchAgents/com.stochamodel.*.plist`):

- **`com.qwentts.api`** — `RunAtLoad=true`, `KeepAlive` on crash. Safe to
  auto-start at boot; the API alone (no worker) does nothing to the shared
  queue.
- **`com.qwentts.worker`** — `RunAtLoad=false` **deliberately**.
  This queue has no server-side dedup by fingerprint — exactly one worker
  may run anywhere in the world at a time, or two instances race the same
  job and produce duplicate/corrupt output. Auto-starting this at every
  boot without a human re-confirming no other instance (the DGX or
  otherwise) is polling would be a real risk. It still gets `KeepAlive`
  crash-restart once it *has* been started — that's just "don't die
  silently," not "come alive on your own."

Both use Mac-adapted launcher scripts (`scripts/start-api-mac.sh`,
`scripts/start-worker-mac.sh`) — adapted from the original
`scripts/run_api.sh`/`run_worker.sh` because those use `ss` for the
port-check guard, which doesn't exist on macOS (replaced with `lsof`), and
hardcode `/workspace/QwenTTS` + `/venv/main/bin/...` paths.

**Starting the worker after a stop** (e.g. after a reboot, or after
manually stopping it): re-run the pre-flight from
`docs/task_worker_deploy.md` / this migration's runbook (confirm no other
instance polling — sample the `Tasks/List` NEW count twice a minute or so
apart with nothing here running; if it drops, something else is
consuming) — **then**:

```bash
launchctl kickstart -k gui/$(id -u)/com.qwentts.worker
curl http://127.0.0.1:8010/health   # confirm last_phrase_poll is advancing
```

**Stopping either service**:

```bash
launchctl bootout gui/$(id -u)/com.qwentts.worker   # or .api
```

Note: a plain `pkill -f "uvicorn server.app:app"` sends `SIGTERM`, which
uvicorn honors gracefully by waiting for any in-flight request — if one is
genuinely stuck (see the latency finding above), that wait can be
effectively unbounded. Use `kill -9 <pid>` if a stop needs to be immediate
and you've verified via `ps`/CPU% that it's the pathological case, not
just a normal, longer-than-usual generation.

## Verification performed before going live

- `/phrases/splice-test` (sync, no S3/queue involvement) — real generated
  audio, checked programmatically (energy envelope, ASR transcript,
  onset/duration artifact detectors) and by ear across 8 different real
  production voice profiles and varied greeting/body phrasing.
- Read-only `Tasks/List` pre-flight against the live queue with the real
  tokens — confirmed auth/fingerprint match, and confirmed via two samples
  45s apart that nothing else was actively consuming before starting the
  worker.
- Worker started 2026-09-17 ~10:00 against the real backlog (555 pending
  `QWEN_TTS_PHRASE` tasks at start); first several real completions
  verified via `task_worker_timing.log` / `server_timing.log` — correct
  shared-body cache reuse observed (16s for a cache-hit phrase vs 143s for
  the cache-miss one that built it), all quality gates reporting clean on
  real production data.
