# Quality Gate — Implementation Notes

Tracking notes for the `qwentts_quality_gate_spec.md` rollout. Records what was
built, where the implementation deviates from the spec and why, and what still
needs validation on the GPU box.

## Status

- **Phase 1 — DONE (pending GPU validation).** ASR layer, unified quality
  evaluation, composite score, diagnostic-mode integration, benchmark harness,
  unit tests.
- **Phase 2 — not started.** best-of-N greeting, ASR hard-gate, `candidate_pool.py`,
  per-voice similarity threshold.
- **Phase 3 — not started.** `reference_validator.py`, body best-of-N, body ASR gate.

## Phase 1 — what was added

New modules:
- `server/asr.py` — lazy Whisper (faster-whisper) singleton + `transcribe_audio`,
  and a pure `analyze_transcript` (WER/CER, prefix-extra, suffix-clipped,
  reference-leak). `ASRReport` dataclass lives here.
- `server/quality.py` — `QualityReport`, `evaluate_candidate`, `composite_score`,
  `diagnostic_phrase_fields`. Combines the existing `tts` heuristic detectors +
  speaker similarity + ASR into one report.

Config (`server/config.py`): ASR settings, greeting ASR thresholds, composite
score weights, `ASR_DIAGNOSTIC_MODE` / `BODY_ASR_DIAGNOSTIC_MODE`. Helpers
`asr_config()` and `score_weights()`. All default OFF/diagnostic.

Integration (flag-gated, **no rejection** in Phase 1):
- `server/worker.py::_handle_phrase` → `_phrase_quality_diagnostics` (full-phrase path).
- `server/app.py::_synthesize_spliced_phrase` → `_phrase_quality_diagnostics`
  (splice path). The function now returns a 10th element `quality_diag`; both
  call sites updated. ASR fields are merged into `phrase_json` only when the
  flag is on.

Benchmark (`bench/`): `run_benchmark.py`, `metrics.py`, `report.py`,
`compare.py`, `voices.json`, `texts.json`, `README.md`.

Tests: `tests/test_asr.py`, `tests/test_quality.py`, `tests/test_bench_metrics.py`,
plus diagnostic-mode integration tests appended to
`tests/test_server_cache_logic.py` (`PhraseAsrDiagnosticModeTests`).

## Deviations from the spec (and why)

1. **WER/CER and fuzzy matching are self-contained, not `jiwer`/`Levenshtein`.**
   `server/asr.py` implements a standard Levenshtein-based WER/CER over our
   already-normalized token lists. This matches the standard WER definition,
   removes two dependencies, and — importantly — keeps the analysis logic
   fully unit-testable in a CPU-only environment. Swapping in `jiwer` later is
   trivial (replace `word_error_rate`/`char_error_rate`). `jiwer` and
   `Levenshtein` were therefore **not** added to `requirements.txt`.

2. **Text normalization strips non-ASCII.** `normalize_text` keeps `[a-z0-9]`.
   Whisper runs with `language="en"`, so a spurious syllable surfaces as a latin
   token (e.g. `ma`), which is what prefix-extra detection keys on. The spec's
   `«ма»` was illustrative Cyrillic; tests use the realistic latin form.

3. **`pyloudnorm` not added yet.** It belongs to Phase 3 (`reference_validator`).
   Only `faster-whisper==1.0.3` was added for Phase 1.

4. **Benchmark drives `/phrases/splice-test` (synchronous) + optional
   `--local-eval`.** This is the prod-critical, ASR-relevant path and returns
   audio synchronously without S3/async polling. ASR metrics come from scoring
   the returned wav locally with `server.quality` on the GPU box (it loads each
   voice's `prompt.pt` for the reference embedding). The async `/phrases` path
   and S3 `phrase_json` round-trip were intentionally not built into the bench
   to keep it runnable and focused; server-side `phrase_json` still gets ASR
   fields independently via `ASR_DIAGNOSTIC_MODE`.

## NOT validated in the dev environment (must be checked on the GPU box)

The dev box is Windows / Python 3.8 / `torch+cpu`, with no `faster-whisper`.
So the following could not be exercised here and are part of GPU validation:

- Real Whisper load + transcription, VRAM headroom (Qwen + Whisper + speaker
  encoder ≤ ~18 GB), and whether `_ASR_LOCK`-style serialization is needed under
  concurrent generation (spec open question 4).
- `evaluate_candidate` against real audio (similarity, real ASR).
- End-to-end `ASR_DIAGNOSTIC_MODE=true` run with `phrase_json` carrying `asr_*`.
- The benchmark end-to-end (1875 generations) and baseline `summary.json`.

What *was* validated here: all pure logic and wiring via mocks — 100+ unit tests
pass, including diagnostic-on/off/failure behavior and the off-path producing a
byte-identical `phrase_json` (no new keys).

## How to turn it on (Phase 1 = diagnostics only)

```
ASR_DIAGNOSTIC_MODE=true        # splice path final-phrase ASR diagnostics
BODY_ASR_DIAGNOSTIC_MODE=true   # full-phrase (worker) ASR diagnostics
ASR_MODEL_SIZE=large-v3
ASR_COMPUTE_TYPE=float16
```

With both off, behavior is identical to pre-Phase-1.

## Carry-over questions for Phase 2 (resolve on box)

- **Qwen batching** (spec open question 1): confirm whether
  `generate_voice_clone(text=[a, b], ...)` returns 2 distinct wavs. Decides
  batched vs sequential in `candidate_pool.py`. Record the finding here.
- **Name normalization** (spec open question 2): start with Levenshtein ≤ 1
  fuzzy match (already implemented in `asr.py`); build a replacement table from
  real benchmark misses if needed.
