# QwenTTS Quality Benchmark

Drives the production splice path over HTTP, saves audio, and computes
quality/timing aggregates. Used to establish a baseline and to compare phases
of the quality-gate rollout.

> **Requires a live server.** With `--local-eval` it also loads the model +
> Whisper to score each wav, so that mode must run on the GPU box. It cannot
> run in a CPU-only dev environment.

## 1. Prepare specs

- `voices.json` — replace the placeholder `support_id`/`voice_id` with **real**
  production profiles (their `prompt.pt` must already exist in S3). The spec
  targets ~15 voices spanning reference quality.
- `texts.json` — greeting + body pools; the driver generates
  `voices × greetings × bodies × runs-per-pair`.

## 2. Run a baseline (Phase 1)

Start the API with diagnostics on (so the server-side `phrase_json` also
carries ASR fields, independent of the benchmark's own local eval):

```bash
ASR_DIAGNOSTIC_MODE=true ./scripts/run_api.sh
```

Then:

```bash
python -m bench.run_benchmark \
    --voices bench/voices.json \
    --texts bench/texts.json \
    --runs-per-pair 5 \
    --base-url http://127.0.0.1:8000 \
    --output-dir bench/results/baseline_v1 \
    --local-eval --asr-diagnostic
```

Outputs in `bench/results/baseline_v1/`:

- `results.jsonl` — one record per generation (timing + quality_report)
- `summary.json` — aggregates (fail rates, distributions, artifact breakdown, worst cases)
- `report.md` / `report.html` — human-readable report
- `audio/` — every generated wav, for manual QA

Commit `summary.json` as the baseline for later comparison.

## 3. Compare phases

After Phase 2/3, run the benchmark again into a new dir and diff:

```bash
python -m bench.compare bench/results/baseline_v1 bench/results/phase2_v1
```

## 4. Quality scoring

- `--local-eval` (recommended): each wav is scored with
  `server.quality.evaluate_candidate` → full QualityReport incl. ASR
  (similarity, WER/CER, prefix-extra, suffix-clipped, reference-leak, artifacts,
  composite_score).
- Without `--local-eval`: the report falls back to the splice-test `X-*`
  response headers only (similarity + greeting artifact flags, no ASR).

A generation counts as a **failure** when `composite_score < 0`, any hard-fail
flag is set, or `similarity_passed` is false (see `metrics.HARD_FAIL_FLAGS`).
