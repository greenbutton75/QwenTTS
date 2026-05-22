"""QwenTTS quality benchmark driver.

Drives the production splice path (``POST /phrases/splice-test``) over HTTP for
every (voice x greeting x body x run) combination, saves the returned audio,
and records timing + quality signals. When ``--local-eval`` is set (and run on
the GPU box that has the model + Whisper), each saved wav is scored with
``server.quality.evaluate_candidate`` to get the full QualityReport including
ASR metrics.

This must run against a live server -- it measures the real pipeline, not the
lab functions. It cannot run in a CPU-only dev environment.

Example:
    python -m bench.run_benchmark \\
        --voices bench/voices.json \\
        --texts bench/texts.json \\
        --runs-per-pair 5 \\
        --base-url http://127.0.0.1:8000 \\
        --output-dir bench/results/baseline_v1 \\
        --local-eval
"""

import argparse
import itertools
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from . import metrics
from . import report as report_mod


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _post_splice_test(
    base_url: str,
    support_id: str,
    voice_id: str,
    greeting: str,
    body: str,
    pause_ms: int,
    crossfade_ms: int,
    target_lufs: float,
    timeout: float,
):
    payload = json.dumps(
        {
            "support_id": support_id,
            "voice_id": voice_id,
            "greeting": greeting,
            "body": body,
            "pause_ms": pause_ms,
            "crossfade_ms": crossfade_ms,
            "content_aware": True,
            "target_lufs": target_lufs,
            "mode": "wav_splice",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/phrases/splice-test",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        wav_bytes = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}
    latency_ms = int((time.time() - started) * 1000)
    return wav_bytes, headers, latency_ms


def _headers_to_quality(headers: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort quality_report derived from splice-test X- headers
    (used when local eval is off)."""
    def _b(name: str) -> Optional[bool]:
        val = headers.get(name)
        return None if val is None else val.strip().lower() == "true"

    sim = headers.get("x-greeting-similarity")
    return {
        "similarity": float(sim) if sim else None,
        "similarity_passed": _b("x-greeting-similarity-passed"),
        "onset_artifact": _b("x-greeting-onset-artifact"),
        "preroll_artifact": _b("x-greeting-preroll-artifact"),
        "ending_artifact": _b("x-greeting-ending-artifact"),
        "asr_wer": None,
        "composite_score": None,
    }


class _LocalEvaluator:
    """Loads voice reference embeddings and scores wavs on the GPU box."""

    def __init__(self) -> None:
        from server.config import S3_PREFIX  # noqa
        from server import quality, tts  # noqa
        from server.s3_store import download_torch  # noqa

        self._S3_PREFIX = S3_PREFIX
        self._quality = quality
        self._tts = tts
        self._download_torch = download_torch
        self._prompt_cache: Dict[str, dict] = {}

    def _prompt(self, support_id: str, voice_id: str) -> dict:
        key = f"{support_id}/{voice_id}"
        if key not in self._prompt_cache:
            path = f"{self._S3_PREFIX}/{support_id}/voices/{voice_id}/prompt.pt"
            self._prompt_cache[key] = self._download_torch(path)
        return self._prompt_cache[key]

    def evaluate(self, support_id, voice_id, wav_bytes, target_text) -> Dict[str, Any]:
        prompt = self._prompt(support_id, voice_id)
        wav, sr = self._tts.wav_from_bytes(wav_bytes)
        started = time.time()
        report = self._quality.evaluate_candidate(
            wav=wav,
            sr=sr,
            text=target_text,
            reference_embedding=prompt["ref_spk_embedding"],
            target_text=target_text,
            ref_text=prompt.get("ref_text"),
            is_greeting=False,
            do_asr=True,
        )
        asr_ms = int((time.time() - started) * 1000)
        asr = report.asr
        qr = {
            "similarity": round(report.similarity, 6),
            "similarity_passed": report.similarity_passed,
            "asr_transcript": asr.transcript_raw if asr else None,
            "asr_wer": round(asr.wer, 6) if asr else None,
            "asr_cer": round(asr.cer, 6) if asr else None,
            "asr_prefix_extra": bool(asr.has_prefix_extra) if asr else None,
            "asr_suffix_clipped": bool(asr.has_suffix_clipped) if asr else None,
            "asr_reference_leak": bool(asr.has_reference_leak) if asr else None,
            "onset_artifact": report.onset_artifact,
            "preroll_artifact": report.preroll_artifact,
            "ending_artifact": report.ending_artifact,
            "start_artifact": report.start_artifact,
            "trailing_rebound_artifact": report.trailing_rebound_artifact,
            "clipped_ending_artifact": report.clipped_ending_artifact,
            "composite_score": round(report.composite_score, 6),
            "all_checks_passed": report.all_checks_passed,
        }
        return qr, asr_ms


def run(args: argparse.Namespace) -> Dict[str, Any]:
    voices = _load_json(args.voices)["voices"]
    texts = _load_json(args.texts)
    greetings = texts["greetings"]
    bodies = texts["bodies"]

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")

    evaluator = _LocalEvaluator() if args.local_eval else None
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    pairs = list(itertools.product(greetings, bodies))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    records: List[dict] = []
    with open(results_path, "w", encoding="utf-8") as out:
        for voice in voices:
            for greeting, body in pairs:
                for run_index in range(args.runs_per_pair):
                    target_text = f"{greeting['text']} {body['text']}".strip()
                    try:
                        wav_bytes, headers, latency_ms = _post_splice_test(
                            args.base_url,
                            voice["support_id"],
                            voice["voice_id"],
                            greeting["text"],
                            body["text"],
                            args.pause_ms,
                            args.crossfade_ms,
                            args.target_lufs,
                            args.timeout,
                        )
                    except Exception as exc:  # noqa: BLE001 -- record and continue
                        record = {
                            "run_id": run_id,
                            "voice_id": voice["voice_id"],
                            "voice_tag": voice.get("tag", voice["voice_id"]),
                            "text_id": greeting.get("id", ""),
                            "text_category": greeting.get("category", "greeting"),
                            "text": target_text,
                            "run_index": run_index,
                            "error": str(exc),
                            "quality_report": None,
                        }
                        records.append(record)
                        out.write(json.dumps(record) + "\n")
                        continue

                    fname = f"{voice.get('tag', voice['voice_id'])}_{greeting.get('id','g')}_{body.get('id','b')}_{run_index}.wav"
                    audio_path = os.path.join(audio_dir, fname)
                    with open(audio_path, "wb") as af:
                        af.write(wav_bytes)

                    asr_ms = None
                    if evaluator is not None:
                        try:
                            quality_report, asr_ms = evaluator.evaluate(
                                voice["support_id"], voice["voice_id"], wav_bytes, target_text
                            )
                        except Exception as exc:  # noqa: BLE001
                            quality_report = {"local_eval_error": str(exc)}
                    else:
                        quality_report = _headers_to_quality(headers)

                    record = {
                        "run_id": run_id,
                        "voice_id": voice["voice_id"],
                        "voice_tag": voice.get("tag", voice["voice_id"]),
                        "text_id": greeting.get("id", ""),
                        "text_category": greeting.get("category", "greeting"),
                        "text": target_text,
                        "run_index": run_index,
                        "audio_path": os.path.relpath(audio_path, args.output_dir),
                        "body_cache": headers.get("x-body-cache"),
                        "latency_total_ms": latency_ms + (asr_ms or 0),
                        "latency_generate_ms": latency_ms,
                        "latency_asr_ms": asr_ms,
                        "quality_report": quality_report,
                    }
                    records.append(record)
                    out.write(json.dumps(record) + "\n")
                    out.flush()

    summary = metrics.aggregate(records)
    summary["run_id"] = run_id
    summary["worst_cases"] = metrics.worst_cases(records, n=15)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_mod.write_report(summary, args.output_dir)
    print(f"Done. {summary['total_generations']} generations, "
          f"overall_fail_rate={summary['overall_fail_rate']}")
    print(f"Results: {results_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QwenTTS quality benchmark")
    p.add_argument("--voices", default="bench/voices.json")
    p.add_argument("--texts", default="bench/texts.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs-per-pair", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=0, help="cap greeting*body pairs (0 = no cap)")
    p.add_argument("--pause-ms", type=int, default=220)
    p.add_argument("--crossfade-ms", type=int, default=10)
    p.add_argument("--target-lufs", type=float, default=-16.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--run-id", default="")
    p.add_argument("--local-eval", action="store_true",
                   help="score each wav locally with server.quality (requires GPU box)")
    p.add_argument("--asr-diagnostic", action="store_true",
                   help="informational flag; ensure server runs with ASR_DIAGNOSTIC_MODE=true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
