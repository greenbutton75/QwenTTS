"""Pure aggregation helpers for the QwenTTS quality benchmark.

Kept dependency-free (numpy only) and side-effect-free so it can be unit
tested without a GPU or a running server.
"""

from typing import Any, Dict, Iterable, List

import numpy as np

# A generation counts as a failure if its composite score drops below this
# OR it trips any hard-fail flag (see HARD_FAIL_FLAGS).
FAIL_SCORE_THRESHOLD = 0.0

HARD_FAIL_FLAGS = (
    "asr_prefix_extra",
    "asr_suffix_clipped",
    "asr_reference_leak",
    "onset_artifact",
    "preroll_artifact",
    "ending_artifact",
    "start_artifact",
    "trailing_rebound_artifact",
    "clipped_ending_artifact",
)


def is_fail(quality_report: Dict[str, Any]) -> bool:
    if quality_report is None:
        return True
    if quality_report.get("similarity_passed") is False:
        return True
    for flag in HARD_FAIL_FLAGS:
        if quality_report.get(flag):
            return True
    score = quality_report.get("composite_score")
    if score is not None and float(score) < FAIL_SCORE_THRESHOLD:
        return True
    return False


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": None, "p10": None, "p90": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.percentile(arr, 50)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def _group_fail_rate(records: List[dict], key: str) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    fails: Dict[str, int] = {}
    for rec in records:
        bucket = str(rec.get(key, "unknown"))
        totals[bucket] = totals.get(bucket, 0) + 1
        if is_fail(rec.get("quality_report")):
            fails[bucket] = fails.get(bucket, 0) + 1
    return {bucket: _rate(fails.get(bucket, 0), total) for bucket, total in sorted(totals.items())}


def aggregate(records: Iterable[dict]) -> Dict[str, Any]:
    """Compute the summary.json aggregates from per-generation records."""
    records = list(records)
    total = len(records)
    qrs = [r.get("quality_report") or {} for r in records]

    overall_fails = sum(1 for r in records if is_fail(r.get("quality_report")))

    artifact_breakdown = {flag: sum(1 for qr in qrs if qr.get(flag)) for flag in HARD_FAIL_FLAGS}

    similarities = [float(qr["similarity"]) for qr in qrs if qr.get("similarity") is not None]
    wers = [float(qr["asr_wer"]) for qr in qrs if qr.get("asr_wer") is not None]
    scores = [float(qr["composite_score"]) for qr in qrs if qr.get("composite_score") is not None]

    latency_total = [float(r["latency_total_ms"]) for r in records if r.get("latency_total_ms") is not None]
    latency_gen = [float(r["latency_generate_ms"]) for r in records if r.get("latency_generate_ms") is not None]
    latency_asr = [float(r["latency_asr_ms"]) for r in records if r.get("latency_asr_ms") is not None]

    return {
        "total_generations": total,
        "overall_fail_rate": _rate(overall_fails, total),
        "overall_fail_count": overall_fails,
        "per_category_fail_rate": _group_fail_rate(records, "text_category"),
        "per_voice_fail_rate": _group_fail_rate(records, "voice_tag"),
        "artifact_breakdown": artifact_breakdown,
        "similarity_distribution": _percentiles(similarities),
        "wer_distribution": _percentiles(wers),
        "composite_score_distribution": _percentiles(scores),
        "latency_total_ms": _percentiles(latency_total),
        "latency_generate_ms": _percentiles(latency_gen),
        "latency_asr_ms": _percentiles(latency_asr),
    }


def worst_cases(records: Iterable[dict], n: int = 15) -> List[dict]:
    """Return the n records with the lowest composite score (for manual QA)."""
    def score(rec):
        qr = rec.get("quality_report") or {}
        s = qr.get("composite_score")
        return float(s) if s is not None else float("inf")

    return sorted(list(records), key=score)[:n]
