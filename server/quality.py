"""Unified per-candidate quality evaluation for the QwenTTS quality gate.

Combines the existing heuristic artifact detectors (from ``server.tts``),
speaker similarity, and the new ASR content checks (from ``server.asr``) into a
single ``QualityReport`` plus a ``composite_score``.

Phase 1: used in diagnostic mode only. ``evaluate_candidate`` performs no
rejection; callers decide what to do with the report.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .asr import ASRReport, transcribe_audio
from .config import (
    GREETING_ASR_MAX_WER,
    GREETING_ASR_PREFIX_EXTRA_REJECT,
    GREETING_ASR_REFERENCE_LEAK_REJECT,
    GREETING_ASR_SUFFIX_CLIPPED_REJECT,
    GREETING_SPEAKER_SIMILARITY_THRESHOLD,
    score_weights,
)
from . import tts

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    similarity: float
    similarity_passed: bool
    asr: Optional[ASRReport]
    asr_passed: bool
    onset_artifact: bool = False
    preroll_artifact: bool = False
    ending_artifact: bool = False
    duration_artifact: bool = False
    start_artifact: bool = False
    trailing_rebound_artifact: bool = False
    clipped_ending_artifact: bool = False
    composite_score: float = 0.0
    all_checks_passed: bool = False
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


def composite_score(report: "QualityReport", weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted quality score (higher is better). See spec section 1.4."""
    w = weights or score_weights()
    asr = report.asr
    asr_wer = asr.wer if asr is not None else 0.0
    has_prefix = bool(asr.has_prefix_extra) if asr is not None else False
    has_suffix = bool(asr.has_suffix_clipped) if asr is not None else False
    has_refleak = bool(asr.has_reference_leak) if asr is not None else False

    score = 0.0
    score += w["w_sim"] * float(report.similarity)
    score += w["w_asr"] * max(0.0, 1.0 - float(asr_wer))
    score -= w["p_prefix"] * int(has_prefix)
    score -= w["p_suffix"] * int(has_suffix)
    score -= w["p_refleak"] * int(has_refleak)
    score -= w["p_onset"] * int(report.onset_artifact)
    score -= w["p_preroll"] * int(report.preroll_artifact)
    score -= w["p_ending"] * int(report.ending_artifact)
    score -= w["p_body_start"] * int(report.start_artifact)
    score -= w["p_body_tail"] * int(report.trailing_rebound_artifact)
    score -= w["p_body_clipped"] * int(report.clipped_ending_artifact)
    return float(score)


def _asr_passed(asr: Optional[ASRReport]) -> bool:
    if asr is None:
        return True
    if asr.wer > GREETING_ASR_MAX_WER:
        return False
    if GREETING_ASR_PREFIX_EXTRA_REJECT and asr.has_prefix_extra:
        return False
    if GREETING_ASR_SUFFIX_CLIPPED_REJECT and asr.has_suffix_clipped:
        return False
    if GREETING_ASR_REFERENCE_LEAK_REJECT and asr.has_reference_leak:
        return False
    return True


def evaluate_candidate(
    wav: np.ndarray,
    sr: int,
    text: str,
    reference_embedding: Any,
    target_text: str,
    ref_text: Optional[str] = None,
    is_greeting: bool = False,
    do_asr: bool = True,
    similarity_threshold: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> QualityReport:
    """Evaluate one generated candidate.

    ``text`` is the spoken text used for the heuristic detectors; ``target_text``
    is what ASR compares against (usually identical).
    """
    threshold = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else GREETING_SPEAKER_SIMILARITY_THRESHOLD
    )

    similarity = tts.speaker_similarity(wav, sr, reference_embedding)
    similarity_passed = similarity >= threshold

    raw_metrics: Dict[str, Any] = {}
    onset_artifact = preroll_artifact = ending_artifact = duration_artifact = False
    start_artifact = trailing_rebound_artifact = clipped_ending_artifact = False

    if is_greeting:
        onset_stats = tts.detect_greeting_onset_artifact(text, wav, sr)
        preroll_stats = tts.detect_greeting_leading_preroll_artifact(text, wav, sr)
        duration_stats = tts.detect_greeting_excessive_duration_artifact(text, wav, sr)
        ending_stats = tts.detect_greeting_clipped_ending_artifact(text, wav, sr)
        onset_artifact = bool(onset_stats.get("artifact", 0))
        preroll_artifact = bool(preroll_stats.get("artifact", 0))
        duration_artifact = bool(duration_stats.get("artifact", 0))
        ending_artifact = bool(ending_stats.get("artifact", 0))
        raw_metrics["onset"] = onset_stats
        raw_metrics["preroll"] = preroll_stats
        raw_metrics["duration"] = duration_stats
        raw_metrics["ending"] = ending_stats
    else:
        body_stats = tts.detect_body_boundary_artifacts(text, wav, sr)
        start_artifact = bool(body_stats.get("start_artifact", 0))
        trailing_rebound_artifact = bool(body_stats.get("trailing_rebound_artifact", 0))
        clipped_ending_artifact = bool(body_stats.get("clipped_ending_artifact", 0))
        raw_metrics["body_boundary"] = body_stats

    asr: Optional[ASRReport] = None
    if do_asr:
        asr = transcribe_audio(wav, sr, target_text, ref_text=ref_text)

    asr_passed = _asr_passed(asr)

    report = QualityReport(
        similarity=float(similarity),
        similarity_passed=bool(similarity_passed),
        asr=asr,
        asr_passed=bool(asr_passed),
        onset_artifact=onset_artifact,
        preroll_artifact=preroll_artifact,
        ending_artifact=ending_artifact,
        duration_artifact=duration_artifact,
        start_artifact=start_artifact,
        trailing_rebound_artifact=trailing_rebound_artifact,
        clipped_ending_artifact=clipped_ending_artifact,
        raw_metrics=raw_metrics,
    )
    report.composite_score = composite_score(report, weights)
    report.all_checks_passed = bool(
        similarity_passed
        and asr_passed
        and not onset_artifact
        and not preroll_artifact
        and not ending_artifact
        and not start_artifact
        and not trailing_rebound_artifact
        and not clipped_ending_artifact
    )
    return report


def diagnostic_phrase_fields(report: QualityReport, gate_decision: str = "skipped") -> Dict[str, Any]:
    """Flatten a QualityReport into the optional phrase_json fields (schema v2).

    Only called when a diagnostic flag is enabled, so these keys never appear
    in baseline phrase_json output.
    """
    asr = report.asr
    return {
        "schema_version": 2,
        "asr_transcript": asr.transcript_raw if asr is not None else None,
        "asr_wer": round(asr.wer, 6) if asr is not None else None,
        "asr_cer": round(asr.cer, 6) if asr is not None else None,
        "asr_prefix_extra": bool(asr.has_prefix_extra) if asr is not None else None,
        "asr_suffix_clipped": bool(asr.has_suffix_clipped) if asr is not None else None,
        "asr_reference_leak": bool(asr.has_reference_leak) if asr is not None else None,
        "asr_no_speech_prob": round(asr.no_speech_prob, 6) if asr is not None else None,
        "composite_score": round(report.composite_score, 6),
        "quality_gate_decision": gate_decision,
    }
