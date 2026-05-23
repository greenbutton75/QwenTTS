"""Best-of-N candidate generation + selection for greetings (Phase 2).

Generates several greeting renders with varied decoding, scores each with the
unified quality evaluator (similarity + greeting artifact heuristics + ASR),
and picks the best by composite score. This replaces the single greedy render +
serial retry for greetings when ``GREETING_BEST_OF_N_ENABLED`` is on.

Generation is sequential with a fixed seed per candidate (deterministic). A
batched path was intentionally not implemented yet -- see IMPLEMENTATION_NOTES.md
(open question: does Qwen3TTSModel.generate_voice_clone support batching).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    BODY_ASR_MAX_WER,
    BODY_BEST_OF_N_MAX_COUNT,
    GREETING_ASR_CHECK,
    GREETING_ASR_MAX_WER,
    GREETING_SPEAKER_SIMILARITY_THRESHOLD,
    VOICE_CLONE_MAX_NEW_TOKENS,
    VOICE_CLONE_NON_STREAMING_MODE,
    VOICE_CLONE_REPETITION_PENALTY,
    GREETING_SPLICE_MAX_NEW_TOKENS,
)
from . import quality
from . import tts

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpec:
    label: str
    generate_config: Dict[str, Any]
    seed: Optional[int] = None


@dataclass
class Candidate:
    spec: CandidateSpec
    wav: np.ndarray
    sr: int
    generate_latency_ms: int
    report: "quality.QualityReport"
    score: float = 0.0


def _base_specs() -> List[CandidateSpec]:
    """Greedy first (most stable), then increasingly diverse sampled renders."""
    common = {
        "non_streaming_mode": VOICE_CLONE_NON_STREAMING_MODE,
        "max_new_tokens": GREETING_SPLICE_MAX_NEW_TOKENS,
        "repetition_penalty": VOICE_CLONE_REPETITION_PENALTY,
    }
    return [
        CandidateSpec("greedy", {**common, "do_sample": False}, seed=None),
        CandidateSpec("sample_lo", {**common, "do_sample": True, "temperature": 0.25, "top_k": 6, "top_p": 0.9}, seed=42),
        CandidateSpec("sample_hi", {**common, "do_sample": True, "temperature": 0.35, "top_k": 10, "top_p": 0.85}, seed=137),
        CandidateSpec("sample_var", {**common, "do_sample": True, "temperature": 0.45, "top_k": 12, "top_p": 0.9}, seed=2027),
    ]


def _build_specs(n: int) -> List[CandidateSpec]:
    base = _base_specs()
    if n <= len(base):
        return base[: max(1, n)]
    # Extra candidates beyond the base set: more diverse sampled renders.
    specs = list(base)
    for i in range(len(base), n):
        specs.append(
            CandidateSpec(
                f"sample_extra_{i}",
                {**base[-1].generate_config, "temperature": 0.4 + 0.05 * (i - len(base) + 1)},
                seed=1000 + i,
            )
        )
    return specs


def _seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def generate_greeting_candidates(
    text: str,
    voice_prompt,
    reference_embedding: Any,
    target_text: Optional[str] = None,
    ref_text: Optional[str] = None,
    n: int = 4,
    similarity_threshold: Optional[float] = None,
    do_asr: Optional[bool] = None,
    adaptive: bool = False,
) -> List[Candidate]:
    """Generate greeting candidates and let the caller pick the best.

    With ``adaptive=True`` the greedy render is generated first and kept alone if
    it is already good (similarity ok, no greeting artifact, not over-long, ASR
    clean). Only a bad greedy triggers the extra sampled renders. This keeps the
    common case at one generation while still protecting the bad cases.
    """
    target = target_text if target_text is not None else text
    do_asr = GREETING_ASR_CHECK if do_asr is None else do_asr
    threshold = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else GREETING_SPEAKER_SIMILARITY_THRESHOLD
    )

    candidates: List[Candidate] = []
    for spec in _build_specs(int(n)):
        _seed(spec.seed)
        started = time.time()
        wav, sr = tts.generate_voice(text, voice_prompt, generate_config=spec.generate_config)
        gen_ms = int((time.time() - started) * 1000)
        report = quality.evaluate_candidate(
            wav=wav,
            sr=sr,
            text=text,
            reference_embedding=reference_embedding,
            target_text=target,
            ref_text=ref_text,
            is_greeting=True,
            do_asr=do_asr,
            similarity_threshold=threshold,
        )
        candidates.append(
            Candidate(spec=spec, wav=wav, sr=sr, generate_latency_ms=gen_ms, report=report, score=report.composite_score)
        )
        if adaptive and len(candidates) == 1 and _greeting_is_good(report):
            break
    return candidates


def _greeting_is_good(report: "quality.QualityReport") -> bool:
    """Whether a greedy greeting is good enough to keep without best-of-N:
    timbre ok, no onset/preroll/ending artifact, not over-long (runaway), and
    ASR clean (right words, name present, no leak)."""
    if not report.similarity_passed:
        return False
    if report.onset_artifact or report.preroll_artifact or report.ending_artifact:
        return False
    if report.duration_artifact:
        return False
    asr = report.asr
    if asr is not None:
        if asr.wer > GREETING_ASR_MAX_WER:
            return False
        if asr.has_prefix_extra or asr.has_suffix_clipped or asr.has_reference_leak:
            return False
    return True


def _body_specs(n: int) -> List[CandidateSpec]:
    """Like _base_specs but with the full body token budget."""
    common = {
        "non_streaming_mode": VOICE_CLONE_NON_STREAMING_MODE,
        "max_new_tokens": VOICE_CLONE_MAX_NEW_TOKENS,
        "repetition_penalty": VOICE_CLONE_REPETITION_PENALTY,
    }
    specs = [
        CandidateSpec("greedy", {**common, "do_sample": False}, seed=None),
        CandidateSpec("sample_lo", {**common, "do_sample": True, "temperature": 0.25, "top_k": 6, "top_p": 0.9}, seed=42),
        CandidateSpec("sample_hi", {**common, "do_sample": True, "temperature": 0.35, "top_k": 10, "top_p": 0.85}, seed=137),
    ]
    while len(specs) < n:
        i = len(specs)
        specs.append(CandidateSpec(f"sample_extra_{i}", {**specs[-1].generate_config, "temperature": 0.4 + 0.05 * i}, seed=1000 + i))
    return specs[: max(1, n)]


def _body_is_good(report: "quality.QualityReport", max_wer: float) -> bool:
    if not report.similarity_passed:
        return False
    if report.start_artifact or report.trailing_rebound_artifact or report.clipped_ending_artifact:
        return False
    if report.asr is not None and report.asr.wer > max_wer:
        return False
    return True


def body_candidate_acceptable(report: "quality.QualityReport") -> bool:
    """Whether a selected body candidate is good enough to ship. Uses the body
    WER bar (BODY_ASR_MAX_WER), NOT the strict greeting bar inside
    QualityReport.all_checks_passed -- spelled-out phone numbers legitimately
    push body WER into the 0.20-0.35 range."""
    return _body_is_good(report, BODY_ASR_MAX_WER)


def generate_body_candidates(
    text: str,
    voice_prompt,
    reference_embedding: Any,
    ref_text: Optional[str] = None,
    max_n: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    do_asr: bool = True,
    max_wer: Optional[float] = None,
) -> List[Candidate]:
    """Adaptive best-of-N for the body / long render.

    Generates the greedy render first; if it is good (similarity ok, no boundary
    artifact, ASR WER within ``max_wer``) it is returned alone. Otherwise up to
    ``max_n`` sampled renders are generated and all are returned for selection.
    Each candidate is post-processed with the tail-preserving cleanup, matching
    the existing body path.
    """
    n = int(max_n or BODY_BEST_OF_N_MAX_COUNT)
    wer_bar = BODY_ASR_MAX_WER if max_wer is None else float(max_wer)
    threshold = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else GREETING_SPEAKER_SIMILARITY_THRESHOLD
    )

    candidates: List[Candidate] = []
    for spec in _body_specs(n):
        _seed(spec.seed)
        started = time.time()
        wav, sr = tts.generate_voice(text, voice_prompt, generate_config=spec.generate_config)
        wav, sr, _ = tts.clean_output_audio_preserve_tail(wav, sr)
        gen_ms = int((time.time() - started) * 1000)
        report = quality.evaluate_candidate(
            wav=wav,
            sr=sr,
            text=text,
            reference_embedding=reference_embedding,
            target_text=text,
            ref_text=ref_text,
            is_greeting=False,
            do_asr=do_asr,
            similarity_threshold=threshold,
        )
        candidates.append(
            Candidate(spec=spec, wav=wav, sr=sr, generate_latency_ms=gen_ms, report=report, score=report.composite_score)
        )
        # Adaptive: keep the greedy render if it is already good.
        if len(candidates) == 1 and _body_is_good(report, wer_bar):
            break
    return candidates


def select_best(candidates: List[Candidate]) -> Candidate:
    """Prefer a candidate that passes all hard checks; else the highest score."""
    if not candidates:
        raise ValueError("no candidates to select from")
    clean = [c for c in candidates if c.report.all_checks_passed]
    pool = clean if clean else candidates
    return max(pool, key=lambda c: c.score)
