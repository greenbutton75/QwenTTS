"""ASR (Whisper) layer for the QwenTTS quality gate.

Phase 1 scope: transcribe generated audio and derive content-level quality
signals (WER/CER, extra prefix, clipped suffix, reference leak). Used in
diagnostic mode only -- no rejection happens here.

Design notes:
- Heavy dependencies (faster-whisper, torch, librosa) are imported lazily
  inside the functions that need them, mirroring ``server.tts._get_model``.
  Importing this module never requires a GPU or Whisper to be installed.
- The transcript analysis is split into a pure function ``analyze_transcript``
  so the detection logic can be unit-tested without running a model.
- WER/CER use a small self-contained edit distance over normalized tokens.
  This matches the standard WER definition (edit distance over word lists /
  reference length) and keeps the module dependency-free and testable. See
  IMPLEMENTATION_NOTES.md for why we did not hard-depend on ``jiwer`` here.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .config import (
    ASR_BEAM_SIZE,
    ASR_COMPUTE_TYPE,
    ASR_MODEL_SIZE,
    ASR_NO_SPEECH_THRESHOLD,
)

logger = logging.getLogger(__name__)

_ASR_MODEL = None
_ASR_LOCK = threading.RLock()
_TARGET_SR = 16000

# How many consecutive ref_text words at the transcript start count as a leak.
_REFERENCE_LEAK_MIN_RUN = 3
# Max edit distance for two words to be considered a fuzzy match.
_FUZZY_WORD_MAX_DISTANCE = 1


@dataclass
class ASRReport:
    transcript_raw: str
    transcript_normalized: str
    target_normalized: str
    wer: float
    cer: float
    has_prefix_extra: bool
    prefix_extra_tokens: List[str] = field(default_factory=list)
    has_suffix_clipped: bool = False
    has_reference_leak: bool = False
    no_speech_prob: float = 0.0
    confidence_score: float = 0.0


# --------------------------------------------------------------------------- #
# Text normalization & distance helpers (pure, no model)
# --------------------------------------------------------------------------- #

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    if not text:
        return ""
    lowered = str(text).lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(cleaned.split())


def _tokens(normalized: str) -> List[str]:
    return normalized.split() if normalized else []


def _edit_distance(a: List[Any], b: List[Any]) -> int:
    """Levenshtein distance over two sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _tokens(reference)
    hyp_words = _tokens(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / float(len(ref_words))


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / float(len(ref_chars))


def _fuzzy_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    return _edit_distance(list(a), list(b)) <= _FUZZY_WORD_MAX_DISTANCE


def _find_target_start(hyp_words: List[str], target_words: List[str]) -> Optional[int]:
    """Index in ``hyp_words`` where the target phrase best begins, or None."""
    if not target_words or not hyp_words:
        return None
    first = target_words[0]
    for idx, word in enumerate(hyp_words):
        if _fuzzy_eq(word, first):
            return idx
    return None


def _has_consecutive_run(hyp_words: List[str], ref_words: List[str], min_run: int) -> bool:
    """True if the start of ``hyp_words`` contains >= min_run consecutive
    ref words appearing consecutively in ref order, within the prefix region."""
    if len(hyp_words) < min_run or len(ref_words) < min_run:
        return False
    # Only inspect the leading region of the hypothesis (a leak shows up first).
    for hyp_start in range(0, min(3, len(hyp_words) - min_run + 1)):
        window = hyp_words[hyp_start : hyp_start + min_run]
        for ref_start in range(0, len(ref_words) - min_run + 1):
            ref_window = ref_words[ref_start : ref_start + min_run]
            if all(_fuzzy_eq(w, r) for w, r in zip(window, ref_window)):
                return True
    return False


def analyze_transcript(
    transcript_raw: str,
    target_text: str,
    ref_text: Optional[str] = None,
    no_speech_prob: float = 0.0,
    confidence_score: float = 0.0,
) -> ASRReport:
    """Pure analysis of an ASR transcript against the target text.

    Separated from the model call so it can be unit-tested directly.
    """
    transcript_normalized = normalize_text(transcript_raw)
    target_normalized = normalize_text(target_text)
    hyp_words = _tokens(transcript_normalized)
    target_words = _tokens(target_normalized)

    wer = word_error_rate(target_normalized, transcript_normalized)
    cer = char_error_rate(target_normalized, transcript_normalized)

    # Empty / no-speech transcript: everything is "missing".
    if not hyp_words:
        return ASRReport(
            transcript_raw=transcript_raw,
            transcript_normalized=transcript_normalized,
            target_normalized=target_normalized,
            wer=1.0 if target_words else 0.0,
            cer=1.0 if target_words else 0.0,
            has_prefix_extra=False,
            prefix_extra_tokens=[],
            has_suffix_clipped=bool(target_words),
            has_reference_leak=False,
            no_speech_prob=float(no_speech_prob),
            confidence_score=float(confidence_score),
        )

    # Prefix extra: words spoken before the target phrase begins.
    has_prefix_extra = False
    prefix_extra_tokens: List[str] = []
    start_idx = _find_target_start(hyp_words, target_words)
    if start_idx is not None and start_idx > 0:
        has_prefix_extra = True
        prefix_extra_tokens = hyp_words[:start_idx]

    # Suffix clipped: the last target word is not present anywhere in the hyp.
    has_suffix_clipped = False
    if target_words:
        last_word = target_words[-1]
        if not any(_fuzzy_eq(w, last_word) for w in hyp_words):
            has_suffix_clipped = True

    # Reference leak: consecutive run of ref_text words at the transcript start.
    has_reference_leak = False
    if ref_text:
        ref_words = _tokens(normalize_text(ref_text))
        has_reference_leak = _has_consecutive_run(
            hyp_words, ref_words, _REFERENCE_LEAK_MIN_RUN
        )

    return ASRReport(
        transcript_raw=transcript_raw,
        transcript_normalized=transcript_normalized,
        target_normalized=target_normalized,
        wer=wer,
        cer=cer,
        has_prefix_extra=has_prefix_extra,
        prefix_extra_tokens=prefix_extra_tokens,
        has_suffix_clipped=has_suffix_clipped,
        has_reference_leak=has_reference_leak,
        no_speech_prob=float(no_speech_prob),
        confidence_score=float(confidence_score),
    )


# --------------------------------------------------------------------------- #
# Whisper model (lazy singleton)
# --------------------------------------------------------------------------- #


def _get_asr_model():
    global _ASR_MODEL
    with _ASR_LOCK:
        if _ASR_MODEL is None:
            from faster_whisper import WhisperModel
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = ASR_COMPUTE_TYPE if device == "cuda" else "int8"
            logger.info(
                "Loading Whisper model size=%s device=%s compute_type=%s",
                ASR_MODEL_SIZE,
                device,
                compute_type,
            )
            _ASR_MODEL = WhisperModel(ASR_MODEL_SIZE, device=device, compute_type=compute_type)
        return _ASR_MODEL


def _to_float_mono(wav: np.ndarray) -> np.ndarray:
    audio = np.asarray(wav)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32)


def _run_whisper(audio_16k: np.ndarray, target_text: Optional[str]) -> Tuple[str, float, float]:
    """Run Whisper and return (transcript_raw, no_speech_prob, confidence)."""
    model = _get_asr_model()
    segments, _info = model.transcribe(
        audio_16k,
        language="en",
        task="transcribe",
        beam_size=ASR_BEAM_SIZE,
        best_of=ASR_BEAM_SIZE,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=(target_text or None),
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200},
        word_timestamps=True,
        no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
    )

    texts: List[str] = []
    no_speech_probs: List[float] = []
    logprobs: List[float] = []
    for seg in segments:
        texts.append(getattr(seg, "text", "") or "")
        no_speech_probs.append(float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
        logprobs.append(float(getattr(seg, "avg_logprob", 0.0) or 0.0))

    transcript_raw = "".join(texts).strip()
    no_speech_prob = max(no_speech_probs) if no_speech_probs else 1.0
    confidence_score = float(np.mean(logprobs)) if logprobs else 0.0
    return transcript_raw, no_speech_prob, confidence_score


def transcribe_audio(
    wav: np.ndarray,
    sr: int,
    target_text: str,
    ref_text: Optional[str] = None,
) -> ASRReport:
    """Transcribe ``wav`` and analyze it against ``target_text``.

    Requires faster-whisper at call time (imported lazily).
    """
    audio = _to_float_mono(wav)
    if int(sr) != _TARGET_SR:
        import librosa

        audio = librosa.resample(y=audio, orig_sr=int(sr), target_sr=_TARGET_SR)

    transcript_raw, no_speech_prob, confidence_score = _run_whisper(audio, target_text)
    return analyze_transcript(
        transcript_raw=transcript_raw,
        target_text=target_text,
        ref_text=ref_text,
        no_speech_prob=no_speech_prob,
        confidence_score=confidence_score,
    )
