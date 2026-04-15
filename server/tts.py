import io
import logging
import re
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from timing_utils import timed_operation

from .config import (
    BODY_QUALITY_CHECK,
    BODY_QUALITY_MAX_ATTEMPTS,
    BODY_QUALITY_REQUIRE_PASS,
    GREETING_ONSET_ARTIFACT_CHECK,
    GREETING_OUTPUT_TRIM_PAD_MS,
    LANGUAGE,
    OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS,
    MODEL_SIZE,
    OUTPUT_AUDIO_TRIM_ENABLED,
    OUTPUT_AUDIO_TRIM_MAX_LEADING_MS,
    OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS,
    OUTPUT_AUDIO_TRIM_PAD_MS,
    REFERENCE_AUDIO_TRIM_ENABLED,
    REFERENCE_AUDIO_TRIM_MAX_LEADING_MS,
    REFERENCE_AUDIO_TRIM_MAX_TRAILING_MS,
    REFERENCE_AUDIO_TRIM_PAD_MS,
    VOICE_CLONE_MAX_NEW_TOKENS,
    body_quality_retry_generate_config,
    greeting_splice_generate_config,
    greeting_splice_retry_generate_config,
    greeting_similarity_retry_generate_config,
    voice_clone_generate_config,
)


_MODEL_CACHE: Dict[str, Qwen3TTSModel] = {}
_VOICE_CLONE_GENERATE_CONFIG = voice_clone_generate_config()
_GREETING_RETRY_GENERATE_CONFIG = greeting_similarity_retry_generate_config()
_GREETING_SPLICE_GENERATE_CONFIG = greeting_splice_generate_config()
_GREETING_SPLICE_RETRY_GENERATE_CONFIG = greeting_splice_retry_generate_config()
_BODY_RETRY_GENERATE_CONFIG = body_quality_retry_generate_config()
_H_GREETING_RE = re.compile(r"^\s*(hi|hello)\b", re.IGNORECASE)
_SHORT_GREETING_RE = re.compile(r"^\s*(hi|hello)\s*[!,]?\s*([a-zA-Z][a-zA-Z'\-]*)\s*[!,.]?\s*$", re.IGNORECASE)
_MODEL_LOCK = threading.RLock()
_FATAL_CUDA_ERROR_MARKERS = (
    "device-side assert triggered",
    "cuda error",
    "cublas",
    "cudnn",
    "illegal memory access",
    "an illegal memory access was encountered",
    "unspecified launch failure",
    "misaligned address",
    "device kernel image is invalid",
)


def _get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> Qwen3TTSModel:
    key = f"Base-{MODEL_SIZE}"
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        import torch
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base")
        device = _get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        _MODEL_CACHE[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        return _MODEL_CACHE[key]


def is_fatal_cuda_error(exc_or_text: Any) -> bool:
    text = str(exc_or_text or "").lower()
    return any(marker in text for marker in _FATAL_CUDA_ERROR_MARKERS)


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + 1e-12)
    else:
        raise TypeError(f"Unsupported audio dtype: {x.dtype}")

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(path, always_2d=False)
        return _normalize_audio(wav), int(sr)
    except Exception:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        sr = audio.frame_rate
        wav = np.array(audio.get_array_of_samples())
        return _normalize_audio(wav), int(sr)


def bytes_to_wav_file(data: bytes, suffix: str = ".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def create_voice_prompt(
    ref_audio: Tuple[np.ndarray, int],
    ref_text: Optional[str],
    x_vector_only: bool,
) -> VoiceClonePromptItem:
    with _MODEL_LOCK:
        tts = _get_model()
        items = tts.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=(ref_text.strip() if ref_text else None),
            x_vector_only_mode=bool(x_vector_only),
        )
    if not items:
        raise RuntimeError("Voice prompt creation returned empty result.")
    return items[0]


def _effective_generate_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = dict(_VOICE_CLONE_GENERATE_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def generate_voice(
    text: str,
    voice_prompt: List[VoiceClonePromptItem],
    generate_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, int]:
    with _MODEL_LOCK:
        tts = _get_model()
        wavs, sr = tts.generate_voice_clone(
            text=text,
            language=LANGUAGE,
            voice_clone_prompt=voice_prompt,
            **_effective_generate_config(generate_config),
        )
    return wavs[0], sr


def _embedding_to_numpy(embedding: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(embedding):
            return embedding.detach().cpu().float().numpy().reshape(-1)
    except Exception:
        pass
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def extract_speaker_embedding(audio: np.ndarray, sr: int):
    with _MODEL_LOCK:
        tts = _get_model()
        wav = _normalize_audio(audio)
        target_sr = int(tts.model.speaker_encoder_sample_rate)
        if int(sr) != target_sr:
            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
        return tts.model.extract_speaker_embedding(audio=wav, sr=target_sr)


def speaker_similarity(audio: np.ndarray, sr: int, reference_embedding: Any) -> float:
    generated = _embedding_to_numpy(extract_speaker_embedding(audio, sr))
    reference = _embedding_to_numpy(reference_embedding)
    denom = float(np.linalg.norm(generated) * np.linalg.norm(reference))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(generated, reference) / denom)


def generate_voice_with_similarity_retry(
    text: str,
    voice_prompt: List[VoiceClonePromptItem],
    reference_embedding: Any,
    min_similarity: float,
    max_attempts: int,
    initial_generate_config: Optional[Dict[str, Any]] = None,
    retry_generate_config: Optional[Dict[str, Any]] = None,
    timing_logger: Optional[logging.Logger] = None,
    attempt_operation: Optional[str] = None,
    timing_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, int, float, int, bool, Dict[str, float]]:
    total_attempts = max(1, int(max_attempts))
    best_wav = None
    best_sr = 0
    best_similarity = float("-inf")
    best_quality = {
        "similarity_passed": 0,
        "onset_artifact": 0,
        "onset_checked": 0,
        "onset_passed": 1,
        "duration_artifact": 0,
        "duration_checked": 0,
        "duration_passed": 1,
        "ending_artifact": 0,
        "ending_checked": 0,
        "ending_passed": 1,
        "preroll_artifact": 0,
        "preroll_checked": 0,
        "preroll_passed": 1,
        "start_passed": 1,
        "greeting_passed": 1,
    }
    best_clean_wav = None
    best_clean_sr = 0
    best_clean_similarity = float("-inf")
    best_clean_quality = dict(best_quality)
    base_timing_fields = dict(timing_fields or {})

    for attempt in range(1, total_attempts + 1):
        if attempt == 1:
            generate_config = None if initial_generate_config is None else dict(initial_generate_config)
        else:
            generate_config = dict(_GREETING_RETRY_GENERATE_CONFIG)
            if retry_generate_config is not None:
                generate_config.update(retry_generate_config)
        quality: Dict[str, float] = {}
        if timing_logger is not None and attempt_operation:
            with timed_operation(
                timing_logger,
                attempt_operation,
                **base_timing_fields,
                attempt=attempt,
                max_attempts=total_attempts,
                generate_max_new_tokens=(generate_config or {}).get("max_new_tokens", VOICE_CLONE_MAX_NEW_TOKENS),
                retry_mode=int(attempt > 1),
            ) as attempt_span:
                wav, sr = generate_voice(text, voice_prompt, generate_config=generate_config)
                similarity = speaker_similarity(wav, sr, reference_embedding)
                onset_stats = detect_greeting_onset_artifact(text, wav, sr)
                preroll_stats = detect_greeting_leading_preroll_artifact(text, wav, sr)
                duration_stats = detect_greeting_excessive_duration_artifact(text, wav, sr)
                ending_stats = detect_greeting_clipped_ending_artifact(text, wav, sr)
                onset_passed = int(not onset_stats.get("artifact", 0))
                preroll_passed = int(not preroll_stats.get("artifact", 0))
                duration_passed = int(not duration_stats.get("artifact", 0))
                ending_passed = int(not ending_stats.get("artifact", 0))
                quality = {
                    "similarity_passed": int(similarity >= float(min_similarity)),
                    "onset_artifact": int(onset_stats.get("artifact", 0)),
                    "onset_checked": int(onset_stats.get("checked", 0)),
                    "onset_passed": onset_passed,
                    "duration_artifact": int(duration_stats.get("artifact", 0)),
                    "duration_checked": int(duration_stats.get("checked", 0)),
                    "duration_passed": duration_passed,
                    "ending_artifact": int(ending_stats.get("artifact", 0)),
                    "ending_checked": int(ending_stats.get("checked", 0)),
                    "ending_passed": ending_passed,
                    "preroll_artifact": int(preroll_stats.get("artifact", 0)),
                    "preroll_checked": int(preroll_stats.get("checked", 0)),
                    "preroll_passed": preroll_passed,
                    "start_passed": int(onset_passed and preroll_passed),
                    "greeting_passed": int(onset_passed and preroll_passed and ending_passed),
                    **onset_stats,
                    **{f"duration_{key}": value for key, value in duration_stats.items() if key not in {"artifact", "checked"}},
                    **{f"ending_{key}": value for key, value in ending_stats.items() if key not in {"artifact", "checked"}},
                    **{f"preroll_{key}": value for key, value in preroll_stats.items() if key not in {"artifact", "checked"}},
                }
                attempt_span.set(
                    similarity=round(float(similarity), 6),
                    similarity_passed=quality["similarity_passed"],
                    start_passed=quality["start_passed"],
                    greeting_passed=quality["greeting_passed"],
                    onset_artifact=quality["onset_artifact"],
                    preroll_artifact=quality["preroll_artifact"],
                    duration_artifact=quality["duration_artifact"],
                    ending_artifact=quality["ending_artifact"],
                )
        else:
            wav, sr = generate_voice(text, voice_prompt, generate_config=generate_config)
            similarity = speaker_similarity(wav, sr, reference_embedding)
            onset_stats = detect_greeting_onset_artifact(text, wav, sr)
            preroll_stats = detect_greeting_leading_preroll_artifact(text, wav, sr)
            duration_stats = detect_greeting_excessive_duration_artifact(text, wav, sr)
            ending_stats = detect_greeting_clipped_ending_artifact(text, wav, sr)
            onset_passed = int(not onset_stats.get("artifact", 0))
            preroll_passed = int(not preroll_stats.get("artifact", 0))
            duration_passed = int(not duration_stats.get("artifact", 0))
            ending_passed = int(not ending_stats.get("artifact", 0))
            quality = {
                "similarity_passed": int(similarity >= float(min_similarity)),
                "onset_artifact": int(onset_stats.get("artifact", 0)),
                "onset_checked": int(onset_stats.get("checked", 0)),
                "onset_passed": onset_passed,
                "duration_artifact": int(duration_stats.get("artifact", 0)),
                "duration_checked": int(duration_stats.get("checked", 0)),
                "duration_passed": duration_passed,
                "ending_artifact": int(ending_stats.get("artifact", 0)),
                "ending_checked": int(ending_stats.get("checked", 0)),
                "ending_passed": ending_passed,
                "preroll_artifact": int(preroll_stats.get("artifact", 0)),
                "preroll_checked": int(preroll_stats.get("checked", 0)),
                "preroll_passed": preroll_passed,
                "start_passed": int(onset_passed and preroll_passed),
                "greeting_passed": int(onset_passed and preroll_passed and ending_passed),
                **onset_stats,
                **{f"duration_{key}": value for key, value in duration_stats.items() if key not in {"artifact", "checked"}},
                **{f"ending_{key}": value for key, value in ending_stats.items() if key not in {"artifact", "checked"}},
                **{f"preroll_{key}": value for key, value in preroll_stats.items() if key not in {"artifact", "checked"}},
            }
        if similarity > best_similarity or best_wav is None:
            best_wav = wav
            best_sr = sr
            best_similarity = similarity
            best_quality = quality
        if quality["greeting_passed"] and (similarity > best_clean_similarity or best_clean_wav is None):
            best_clean_wav = wav
            best_clean_sr = sr
            best_clean_similarity = similarity
            best_clean_quality = quality
        if similarity >= float(min_similarity) and quality["greeting_passed"]:
            return wav, sr, similarity, attempt, True, quality

    if best_clean_wav is not None:
        return best_clean_wav, best_clean_sr, best_clean_similarity, total_attempts, False, best_clean_quality
    return best_wav, best_sr, best_similarity, total_attempts, False, best_quality


def greeting_splice_generate_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return dict(_GREETING_SPLICE_GENERATE_CONFIG), dict(_GREETING_SPLICE_RETRY_GENERATE_CONFIG)


def detect_body_boundary_artifacts(text: str, wav: np.ndarray, sr: int) -> Dict[str, float]:
    audio = _normalize_audio(wav)
    default = {
        "checked": 0,
        "passed": 1,
        "start_artifact": 0,
        "trailing_rebound_artifact": 0,
        "clipped_ending_artifact": 0,
        "start_gap_ms": 0,
        "start_first_run_ms": 0,
        "start_second_run_ms": 0,
        "tail_gap_ms": 0,
        "tail_last_run_ms": 0,
        "duration_ms": 0,
        "expected_min_ms": 0,
        "ending_tail_speech_ratio": 0.0,
        "ending_tail_to_pre_ratio": 0.0,
        "ending_tail_to_global_ratio": 0.0,
    }
    if not BODY_QUALITY_CHECK:
        return default
    if not isinstance(text, str) or not text.strip():
        return default
    if audio.size < max(1, int(sr * 1.6)):
        return {**default, "checked": 1}

    speech_like, _, hop_samples = _speech_frame_mask(audio, sr=int(sr), frame_ms=20, hop_ms=10)
    if speech_like.size < 20:
        return {**default, "checked": 1}

    runs = _active_runs(speech_like, min_run=3)
    if not runs:
        return {**default, "checked": 1}

    start_artifact = 0
    start_gap_ms = 0
    start_first_run_ms = 0
    start_second_run_ms = 0
    early_runs = [run for run in runs if run[0] * hop_samples < int(sr * 1.4)]
    if len(early_runs) >= 2:
        first_start, first_end = early_runs[0]
        second_start, second_end = early_runs[1]
        first_start_ms = int(round(first_start * hop_samples * 1000.0 / sr))
        start_gap_ms = int(round((second_start - first_end - 1) * hop_samples * 1000.0 / sr))
        start_first_run_ms = int(round((first_end - first_start + 1) * hop_samples * 1000.0 / sr))
        start_second_run_ms = int(round((second_end - second_start + 1) * hop_samples * 1000.0 / sr))
        start_artifact = int(
            first_start_ms <= 250
            and start_first_run_ms <= 420
            and start_gap_ms >= 120
            and start_second_run_ms >= 260
        )

    trailing_rebound_artifact = 0
    tail_gap_ms = 0
    tail_last_run_ms = 0
    late_runs = [run for run in runs if run[1] * hop_samples >= max(0, audio.size - int(sr * 2.0))]
    if len(late_runs) >= 2:
        prev_start, prev_end = late_runs[-2]
        last_start, last_end = late_runs[-1]
        tail_gap_ms = int(round((last_start - prev_end - 1) * hop_samples * 1000.0 / sr))
        tail_last_run_ms = int(round((last_end - last_start + 1) * hop_samples * 1000.0 / sr))
        trailing_from_end_ms = int(round((speech_like.size - 1 - last_end) * hop_samples * 1000.0 / sr))
        trailing_rebound_artifact = int(
            tail_gap_ms >= 220
            and tail_last_run_ms <= 520
            and trailing_from_end_ms <= 420
        )

    clipped_ending_artifact = 0
    duration_ms = int(round(audio.shape[0] * 1000.0 / sr))
    expected_min_ms = 0
    ending_tail_speech_ratio = 0.0
    ending_tail_to_pre_ratio = 0.0
    ending_tail_to_global_ratio = 0.0
    stripped_text = text.strip()
    if stripped_text.endswith((".", "!", "?")):
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", stripped_text))
        char_count = len(re.findall(r"[A-Za-z0-9]", stripped_text))
        expected_min_ms = int(min(25000, max(3000, max(word_count * 220, char_count * 42))))

        rms = _rms_envelope(audio, frame_samples=int(sr * 0.02), hop_samples=hop_samples)
        min_len = min(rms.size, speech_like.size)
        if min_len >= 8:
            rms = rms[:min_len]
            speech_like_tail = speech_like[:min_len]
            tail_frames = max(4, int(round(0.12 * sr / hop_samples)))
            pre_tail_frames = max(tail_frames, int(round(0.20 * sr / hop_samples)))
            tail = rms[-tail_frames:]
            pre_tail_end = max(0, rms.size - tail_frames)
            pre_tail_start = max(0, pre_tail_end - pre_tail_frames)
            pre_tail = rms[pre_tail_start:pre_tail_end]
            if pre_tail.size == 0:
                pre_tail = rms[:-tail_frames] if rms.size > tail_frames else rms
            ending_tail_speech_ratio = float(np.mean(speech_like_tail[-tail_frames:])) if tail_frames > 0 else 0.0
            tail_mean = float(np.mean(tail)) if tail.size else 0.0
            pre_tail_mean = float(np.mean(pre_tail)) if pre_tail.size else 0.0
            global_p90 = float(np.percentile(rms, 90))
            ending_tail_to_pre_ratio = tail_mean / max(pre_tail_mean, 1e-6)
            ending_tail_to_global_ratio = tail_mean / max(global_p90, 1e-6)
            clipped_ending_artifact = int(
                duration_ms <= expected_min_ms
                and ending_tail_speech_ratio >= 0.72
                and ending_tail_to_pre_ratio >= 0.88
                and ending_tail_to_global_ratio >= 0.34
            )

    passed = int(not start_artifact and not trailing_rebound_artifact and not clipped_ending_artifact)
    return {
        "checked": 1,
        "passed": passed,
        "start_artifact": start_artifact,
        "trailing_rebound_artifact": trailing_rebound_artifact,
        "clipped_ending_artifact": clipped_ending_artifact,
        "start_gap_ms": start_gap_ms,
        "start_first_run_ms": start_first_run_ms,
        "start_second_run_ms": start_second_run_ms,
        "tail_gap_ms": tail_gap_ms,
        "tail_last_run_ms": tail_last_run_ms,
        "duration_ms": duration_ms,
        "expected_min_ms": expected_min_ms,
        "ending_tail_speech_ratio": ending_tail_speech_ratio,
        "ending_tail_to_pre_ratio": ending_tail_to_pre_ratio,
        "ending_tail_to_global_ratio": ending_tail_to_global_ratio,
    }


def generate_body_with_quality_retry(
    text: str,
    voice_prompt: List[VoiceClonePromptItem],
    max_attempts: Optional[int] = None,
) -> Tuple[np.ndarray, int, int, bool, Dict[str, float], Dict[str, int]]:
    total_attempts = max(1, int(max_attempts or BODY_QUALITY_MAX_ATTEMPTS))
    best_wav = None
    best_sr = 0
    best_quality: Dict[str, float] = {
        "checked": 0,
        "passed": 1,
        "start_artifact": 0,
        "trailing_rebound_artifact": 0,
        "clipped_ending_artifact": 0,
    }
    best_trim: Dict[str, int] = {
        "trimmed": 0,
        "leading_ms": 0,
        "trailing_ms": 0,
        "original_ms": 0,
        "cleaned_ms": 0,
    }

    for attempt in range(1, total_attempts + 1):
        generate_config = None if attempt == 1 else dict(_BODY_RETRY_GENERATE_CONFIG)
        wav, sr = generate_voice(text, voice_prompt, generate_config=generate_config)
        cleaned_wav, cleaned_sr, trim_stats = clean_output_audio(wav, sr)
        quality = detect_body_boundary_artifacts(text, cleaned_wav, cleaned_sr)
        if best_wav is None:
            best_wav = cleaned_wav
            best_sr = cleaned_sr
            best_quality = quality
            best_trim = trim_stats
        if quality.get("passed", 1):
            return cleaned_wav, cleaned_sr, attempt, True, quality, trim_stats
        if not best_quality.get("passed", 1):
            best_wav = cleaned_wav
            best_sr = cleaned_sr
            best_quality = quality
            best_trim = trim_stats

    return best_wav, best_sr, total_attempts, False, best_quality, best_trim


def write_wav_temp(wav: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, wav, sr)
    return tmp.name


def wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, wav, sr, format="WAV")
    return bio.getvalue()


def wav_from_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    bio = io.BytesIO(data)
    wav, sr = sf.read(bio, always_2d=False)
    return _normalize_audio(wav), int(sr)


def _rms_envelope(signal: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0,), dtype=np.float32)
    frame_samples = max(1, frame_samples)
    hop_samples = max(1, hop_samples)
    if signal.size <= frame_samples:
        return np.array([float(np.sqrt(np.mean(np.square(signal)) + 1e-12))], dtype=np.float32)

    rms = []
    for start in range(0, signal.size - frame_samples + 1, hop_samples):
        frame = signal[start : start + frame_samples]
        rms.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-12)))
    return np.asarray(rms, dtype=np.float32)


def _spectral_flatness_envelope(signal: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0,), dtype=np.float32)
    frame_samples = max(1, int(frame_samples))
    hop_samples = max(1, int(hop_samples))

    window = np.hanning(frame_samples).astype(np.float32)
    if not np.any(window):
        window = np.ones((frame_samples,), dtype=np.float32)

    frames = []
    if signal.size <= frame_samples:
        frame = np.zeros((frame_samples,), dtype=np.float32)
        frame[: signal.size] = signal.astype(np.float32)
        frames.append(frame)
    else:
        for start in range(0, signal.size - frame_samples + 1, hop_samples):
            frames.append(signal[start : start + frame_samples].astype(np.float32))

    flatness = []
    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        spectrum = np.maximum(spectrum, 1e-8)
        geo_mean = float(np.exp(np.mean(np.log(spectrum))))
        ar_mean = float(np.mean(spectrum))
        flatness.append(0.0 if ar_mean <= 1e-12 else geo_mean / ar_mean)
    return np.asarray(flatness, dtype=np.float32)


def _adaptive_vad_threshold(rms: np.ndarray) -> float:
    if rms.size == 0:
        return 0.0
    base = float(np.percentile(rms, 15))
    high = float(np.percentile(rms, 90))
    # Conservative threshold to pick near-silence regions.
    return max(1e-5, base + 0.2 * max(0.0, high - base))


def _spectral_centroid_envelope(signal: np.ndarray, sr: int, frame_samples: int, hop_samples: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0,), dtype=np.float32)
    frame_samples = max(1, int(frame_samples))
    hop_samples = max(1, int(hop_samples))
    window = np.hanning(frame_samples).astype(np.float32)
    if not np.any(window):
        window = np.ones((frame_samples,), dtype=np.float32)
    freqs = np.fft.rfftfreq(frame_samples, d=1.0 / float(sr)).astype(np.float32)

    centroids = []
    if signal.size <= frame_samples:
        frame = np.zeros((frame_samples,), dtype=np.float32)
        frame[: signal.size] = signal.astype(np.float32)
        frames = [frame]
    else:
        frames = [signal[start : start + frame_samples].astype(np.float32) for start in range(0, signal.size - frame_samples + 1, hop_samples)]

    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        denom = float(np.sum(spectrum))
        centroids.append(0.0 if denom <= 1e-12 else float(np.sum(spectrum * freqs) / denom))
    return np.asarray(centroids, dtype=np.float32)


def _speech_frame_mask(
    signal: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
) -> Tuple[np.ndarray, int, int]:
    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    hop_samples = max(1, int(sr * hop_ms / 1000.0))
    rms = _rms_envelope(signal, frame_samples=frame_samples, hop_samples=hop_samples)
    flatness = _spectral_flatness_envelope(signal, frame_samples=frame_samples, hop_samples=hop_samples)
    if rms.size == 0 or flatness.size == 0:
        return np.zeros((0,), dtype=bool), frame_samples, hop_samples
    if flatness.size != rms.size:
        min_len = min(rms.size, flatness.size)
        rms = rms[:min_len]
        flatness = flatness[:min_len]
    if rms.size == 0:
        return np.zeros((0,), dtype=bool), frame_samples, hop_samples

    threshold = _adaptive_vad_threshold(rms)
    active = rms > threshold
    strong_threshold = max(threshold * 2.5, threshold + 1e-4)
    flatness_threshold = min(0.5, float(np.percentile(flatness, 40)) + 0.12)
    speech_like = active & ((flatness <= flatness_threshold) | (rms >= strong_threshold))
    return speech_like, frame_samples, hop_samples


def detect_greeting_onset_artifact(text: str, wav: np.ndarray, sr: int) -> Dict[str, float]:
    audio = _normalize_audio(wav)
    default = {
        "artifact": 0,
        "checked": 0,
        "start_ms": -1,
        "mean_rms": 0.0,
        "head_to_mid_rms_ratio": 0.0,
        "mean_flatness": 0.0,
        "flatness_cv": 0.0,
        "rms_cv": 0.0,
        "centroid_std": 0.0,
    }
    if not GREETING_ONSET_ARTIFACT_CHECK:
        return default
    if not isinstance(text, str) or not _H_GREETING_RE.match(text):
        return default
    if audio.size < max(1, int(sr * 0.22)):
        return default

    speech_like, frame_samples, hop_samples = _speech_frame_mask(audio, sr=int(sr), frame_ms=20, hop_ms=10)
    start_frame = _find_active_run_start(speech_like, min_run=3) if speech_like.size else None
    start_ms = -1 if start_frame is None else int(round(start_frame * hop_samples * 1000.0 / sr))

    inspect = audio[: min(audio.shape[0], int(sr * 0.35))]
    if inspect.size < max(1, int(sr * 0.22)):
        return {**default, "checked": 1, "start_ms": start_ms}

    rms = _rms_envelope(inspect, frame_samples=frame_samples, hop_samples=hop_samples)
    flatness = _spectral_flatness_envelope(inspect, frame_samples=frame_samples, hop_samples=hop_samples)
    centroid = _spectral_centroid_envelope(inspect, sr=int(sr), frame_samples=frame_samples, hop_samples=hop_samples)
    min_len = min(rms.size, flatness.size, centroid.size)
    if min_len < 6:
        return {**default, "checked": 1, "start_ms": start_ms}
    rms = rms[:min_len]
    flatness = flatness[:min_len]
    centroid = centroid[:min_len]

    head = inspect[: max(1, int(sr * 0.08))]
    mid_start = int(sr * 0.08)
    mid_end = min(inspect.shape[0], int(sr * 0.22))
    mid = inspect[mid_start:mid_end]
    head_rms = float(np.sqrt(np.mean(np.square(head)) + 1e-12)) if head.size else 0.0
    mid_rms = float(np.sqrt(np.mean(np.square(mid)) + 1e-12)) if mid.size else 0.0
    mean_rms = float(np.mean(rms))
    mean_flatness = float(np.mean(flatness))
    rms_cv = float(np.std(rms) / (mean_rms + 1e-8))
    flatness_cv = float(np.std(flatness) / (mean_flatness + 1e-8)) if mean_flatness > 1e-8 else 0.0
    centroid_std = float(np.std(centroid))
    ratio = head_rms / max(mid_rms, 1e-6)

    artifact = int(
        mean_rms > 0.025
        and ratio > 0.75
        and mean_flatness < 0.18
        and rms_cv < 0.22
        and flatness_cv < 0.5
        and centroid_std < 220.0
    )
    return {
        "artifact": artifact,
        "checked": 1,
        "start_ms": start_ms,
        "mean_rms": mean_rms,
        "head_to_mid_rms_ratio": ratio,
        "mean_flatness": mean_flatness,
        "flatness_cv": flatness_cv,
        "rms_cv": rms_cv,
        "centroid_std": centroid_std,
    }


def detect_greeting_leading_preroll_artifact(text: str, wav: np.ndarray, sr: int) -> Dict[str, float]:
    audio = _normalize_audio(wav)
    default = {
        "artifact": 0,
        "checked": 0,
        "speech_start_ms": -1,
        "strong_start_ms": -1,
        "preroll_ms": 0,
        "strong_threshold": 0.0,
        "leading_mean_rms": 0.0,
        "leading_p95_rms": 0.0,
        "leading_speech_ratio": 0.0,
        "preroll_mean_rms": 0.0,
        "preroll_p95_rms": 0.0,
        "preroll_speech_ratio": 0.0,
        "global_p95_rms": 0.0,
        "leading_to_global_ratio": 0.0,
        "preroll_to_global_ratio": 0.0,
    }
    if not GREETING_ONSET_ARTIFACT_CHECK:
        return default
    if not isinstance(text, str) or not _H_GREETING_RE.match(text):
        return default
    if audio.size < max(1, int(sr * 1.5)):
        return default

    speech_like, frame_samples, hop_samples = _speech_frame_mask(audio, sr=int(sr), frame_ms=20, hop_ms=10)
    if speech_like.size == 0:
        return default

    rms = _rms_envelope(audio, frame_samples=frame_samples, hop_samples=hop_samples)
    min_len = min(rms.size, speech_like.size)
    if min_len < 12:
        return {**default, "checked": 1}
    rms = rms[:min_len]
    speech_like = speech_like[:min_len]

    speech_start_frame = _find_active_run_start(speech_like, min_run=3)
    speech_start_ms = -1 if speech_start_frame is None else int(round(speech_start_frame * hop_samples * 1000.0 / sr))
    if speech_start_frame is None:
        return {**default, "checked": 1, "speech_start_ms": speech_start_ms}

    global_p90 = float(np.percentile(rms, 90))
    global_p95 = float(np.percentile(rms, 95))
    global_p98 = float(np.percentile(rms, 98))
    strong_threshold = max(0.018, global_p90 * 0.45, global_p95 * 0.33, global_p98 * 0.28)
    strong_mask = speech_like & (rms >= strong_threshold)
    strong_start_frame = _find_active_run_start(strong_mask, min_run=8)
    strong_start_ms = -1 if strong_start_frame is None else int(round(strong_start_frame * hop_samples * 1000.0 / sr))
    if strong_start_frame is None:
        return {
            **default,
            "checked": 1,
            "speech_start_ms": speech_start_ms,
            "strong_start_ms": strong_start_ms,
            "strong_threshold": strong_threshold,
            "global_p95_rms": global_p95,
        }

    leading = rms[:strong_start_frame]
    leading_mean = float(np.mean(leading)) if leading.size else 0.0
    leading_p95 = float(np.percentile(leading, 95)) if leading.size else 0.0
    leading_speech_ratio = float(np.mean(speech_like[:strong_start_frame])) if strong_start_frame > 0 else 0.0
    leading_to_global_ratio = leading_p95 / max(global_p95, 1e-6)
    if strong_start_frame <= speech_start_frame:
        delayed_strong_start = bool(
            strong_start_ms >= 1800
            and leading.size >= 80
            and leading_speech_ratio <= 0.20
            and leading_mean <= strong_threshold * 0.55
            and leading_p95 <= strong_threshold * 0.85
            and leading_to_global_ratio <= 0.40
        )
        return {
            **default,
            "artifact": int(delayed_strong_start),
            "checked": 1,
            "speech_start_ms": speech_start_ms,
            "strong_start_ms": strong_start_ms,
            "strong_threshold": strong_threshold,
            "leading_mean_rms": leading_mean,
            "leading_p95_rms": leading_p95,
            "leading_speech_ratio": leading_speech_ratio,
            "global_p95_rms": global_p95,
            "leading_to_global_ratio": leading_to_global_ratio,
        }

    preroll = rms[speech_start_frame:strong_start_frame]
    if preroll.size == 0:
        return {
            **default,
            "checked": 1,
            "speech_start_ms": speech_start_ms,
            "strong_start_ms": strong_start_ms,
            "strong_threshold": strong_threshold,
            "leading_mean_rms": leading_mean,
            "leading_p95_rms": leading_p95,
            "leading_speech_ratio": leading_speech_ratio,
            "global_p95_rms": global_p95,
            "leading_to_global_ratio": leading_to_global_ratio,
        }

    preroll_ms = int(round((strong_start_frame - speech_start_frame) * hop_samples * 1000.0 / sr))
    preroll_mean = float(np.mean(preroll))
    preroll_p95 = float(np.percentile(preroll, 95))
    preroll_speech_ratio = float(np.mean(speech_like[speech_start_frame:strong_start_frame]))
    preroll_to_global_ratio = preroll_p95 / max(global_p95, 1e-6)
    delayed_strong_start = bool(
        strong_start_ms >= 1800
        and leading.size >= 80
        and leading_speech_ratio <= 0.20
        and leading_mean <= strong_threshold * 0.55
        and leading_p95 <= strong_threshold * 0.85
        and leading_to_global_ratio <= 0.40
    )
    low_energy_speech_preroll = bool(
        preroll_ms >= 1200
        and preroll.size >= 40
        and preroll_speech_ratio <= 0.35
        and preroll_mean <= strong_threshold * 0.55
        and preroll_p95 <= strong_threshold * 0.80
        and preroll_to_global_ratio <= 0.40
    )
    artifact = int(delayed_strong_start or low_energy_speech_preroll)
    return {
        "artifact": artifact,
        "checked": 1,
        "speech_start_ms": speech_start_ms,
        "strong_start_ms": strong_start_ms,
        "preroll_ms": preroll_ms,
        "strong_threshold": strong_threshold,
        "leading_mean_rms": leading_mean,
        "leading_p95_rms": leading_p95,
        "leading_speech_ratio": leading_speech_ratio,
        "preroll_mean_rms": preroll_mean,
        "preroll_p95_rms": preroll_p95,
        "preroll_speech_ratio": preroll_speech_ratio,
        "global_p95_rms": global_p95,
        "leading_to_global_ratio": leading_to_global_ratio,
        "preroll_to_global_ratio": preroll_to_global_ratio,
    }


def detect_greeting_clipped_ending_artifact(text: str, wav: np.ndarray, sr: int) -> Dict[str, float]:
    audio = _normalize_audio(wav)
    default = {
        "artifact": 0,
        "checked": 0,
        "duration_ms": 0,
        "expected_min_ms": 0,
        "tail_mean_rms": 0.0,
        "pre_tail_mean_rms": 0.0,
        "tail_to_pre_ratio": 0.0,
        "tail_to_global_ratio": 0.0,
        "tail_speech_ratio": 0.0,
        "tail_run_ms": 0,
    }
    if not GREETING_ONSET_ARTIFACT_CHECK:
        return default

    match = _SHORT_GREETING_RE.match(text or "")
    if not match:
        return default
    if audio.size < max(1, int(sr * 0.14)):
        return {**default, "checked": 1, "duration_ms": int(round(audio.shape[0] * 1000.0 / max(int(sr), 1)))}

    greeting_word = (match.group(1) or "").strip().lower()
    name_letters = re.sub(r"[^a-zA-Z]", "", match.group(2) or "")
    letter_count = max(1, len(name_letters)) + (2 if greeting_word == "hello" else 1)
    duration_ms = int(round(audio.shape[0] * 1000.0 / sr))
    expected_min_ms = int(min(950, max(430, 360 + letter_count * 32)))

    speech_like, frame_samples, hop_samples = _speech_frame_mask(audio, sr=int(sr), frame_ms=20, hop_ms=10)
    rms = _rms_envelope(audio, frame_samples=frame_samples, hop_samples=hop_samples)
    min_len = min(rms.size, speech_like.size)
    if min_len < 8:
        return {
            **default,
            "checked": 1,
            "duration_ms": duration_ms,
            "expected_min_ms": expected_min_ms,
        }
    rms = rms[:min_len]
    speech_like = speech_like[:min_len]

    tail_frames = max(4, int(round(0.08 * sr / hop_samples)))
    pre_tail_frames = max(tail_frames, int(round(0.12 * sr / hop_samples)))
    tail = rms[-tail_frames:]
    pre_tail_end = max(0, rms.size - tail_frames)
    pre_tail_start = max(0, pre_tail_end - pre_tail_frames)
    pre_tail = rms[pre_tail_start:pre_tail_end]
    if pre_tail.size == 0:
        pre_tail = rms[:-tail_frames] if rms.size > tail_frames else rms

    tail_mean = float(np.mean(tail)) if tail.size else 0.0
    pre_tail_mean = float(np.mean(pre_tail)) if pre_tail.size else 0.0
    global_p90 = float(np.percentile(rms, 90))
    tail_to_pre_ratio = tail_mean / max(pre_tail_mean, 1e-6)
    tail_to_global_ratio = tail_mean / max(global_p90, 1e-6)
    tail_speech_ratio = float(np.mean(speech_like[-tail_frames:])) if tail_frames > 0 else 0.0

    tail_run_ms = 0
    runs = _active_runs(speech_like, min_run=1)
    if runs:
        last_start, last_end = runs[-1]
        if last_end >= speech_like.size - tail_frames - 1:
            tail_run_ms = int(round((last_end - last_start + 1) * hop_samples * 1000.0 / sr))

    artifact = int(
        duration_ms <= expected_min_ms
        and tail_speech_ratio >= 0.65
        and tail_to_pre_ratio >= 0.82
        and tail_to_global_ratio >= 0.42
    )
    return {
        "artifact": artifact,
        "checked": 1,
        "duration_ms": duration_ms,
        "expected_min_ms": expected_min_ms,
        "tail_mean_rms": tail_mean,
        "pre_tail_mean_rms": pre_tail_mean,
        "tail_to_pre_ratio": tail_to_pre_ratio,
        "tail_to_global_ratio": tail_to_global_ratio,
        "tail_speech_ratio": tail_speech_ratio,
        "tail_run_ms": tail_run_ms,
    }


def detect_greeting_excessive_duration_artifact(text: str, wav: np.ndarray, sr: int) -> Dict[str, float]:
    audio = _normalize_audio(wav)
    default = {
        "artifact": 0,
        "checked": 0,
        "duration_ms": 0,
        "expected_max_ms": 0,
        "duration_to_limit_ratio": 0.0,
    }
    if not GREETING_ONSET_ARTIFACT_CHECK:
        return default

    match = _SHORT_GREETING_RE.match(text or "")
    if not match:
        return default

    greeting_word = (match.group(1) or "").strip().lower()
    name_letters = re.sub(r"[^a-zA-Z]", "", match.group(2) or "")
    letter_count = max(1, len(name_letters)) + (2 if greeting_word == "hello" else 1)
    duration_ms = int(round(audio.shape[0] * 1000.0 / max(int(sr), 1)))
    expected_max_ms = int(min(3200, max(1800, 1000 + letter_count * 180)))
    ratio = duration_ms / max(expected_max_ms, 1)
    artifact = int(duration_ms >= expected_max_ms)
    return {
        "artifact": artifact,
        "checked": 1,
        "duration_ms": duration_ms,
        "expected_max_ms": expected_max_ms,
        "duration_to_limit_ratio": ratio,
    }


def _find_active_run_start(active: np.ndarray, min_run: int) -> Optional[int]:
    run = 0
    for idx, value in enumerate(active):
        run = run + 1 if value else 0
        if run >= min_run:
            return idx - min_run + 1
    return None


def _find_active_run_end(active: np.ndarray, min_run: int) -> Optional[int]:
    run = 0
    for idx in range(active.size - 1, -1, -1):
        if active[idx]:
            run += 1
            if run >= min_run:
                return idx + min_run - 1
        else:
            run = 0
    return None


def _active_runs(active: np.ndarray, min_run: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    run = 0
    for idx, value in enumerate(active):
        if value:
            if start is None:
                start = idx
            run += 1
            continue
        if start is not None and run >= min_run:
            runs.append((start, idx - 1))
        start = None
        run = 0
    if start is not None and run >= min_run:
        runs.append((start, active.size - 1))
    return runs


def trim_audio_edges(
    wav: np.ndarray,
    sr: int,
    pad_ms: int,
    max_leading_ms: int,
    max_trailing_ms: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_run_frames: int = 3,
) -> Tuple[np.ndarray, Dict[str, int]]:
    audio = _normalize_audio(wav)
    total_samples = int(audio.shape[0])
    if total_samples == 0:
        return audio, {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "original_ms": 0,
            "cleaned_ms": 0,
        }

    speech_like, frame_samples, hop_samples = _speech_frame_mask(
        audio,
        sr=int(sr),
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    if speech_like.size == 0:
        return audio, {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    start_frame = _find_active_run_start(speech_like, min_run=min_run_frames)
    end_frame = _find_active_run_end(speech_like, min_run=min_run_frames)
    if start_frame is None or end_frame is None:
        return audio, {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    pad_samples = max(0, int(sr * pad_ms / 1000.0))
    proposed_start = max(0, start_frame * hop_samples - pad_samples)
    proposed_end = min(total_samples, end_frame * hop_samples + frame_samples + pad_samples)
    max_leading_samples = max(0, int(sr * max_leading_ms / 1000.0))
    max_trailing_samples = max(0, int(sr * max_trailing_ms / 1000.0))
    start_sample = min(proposed_start, max_leading_samples)
    end_sample = max(proposed_end, total_samples - max_trailing_samples)
    if end_sample <= start_sample + max(1, frame_samples):
        return audio, {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    trimmed = audio[start_sample:end_sample]
    leading_ms = int(round(start_sample * 1000.0 / sr))
    trailing_ms = int(round((total_samples - end_sample) * 1000.0 / sr))
    return trimmed, {
        "trimmed": int(start_sample > 0 or end_sample < total_samples),
        "leading_ms": leading_ms,
        "trailing_ms": trailing_ms,
        "original_ms": int(round(total_samples * 1000.0 / sr)),
        "cleaned_ms": int(round(trimmed.shape[0] * 1000.0 / sr)),
    }


def compact_internal_silences(
    wav: np.ndarray,
    sr: int,
    max_internal_silence_ms: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_run_frames: int = 3,
) -> Tuple[np.ndarray, Dict[str, int]]:
    audio = _normalize_audio(wav)
    total_samples = int(audio.shape[0])
    keep_samples = max(0, int(sr * max_internal_silence_ms / 1000.0))
    if total_samples == 0 or keep_samples <= 0:
        return audio, {
            "compressed": 0,
            "spans": 0,
            "removed_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
        }

    speech_like, frame_samples, hop_samples = _speech_frame_mask(
        audio,
        sr=int(sr),
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    if speech_like.size == 0:
        return audio, {
            "compressed": 0,
            "spans": 0,
            "removed_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    speech_run_start = _find_active_run_start(speech_like, min_run=min_run_frames)
    speech_run_end = _find_active_run_end(speech_like, min_run=min_run_frames)
    if speech_run_start is None or speech_run_end is None or speech_run_end <= speech_run_start:
        return audio, {
            "compressed": 0,
            "spans": 0,
            "removed_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    parts = []
    cursor = 0
    removed_samples = 0
    compressed_spans = 0
    in_silence = False
    silence_start = 0

    for idx in range(speech_run_start, speech_run_end + 1):
        is_speech = bool(speech_like[idx])
        if not is_speech and not in_silence:
            in_silence = True
            silence_start = idx
            continue
        if is_speech and in_silence:
            in_silence = False
            run_start = silence_start
            run_end = idx - 1
            silence_sample_start = max(0, run_start * hop_samples)
            silence_sample_end = min(total_samples, run_end * hop_samples + frame_samples)
            silence_len = max(0, silence_sample_end - silence_sample_start)
            if silence_len > keep_samples:
                left_keep = keep_samples // 2
                right_keep = keep_samples - left_keep
                remove_start = silence_sample_start + left_keep
                remove_end = silence_sample_end - right_keep
                if remove_end > remove_start:
                    parts.append(audio[cursor:remove_start])
                    cursor = remove_end
                    removed_samples += remove_end - remove_start
                    compressed_spans += 1
    if cursor == 0:
        return audio, {
            "compressed": 0,
            "spans": 0,
            "removed_ms": 0,
            "original_ms": int(round(total_samples * 1000.0 / sr)),
            "cleaned_ms": int(round(total_samples * 1000.0 / sr)),
        }

    parts.append(audio[cursor:])
    compacted = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return compacted.astype(np.float32), {
        "compressed": int(compressed_spans > 0),
        "spans": int(compressed_spans),
        "removed_ms": int(round(removed_samples * 1000.0 / sr)),
        "original_ms": int(round(total_samples * 1000.0 / sr)),
        "cleaned_ms": int(round(compacted.shape[0] * 1000.0 / sr)),
    }


def trim_low_energy_boundary_artifacts(
    wav: np.ndarray,
    sr: int,
    *,
    pad_ms: int,
    max_leading_ms: int,
    max_trailing_ms: int,
    allow_leading_artifact_trim: bool = True,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_run_frames: int = 3,
    min_strong_run_frames: int = 8,
) -> Tuple[np.ndarray, Dict[str, int]]:
    audio = _normalize_audio(wav)
    total_samples = int(audio.shape[0])
    default = {
        "trimmed": 0,
        "leading_ms": 0,
        "trailing_ms": 0,
        "original_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
        "cleaned_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
    }
    if total_samples == 0:
        return audio, default

    speech_like, frame_samples, hop_samples = _speech_frame_mask(
        audio,
        sr=int(sr),
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    if speech_like.size == 0:
        return audio, default

    rms = _rms_envelope(audio, frame_samples=frame_samples, hop_samples=hop_samples)
    min_len = min(rms.size, speech_like.size)
    if min_len < max(12, min_strong_run_frames):
        return audio, default
    rms = rms[:min_len]
    speech_like = speech_like[:min_len]

    speech_start_frame = _find_active_run_start(speech_like, min_run=min_run_frames)
    speech_end_frame = _find_active_run_end(speech_like, min_run=min_run_frames)
    if speech_start_frame is None or speech_end_frame is None:
        return audio, default

    global_p95 = float(np.percentile(rms, 95))
    global_p98 = float(np.percentile(rms, 98))
    global_p90 = float(np.percentile(rms, 90))
    strong_threshold = max(0.018, global_p90 * 0.45, global_p95 * 0.33, global_p98 * 0.28)
    strong_mask = speech_like & (rms >= strong_threshold)
    strong_start_frame = _find_active_run_start(strong_mask, min_run=min_strong_run_frames)
    strong_end_frame = _find_active_run_end(strong_mask, min_run=min_strong_run_frames)
    if strong_start_frame is None:
        strong_start_frame = speech_start_frame
    if strong_end_frame is None:
        strong_end_frame = speech_end_frame

    pad_samples = max(0, int(sr * pad_ms / 1000.0))
    max_leading_samples = max(0, int(sr * max_leading_ms / 1000.0))
    max_trailing_samples = max(0, int(sr * max_trailing_ms / 1000.0))
    start_sample = 0
    end_sample = total_samples

    if allow_leading_artifact_trim and strong_start_frame is not None:
        leading = rms[:strong_start_frame]
        leading_p95 = float(np.percentile(leading, 95)) if leading.size else 0.0
        leading_speech_ratio = float(np.mean(speech_like[:strong_start_frame])) if strong_start_frame > 0 else 0.0
        strong_start_ms = int(round(strong_start_frame * hop_samples * 1000.0 / sr))
        delayed_strong_start = bool(
            strong_start_ms >= 450
            and leading.size >= 20
            and leading_speech_ratio <= 0.35
            and leading_p95 <= global_p95 * 0.35
        )

        preroll_artifact = False
        if strong_start_frame > speech_start_frame:
            preroll = rms[speech_start_frame:strong_start_frame]
            preroll_ms = int(round((strong_start_frame - speech_start_frame) * hop_samples * 1000.0 / sr))
            preroll_p95 = float(np.percentile(preroll, 95)) if preroll.size else 0.0
            preroll_speech_ratio = float(np.mean(speech_like[speech_start_frame:strong_start_frame])) if preroll.size else 0.0
            preroll_artifact = bool(
                preroll_ms >= 1200
                and preroll.size >= 40
                and preroll_speech_ratio <= 0.40
                and preroll_p95 <= global_p95 * 0.45
            )

        if delayed_strong_start or preroll_artifact:
            proposed_start = max(0, strong_start_frame * hop_samples - pad_samples)
            start_sample = min(proposed_start, max_leading_samples)

    if strong_end_frame is not None and strong_end_frame < (rms.size - 1):
        tail = rms[strong_end_frame + 1 :]
        tail_ms = int(round((rms.size - 1 - strong_end_frame) * hop_samples * 1000.0 / sr))
        tail_mean = float(np.mean(tail)) if tail.size else 0.0
        tail_p95 = float(np.percentile(tail, 95)) if tail.size else 0.0
        tail_strong_ratio = float(np.mean(strong_mask[strong_end_frame + 1 :])) if tail.size else 0.0
        weak_trailing_tail = bool(
            tail_ms >= 400
            and tail.size >= 20
            and tail_mean <= global_p95 * 0.35
            and tail_p95 <= global_p95 * 0.75
            and tail_strong_ratio <= 0.20
        )
        if weak_trailing_tail:
            proposed_end = min(total_samples, strong_end_frame * hop_samples + frame_samples + pad_samples)
            end_sample = max(proposed_end, total_samples - max_trailing_samples)

    if end_sample <= start_sample + max(1, frame_samples):
        return audio, default

    cleaned = audio[start_sample:end_sample].astype(np.float32)
    leading_ms = int(round(start_sample * 1000.0 / sr))
    trailing_ms = int(round((total_samples - end_sample) * 1000.0 / sr))
    return cleaned, {
        "trimmed": int(start_sample > 0 or end_sample < total_samples),
        "leading_ms": leading_ms,
        "trailing_ms": trailing_ms,
        "original_ms": int(round(total_samples * 1000.0 / sr)),
        "cleaned_ms": int(round(cleaned.shape[0] * 1000.0 / sr)),
    }


def trim_low_clarity_boundary_blocks(
    wav: np.ndarray,
    sr: int,
    *,
    pad_ms: int,
    min_boundary_ms: int = 10000,
    chunk_ms: int = 10000,
    hop_ms: int = 5000,
    frame_ms: int = 40,
    feature_hop_ms: int = 20,
) -> Tuple[np.ndarray, Dict[str, int]]:
    audio = _normalize_audio(wav)
    total_samples = int(audio.shape[0])
    default = {
        "trimmed": 0,
        "leading_ms": 0,
        "trailing_ms": 0,
        "original_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
        "cleaned_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
    }
    if total_samples == 0:
        return audio, default

    chunk_samples = max(1, int(sr * chunk_ms / 1000.0))
    hop_samples = max(1, int(sr * hop_ms / 1000.0))
    if total_samples < chunk_samples + hop_samples:
        return audio, default

    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    feature_hop_samples = max(1, int(sr * feature_hop_ms / 1000.0))
    chunk_metrics = []
    for start in range(0, total_samples - chunk_samples + 1, hop_samples):
        segment = audio[start : start + chunk_samples]
        rms = _rms_envelope(segment, frame_samples=frame_samples, hop_samples=feature_hop_samples)
        flat = _spectral_flatness_envelope(segment, frame_samples=frame_samples, hop_samples=feature_hop_samples)
        cent = _spectral_centroid_envelope(
            segment,
            sr=int(sr),
            frame_samples=frame_samples,
            hop_samples=feature_hop_samples,
        )
        min_len = min(rms.size, flat.size, cent.size)
        if min_len < 8:
            continue
        rms = rms[:min_len]
        flat = flat[:min_len]
        cent = cent[:min_len]

        def _cv(values: np.ndarray) -> float:
            mean = float(np.mean(values))
            return 0.0 if mean <= 1e-8 else float(np.std(values) / (mean + 1e-8))

        chunk_metrics.append(
            {
                "start": start,
                "end": start + chunk_samples,
                "rms": float(np.mean(rms)),
                "flat": float(np.mean(flat)),
                "cent": float(np.mean(cent)),
                "rms_cv": _cv(rms),
                "flat_cv": _cv(flat),
                "cent_cv": _cv(cent),
            }
        )

    if len(chunk_metrics) < 3:
        return audio, default

    metrics = np.asarray(
        [
            [item["rms"], item["flat"], item["cent"], item["rms_cv"], item["flat_cv"], item["cent_cv"]]
            for item in chunk_metrics
        ],
        dtype=np.float32,
    )
    reference = metrics[metrics[:, 0] >= float(np.percentile(metrics[:, 0], 70))]
    if reference.size == 0:
        reference = metrics
    ref_rms, ref_flat, ref_cent, ref_rms_cv, ref_flat_cv, ref_cent_cv = np.median(reference, axis=0)

    scores = []
    for item in chunk_metrics:
        score = (
            0.20 * min(item["rms"] / max(float(ref_rms), 1e-6), 1.5)
            + 0.25 * min(item["flat"] / max(float(ref_flat), 1e-6), 1.5)
            + 0.20 * min(item["cent"] / max(float(ref_cent), 1e-6), 1.5)
            + 0.15 * min(item["rms_cv"] / max(float(ref_rms_cv), 1e-6), 1.5)
            + 0.10 * min(item["flat_cv"] / max(float(ref_flat_cv), 1e-6), 1.5)
            + 0.10 * min(item["cent_cv"] / max(float(ref_cent_cv), 1e-6), 1.5)
        )
        scores.append(float(score))
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size >= 3:
        scores = np.convolve(scores, np.asarray([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    good = scores >= 0.80
    if not np.any(good):
        return audio, default

    first_good = int(np.argmax(good))
    last_good = int(good.size - 1 - np.argmax(good[::-1]))
    leading_ms = int(round(chunk_metrics[first_good]["start"] * 1000.0 / sr))
    trailing_ms = int(round((total_samples - chunk_metrics[last_good]["end"]) * 1000.0 / sr))

    start_sample = 0
    end_sample = total_samples
    pad_samples = max(0, int(sr * pad_ms / 1000.0))
    if leading_ms >= int(min_boundary_ms):
        start_sample = max(0, int(chunk_metrics[first_good]["start"]) - pad_samples)
    if trailing_ms >= int(min_boundary_ms):
        end_sample = min(total_samples, int(chunk_metrics[last_good]["end"]) + pad_samples)
    if end_sample <= start_sample + max(1, frame_samples):
        return audio, default
    if start_sample == 0 and end_sample == total_samples:
        return audio, default

    cleaned = audio[start_sample:end_sample].astype(np.float32)
    return cleaned, {
        "trimmed": 1,
        "leading_ms": int(round(start_sample * 1000.0 / sr)),
        "trailing_ms": int(round((total_samples - end_sample) * 1000.0 / sr)),
        "original_ms": int(round(total_samples * 1000.0 / sr)),
        "cleaned_ms": int(round(cleaned.shape[0] * 1000.0 / sr)),
    }


def refine_local_clarity_boundaries(
    wav: np.ndarray,
    sr: int,
    *,
    pad_ms: int,
    inspect_ms: int = 8000,
    min_leading_gap_ms: int = 1000,
    min_trailing_gap_ms: int = 2000,
    window_ms: int = 1000,
    hop_ms: int = 500,
) -> Tuple[np.ndarray, Dict[str, int]]:
    audio = _normalize_audio(wav)
    total_samples = int(audio.shape[0])
    default = {
        "trimmed": 0,
        "leading_ms": 0,
        "trailing_ms": 0,
        "original_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
        "cleaned_ms": int(round(total_samples * 1000.0 / sr)) if sr else 0,
    }
    if total_samples < int(sr * 12.0):
        return audio, default

    window_samples = max(1, int(sr * window_ms / 1000.0))
    hop_samples = max(1, int(sr * hop_ms / 1000.0))
    inspect_samples = min(total_samples, max(window_samples, int(sr * inspect_ms / 1000.0)))
    pad_samples = max(0, int(sr * pad_ms / 1000.0))

    mid_start = int(total_samples * 0.25)
    mid_end = int(total_samples * 0.75)
    if mid_end - mid_start < window_samples:
        return audio, default

    def _window_metrics(segment: np.ndarray) -> Tuple[float, float, float]:
        spec = np.abs(np.fft.rfft(segment * np.hanning(segment.shape[0]).astype(np.float32))).astype(np.float32)
        spec = np.maximum(spec, 1e-8)
        rms = float(np.sqrt(np.mean(np.square(segment)) + 1e-12))
        flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        freqs = np.fft.rfftfreq(segment.shape[0], d=1.0 / float(sr)).astype(np.float32)
        centroid = float(np.sum(spec * freqs) / np.sum(spec))
        return rms, flat, centroid

    ref_metrics = []
    mid_segment = audio[mid_start:mid_end]
    for start in range(0, mid_segment.shape[0] - window_samples + 1, hop_samples):
        ref_metrics.append(_window_metrics(mid_segment[start : start + window_samples]))
    if not ref_metrics:
        return audio, default
    ref_rms, ref_flat, ref_cent = np.median(np.asarray(ref_metrics, dtype=np.float32), axis=0)

    def _score(metrics: Tuple[float, float, float]) -> float:
        rms, flat, cent = metrics
        return float(
            0.45 * min(rms / max(float(ref_rms), 1e-6), 1.6)
            + 0.25 * min(flat / max(float(ref_flat), 1e-6), 1.6)
            + 0.30 * min(cent / max(float(ref_cent), 1e-6), 1.6)
        )

    head_scores = []
    head_offsets = []
    head_segment = audio[:inspect_samples]
    for start in range(0, head_segment.shape[0] - window_samples + 1, hop_samples):
        head_offsets.append(start)
        head_scores.append(_score(_window_metrics(head_segment[start : start + window_samples])))

    tail_scores = []
    tail_offsets = []
    tail_segment = audio[total_samples - inspect_samples :]
    for start in range(0, tail_segment.shape[0] - window_samples + 1, hop_samples):
        tail_offsets.append(start)
        tail_scores.append(_score(_window_metrics(tail_segment[start : start + window_samples])))

    start_sample = 0
    end_sample = total_samples

    if head_scores:
        head_scores_arr = np.asarray(head_scores, dtype=np.float32)
        good_idx = np.where(head_scores_arr >= 1.0)[0]
        if good_idx.size > 0:
            first_good = int(good_idx[0])
            first_good_ms = int(round(head_offsets[first_good] * 1000.0 / sr))
            if first_good_ms >= int(min_leading_gap_ms):
                prior = head_scores_arr[:first_good]
                if prior.size > 0 and float(np.max(prior)) <= 0.90:
                    start_sample = max(0, head_offsets[first_good] - pad_samples)

    if tail_scores:
        tail_scores_arr = np.asarray(tail_scores, dtype=np.float32)
        good_idx = np.where(tail_scores_arr >= 1.0)[0]
        if good_idx.size > 0:
            last_good = int(good_idx[-1])
            tail_base = total_samples - inspect_samples
            last_good_end = tail_base + tail_offsets[last_good] + window_samples
            trailing_gap_ms = int(round((total_samples - last_good_end) * 1000.0 / sr))
            if trailing_gap_ms >= int(min_trailing_gap_ms):
                after = tail_scores_arr[last_good + 1 :]
                if after.size > 0 and float(np.max(after)) <= 0.90:
                    end_sample = min(total_samples, last_good_end + pad_samples)

    if end_sample <= start_sample + window_samples:
        return audio, default
    if start_sample == 0 and end_sample == total_samples:
        return audio, default

    cleaned = audio[start_sample:end_sample].astype(np.float32)
    return cleaned, {
        "trimmed": 1,
        "leading_ms": int(round(start_sample * 1000.0 / sr)),
        "trailing_ms": int(round((total_samples - end_sample) * 1000.0 / sr)),
        "original_ms": int(round(total_samples * 1000.0 / sr)),
        "cleaned_ms": int(round(cleaned.shape[0] * 1000.0 / sr)),
    }


def clean_reference_audio(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    audio = _normalize_audio(wav)
    if not REFERENCE_AUDIO_TRIM_ENABLED:
        stats = {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "original_ms": int(round(audio.shape[0] * 1000.0 / sr)) if sr else 0,
            "cleaned_ms": int(round(audio.shape[0] * 1000.0 / sr)) if sr else 0,
        }
        return audio, int(sr), stats
    cleaned, stats = trim_audio_edges(
        audio,
        sr=int(sr),
        pad_ms=REFERENCE_AUDIO_TRIM_PAD_MS,
        max_leading_ms=REFERENCE_AUDIO_TRIM_MAX_LEADING_MS,
        max_trailing_ms=REFERENCE_AUDIO_TRIM_MAX_TRAILING_MS,
    )
    return cleaned, int(sr), stats


def _clean_output_audio_impl(
    wav: np.ndarray,
    sr: int,
    *,
    pad_ms: int,
    max_leading_ms: int,
    max_trailing_ms: int,
    allow_leading_artifact_trim: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    audio = _normalize_audio(wav)
    if not OUTPUT_AUDIO_TRIM_ENABLED:
        stats = {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "boundary_artifact_trimmed": 0,
            "boundary_leading_ms": 0,
            "boundary_trailing_ms": 0,
            "internal_silence_compressed": 0,
            "internal_silence_spans": 0,
            "internal_silence_removed_ms": 0,
            "original_ms": int(round(audio.shape[0] * 1000.0 / sr)) if sr else 0,
            "cleaned_ms": int(round(audio.shape[0] * 1000.0 / sr)) if sr else 0,
        }
        return audio, int(sr), stats

    edge_cleaned, edge_stats = trim_audio_edges(
        audio,
        sr=int(sr),
        pad_ms=pad_ms,
        max_leading_ms=max_leading_ms,
        max_trailing_ms=max_trailing_ms,
    )
    boundary_cleaned, boundary_stats = trim_low_energy_boundary_artifacts(
        edge_cleaned,
        sr=int(sr),
        pad_ms=pad_ms,
        max_leading_ms=(OUTPUT_AUDIO_TRIM_MAX_LEADING_MS if allow_leading_artifact_trim else 0),
        max_trailing_ms=max_trailing_ms,
        allow_leading_artifact_trim=allow_leading_artifact_trim,
    )
    clarity_cleaned, clarity_stats = trim_low_clarity_boundary_blocks(
        boundary_cleaned,
        sr=int(sr),
        pad_ms=pad_ms,
    )
    local_clarity_cleaned, local_clarity_stats = refine_local_clarity_boundaries(
        clarity_cleaned,
        sr=int(sr),
        pad_ms=pad_ms,
    )
    compacted, silence_stats = compact_internal_silences(
        local_clarity_cleaned,
        sr=int(sr),
        max_internal_silence_ms=OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS,
    )
    stats = dict(edge_stats)
    stats["trimmed"] = int(
        bool(edge_stats.get("trimmed"))
        or bool(boundary_stats.get("trimmed"))
        or bool(clarity_stats.get("trimmed"))
        or bool(local_clarity_stats.get("trimmed"))
        or bool(silence_stats.get("compressed"))
    )
    stats["boundary_artifact_trimmed"] = int(boundary_stats.get("trimmed", 0))
    stats["boundary_leading_ms"] = int(boundary_stats.get("leading_ms", 0))
    stats["boundary_trailing_ms"] = int(boundary_stats.get("trailing_ms", 0))
    stats["clarity_boundary_trimmed"] = int(clarity_stats.get("trimmed", 0))
    stats["clarity_leading_ms"] = int(clarity_stats.get("leading_ms", 0))
    stats["clarity_trailing_ms"] = int(clarity_stats.get("trailing_ms", 0))
    stats["local_clarity_boundary_trimmed"] = int(local_clarity_stats.get("trimmed", 0))
    stats["local_clarity_leading_ms"] = int(local_clarity_stats.get("leading_ms", 0))
    stats["local_clarity_trailing_ms"] = int(local_clarity_stats.get("trailing_ms", 0))
    stats["leading_ms"] = (
        int(edge_stats.get("leading_ms", 0))
        + int(boundary_stats.get("leading_ms", 0))
        + int(clarity_stats.get("leading_ms", 0))
        + int(local_clarity_stats.get("leading_ms", 0))
    )
    stats["trailing_ms"] = (
        int(edge_stats.get("trailing_ms", 0))
        + int(boundary_stats.get("trailing_ms", 0))
        + int(clarity_stats.get("trailing_ms", 0))
        + int(local_clarity_stats.get("trailing_ms", 0))
    )
    stats["internal_silence_compressed"] = int(silence_stats["compressed"])
    stats["internal_silence_spans"] = int(silence_stats["spans"])
    stats["internal_silence_removed_ms"] = int(silence_stats["removed_ms"])
    stats["cleaned_ms"] = int(round(compacted.shape[0] * 1000.0 / sr)) if sr else 0
    return compacted, int(sr), stats


def clean_output_audio(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    return _clean_output_audio_impl(
        wav,
        sr,
        pad_ms=OUTPUT_AUDIO_TRIM_PAD_MS,
        max_leading_ms=OUTPUT_AUDIO_TRIM_MAX_LEADING_MS,
        max_trailing_ms=OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS,
        allow_leading_artifact_trim=True,
    )


def clean_output_audio_preserve_start(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    return _clean_output_audio_impl(
        wav,
        sr,
        pad_ms=max(int(OUTPUT_AUDIO_TRIM_PAD_MS), int(GREETING_OUTPUT_TRIM_PAD_MS)),
        max_leading_ms=OUTPUT_AUDIO_TRIM_MAX_LEADING_MS,
        max_trailing_ms=OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS,
        allow_leading_artifact_trim=True,
    )


def clean_output_audio_for_greeting(text: str, wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    if isinstance(text, str) and _SHORT_GREETING_RE.match(text):
        return _clean_output_audio_impl(
            wav,
            sr,
            pad_ms=max(int(OUTPUT_AUDIO_TRIM_PAD_MS), int(GREETING_OUTPUT_TRIM_PAD_MS)),
            max_leading_ms=OUTPUT_AUDIO_TRIM_MAX_LEADING_MS,
            max_trailing_ms=0,
            allow_leading_artifact_trim=True,
        )
    return clean_output_audio_preserve_start(wav, sr)


def clean_output_audio_without_leading_trim(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    return _clean_output_audio_impl(
        wav,
        sr,
        pad_ms=OUTPUT_AUDIO_TRIM_PAD_MS,
        max_leading_ms=0,
        max_trailing_ms=OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS,
        allow_leading_artifact_trim=True,
    )


def clean_output_audio_for_spliced_phrase(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    audio = _normalize_audio(wav)
    original_ms = int(round(audio.shape[0] * 1000.0 / sr)) if sr else 0
    if not OUTPUT_AUDIO_TRIM_ENABLED:
        stats = {
            "trimmed": 0,
            "leading_ms": 0,
            "trailing_ms": 0,
            "boundary_artifact_trimmed": 0,
            "boundary_leading_ms": 0,
            "boundary_trailing_ms": 0,
            "clarity_boundary_trimmed": 0,
            "clarity_leading_ms": 0,
            "clarity_trailing_ms": 0,
            "local_clarity_boundary_trimmed": 0,
            "local_clarity_leading_ms": 0,
            "local_clarity_trailing_ms": 0,
            "internal_silence_compressed": 0,
            "internal_silence_spans": 0,
            "internal_silence_removed_ms": 0,
            "original_ms": original_ms,
            "cleaned_ms": original_ms,
        }
        return audio, int(sr), stats

    compacted, silence_stats = compact_internal_silences(
        audio,
        sr=int(sr),
        max_internal_silence_ms=OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS,
    )
    stats = {
        "trimmed": int(silence_stats.get("compressed", 0)),
        "leading_ms": 0,
        "trailing_ms": 0,
        "boundary_artifact_trimmed": 0,
        "boundary_leading_ms": 0,
        "boundary_trailing_ms": 0,
        "clarity_boundary_trimmed": 0,
        "clarity_leading_ms": 0,
        "clarity_trailing_ms": 0,
        "local_clarity_boundary_trimmed": 0,
        "local_clarity_leading_ms": 0,
        "local_clarity_trailing_ms": 0,
        "internal_silence_compressed": int(silence_stats.get("compressed", 0)),
        "internal_silence_spans": int(silence_stats.get("spans", 0)),
        "internal_silence_removed_ms": int(silence_stats.get("removed_ms", 0)),
        "original_ms": original_ms,
        "cleaned_ms": int(round(compacted.shape[0] * 1000.0 / sr)) if sr else 0,
    }
    return compacted, int(sr), stats


def _find_boundary_sample(
    signal: np.ndarray,
    sr: int,
    search_from_end: bool,
    window_ms: int = 300,
    frame_ms: int = 10,
    hop_ms: int = 5,
) -> int:
    if signal.size == 0:
        return 0

    speech_like, frame_samples, hop_samples = _speech_frame_mask(
        signal,
        sr=int(sr),
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    low_mask = ~speech_like

    search_samples = max(1, int(sr * window_ms / 1000.0))
    frames_in_window = max(1, search_samples // hop_samples)
    total_frames = low_mask.size

    if search_from_end:
        start_idx = max(0, total_frames - frames_in_window)
        candidates = np.where(low_mask[start_idx:])[0]
        if candidates.size > 0:
            frame_idx = start_idx + int(candidates[-1])
            return min(signal.size, frame_idx * hop_samples + frame_samples // 2)
        return signal.size

    end_idx = min(total_frames, frames_in_window)
    candidates = np.where(low_mask[:end_idx])[0]
    if candidates.size > 0:
        frame_idx = int(candidates[0])
        return min(signal.size, frame_idx * hop_samples + frame_samples // 2)
    return 0


def _equal_power_fades(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    n = max(1, int(num_samples))
    idx = np.arange(n, dtype=np.float32)
    # equal-power fade pair
    fade_out = np.cos(np.pi * idx / (2.0 * n)).astype(np.float32)
    fade_in = np.sin(np.pi * idx / (2.0 * n)).astype(np.float32)
    return fade_out, fade_in


def _normalize_loudness_rms(signal: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
    if signal.size == 0:
        return signal
    target_rms = 10.0 ** (float(target_lufs) / 20.0)
    cur_rms = float(np.sqrt(np.mean(np.square(signal)) + 1e-12))
    if cur_rms <= 1e-8:
        return signal
    out = signal * (target_rms / cur_rms)
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def _simple_splice(
    greeting_wav: np.ndarray,
    body_wav: np.ndarray,
    sr: int,
    pause_ms: int,
    crossfade_ms: int,
) -> np.ndarray:
    pause_samples = int(sr * (pause_ms / 1000.0))
    silence = np.zeros((pause_samples,), dtype=np.float32)
    base = np.concatenate([greeting_wav, silence], axis=0)

    fade_samples = int(sr * (crossfade_ms / 1000.0))
    fade_samples = max(0, min(fade_samples, base.shape[0], body_wav.shape[0]))
    if fade_samples == 0:
        return np.concatenate([base, body_wav], axis=0)
    fade_out, fade_in = _equal_power_fades(fade_samples)
    overlap = base[-fade_samples:] * fade_out + body_wav[:fade_samples] * fade_in
    return np.concatenate([base[:-fade_samples], overlap, body_wav[fade_samples:]], axis=0)


def splice_speech_segments(
    greeting_wav: np.ndarray,
    body_wav: np.ndarray,
    sample_rate: int,
    pause_ms: int = 120,
    crossfade_ms: int = 10,
    content_aware: bool = True,
    target_lufs: float = -16.0,
) -> np.ndarray:
    g = _normalize_audio(greeting_wav)
    b = _normalize_audio(body_wav)
    sr = int(sample_rate)
    pause_ms = max(0, int(pause_ms))
    crossfade_ms = max(0, int(crossfade_ms))

    if not content_aware:
        out = _simple_splice(g, b, sr=sr, pause_ms=pause_ms, crossfade_ms=crossfade_ms)
        return _normalize_loudness_rms(out, target_lufs=target_lufs)

    b_cut = _find_boundary_sample(b, sr=sr, search_from_end=False, window_ms=300)
    b_cut = max(0, min(b_cut, b.shape[0]))
    g_trim = g
    b_trim = b[b_cut:] if b_cut < b.shape[0] else b

    if b_trim.size == 0:
        b_trim = b
    out = _simple_splice(g_trim, b_trim, sr=sr, pause_ms=pause_ms, crossfade_ms=crossfade_ms)
    return _normalize_loudness_rms(out, target_lufs=target_lufs)


def splice_wavs(
    greeting_wav: np.ndarray,
    body_wav: np.ndarray,
    sr: int,
    pause_ms: int = 180,
    crossfade_ms: int = 20,
) -> np.ndarray:
    """Backward-compatible wrapper around the newer splice implementation."""
    return splice_speech_segments(
        greeting_wav=greeting_wav,
        body_wav=body_wav,
        sample_rate=sr,
        pause_ms=pause_ms,
        crossfade_ms=crossfade_ms,
        content_aware=False,
        target_lufs=-16.0,
    )
