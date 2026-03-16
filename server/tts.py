import io
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

from .config import LANGUAGE, MODEL_SIZE


_MODEL_CACHE: Dict[str, Qwen3TTSModel] = {}


def _get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> Qwen3TTSModel:
    key = f"Base-{MODEL_SIZE}"
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
    tts = _get_model()
    items = tts.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=(ref_text.strip() if ref_text else None),
        x_vector_only_mode=bool(x_vector_only),
    )
    if not items:
        raise RuntimeError("Voice prompt creation returned empty result.")
    return items[0]


def generate_voice(
    text: str,
    voice_prompt: List[VoiceClonePromptItem],
) -> Tuple[np.ndarray, int]:
    tts = _get_model()
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        voice_clone_prompt=voice_prompt,
    )
    return wavs[0], sr


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


def _adaptive_vad_threshold(rms: np.ndarray) -> float:
    if rms.size == 0:
        return 0.0
    base = float(np.percentile(rms, 15))
    high = float(np.percentile(rms, 90))
    # Conservative threshold to pick near-silence regions.
    return max(1e-5, base + 0.2 * max(0.0, high - base))


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

    frame_samples = max(1, int(sr * frame_ms / 1000.0))
    hop_samples = max(1, int(sr * hop_ms / 1000.0))
    rms = _rms_envelope(signal, frame_samples=frame_samples, hop_samples=hop_samples)
    thr = _adaptive_vad_threshold(rms)
    low_mask = rms <= thr

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

    g_cut = _find_boundary_sample(g, sr=sr, search_from_end=True, window_ms=300)
    b_cut = _find_boundary_sample(b, sr=sr, search_from_end=False, window_ms=300)
    g_cut = max(0, min(g_cut, g.shape[0]))
    b_cut = max(0, min(b_cut, b.shape[0]))
    g_trim = g[:g_cut] if g_cut > 0 else g
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
