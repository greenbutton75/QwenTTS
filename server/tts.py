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
