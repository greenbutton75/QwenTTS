import hashlib
import json
from typing import Any, Mapping, Optional

import numpy as np


def _hash_scalar(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _update_hash(hasher: "hashlib._Hash", value: Any) -> None:
    if value is None:
        hasher.update(b"n:")
        return

    if isinstance(value, (bool, int, float, str)):
        hasher.update(b"s:")
        hasher.update(_hash_scalar(value))
        return

    if isinstance(value, Mapping):
        hasher.update(b"m{")
        for key in sorted(value):
            hasher.update(str(key).encode("utf-8"))
            hasher.update(b"=")
            _update_hash(hasher, value[key])
            hasher.update(b";")
        hasher.update(b"}")
        return

    if isinstance(value, (list, tuple)):
        hasher.update(b"l[")
        for item in value:
            _update_hash(hasher, item)
            hasher.update(b",")
        hasher.update(b"]")
        return

    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            array = tensor.numpy()
            hasher.update(b"t:")
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(array.tobytes())
            return
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"a:")
        hasher.update(str(tuple(array.shape)).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
        return

    hasher.update(b"r:")
    hasher.update(repr(value).encode("utf-8"))


def prompt_fingerprint(prompt_data: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, dict(prompt_data))
    return hasher.hexdigest()


def body_cache_hash(
    support_id: str,
    voice_id: str,
    body: str,
    model_size: str,
    language: str,
    prompt_data: Mapping[str, Any],
    generation_config: Optional[Mapping[str, Any]] = None,
    engine: str = "qwen3_tts_generate_voice_clone_defaults",
) -> str:
    payload = {
        "support_id": support_id,
        "voice_id": voice_id,
        "body": body.strip(),
        "model_params": {
            "model_size": model_size,
            "language": language,
            "engine": engine,
        },
        "generation_config": dict(generation_config or {}),
        "prompt_fingerprint": prompt_fingerprint(prompt_data),
        "prompt_meta": {
            "x_vector_only_mode": bool(prompt_data.get("x_vector_only_mode", False)),
            "icl_mode": bool(prompt_data.get("icl_mode", False)),
            "ref_text": prompt_data.get("ref_text") or "",
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
