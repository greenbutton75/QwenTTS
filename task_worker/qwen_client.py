from typing import Optional

import requests

from .config import (
    LOG_BACKUPS,
    LOG_DIR,
    LOG_MAX_BYTES,
    QWEN_TTS_BASE_URL,
    QWEN_TTS_HEALTH_TIMEOUT_SECONDS,
    QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS,
    QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS,
    QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS,
    QWEN_TTS_STATUS_TIMEOUT_SECONDS,
)
from timing_utils import setup_timing_logger, timed_operation


TIMING_LOGGER = setup_timing_logger(
    logger_name="task_worker.timing",
    log_dir=LOG_DIR,
    filename="task_worker_timing.log",
    max_bytes=LOG_MAX_BYTES,
    backups=LOG_BACKUPS,
)


def _request_json(method: str, url: str, *, operation: str, timeout: int, **kwargs) -> dict:
    with timed_operation(
        TIMING_LOGGER,
        operation,
        method=method.upper(),
        url=url,
        timeout_seconds=timeout,
    ) as span:
        request = getattr(requests, method.lower())
        response = request(url, timeout=timeout, **kwargs)
        span.set(status_code=response.status_code)
        response.raise_for_status()
        return response.json()


def health_check() -> dict:
    url = f"{QWEN_TTS_BASE_URL}/health"
    return _request_json(
        "get",
        url,
        operation="task_worker.http.health_check",
        timeout=QWEN_TTS_HEALTH_TIMEOUT_SECONDS,
    )


def create_profile(
    support_id: str,
    voice_id: str,
    voice_name: str,
    ref_text: Optional[str],
    xvector_only: bool,
) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/profiles"
    data = {
        "support_id": support_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "ref_text": ref_text or "",
        "xvector_only": str(bool(xvector_only)).lower(),
    }
    return _request_json(
        "post",
        url,
        operation="task_worker.http.create_profile",
        timeout=QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS,
        data=data,
    )


def get_profile_status(support_id: str, voice_id: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/profiles/{voice_id}"
    return _request_json(
        "get",
        url,
        operation="task_worker.http.get_profile_status",
        timeout=QWEN_TTS_STATUS_TIMEOUT_SECONDS,
        params={"support_id": support_id},
    )


def create_phrase(support_id: str, voice_id: str, phrase_id: str, text: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/phrases"
    payload = {
        "support_id": support_id,
        "voice_id": voice_id,
        "phrase_id": phrase_id,
        "text": text,
    }
    return _request_json(
        "post",
        url,
        operation="task_worker.http.create_phrase",
        timeout=QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS,
        json=payload,
    )


def create_phrase_splice(
    support_id: str,
    voice_id: str,
    phrase_id: str,
    greeting: str,
    body: str,
    pause_ms: int = 120,
    crossfade_ms: int = 10,
    content_aware: bool = True,
    target_lufs: float = -16.0,
) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/phrases/splice-prod"
    payload = {
        "support_id": support_id,
        "voice_id": voice_id,
        "phrase_id": phrase_id,
        "greeting": greeting,
        "body": body,
        "pause_ms": int(pause_ms),
        "crossfade_ms": int(crossfade_ms),
        "content_aware": bool(content_aware),
        "target_lufs": float(target_lufs),
    }
    return _request_json(
        "post",
        url,
        operation="task_worker.http.create_phrase_splice",
        timeout=QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS,
        json=payload,
    )


def get_phrase_status(support_id: str, phrase_id: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/phrases/{phrase_id}"
    return _request_json(
        "get",
        url,
        operation="task_worker.http.get_phrase_status",
        timeout=QWEN_TTS_STATUS_TIMEOUT_SECONDS,
        params={"support_id": support_id},
    )
