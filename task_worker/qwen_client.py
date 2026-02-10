from typing import Optional

import requests

from .config import QWEN_TTS_BASE_URL


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
    r = requests.post(url, data=data, timeout=180)
    r.raise_for_status()
    return r.json()


def get_profile_status(support_id: str, voice_id: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/profiles/{voice_id}"
    r = requests.get(url, params={"support_id": support_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def create_phrase(support_id: str, voice_id: str, phrase_id: str, text: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/phrases"
    payload = {
        "support_id": support_id,
        "voice_id": voice_id,
        "phrase_id": phrase_id,
        "text": text,
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def get_phrase_status(support_id: str, phrase_id: str) -> dict:
    url = f"{QWEN_TTS_BASE_URL}/phrases/{phrase_id}"
    r = requests.get(url, params={"support_id": support_id}, timeout=30)
    r.raise_for_status()
    return r.json()
