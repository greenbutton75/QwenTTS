import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val


BASE_URL = _get_env("TASK_BASE_URL", "https://rixtrema.net/api/async_task_manager")
USER_TOKEN = _get_env("USER_TOKEN")
SYSTEM_TOKEN = _get_env("SYSTEM_TOKEN")
FINGERPRINT = _get_env("FINGERPRINT")

QWEN_TTS_BASE_URL = _get_env("QWEN_TTS_BASE_URL", "http://127.0.0.1:8000")

AWS_ACCESS_KEY_ID = _get_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get_env("AWS_SECRET_ACCESS_KEY")
AWS_REGION = _get_env("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = _get_env("S3_BUCKET_NAME")

PROFILE_POLL_INTERVAL = int(os.getenv("PROFILE_POLL_INTERVAL", "15"))
PHRASE_POLL_INTERVAL = int(os.getenv("PHRASE_POLL_INTERVAL", "15"))
PHRASE_PAGE_SIZE = int(os.getenv("PHRASE_PAGE_SIZE", "50"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "1800"))

STAGE_PROCESSING = os.getenv("STAGE_PROCESSING", "PROCESSING")

