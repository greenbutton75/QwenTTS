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
PHRASE_POLL_INTERVAL = int(os.getenv("PHRASE_POLL_INTERVAL", "5"))
PHRASE_PAGE_SIZE = int(os.getenv("PHRASE_PAGE_SIZE", "50"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "1800"))
QWEN_TTS_READY_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_READY_TIMEOUT_SECONDS", "300"))
QWEN_TTS_READY_POLL_INTERVAL = int(os.getenv("QWEN_TTS_READY_POLL_INTERVAL", "2"))
QWEN_TTS_HEALTH_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_HEALTH_TIMEOUT_SECONDS", "5"))
QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS", "180"))
QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS", "150"))
QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS", "150"))
QWEN_TTS_STATUS_TIMEOUT_SECONDS = int(os.getenv("QWEN_TTS_STATUS_TIMEOUT_SECONDS", "30"))
PHRASE_SPLICE_PAUSE_MS = int(os.getenv("PHRASE_SPLICE_PAUSE_MS", "220"))
PHRASE_SPLICE_CROSSFADE_MS = int(os.getenv("PHRASE_SPLICE_CROSSFADE_MS", "10"))

STAGE_PROCESSING = os.getenv("STAGE_PROCESSING", "PROCESSING")
ENABLE_PHRASE_SPLICE_GROUPING = os.getenv("ENABLE_PHRASE_SPLICE_GROUPING", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

HEALTH_PORT = int(os.getenv("TASK_WORKER_HEALTH_PORT", "8010"))

LOG_DIR = os.getenv("TASK_WORKER_LOG_DIR", "/workspace/QwenTTS/logs")
LOG_MAX_BYTES = int(os.getenv("TASK_WORKER_LOG_MAX_BYTES", "10485760"))
LOG_BACKUPS = int(os.getenv("TASK_WORKER_LOG_BACKUPS", "7"))
