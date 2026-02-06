import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val


S3_BUCKET_NAME = _get_env("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = _get_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get_env("AWS_SECRET_ACCESS_KEY")
AWS_REGION = _get_env("AWS_REGION", "us-east-1")

S3_PREFIX = os.getenv("S3_PREFIX", "support").strip().strip("/")

ADMIN_USER = _get_env("ADMIN_USER")
ADMIN_PASSWORD = _get_env("ADMIN_PASSWORD")

MODEL_SIZE = os.getenv("MODEL_SIZE", "1.7B")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_SECONDS = int(os.getenv("RETRY_BASE_SECONDS", "30"))

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", "7"))

SQLITE_PATH = os.getenv("SQLITE_PATH", "data/queue.db")

LANGUAGE = os.getenv("LANGUAGE", "English")
