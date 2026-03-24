import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return bool(default)
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return int(default)
    return int(val)


def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return float(default)
    return float(val)


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

# Production voice clone should prefer speaker stability over variation.
VOICE_CLONE_DO_SAMPLE = _get_env_bool("VOICE_CLONE_DO_SAMPLE", False)
VOICE_CLONE_NON_STREAMING_MODE = _get_env_bool("VOICE_CLONE_NON_STREAMING_MODE", True)
VOICE_CLONE_MAX_NEW_TOKENS = _get_env_int("VOICE_CLONE_MAX_NEW_TOKENS", 2048)
VOICE_CLONE_REPETITION_PENALTY = _get_env_float("VOICE_CLONE_REPETITION_PENALTY", 1.02)
VOICE_CLONE_TEMPERATURE = _get_env_float("VOICE_CLONE_TEMPERATURE", 0.2)
VOICE_CLONE_TOP_K = _get_env_int("VOICE_CLONE_TOP_K", 1)
VOICE_CLONE_TOP_P = _get_env_float("VOICE_CLONE_TOP_P", 1.0)


def voice_clone_generate_config() -> dict:
    config = {
        "do_sample": VOICE_CLONE_DO_SAMPLE,
        "non_streaming_mode": VOICE_CLONE_NON_STREAMING_MODE,
        "max_new_tokens": VOICE_CLONE_MAX_NEW_TOKENS,
        "repetition_penalty": VOICE_CLONE_REPETITION_PENALTY,
    }
    if VOICE_CLONE_DO_SAMPLE:
        config.update(
            temperature=VOICE_CLONE_TEMPERATURE,
            top_k=VOICE_CLONE_TOP_K,
            top_p=VOICE_CLONE_TOP_P,
        )
    return config
