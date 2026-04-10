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
GREETING_SPEAKER_SIMILARITY_CHECK = _get_env_bool("GREETING_SPEAKER_SIMILARITY_CHECK", True)
GREETING_SPEAKER_SIMILARITY_THRESHOLD = _get_env_float("GREETING_SPEAKER_SIMILARITY_THRESHOLD", 0.55)
GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS = _get_env_int("GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS", 3)
GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS = _get_env_bool("GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS", False)
GREETING_ONSET_ARTIFACT_CHECK = _get_env_bool("GREETING_ONSET_ARTIFACT_CHECK", True)
GREETING_ONSET_ARTIFACT_REQUIRE_PASS = _get_env_bool("GREETING_ONSET_ARTIFACT_REQUIRE_PASS", True)
GREETING_SPEAKER_SIMILARITY_RETRY_DO_SAMPLE = _get_env_bool("GREETING_SPEAKER_SIMILARITY_RETRY_DO_SAMPLE", True)
GREETING_SPEAKER_SIMILARITY_RETRY_TEMPERATURE = _get_env_float("GREETING_SPEAKER_SIMILARITY_RETRY_TEMPERATURE", 0.3)
GREETING_SPEAKER_SIMILARITY_RETRY_TOP_K = _get_env_int("GREETING_SPEAKER_SIMILARITY_RETRY_TOP_K", 8)
GREETING_SPEAKER_SIMILARITY_RETRY_TOP_P = _get_env_float("GREETING_SPEAKER_SIMILARITY_RETRY_TOP_P", 0.9)
BODY_QUALITY_CHECK = _get_env_bool("BODY_QUALITY_CHECK", True)
BODY_QUALITY_REQUIRE_PASS = _get_env_bool("BODY_QUALITY_REQUIRE_PASS", True)
BODY_QUALITY_MAX_ATTEMPTS = _get_env_int("BODY_QUALITY_MAX_ATTEMPTS", 3)
BODY_QUALITY_RETRY_DO_SAMPLE = _get_env_bool("BODY_QUALITY_RETRY_DO_SAMPLE", True)
BODY_QUALITY_RETRY_TEMPERATURE = _get_env_float("BODY_QUALITY_RETRY_TEMPERATURE", 0.25)
BODY_QUALITY_RETRY_TOP_K = _get_env_int("BODY_QUALITY_RETRY_TOP_K", 6)
BODY_QUALITY_RETRY_TOP_P = _get_env_float("BODY_QUALITY_RETRY_TOP_P", 0.85)
BODY_QUALITY_ALGORITHM_VERSION = os.getenv("BODY_QUALITY_ALGORITHM_VERSION", "body_boundary_v1")
REFERENCE_AUDIO_TRIM_ENABLED = _get_env_bool("REFERENCE_AUDIO_TRIM_ENABLED", True)
REFERENCE_AUDIO_TRIM_PAD_MS = _get_env_int("REFERENCE_AUDIO_TRIM_PAD_MS", 80)
REFERENCE_AUDIO_TRIM_MAX_LEADING_MS = _get_env_int("REFERENCE_AUDIO_TRIM_MAX_LEADING_MS", 2500)
REFERENCE_AUDIO_TRIM_MAX_TRAILING_MS = _get_env_int("REFERENCE_AUDIO_TRIM_MAX_TRAILING_MS", 1500)
OUTPUT_AUDIO_TRIM_ENABLED = _get_env_bool("OUTPUT_AUDIO_TRIM_ENABLED", True)
OUTPUT_AUDIO_TRIM_PAD_MS = _get_env_int("OUTPUT_AUDIO_TRIM_PAD_MS", 30)
GREETING_OUTPUT_TRIM_PAD_MS = _get_env_int("GREETING_OUTPUT_TRIM_PAD_MS", 160)
OUTPUT_AUDIO_TRIM_MAX_LEADING_MS = _get_env_int("OUTPUT_AUDIO_TRIM_MAX_LEADING_MS", 15000)
OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS = _get_env_int("OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS", 2000)
OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS = _get_env_int("OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS", 600)
OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION = os.getenv("OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION", "rms_flatness_pause_compact_v4")


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


def greeting_similarity_retry_generate_config() -> dict:
    config = {
        "do_sample": GREETING_SPEAKER_SIMILARITY_RETRY_DO_SAMPLE,
        "non_streaming_mode": VOICE_CLONE_NON_STREAMING_MODE,
        "max_new_tokens": VOICE_CLONE_MAX_NEW_TOKENS,
        "repetition_penalty": VOICE_CLONE_REPETITION_PENALTY,
    }
    if GREETING_SPEAKER_SIMILARITY_RETRY_DO_SAMPLE:
        config.update(
            temperature=GREETING_SPEAKER_SIMILARITY_RETRY_TEMPERATURE,
            top_k=GREETING_SPEAKER_SIMILARITY_RETRY_TOP_K,
            top_p=GREETING_SPEAKER_SIMILARITY_RETRY_TOP_P,
        )
    return config


def body_quality_retry_generate_config() -> dict:
    config = {
        "do_sample": BODY_QUALITY_RETRY_DO_SAMPLE,
        "non_streaming_mode": VOICE_CLONE_NON_STREAMING_MODE,
        "max_new_tokens": VOICE_CLONE_MAX_NEW_TOKENS,
        "repetition_penalty": VOICE_CLONE_REPETITION_PENALTY,
    }
    if BODY_QUALITY_RETRY_DO_SAMPLE:
        config.update(
            temperature=BODY_QUALITY_RETRY_TEMPERATURE,
            top_k=BODY_QUALITY_RETRY_TOP_K,
            top_p=BODY_QUALITY_RETRY_TOP_P,
        )
    return config


def body_quality_config() -> dict:
    return {
        "enabled": BODY_QUALITY_CHECK,
        "require_pass": BODY_QUALITY_REQUIRE_PASS,
        "max_attempts": BODY_QUALITY_MAX_ATTEMPTS,
        "algorithm_version": BODY_QUALITY_ALGORITHM_VERSION,
    }


def output_audio_trim_config() -> dict:
    return {
        "enabled": OUTPUT_AUDIO_TRIM_ENABLED,
        "pad_ms": OUTPUT_AUDIO_TRIM_PAD_MS,
        "max_leading_ms": OUTPUT_AUDIO_TRIM_MAX_LEADING_MS,
        "max_trailing_ms": OUTPUT_AUDIO_TRIM_MAX_TRAILING_MS,
        "max_internal_silence_ms": OUTPUT_AUDIO_MAX_INTERNAL_SILENCE_MS,
        "algorithm_version": OUTPUT_AUDIO_TRIM_ALGORITHM_VERSION,
    }
