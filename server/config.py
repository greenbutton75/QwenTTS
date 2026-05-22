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
GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS = _get_env_int("GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS", 5)
GREETING_FULL_PHRASE_MAX_ATTEMPTS = _get_env_int(
    "GREETING_FULL_PHRASE_MAX_ATTEMPTS",
    min(GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS, 2),
)
GREETING_SPLICE_MAX_ATTEMPTS = _get_env_int(
    "GREETING_SPLICE_MAX_ATTEMPTS",
    min(GREETING_SPEAKER_SIMILARITY_MAX_ATTEMPTS, 3),
)
GREETING_SPLICE_MAX_NEW_TOKENS = _get_env_int(
    "GREETING_SPLICE_MAX_NEW_TOKENS",
    min(VOICE_CLONE_MAX_NEW_TOKENS, 256),
)
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
BODY_QUALITY_ALGORITHM_VERSION = os.getenv("BODY_QUALITY_ALGORITHM_VERSION", "body_boundary_v2")
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

# --- Quality Gate (Phase 1): ASR diagnostics + composite score ---
# All flags default OFF / diagnostic-only so a fresh deploy behaves exactly
# like the pre-Phase-1 service.

# ASR / Whisper
ASR_DIAGNOSTIC_MODE = _get_env_bool("ASR_DIAGNOSTIC_MODE", False)
BODY_ASR_DIAGNOSTIC_MODE = _get_env_bool("BODY_ASR_DIAGNOSTIC_MODE", False)
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "large-v3")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
ASR_BEAM_SIZE = _get_env_int("ASR_BEAM_SIZE", 5)
ASR_NO_SPEECH_THRESHOLD = _get_env_float("ASR_NO_SPEECH_THRESHOLD", 0.6)

# Greeting ASR gate thresholds (used to compute asr_passed; enforcement is Phase 2)
GREETING_ASR_MAX_WER = _get_env_float("GREETING_ASR_MAX_WER", 0.20)
GREETING_ASR_PREFIX_EXTRA_REJECT = _get_env_bool("GREETING_ASR_PREFIX_EXTRA_REJECT", True)
GREETING_ASR_SUFFIX_CLIPPED_REJECT = _get_env_bool("GREETING_ASR_SUFFIX_CLIPPED_REJECT", True)
GREETING_ASR_REFERENCE_LEAK_REJECT = _get_env_bool("GREETING_ASR_REFERENCE_LEAK_REJECT", True)

# Composite score weights (TODO: calibrate after baseline measurement)
SCORE_W_SIM = _get_env_float("SCORE_W_SIM", 1.0)
SCORE_W_ASR = _get_env_float("SCORE_W_ASR", 1.5)
SCORE_P_PREFIX = _get_env_float("SCORE_P_PREFIX", 0.50)
SCORE_P_SUFFIX = _get_env_float("SCORE_P_SUFFIX", 0.50)
SCORE_P_REFLEAK = _get_env_float("SCORE_P_REFLEAK", 1.00)
SCORE_P_ONSET = _get_env_float("SCORE_P_ONSET", 0.30)
SCORE_P_PREROLL = _get_env_float("SCORE_P_PREROLL", 0.30)
SCORE_P_ENDING = _get_env_float("SCORE_P_ENDING", 0.30)
SCORE_P_BODY_START = _get_env_float("SCORE_P_BODY_START", 0.30)
SCORE_P_BODY_TAIL = _get_env_float("SCORE_P_BODY_TAIL", 0.30)
SCORE_P_BODY_CLIPPED = _get_env_float("SCORE_P_BODY_CLIPPED", 0.30)
# Over-long greeting (leading silence/noise). Penalty only, never a hard fail --
# best-of-N then picks a shorter clean candidate (see candidate_pool).
SCORE_P_DURATION = _get_env_float("SCORE_P_DURATION", 0.30)

# --- Quality Gate (Phase 2): best-of-N greeting selection ---
# Default ON: rolled out to prod after validation. onstart re-clones main and
# rebuilds /etc/qwentts.env from the template, so a code default is the durable
# switch (env-only flags would silently revert on instance recreate). Set the
# env var to false to disable.
GREETING_BEST_OF_N_ENABLED = _get_env_bool("GREETING_BEST_OF_N_ENABLED", True)
GREETING_BEST_OF_N_COUNT = _get_env_int("GREETING_BEST_OF_N_COUNT", 4)
GREETING_ASR_CHECK = _get_env_bool("GREETING_ASR_CHECK", True)
GREETING_ASR_REQUIRE_PASS = _get_env_bool("GREETING_ASR_REQUIRE_PASS", False)

# --- Quality Gate (Phase 3): adaptive best-of-N for the body / long render ---
# Adaptive: keep a good greedy render (1 generation); regenerate only when the
# body is bad (ASR garble / artifact), then pick the best candidate.
# Default ON (rolled out after validation; see GREETING_BEST_OF_N_ENABLED note).
BODY_BEST_OF_N_ENABLED = _get_env_bool("BODY_BEST_OF_N_ENABLED", True)
BODY_BEST_OF_N_MAX_COUNT = _get_env_int("BODY_BEST_OF_N_MAX_COUNT", 3)
# WER bar for accepting a body greedy render. Looser than the greeting bar
# because spelled-out phone numbers inflate WER on legitimate audio.
BODY_ASR_MAX_WER = _get_env_float("BODY_ASR_MAX_WER", 0.35)

# Conservative trailing-silence trim on the final merged phrase. VAD finds the
# last real speech; only clearly non-speech tail beyond the pad is removed, so a
# final word cannot be cut. Default on (it only ever removes silence/noise).
SPLICE_TRAILING_SILENCE_TRIM_ENABLED = _get_env_bool("SPLICE_TRAILING_SILENCE_TRIM_ENABLED", True)
SPLICE_TRAILING_SILENCE_PAD_MS = _get_env_int("SPLICE_TRAILING_SILENCE_PAD_MS", 500)
SPLICE_TRAILING_SILENCE_MIN_MS = _get_env_int("SPLICE_TRAILING_SILENCE_MIN_MS", 700)


def asr_config() -> dict:
    return {
        "diagnostic_mode": ASR_DIAGNOSTIC_MODE,
        "body_diagnostic_mode": BODY_ASR_DIAGNOSTIC_MODE,
        "model_size": ASR_MODEL_SIZE,
        "compute_type": ASR_COMPUTE_TYPE,
        "beam_size": ASR_BEAM_SIZE,
        "no_speech_threshold": ASR_NO_SPEECH_THRESHOLD,
        "max_wer": GREETING_ASR_MAX_WER,
        "prefix_extra_reject": GREETING_ASR_PREFIX_EXTRA_REJECT,
        "suffix_clipped_reject": GREETING_ASR_SUFFIX_CLIPPED_REJECT,
        "reference_leak_reject": GREETING_ASR_REFERENCE_LEAK_REJECT,
    }


def score_weights() -> dict:
    return {
        "w_sim": SCORE_W_SIM,
        "w_asr": SCORE_W_ASR,
        "p_prefix": SCORE_P_PREFIX,
        "p_suffix": SCORE_P_SUFFIX,
        "p_refleak": SCORE_P_REFLEAK,
        "p_onset": SCORE_P_ONSET,
        "p_preroll": SCORE_P_PREROLL,
        "p_ending": SCORE_P_ENDING,
        "p_body_start": SCORE_P_BODY_START,
        "p_body_tail": SCORE_P_BODY_TAIL,
        "p_body_clipped": SCORE_P_BODY_CLIPPED,
        "p_duration": SCORE_P_DURATION,
    }


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


def greeting_splice_generate_config() -> dict:
    config = voice_clone_generate_config()
    config["max_new_tokens"] = GREETING_SPLICE_MAX_NEW_TOKENS
    return config


def greeting_splice_retry_generate_config() -> dict:
    config = greeting_similarity_retry_generate_config()
    config["max_new_tokens"] = GREETING_SPLICE_MAX_NEW_TOKENS
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
