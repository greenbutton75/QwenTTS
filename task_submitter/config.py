import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val


BASE_URL = _get_env("TASK_BASE_URL", "https://rixtrema.net/api/async_task_manager")
USER_TOKEN = _get_env("USER_TOKEN")

AWS_ACCESS_KEY_ID = _get_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get_env("AWS_SECRET_ACCESS_KEY")
AWS_REGION = _get_env("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = _get_env("S3_BUCKET_NAME")

