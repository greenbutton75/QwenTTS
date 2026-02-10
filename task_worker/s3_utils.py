import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from .config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME


_s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)


def object_exists(key: str) -> bool:
    try:
        _s3.head_object(Bucket=S3_BUCKET_NAME, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def read_json(key: str) -> Dict[str, Any]:
    resp = _s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    raw = resp["Body"].read().decode("utf-8")
    return json.loads(raw)

