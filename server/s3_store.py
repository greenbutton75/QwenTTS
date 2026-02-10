import io
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


def upload_bytes(key: str, data: bytes, content_type: str) -> None:
    _s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def upload_file(key: str, path: str, content_type: str) -> None:
    with open(path, "rb") as f:
        upload_bytes(key, f.read(), content_type)


def download_bytes(key: str) -> bytes:
    resp = _s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    return resp["Body"].read()


def object_exists(key: str) -> bool:
    try:
        _s3.head_object(Bucket=S3_BUCKET_NAME, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def write_json(key: str, data: Dict[str, Any]) -> None:
    buf = json.dumps(data, ensure_ascii=False).encode("utf-8")
    upload_bytes(key, buf, "application/json")


def read_json(key: str) -> Dict[str, Any]:
    raw = download_bytes(key)
    return json.loads(raw.decode("utf-8"))


def upload_torch(key: str, data: Any) -> None:
    import torch

    bio = io.BytesIO()
    torch.save(data, bio)
    upload_bytes(key, bio.getvalue(), "application/octet-stream")


def download_torch(key: str) -> Any:
    import torch

    raw = download_bytes(key)
    bio = io.BytesIO(raw)
    return torch.load(bio, map_location="cpu")


def create_presigned_url(key: str, expires_seconds: int) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_seconds,
    )
