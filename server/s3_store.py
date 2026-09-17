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


def _to_inference_dtype(obj: Any) -> Any:
    """Recursively downcast bfloat16 tensors to float32 on non-CUDA hosts.

    All cached torch objects in S3 (prompt.pt speaker embeddings, splice
    body cache) were produced on the CUDA box in bfloat16. The live model
    itself already loads as float32 on any non-CUDA device (see
    server/tts.py::_get_model), and MPS (this Mac's backend, torch 2.2.2)
    cannot hold a bfloat16 tensor on-device at all -- moving one over
    raises "BFloat16 is not supported on MPS" before any dtype cast even
    runs. Downcasting here, once, right after load, keeps every downstream
    call site consistent with the live model's dtype without hunting down
    each individual .to(device) call in the model code. CUDA path is
    untouched (bfloat16 there is intended, not a bug).
    """
    import torch

    if torch.cuda.is_available():
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.float32) if obj.dtype == torch.bfloat16 else obj
    if isinstance(obj, dict):
        return {k: _to_inference_dtype(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_inference_dtype(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_inference_dtype(v) for v in obj)
    return obj


def download_torch(key: str) -> Any:
    import torch

    raw = download_bytes(key)
    bio = io.BytesIO(raw)
    data = torch.load(bio, map_location="cpu")
    return _to_inference_dtype(data)


def delete_prefix(prefix: str) -> int:
    deleted = 0
    continuation_token = None
    while True:
        kwargs = {
            "Bucket": S3_BUCKET_NAME,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = _s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        if contents:
            objects = [{"Key": item["Key"]} for item in contents]
            _s3.delete_objects(
                Bucket=S3_BUCKET_NAME,
                Delete={"Objects": objects, "Quiet": True},
            )
            deleted += len(objects)

        if not resp.get("IsTruncated"):
            return deleted
        continuation_token = resp.get("NextContinuationToken")


def create_presigned_url(key: str, expires_seconds: int) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_seconds,
    )
