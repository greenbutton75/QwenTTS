import os
import tempfile
from typing import Tuple

import boto3
from pydub import AudioSegment

from .config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME


_s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)


def _to_wav_if_needed(path: str) -> Tuple[str, bool]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path, False
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio.export(tmp.name, format="wav")
    return tmp.name, True


def upload_sample(support_id: str, voice_id: str, local_path: str) -> str:
    wav_path, is_temp = _to_wav_if_needed(local_path)
    key = f"support/{support_id}/voices/{voice_id}/sample.wav"
    with open(wav_path, "rb") as f:
        _s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=f.read(),
            ContentType="audio/wav",
        )
    if is_temp:
        os.remove(wav_path)
    return key
