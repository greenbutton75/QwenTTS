from typing import Literal, Optional

from pydantic import BaseModel


class CreateProfileResponse(BaseModel):
    support_id: str
    voice_id: str
    status: str


class CreatePhraseRequest(BaseModel):
    support_id: str
    voice_id: str
    phrase_id: str
    text: str


class SpliceTestRequest(BaseModel):
    support_id: str
    voice_id: str
    greeting: str
    body: str
    pause_ms: int = 180
    crossfade_ms: int = 20
    content_aware: bool = True
    target_lufs: float = -16.0
    mode: Literal["wav_splice", "latent_concat"] = "wav_splice"


class CreatePhraseResponse(BaseModel):
    phrase_id: str
    status: str


class PhraseStatusResponse(BaseModel):
    phrase_id: str
    status: str
    result_key: Optional[str] = None
    public_url: Optional[str] = None
    text: Optional[str] = None
    voice_id: Optional[str] = None
    error: Optional[str] = None


class ProfileStatusResponse(BaseModel):
    voice_id: str
    status: str
    voice_name: Optional[str] = None
    ref_text: Optional[str] = None
    error: Optional[str] = None
