from typing import Optional

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


class CreatePhraseResponse(BaseModel):
    phrase_id: str
    status: str


class PhraseStatusResponse(BaseModel):
    phrase_id: str
    status: str
    result_key: Optional[str] = None
    text: Optional[str] = None
    voice_id: Optional[str] = None
    error: Optional[str] = None


class ProfileStatusResponse(BaseModel):
    voice_id: str
    status: str
    voice_name: Optional[str] = None
    ref_text: Optional[str] = None
    error: Optional[str] = None
