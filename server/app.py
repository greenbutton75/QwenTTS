import json
import os
import threading
import time
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import ADMIN_PASSWORD, ADMIN_USER, S3_PREFIX, SQLITE_PATH
from .db import TaskDB
from .logging_setup import setup_logging
from .models import (
    CreatePhraseRequest,
    CreatePhraseResponse,
    CreateProfileResponse,
    PhraseStatusResponse,
    ProfileStatusResponse,
)
from .s3_store import object_exists, read_json, write_json
from .worker import Worker


os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
logger = setup_logging()
app = FastAPI(title="QwenTTS VVK API", version="1.0.0")
security = HTTPBasic()
db = TaskDB(SQLITE_PATH)
worker = Worker(db, logger)


def _voice_paths(support_id: str, voice_id: str) -> dict:
    base = f"{S3_PREFIX}/{support_id}/voices/{voice_id}"
    return {
        "sample": f"{base}/sample.wav",
        "voice_json": f"{base}/voice.json",
        "reference": f"{base}/reference.wav",
        "prompt": f"{base}/prompt.pt",
    }


def _phrase_paths(support_id: str, phrase_id: str) -> dict:
    base = f"{S3_PREFIX}/{support_id}/phrases/{phrase_id}"
    return {
        "phrase_json": f"{base}.json",
        "audio": f"{base}.wav",
    }


def _check_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    if credentials.username != ADMIN_USER or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username


@app.on_event("startup")
def _startup() -> None:
    t = threading.Thread(target=worker.run_forever, daemon=True)
    t.start()
    logger.info("API started.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/profiles", response_model=CreateProfileResponse)
async def create_profile(
    support_id: str = Form(...),
    voice_id: str = Form(...),
    voice_name: str = Form(...),
    ref_text: Optional[str] = Form(None),
    xvector_only: Optional[bool] = Form(False),
) -> CreateProfileResponse:
    if not support_id.strip():
        raise HTTPException(status_code=400, detail="support_id is required")
    if not voice_id.strip():
        raise HTTPException(status_code=400, detail="voice_id is required")
    if not voice_name.strip():
        raise HTTPException(status_code=400, detail="voice_name is required")

    paths = _voice_paths(support_id, voice_id)

    sample_key = paths["sample"]
    if not object_exists(sample_key):
        raise HTTPException(status_code=404, detail=f"sample not found at s3://{sample_key}")

    voice_json = {
        "support_id": support_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "status": "queued",
        "ref_text": ref_text,
        "x_vector_only": bool(xvector_only),
        "sample_key": sample_key,
        "reference_key": paths["reference"],
        "prompt_key": paths["prompt"],
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    write_json(paths["voice_json"], voice_json)

    db.enqueue(
        "profile",
        {
            "support_id": support_id,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "ref_text": ref_text,
            "x_vector_only": bool(xvector_only),
            "s3_sample_key": sample_key,
        },
    )

    return CreateProfileResponse(support_id=support_id, voice_id=voice_id, status="queued")


@app.get("/profiles/{voice_id}", response_model=ProfileStatusResponse)
def get_profile_status(voice_id: str, support_id: str) -> ProfileStatusResponse:
    if not support_id.strip():
        raise HTTPException(status_code=400, detail="support_id is required")
    paths = _voice_paths(support_id, voice_id)
    try:
        data = read_json(paths["voice_json"])
    except Exception:
        raise HTTPException(status_code=404, detail="Profile not found")
    return ProfileStatusResponse(
        voice_id=voice_id,
        status=data.get("status", "unknown"),
        voice_name=data.get("voice_name"),
        ref_text=data.get("ref_text"),
        error=data.get("error"),
    )


@app.post("/phrases", response_model=CreatePhraseResponse)
def create_phrase(req: CreatePhraseRequest) -> CreatePhraseResponse:
    if not req.support_id.strip():
        raise HTTPException(status_code=400, detail="support_id is required")
    if not req.voice_id.strip():
        raise HTTPException(status_code=400, detail="voice_id is required")
    if not req.phrase_id.strip():
        raise HTTPException(status_code=400, detail="phrase_id is required")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    paths = _phrase_paths(req.support_id, req.phrase_id)
    phrase_json = {
        "support_id": req.support_id,
        "voice_id": req.voice_id,
        "phrase_id": req.phrase_id,
        "text": req.text,
        "status": "queued",
        "result_key": paths["audio"],
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    write_json(paths["phrase_json"], phrase_json)

    db.enqueue(
        "phrase",
        {
            "support_id": req.support_id,
            "voice_id": req.voice_id,
            "phrase_id": req.phrase_id,
            "text": req.text,
        },
    )
    return CreatePhraseResponse(phrase_id=req.phrase_id, status="queued")


@app.get("/phrases/{phrase_id}", response_model=PhraseStatusResponse)
def get_phrase_status(phrase_id: str, support_id: str) -> PhraseStatusResponse:
    if not support_id.strip():
        raise HTTPException(status_code=400, detail="support_id is required")
    paths = _phrase_paths(support_id, phrase_id)
    try:
        data = read_json(paths["phrase_json"])
    except Exception:
        raise HTTPException(status_code=404, detail="Phrase not found")

    return PhraseStatusResponse(
        phrase_id=phrase_id,
        status=data.get("status", "unknown"),
        result_key=data.get("result_key"),
        public_url=data.get("public_url"),
        text=data.get("text"),
        voice_id=data.get("voice_id"),
        error=data.get("error"),
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(_: str = Depends(_check_admin)) -> HTMLResponse:
    stats = db.stats()
    stats_by_type = db.stats_by_type()
    recent = db.list_recent(50)
    rows = []
    for r in recent:
        payload = {}
        try:
            payload = json.loads(r.get("payload_json", "{}"))
        except Exception:
            payload = {}
        rows.append(
            "<tr>"
            f"<td>{r['id']}</td>"
            f"<td>{r['task_type']}</td>"
            f"<td>{r['status']}</td>"
            f"<td>{r['attempts']}</td>"
            f"<td>{payload.get('support_id','')}</td>"
            f"<td>{payload.get('voice_id','')}</td>"
            f"<td>{payload.get('phrase_id','')}</td>"
            f"<td>{int(r['updated_at'])}</td>"
            "</tr>"
        )
    rows = "\n".join(rows)
    by_type_rows = "\n".join(
        f"<tr><td>{t}</td><td>{v.get('queued',0)}</td><td>{v.get('running',0)}</td>"
        f"<td>{v.get('done',0)}</td><td>{v.get('failed',0)}</td></tr>"
        for t, v in stats_by_type.items()
    )
    html = f"""
    <html>
      <head><title>QwenTTS Admin</title></head>
      <body>
        <h1>QwenTTS Admin</h1>
        <h2>Queue Stats</h2>
        <ul>
          <li>Queued: {stats.get('queued', 0)}</li>
          <li>Running: {stats.get('running', 0)}</li>
          <li>Done: {stats.get('done', 0)}</li>
          <li>Failed: {stats.get('failed', 0)}</li>
        </ul>
        <h2>Stats by Task Type</h2>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr><th>Type</th><th>Queued</th><th>Running</th><th>Done</th><th>Failed</th></tr>
          {by_type_rows}
        </table>
        <h2>Recent Tasks</h2>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr><th>ID</th><th>Type</th><th>Status</th><th>Attempts</th><th>Support</th><th>Voice</th><th>Phrase</th><th>Updated</th></tr>
          {rows}
        </table>
      </body>
    </html>
    """
    return HTMLResponse(html)
