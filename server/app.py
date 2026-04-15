import json
import os
import threading
import time
import urllib.request
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from qwen_tts import VoiceClonePromptItem

from .config import ADMIN_PASSWORD, ADMIN_USER, LANGUAGE, MODEL_SIZE, S3_PREFIX, SQLITE_PATH
from .config import voice_clone_generate_config
from .config import (
    BODY_QUALITY_REQUIRE_PASS,
    GREETING_ONSET_ARTIFACT_REQUIRE_PASS,
    GREETING_SPEAKER_SIMILARITY_CHECK,
    GREETING_SPLICE_MAX_ATTEMPTS,
    GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS,
    GREETING_SPEAKER_SIMILARITY_THRESHOLD,
    body_quality_config,
    output_audio_trim_config,
)
from .cache_utils import body_cache_hash, prompt_fingerprint
from .db import TaskDB
from .logging_setup import setup_logging, setup_timing_logging
from .models import (
    CreatePhraseRequest,
    CreatePhraseResponse,
    CreateProfileResponse,
    CreateSplicePhraseRequest,
    PhraseStatusResponse,
    ProfileStatusResponse,
    SpliceTestRequest,
)
from .s3_store import (
    create_presigned_url,
    download_bytes,
    download_torch,
    object_exists,
    read_json,
    upload_bytes,
    write_json,
)
from .tts import (
    clean_output_audio,
    clean_output_audio_preserve_start,
    clean_output_audio_for_greeting,
    clean_output_audio_for_spliced_phrase,
    clean_output_audio_without_leading_trim,
    generate_body_with_quality_retry,
    generate_voice,
    generate_voice_with_similarity_retry,
    greeting_splice_generate_configs,
    is_fatal_cuda_error,
    splice_speech_segments,
    wav_from_bytes,
    wav_to_bytes,
)
from .worker import Worker
from timing_utils import timed_operation


os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
logger = setup_logging()
timing_logger = setup_timing_logging()
app = FastAPI(title="QwenTTS VVK API", version="1.0.0")
security = HTTPBasic()
db = TaskDB(SQLITE_PATH)
worker = Worker(db, logger)
VOICE_CLONE_GENERATE_CONFIG = voice_clone_generate_config()
OUTPUT_AUDIO_TRIM_CONFIG = output_audio_trim_config()
BODY_QUALITY_CONFIG = body_quality_config()


def _crash_process_on_fatal_cuda(exc: Exception, context: str) -> None:
    if not is_fatal_cuda_error(exc):
        return
    logger.critical(
        "Fatal CUDA error in %s. Uvicorn process will exit so run_api.sh can restart it. Error: %s",
        context,
        exc,
    )
    os._exit(86)


def _voice_paths(support_id: str, voice_id: str) -> dict:
    base = f"{S3_PREFIX}/{support_id}/voices/{voice_id}"
    return {
        "sample": f"{base}/sample.wav",
        "voice_json": f"{base}/voice.json",
        "reference": f"{base}/reference.wav",
        "prompt": f"{base}/prompt.pt",
        "splice_cache_prefix": f"{base}/splice_cache/",
    }


def _phrase_paths(support_id: str, phrase_id: str) -> dict:
    base = f"{S3_PREFIX}/{support_id}/phrases/{phrase_id}"
    return {
        "phrase_json": f"{base}.json",
        "audio": f"{base}.wav",
    }


def _body_cache_paths(support_id: str, voice_id: str, cache_hash: str) -> dict:
    base = f"{S3_PREFIX}/{support_id}/voices/{voice_id}/splice_cache"
    return {
        "audio": f"{base}/body_{cache_hash}.wav",
        "meta": f"{base}/body_{cache_hash}.json",
    }


def _body_cache_hash(
    support_id: str,
    voice_id: str,
    body: str,
    prompt_data: dict,
) -> str:
    return body_cache_hash(
        support_id=support_id,
        voice_id=voice_id,
        body=body,
        model_size=MODEL_SIZE,
        language=LANGUAGE,
        prompt_data=prompt_data,
        generation_config={
            "voice_clone": dict(VOICE_CLONE_GENERATE_CONFIG),
            "output_trim": dict(OUTPUT_AUDIO_TRIM_CONFIG),
            "body_quality": dict(BODY_QUALITY_CONFIG),
        },
    )


def _load_voice_prompt(support_id: str, voice_id: str):
    with timed_operation(
        timing_logger,
        "api.load_voice_prompt",
        support_id=support_id,
        voice_id=voice_id,
    ):
        voice_paths = _voice_paths(support_id, voice_id)
        if not object_exists(voice_paths["prompt"]):
            raise HTTPException(status_code=404, detail="voice prompt not found, profile is not ready")
        prompt_data = download_torch(voice_paths["prompt"])
        voice_prompt = [
            VoiceClonePromptItem(
                ref_code=prompt_data["ref_code"],
                ref_spk_embedding=prompt_data["ref_spk_embedding"],
                x_vector_only_mode=prompt_data["x_vector_only_mode"],
                icl_mode=prompt_data["icl_mode"],
                ref_text=prompt_data.get("ref_text"),
            )
        ]
        return prompt_data, voice_prompt


def _load_or_generate_body_wav(
    support_id: str,
    voice_id: str,
    body: str,
    prompt_data: dict,
    voice_prompt,
):
    with timed_operation(
        timing_logger,
        "api.body_cache.prepare",
        support_id=support_id,
        voice_id=voice_id,
        body_chars=len(body or ""),
    ) as span:
        prompt_digest = prompt_fingerprint(prompt_data)
        body_hash = _body_cache_hash(support_id, voice_id, body, prompt_data)
        cache_paths = _body_cache_paths(support_id, voice_id, body_hash)
        body_cache_hit = object_exists(cache_paths["audio"])
        body_wav = None
        sr_body = None
        if body_cache_hit:
            with timed_operation(
                timing_logger,
                "api.body_cache.download",
                support_id=support_id,
                voice_id=voice_id,
                body_hash=body_hash,
            ):
                try:
                    body_wav, sr_body = wav_from_bytes(download_bytes(cache_paths["audio"]))
                except Exception:
                    body_cache_hit = False

        if not body_cache_hit:
            with timed_operation(
                timing_logger,
                "api.body_cache.generate",
                support_id=support_id,
                voice_id=voice_id,
                body_hash=body_hash,
                body_chars=len(body or ""),
            ):
                body_wav, sr_body, body_attempts, body_passed, body_quality, body_trim = generate_body_with_quality_retry(
                    body,
                    voice_prompt,
                )
                if BODY_QUALITY_REQUIRE_PASS and not body_passed:
                    raise HTTPException(status_code=422, detail="body boundary artifact detected in all attempts")
                body_bytes = wav_to_bytes(body_wav, int(sr_body))
                upload_bytes(cache_paths["audio"], body_bytes, "audio/wav")
                write_json(
                    cache_paths["meta"],
                    {
                        "support_id": support_id,
                        "voice_id": voice_id,
                        "body_hash": body_hash,
                        "body_text": body,
                        "prompt_fingerprint": prompt_digest,
                        "model_size": MODEL_SIZE,
                        "language": LANGUAGE,
                        "generation_config": dict(VOICE_CLONE_GENERATE_CONFIG),
                        "body_quality": body_quality,
                        "body_attempts": body_attempts,
                        "output_trim": body_trim,
                        "created_at": int(time.time()),
                    },
                )
        span.set(body_hash=body_hash, body_cache_hit=body_cache_hit)
        return body_wav, int(sr_body), body_hash, body_cache_hit


def _synthesize_spliced_phrase(
    support_id: str,
    voice_id: str,
    greeting: str,
    body: str,
    pause_ms: int,
    crossfade_ms: int,
    content_aware: bool,
    target_lufs: float,
):
    with timed_operation(
        timing_logger,
        "api.splice_synthesize.total",
        support_id=support_id,
        voice_id=voice_id,
        greeting_chars=len(greeting or ""),
        body_chars=len(body or ""),
        pause_ms=pause_ms,
        crossfade_ms=crossfade_ms,
        content_aware=content_aware,
    ) as span:
        prompt_data, voice_prompt = _load_voice_prompt(support_id, voice_id)
        prompt_digest = prompt_fingerprint(prompt_data)
        greeting_similarity = None
        greeting_attempts = 1
        greeting_similarity_passed = None
        greeting_quality = {
            "similarity_passed": 0,
            "onset_artifact": 0,
            "onset_checked": 0,
            "onset_passed": 1,
            "duration_artifact": 0,
            "duration_checked": 0,
            "duration_passed": 1,
            "ending_artifact": 0,
            "ending_checked": 0,
            "ending_passed": 1,
            "preroll_artifact": 0,
            "preroll_checked": 0,
            "preroll_passed": 1,
            "start_passed": 1,
            "greeting_passed": 1,
        }
        with timed_operation(
            timing_logger,
            "api.splice_synthesize.greeting",
            support_id=support_id,
            voice_id=voice_id,
            greeting_chars=len(greeting or ""),
            similarity_check=GREETING_SPEAKER_SIMILARITY_CHECK,
        ) as greeting_span:
            if GREETING_SPEAKER_SIMILARITY_CHECK:
                splice_initial_generate_config, splice_retry_generate_config = greeting_splice_generate_configs()
                greeting_wav, sr_greeting, greeting_similarity, greeting_attempts, greeting_similarity_passed, greeting_quality = (
                    generate_voice_with_similarity_retry(
                        text=greeting,
                        voice_prompt=voice_prompt,
                        reference_embedding=prompt_data["ref_spk_embedding"],
                        min_similarity=GREETING_SPEAKER_SIMILARITY_THRESHOLD,
                        max_attempts=GREETING_SPLICE_MAX_ATTEMPTS,
                        initial_generate_config=splice_initial_generate_config,
                        retry_generate_config=splice_retry_generate_config,
                        timing_logger=timing_logger,
                        attempt_operation="api.splice_synthesize.greeting_attempt",
                        timing_fields={
                            "support_id": support_id,
                            "voice_id": voice_id,
                            "greeting_chars": len(greeting or ""),
                        },
                    )
                )
                greeting_span.set(
                    greeting_attempts=greeting_attempts,
                    greeting_similarity=greeting_similarity,
                    greeting_similarity_passed=greeting_similarity_passed,
                    greeting_start_passed=bool(greeting_quality.get("start_passed", 1)),
                    greeting_passed=bool(greeting_quality.get("greeting_passed", greeting_quality.get("start_passed", 1))),
                    greeting_onset_artifact=bool(greeting_quality.get("onset_artifact", 0)),
                    greeting_duration_artifact=bool(greeting_quality.get("duration_artifact", 0)),
                    greeting_ending_artifact=bool(greeting_quality.get("ending_artifact", 0)),
                    greeting_preroll_artifact=bool(greeting_quality.get("preroll_artifact", 0)),
                )
                span.set(
                    greeting_attempts=greeting_attempts,
                    greeting_similarity=greeting_similarity,
                    greeting_similarity_passed=greeting_similarity_passed,
                    greeting_start_passed=bool(greeting_quality.get("start_passed", 1)),
                    greeting_passed=bool(greeting_quality.get("greeting_passed", greeting_quality.get("start_passed", 1))),
                )
                if GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS and not greeting_similarity_passed:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "greeting speaker similarity below threshold: "
                            f"{greeting_similarity:.4f} < {GREETING_SPEAKER_SIMILARITY_THRESHOLD:.4f}"
                        ),
                    )
                if GREETING_ONSET_ARTIFACT_REQUIRE_PASS and not greeting_quality.get(
                    "greeting_passed",
                    greeting_quality.get("start_passed", 1),
                ):
                    raise HTTPException(status_code=422, detail="greeting quality artifact detected in all attempts")
                greeting_wav, sr_greeting, _ = clean_output_audio_for_greeting(greeting, greeting_wav, sr_greeting)
            else:
                greeting_wav, sr_greeting = generate_voice(greeting, voice_prompt)
                greeting_wav, sr_greeting, _ = clean_output_audio_for_greeting(greeting, greeting_wav, sr_greeting)
        body_wav, sr_body, body_hash, body_cache_hit = _load_or_generate_body_wav(
            support_id=support_id,
            voice_id=voice_id,
            body=body,
            prompt_data=prompt_data,
            voice_prompt=voice_prompt,
        )
        if int(sr_greeting) != int(sr_body):
            with timed_operation(
                timing_logger,
                "api.body_cache.regenerate_sample_rate_mismatch",
                support_id=support_id,
                voice_id=voice_id,
                body_hash=body_hash,
            ):
                body_wav, sr_body, body_attempts, body_passed, body_quality, body_trim = generate_body_with_quality_retry(
                    body,
                    voice_prompt,
                )
                if BODY_QUALITY_REQUIRE_PASS and not body_passed:
                    raise HTTPException(status_code=422, detail="body boundary artifact detected in all attempts")
                body_bytes = wav_to_bytes(body_wav, int(sr_body))
                cache_paths = _body_cache_paths(support_id, voice_id, body_hash)
                upload_bytes(cache_paths["audio"], body_bytes, "audio/wav")
                write_json(
                    cache_paths["meta"],
                    {
                        "support_id": support_id,
                        "voice_id": voice_id,
                        "body_hash": body_hash,
                        "body_text": body,
                        "prompt_fingerprint": prompt_digest,
                        "model_size": MODEL_SIZE,
                        "language": LANGUAGE,
                        "generation_config": dict(VOICE_CLONE_GENERATE_CONFIG),
                        "body_quality": body_quality,
                        "body_attempts": body_attempts,
                        "output_trim": body_trim,
                        "created_at": int(time.time()),
                    },
                )
                if int(sr_greeting) != int(sr_body):
                    raise HTTPException(status_code=500, detail="internal error: sample rates mismatch")

        splice_strategy = "content_aware" if content_aware else "simple"
        with timed_operation(
            timing_logger,
            "api.splice_synthesize.merge",
            support_id=support_id,
            voice_id=voice_id,
            body_cache_hit=body_cache_hit,
            splice_strategy=splice_strategy,
        ):
            merged_wav = splice_speech_segments(
                greeting_wav=greeting_wav,
                body_wav=body_wav,
                sample_rate=int(sr_greeting),
                pause_ms=pause_ms,
                crossfade_ms=crossfade_ms,
                content_aware=content_aware,
                target_lufs=target_lufs,
            )
            merged_wav, merged_sr, output_trim = clean_output_audio_for_spliced_phrase(merged_wav, int(sr_greeting))
        span.set(
            body_hash=body_hash,
            body_cache_hit=body_cache_hit,
            greeting_attempts=greeting_attempts,
            greeting_similarity=greeting_similarity,
            greeting_similarity_passed=greeting_similarity_passed,
            greeting_start_passed=bool(greeting_quality.get("start_passed", 1)),
            greeting_passed=bool(greeting_quality.get("greeting_passed", greeting_quality.get("start_passed", 1))),
            splice_strategy=splice_strategy,
        )
        return (
            wav_to_bytes(merged_wav, int(merged_sr)),
            body_hash,
            body_cache_hit,
            splice_strategy,
            greeting_similarity,
            greeting_attempts,
            greeting_similarity_passed,
            greeting_quality,
            output_trim,
        )


def _check_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    if credentials.username != ADMIN_USER or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username


def _fetch_task_worker_health() -> dict:
    url = os.getenv("TASK_WORKER_HEALTH_URL", "http://127.0.0.1:8010/health")
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception:
        return {}


@app.on_event("startup")
def _startup() -> None:
    recovered = db.requeue_running_tasks()
    if recovered:
        logger.warning("Recovered %s stale running local queue task(s) after startup.", recovered)
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
    with timed_operation(
        timing_logger,
        "api.create_profile.enqueue",
        support_id=support_id,
        voice_id=voice_id,
    ):
        if not support_id.strip():
            raise HTTPException(status_code=400, detail="support_id is required")
        if not voice_id.strip():
            raise HTTPException(status_code=400, detail="voice_id is required")
        if not voice_name.strip():
            raise HTTPException(status_code=400, detail="voice_name is required")

        paths = _voice_paths(support_id, voice_id)
        if object_exists(paths["voice_json"]):
            try:
                existing = read_json(paths["voice_json"])
                if existing.get("status") in ("queued", "processing", "running"):
                    return CreateProfileResponse(support_id=support_id, voice_id=voice_id, status=str(existing.get("status")))
            except Exception:
                pass

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
    with timed_operation(
        timing_logger,
        "api.create_phrase.enqueue",
        support_id=req.support_id,
        voice_id=req.voice_id,
        phrase_id=req.phrase_id,
        text_chars=len(req.text or ""),
    ):
        if not req.support_id.strip():
            raise HTTPException(status_code=400, detail="support_id is required")
        if not req.voice_id.strip():
            raise HTTPException(status_code=400, detail="voice_id is required")
        if not req.phrase_id.strip():
            raise HTTPException(status_code=400, detail="phrase_id is required")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text is required")

        paths = _phrase_paths(req.support_id, req.phrase_id)
        if object_exists(paths["phrase_json"]):
            try:
                existing = read_json(paths["phrase_json"])
                if existing.get("status") in ("queued", "processing", "running"):
                    return CreatePhraseResponse(phrase_id=req.phrase_id, status=str(existing.get("status")))
            except Exception:
                pass
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


@app.post("/phrases/splice-test")
def splice_test_phrase(req: SpliceTestRequest) -> Response:
    with timed_operation(
        timing_logger,
        "api.splice_test.request",
        support_id=req.support_id,
        voice_id=req.voice_id,
        mode=req.mode,
        greeting_chars=len(req.greeting or ""),
        body_chars=len(req.body or ""),
    ):
        if not req.support_id.strip():
            raise HTTPException(status_code=400, detail="support_id is required")
        if not req.voice_id.strip():
            raise HTTPException(status_code=400, detail="voice_id is required")
        if not req.greeting.strip():
            raise HTTPException(status_code=400, detail="greeting is required")
        if not req.body.strip():
            raise HTTPException(status_code=400, detail="body is required")
        if req.pause_ms < 0:
            raise HTTPException(status_code=400, detail="pause_ms must be >= 0")
        if req.crossfade_ms < 0:
            raise HTTPException(status_code=400, detail="crossfade_ms must be >= 0")
        if req.mode not in ("wav_splice", "latent_concat"):
            raise HTTPException(status_code=400, detail="mode must be 'wav_splice' or 'latent_concat'")

        if req.mode == "latent_concat":
            enabled = os.getenv("ENABLE_EXPERIMENTAL_LATENT_CONCAT", "false").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not enabled:
                raise HTTPException(
                    status_code=400,
                    detail="latent_concat is disabled. Set ENABLE_EXPERIMENTAL_LATENT_CONCAT=true to enable experiments.",
                )
            raise HTTPException(status_code=501, detail="latent_concat experimental path is not implemented yet")

        try:
            (
                wav_bytes,
                body_hash,
                body_cache_hit,
                splice_strategy,
                greeting_similarity,
                greeting_attempts,
                greeting_similarity_passed,
                greeting_quality,
                output_trim,
            ) = _synthesize_spliced_phrase(
                support_id=req.support_id,
                voice_id=req.voice_id,
                greeting=req.greeting,
                body=req.body,
                pause_ms=req.pause_ms,
                crossfade_ms=req.crossfade_ms,
                content_aware=req.content_aware,
                target_lufs=req.target_lufs,
            )
        except Exception as exc:
            _crash_process_on_fatal_cuda(exc, "splice-test")
            raise

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="{req.voice_id}_splice_test.wav"',
                "X-Body-Cache": "hit" if body_cache_hit else "miss",
                "X-Body-Cache-Key": body_hash,
                "X-Mode": req.mode,
                "X-Splice-Strategy": splice_strategy,
                "X-Target-Lufs": f"{req.target_lufs:.2f}",
                "X-Greeting-Similarity": "" if greeting_similarity is None else f"{greeting_similarity:.4f}",
                "X-Greeting-Attempts": str(greeting_attempts),
                "X-Greeting-Similarity-Passed": "" if greeting_similarity_passed is None else str(bool(greeting_similarity_passed)).lower(),
                "X-Greeting-Onset-Checked": str(bool(greeting_quality.get("onset_checked", 0))).lower(),
                "X-Greeting-Onset-Passed": str(bool(greeting_quality.get("onset_passed", 1))).lower(),
                "X-Greeting-Onset-Artifact": str(bool(greeting_quality.get("onset_artifact", 0))).lower(),
                "X-Greeting-Ending-Checked": str(bool(greeting_quality.get("ending_checked", 0))).lower(),
                "X-Greeting-Ending-Passed": str(bool(greeting_quality.get("ending_passed", 1))).lower(),
                "X-Greeting-Ending-Artifact": str(bool(greeting_quality.get("ending_artifact", 0))).lower(),
                "X-Greeting-Preroll-Checked": str(bool(greeting_quality.get("preroll_checked", 0))).lower(),
                "X-Greeting-Preroll-Passed": str(bool(greeting_quality.get("preroll_passed", 1))).lower(),
                "X-Greeting-Preroll-Artifact": str(bool(greeting_quality.get("preroll_artifact", 0))).lower(),
                "X-Greeting-Start-Passed": str(bool(greeting_quality.get("start_passed", 1))).lower(),
                "X-Greeting-Passed": str(bool(greeting_quality.get("greeting_passed", greeting_quality.get("start_passed", 1)))).lower(),
                "X-Output-Trim-Leading-Ms": str(output_trim["leading_ms"]),
                "X-Output-Trim-Trailing-Ms": str(output_trim["trailing_ms"]),
                "X-Output-Trim-Applied": str(bool(output_trim["trimmed"])).lower(),
            },
        )


@app.post("/phrases/splice-prod", response_model=CreatePhraseResponse)
def create_phrase_splice_prod(req: CreateSplicePhraseRequest) -> CreatePhraseResponse:
    with timed_operation(
        timing_logger,
        "api.splice_prod.request",
        support_id=req.support_id,
        voice_id=req.voice_id,
        phrase_id=req.phrase_id,
        greeting_chars=len(req.greeting or ""),
        body_chars=len(req.body or ""),
    ):
        if not req.support_id.strip():
            raise HTTPException(status_code=400, detail="support_id is required")
        if not req.voice_id.strip():
            raise HTTPException(status_code=400, detail="voice_id is required")
        if not req.phrase_id.strip():
            raise HTTPException(status_code=400, detail="phrase_id is required")
        if not req.greeting.strip():
            raise HTTPException(status_code=400, detail="greeting is required")
        if not req.body.strip():
            raise HTTPException(status_code=400, detail="body is required")
        if req.pause_ms < 0:
            raise HTTPException(status_code=400, detail="pause_ms must be >= 0")
        if req.crossfade_ms < 0:
            raise HTTPException(status_code=400, detail="crossfade_ms must be >= 0")

        paths = _phrase_paths(req.support_id, req.phrase_id)
        write_json(
            paths["phrase_json"],
            {
                "support_id": req.support_id,
                "voice_id": req.voice_id,
                "phrase_id": req.phrase_id,
                "text": f"{req.greeting} {req.body}".strip(),
                "status": "processing",
                "result_key": paths["audio"],
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            },
        )

        try:
            (
                wav_bytes,
                _,
                body_cache_hit,
                splice_strategy,
                greeting_similarity,
                greeting_attempts,
                greeting_similarity_passed,
                greeting_quality,
                output_trim,
            ) = _synthesize_spliced_phrase(
                support_id=req.support_id,
                voice_id=req.voice_id,
                greeting=req.greeting,
                body=req.body,
                pause_ms=req.pause_ms,
                crossfade_ms=req.crossfade_ms,
                content_aware=req.content_aware,
                target_lufs=req.target_lufs,
            )
            with timed_operation(
                timing_logger,
                "api.splice_prod.persist",
                support_id=req.support_id,
                voice_id=req.voice_id,
                phrase_id=req.phrase_id,
                body_cache_hit=body_cache_hit,
            ):
                upload_bytes(paths["audio"], wav_bytes, "audio/wav")
                public_url = create_presigned_url(paths["audio"], expires_seconds=60 * 24 * 3600)
                write_json(
                    paths["phrase_json"],
                    {
                        "support_id": req.support_id,
                        "voice_id": req.voice_id,
                        "phrase_id": req.phrase_id,
                        "text": f"{req.greeting} {req.body}".strip(),
                        "status": "done",
                        "result_key": paths["audio"],
                        "public_url": public_url,
                        "splice_strategy": splice_strategy,
                        "body_cache": "hit" if body_cache_hit else "miss",
                        "greeting_similarity": greeting_similarity,
                        "greeting_attempts": greeting_attempts,
                        "greeting_similarity_passed": greeting_similarity_passed,
                        "greeting_onset_checked": bool(greeting_quality.get("onset_checked", 0)),
                        "greeting_onset_passed": bool(greeting_quality.get("onset_passed", 1)),
                        "greeting_onset_artifact": bool(greeting_quality.get("onset_artifact", 0)),
                        "greeting_duration_checked": bool(greeting_quality.get("duration_checked", 0)),
                        "greeting_duration_passed": bool(greeting_quality.get("duration_passed", 1)),
                        "greeting_duration_artifact": bool(greeting_quality.get("duration_artifact", 0)),
                        "greeting_ending_checked": bool(greeting_quality.get("ending_checked", 0)),
                        "greeting_ending_passed": bool(greeting_quality.get("ending_passed", 1)),
                        "greeting_ending_artifact": bool(greeting_quality.get("ending_artifact", 0)),
                        "greeting_preroll_checked": bool(greeting_quality.get("preroll_checked", 0)),
                        "greeting_preroll_passed": bool(greeting_quality.get("preroll_passed", 1)),
                        "greeting_preroll_artifact": bool(greeting_quality.get("preroll_artifact", 0)),
                        "greeting_start_passed": bool(greeting_quality.get("start_passed", 1)),
                        "greeting_passed": bool(greeting_quality.get("greeting_passed", greeting_quality.get("start_passed", 1))),
                        "output_trim": output_trim,
                        "updated_at": int(time.time()),
                    },
                )
        except Exception as exc:
            write_json(
                paths["phrase_json"],
                {
                    "support_id": req.support_id,
                    "voice_id": req.voice_id,
                    "phrase_id": req.phrase_id,
                    "text": f"{req.greeting} {req.body}".strip(),
                    "status": "failed",
                    "result_key": paths["audio"],
                    "error": str(exc),
                    "updated_at": int(time.time()),
                },
            )
            _crash_process_on_fatal_cuda(exc, "splice-prod")
            raise

        return CreatePhraseResponse(phrase_id=req.phrase_id, status="done")


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(_: str = Depends(_check_admin)) -> HTMLResponse:
    stats = db.stats()
    stats_by_type = db.stats_by_type()
    recent = db.list_recent(50)
    worker_health = _fetch_task_worker_health()
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
        <h2>Task Worker Phrase Metrics</h2>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>phrase_grouped</td><td>{worker_health.get('phrase_grouped', 0)}</td></tr>
          <tr><td>phrase_splice_path</td><td>{worker_health.get('phrase_splice_path', 0)}</td></tr>
          <tr><td>phrase_fallback_full</td><td>{worker_health.get('phrase_fallback_full', 0)}</td></tr>
          <tr><td>splice_failures</td><td>{worker_health.get('splice_failures', 0)}</td></tr>
        </table>
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
