import os
import time
import traceback
from typing import Any, Dict
import re

from .cache_utils import prompt_fingerprint
from .config import LOG_BACKUPS, LOG_DIR, LOG_MAX_BYTES
from .config import (
    GREETING_ONSET_ARTIFACT_REQUIRE_PASS,
    GREETING_FULL_PHRASE_MAX_ATTEMPTS,
    GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS,
    GREETING_SPEAKER_SIMILARITY_THRESHOLD,
    MAX_RETRIES,
    RETRY_BASE_SECONDS,
    S3_PREFIX,
)
from .db import TaskDB
from qwen_tts import VoiceClonePromptItem

from .s3_store import (
    create_presigned_url,
    delete_prefix,
    download_bytes,
    download_torch,
    upload_file,
    upload_torch,
    write_json,
)
from .tts import (
    bytes_to_wav_file,
    clean_output_audio,
    clean_output_audio_for_greeting,
    clean_output_audio_preserve_start,
    clean_reference_audio,
    create_voice_prompt,
    generate_voice,
    generate_voice_with_similarity_retry,
    is_fatal_cuda_error,
    load_audio,
    write_wav_temp,
    greeting_splice_generate_configs,
)
from timing_utils import setup_timing_logger, timed_operation


GREETING_SPLIT_RE = re.compile(
    r"^\s*(hi|hello)(?:\s+[!,.]?\s*|[!,.]\s*)([a-zA-Z][a-zA-Z'\-]*)\s*([!,.]?)\s*",
    re.IGNORECASE,
)


def _split_greeting_body(text: str):
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    match = GREETING_SPLIT_RE.match(raw)
    if not match:
        return None
    greeting = raw[: match.end()].strip()
    body = raw[match.end() :].strip()
    if not greeting or not body:
        return None
    return greeting, body


def _voice_paths(support_id: str, voice_id: str) -> Dict[str, str]:
    base = f"{S3_PREFIX}/{support_id}/voices/{voice_id}"
    return {
        "reference": f"{base}/reference.wav",
        "voice_json": f"{base}/voice.json",
        "prompt": f"{base}/prompt.pt",
        "sample": f"{base}/sample.wav",
        "splice_cache_prefix": f"{base}/splice_cache/",
    }


def _phrase_paths(support_id: str, phrase_id: str) -> Dict[str, str]:
    base = f"{S3_PREFIX}/{support_id}/phrases/{phrase_id}"
    return {
        "audio": f"{base}.wav",
        "phrase_json": f"{base}.json",
    }


class Worker:
    def __init__(self, db: TaskDB, logger) -> None:
        self.db = db
        self.logger = logger
        self.timing_logger = setup_timing_logger(
            logger_name="qwentts.api_worker.timing",
            log_dir=LOG_DIR,
            filename="server_timing.log",
            max_bytes=LOG_MAX_BYTES,
            backups=LOG_BACKUPS,
        )
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        self.logger.info("Worker started.")
        while not self._stop:
            task = self.db.get_next_due()
            if not task:
                time.sleep(1)
                continue

            self.db.mark_running(task.task_id)
            try:
                if task.task_type == "profile":
                    self._handle_profile(task.payload)
                elif task.task_type == "phrase":
                    self._handle_phrase(task.payload)
                else:
                    raise RuntimeError(f"Unknown task type: {task.task_type}")
                self.db.mark_done(task.task_id)
            except Exception as exc:
                attempts = task.attempts + 1
                self.logger.error("Task failed: %s", exc)
                self.logger.debug("Traceback:\n%s", traceback.format_exc())

                if is_fatal_cuda_error(exc):
                    try:
                        self.db.requeue_with_backoff(task.task_id, attempts, 5)
                    finally:
                        self.logger.critical(
                            "Fatal CUDA error in API worker task_id=%s type=%s. Exiting process for supervisor restart.",
                            task.task_id,
                            task.task_type,
                        )
                        os._exit(86)

                self._update_status_on_failure(task.task_type, task.payload, str(exc), attempts)

                if attempts >= MAX_RETRIES:
                    self.db.mark_failed(task.task_id)
                else:
                    delay = RETRY_BASE_SECONDS * (2 ** (attempts - 1))
                    self.db.requeue_with_backoff(task.task_id, attempts, delay)

    def _handle_profile(self, payload: Dict[str, Any]) -> None:
        support_id = payload["support_id"]
        voice_id = payload["voice_id"]
        voice_name = payload.get("voice_name") or voice_id
        ref_text = payload.get("ref_text") or None
        x_vector_only = bool(payload.get("x_vector_only", False))
        s3_sample_key = payload["s3_sample_key"]

        with timed_operation(
            self.timing_logger,
            "api.worker.profile.total",
            support_id=support_id,
            voice_id=voice_id,
        ):
            paths = _voice_paths(support_id, voice_id)

            with timed_operation(self.timing_logger, "api.worker.profile.download_sample", support_id=support_id, voice_id=voice_id):
                sample_bytes = download_bytes(s3_sample_key)
                sample_path = bytes_to_wav_file(sample_bytes, suffix=".wav")
                wav, sr = load_audio(sample_path)
            with timed_operation(self.timing_logger, "api.worker.profile.prepare_prompt", support_id=support_id, voice_id=voice_id):
                cleaned_wav, cleaned_sr, reference_trim = clean_reference_audio(wav, sr)
                prompt_item = create_voice_prompt((cleaned_wav, cleaned_sr), ref_text, x_vector_only)
                prompt_payload = {
                    "ref_code": prompt_item.ref_code,
                    "ref_spk_embedding": prompt_item.ref_spk_embedding,
                    "x_vector_only_mode": prompt_item.x_vector_only_mode,
                    "icl_mode": prompt_item.icl_mode,
                    "ref_text": prompt_item.ref_text,
                }
                prompt_digest = prompt_fingerprint(prompt_payload)
                reference_path = write_wav_temp(cleaned_wav, cleaned_sr)

            with timed_operation(self.timing_logger, "api.worker.profile.persist", support_id=support_id, voice_id=voice_id):
                delete_prefix(paths["splice_cache_prefix"])
                upload_file(paths["reference"], reference_path, "audio/wav")
                upload_torch(paths["prompt"], prompt_payload)

                voice_json = {
                    "support_id": support_id,
                    "voice_id": voice_id,
                    "voice_name": voice_name,
                    "status": "done",
                    "ref_text": ref_text,
                    "x_vector_only": x_vector_only,
                    "reference_key": paths["reference"],
                    "prompt_key": paths["prompt"],
                    "prompt_fingerprint": prompt_digest,
                    "reference_trim": reference_trim,
                    "updated_at": int(time.time()),
                }
                write_json(paths["voice_json"], voice_json)

            for path in (sample_path, reference_path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def _handle_phrase(self, payload: Dict[str, Any]) -> None:
        support_id = payload["support_id"]
        voice_id = payload["voice_id"]
        phrase_id = payload["phrase_id"]
        text = payload["text"]

        voice_paths = _voice_paths(support_id, voice_id)
        phrase_paths = _phrase_paths(support_id, phrase_id)

        with timed_operation(
            self.timing_logger,
            "api.worker.phrase.total",
            support_id=support_id,
            voice_id=voice_id,
            phrase_id=phrase_id,
            text_chars=len(text or ""),
        ):
            with timed_operation(self.timing_logger, "api.worker.phrase.load_prompt", support_id=support_id, voice_id=voice_id, phrase_id=phrase_id):
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
            with timed_operation(self.timing_logger, "api.worker.phrase.generate", support_id=support_id, voice_id=voice_id, phrase_id=phrase_id) as phrase_span:
                split = _split_greeting_body(text)
                if split is not None:
                    greeting_text, _ = split
                    probe_initial_generate_config, probe_retry_generate_config = greeting_splice_generate_configs()
                    (
                        _probe_wav,
                        _probe_sr,
                        greeting_similarity,
                        greeting_attempts,
                        greeting_similarity_passed,
                        greeting_quality,
                    ) = generate_voice_with_similarity_retry(
                        text=greeting_text,
                        voice_prompt=voice_prompt,
                        reference_embedding=prompt_data["ref_spk_embedding"],
                        min_similarity=GREETING_SPEAKER_SIMILARITY_THRESHOLD,
                        max_attempts=GREETING_FULL_PHRASE_MAX_ATTEMPTS,
                        initial_generate_config=probe_initial_generate_config,
                        retry_generate_config=probe_retry_generate_config,
                        timing_logger=self.timing_logger,
                        attempt_operation="api.worker.phrase.greeting_attempt",
                        timing_fields={
                            "support_id": support_id,
                            "voice_id": voice_id,
                            "phrase_id": phrase_id,
                            "probe_mode": 1,
                            "text_chars": len(text or ""),
                            "greeting_chars": len(greeting_text or ""),
                        },
                    )
                    phrase_span.set(
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
                    if GREETING_SPEAKER_SIMILARITY_REQUIRE_PASS and not greeting_similarity_passed:
                        raise RuntimeError(
                            "phrase greeting speaker similarity below threshold: "
                            f"{greeting_similarity:.4f} < {GREETING_SPEAKER_SIMILARITY_THRESHOLD:.4f}"
                        )
                    if GREETING_ONSET_ARTIFACT_REQUIRE_PASS and not greeting_quality.get(
                        "greeting_passed",
                        greeting_quality.get("start_passed", 1),
                    ):
                        raise RuntimeError("phrase greeting quality artifact detected in all attempts")
                    wav, sr = generate_voice(text, voice_prompt)
                    wav, sr, output_trim = clean_output_audio(wav, sr)
                else:
                    wav, sr = generate_voice(text, voice_prompt)
                    wav, sr, output_trim = clean_output_audio(wav, sr)

            with timed_operation(self.timing_logger, "api.worker.phrase.persist", support_id=support_id, voice_id=voice_id, phrase_id=phrase_id):
                tmp_path = write_wav_temp(wav, sr)
                upload_file(phrase_paths["audio"], tmp_path, "audio/wav")
                public_url = create_presigned_url(phrase_paths["audio"], expires_seconds=60 * 24 * 3600)

                phrase_json = {
                    "support_id": support_id,
                    "voice_id": voice_id,
                    "phrase_id": phrase_id,
                    "text": text,
                    "status": "done",
                    "result_key": phrase_paths["audio"],
                    "public_url": public_url,
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
                }
                write_json(phrase_paths["phrase_json"], phrase_json)

            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _update_status_on_failure(
        self, task_type: str, payload: Dict[str, Any], error: str, attempts: int
    ) -> None:
        now = int(time.time())
        if task_type == "profile":
            paths = _voice_paths(payload["support_id"], payload["voice_id"])
            voice_json = {
                "support_id": payload["support_id"],
                "voice_id": payload["voice_id"],
                "voice_name": payload.get("voice_name") or payload["voice_id"],
                "status": "failed",
                "ref_text": payload.get("ref_text"),
                "x_vector_only": bool(payload.get("x_vector_only", False)),
                "error": error,
                "attempts": attempts,
                "updated_at": now,
            }
            write_json(paths["voice_json"], voice_json)
            return

        if task_type == "phrase":
            paths = _phrase_paths(payload["support_id"], payload["phrase_id"])
            phrase_json = {
                "support_id": payload["support_id"],
                "voice_id": payload["voice_id"],
                "phrase_id": payload["phrase_id"],
                "text": payload.get("text"),
                "status": "failed",
                "error": error,
                "attempts": attempts,
                "updated_at": now,
            }
            write_json(paths["phrase_json"], phrase_json)
            return
