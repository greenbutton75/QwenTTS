import os
import time
import traceback
from typing import Any, Dict

from .config import MAX_RETRIES, RETRY_BASE_SECONDS, S3_PREFIX
from .db import TaskDB
from qwen_tts import VoiceClonePromptItem

from .s3_store import (
    create_presigned_url,
    download_bytes,
    download_torch,
    upload_file,
    upload_torch,
    write_json,
)
from .tts import bytes_to_wav_file, create_voice_prompt, generate_voice, load_audio, write_wav_temp


def _voice_paths(support_id: str, voice_id: str) -> Dict[str, str]:
    base = f"{S3_PREFIX}/{support_id}/voices/{voice_id}"
    return {
        "reference": f"{base}/reference.wav",
        "voice_json": f"{base}/voice.json",
        "prompt": f"{base}/prompt.pt",
        "sample": f"{base}/sample.wav",
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

        paths = _voice_paths(support_id, voice_id)

        sample_bytes = download_bytes(s3_sample_key)
        sample_path = bytes_to_wav_file(sample_bytes, suffix=".wav")
        wav, sr = load_audio(sample_path)
        prompt_item = create_voice_prompt((wav, sr), ref_text, x_vector_only)

        upload_file(paths["reference"], sample_path, "audio/wav")
        upload_torch(
            paths["prompt"],
            {
                "ref_code": prompt_item.ref_code,
                "ref_spk_embedding": prompt_item.ref_spk_embedding,
                "x_vector_only_mode": prompt_item.x_vector_only_mode,
                "icl_mode": prompt_item.icl_mode,
                "ref_text": prompt_item.ref_text,
            },
        )

        voice_json = {
            "support_id": support_id,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "status": "done",
            "ref_text": ref_text,
            "x_vector_only": x_vector_only,
            "reference_key": paths["reference"],
            "prompt_key": paths["prompt"],
            "updated_at": int(time.time()),
        }
        write_json(paths["voice_json"], voice_json)

        try:
            os.remove(sample_path)
        except Exception:
            pass

    def _handle_phrase(self, payload: Dict[str, Any]) -> None:
        support_id = payload["support_id"]
        voice_id = payload["voice_id"]
        phrase_id = payload["phrase_id"]
        text = payload["text"]

        voice_paths = _voice_paths(support_id, voice_id)
        phrase_paths = _phrase_paths(support_id, phrase_id)

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

        wav, sr = generate_voice(text, voice_prompt)
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
