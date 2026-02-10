import logging
import time
from typing import Any, Dict

from .config import MAX_WAIT_SECONDS, PHRASE_PAGE_SIZE, PHRASE_POLL_INTERVAL, PROFILE_POLL_INTERVAL, STAGE_PROCESSING
from .health import HealthState
from .qwen_client import create_phrase, create_profile, get_phrase_status, get_profile_status
from .s3_utils import object_exists, read_json
from .task_api import (
    complete_task,
    create_task,
    failed_task,
    list_tasks,
    task_id_from_record,
    task_params_from_record,
    update_progress,
)


logger = logging.getLogger("task_worker")


def _sample_key(support_id: str, voice_id: str) -> str:
    return f"support/{support_id}/voices/{voice_id}/sample.wav"


def _voice_json_key(support_id: str, voice_id: str) -> str:
    return f"support/{support_id}/voices/{voice_id}/voice.json"


def _phrase_json_key(support_id: str, phrase_id: str) -> str:
    return f"support/{support_id}/phrases/{phrase_id}.json"


def _wait_profile(support_id: str, voice_id: str) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < MAX_WAIT_SECONDS:
        data = get_profile_status(support_id, voice_id)
        status = data.get("status")
        if status in ("done", "failed"):
            return data
        time.sleep(PROFILE_POLL_INTERVAL)
    return {"status": "failed", "error": "timeout waiting for profile"}


def _wait_phrase(support_id: str, phrase_id: str) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < MAX_WAIT_SECONDS:
        data = get_phrase_status(support_id, phrase_id)
        status = data.get("status")
        if status in ("done", "failed"):
            return data
        time.sleep(PHRASE_POLL_INTERVAL)
    return {"status": "failed", "error": "timeout waiting for phrase"}


def _profile_ready_in_s3(support_id: str, voice_id: str) -> bool:
    key = _voice_json_key(support_id, voice_id)
    if not object_exists(key):
        return False
    try:
        data = read_json(key)
        return data.get("status") == "done"
    except Exception:
        return False


def process_create_profiles(state: HealthState) -> None:
    tasks = list_tasks(task_type="QWEN_TTS_CREATE_PROFILE", statuses=["NEW"])
    state.mark_profile_poll(len(tasks))
    if not tasks:
        return

    for rec in tasks:
        task_id = task_id_from_record(rec)
        params = task_params_from_record(rec)
        support_id = params.get("support_id")
        voice_id = params.get("voice_id")
        voice_name = params.get("voice_name")
        ref_text = params.get("ref_text")
        xvector_only = bool(params.get("xvector_only", False))

        if not support_id or not voice_id or not voice_name:
            failed_task(task_id, error="missing support_id/voice_id/voice_name")
            state.inc_profile_failed()
            continue

        sample_key = _sample_key(support_id, voice_id)
        if not object_exists(sample_key):
            failed_task(task_id, error=f"sample not found: {sample_key}")
            state.inc_profile_failed()
            continue

        try:
            create_profile(support_id, voice_id, voice_name, ref_text, xvector_only)
            update_progress(task_id, progress=10, stage=STAGE_PROCESSING, data="profile submitted")
            result = _wait_profile(support_id, voice_id)
            if result.get("status") == "done":
                complete_task(task_id, data="done")
                state.inc_profile_success()
                create_task(
                    "QWEN_TTS_READY_PROFILE",
                    None,
                    {"support_id": support_id, "voice_id": voice_id},
                )
            else:
                failed_task(task_id, error=result.get("error", "profile failed"))
                state.inc_profile_failed()
        except Exception as exc:
            failed_task(task_id, error=str(exc))
            state.inc_profile_failed()
            state.set_error(str(exc))


def process_phrases_batch(state: HealthState) -> None:
    tasks = list_tasks(
        task_type="QWEN_TTS_PHRASE",
        statuses=["NEW"],
        page_size=PHRASE_PAGE_SIZE,
    )
    state.mark_phrase_poll(len(tasks))
    if not tasks:
        return

    for rec in tasks:
        task_id = task_id_from_record(rec)
        params = task_params_from_record(rec)
        support_id = params.get("support_id")
        voice_id = params.get("voice_id")
        phrase_id = params.get("phrase_id")
        text = params.get("text")

        if not support_id or not voice_id or not phrase_id or not text:
            failed_task(task_id, error="missing support_id/voice_id/phrase_id/text")
            state.inc_phrase_failed()
            continue

        if not _profile_ready_in_s3(support_id, voice_id):
            # profile not ready yet -> skip for now
            continue

        try:
            create_phrase(support_id, voice_id, phrase_id, text)
            update_progress(task_id, progress=10, stage=STAGE_PROCESSING, data="phrase submitted")
            result = _wait_phrase(support_id, phrase_id)
            if result.get("status") == "done":
                public_url = result.get("public_url", "")
                complete_task(task_id, data=public_url)
                state.inc_phrase_success()
                create_task(
                    "QWEN_TTS_READY_PHRASE",
                    None,
                    {"support_id": support_id, "voice_id": voice_id, "phrase_id": phrase_id},
                )
            else:
                failed_task(task_id, error=result.get("error", "phrase failed"))
                state.inc_phrase_failed()
        except Exception as exc:
            failed_task(task_id, error=str(exc))
            state.inc_phrase_failed()
            state.set_error(str(exc))


def run_loop(state: HealthState) -> None:
    while True:
        process_create_profiles(state)
        process_phrases_batch(state)
        time.sleep(15)
