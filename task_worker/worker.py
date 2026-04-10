import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ENABLE_PHRASE_SPLICE_GROUPING,
    LOG_BACKUPS,
    LOG_DIR,
    LOG_MAX_BYTES,
    MAX_WAIT_SECONDS,
    PHRASE_PAGE_SIZE,
    PHRASE_POLL_INTERVAL,
    PHRASE_SPLICE_CROSSFADE_MS,
    PHRASE_SPLICE_PAUSE_MS,
    PROFILE_POLL_INTERVAL,
    QWEN_TTS_READY_POLL_INTERVAL,
    QWEN_TTS_READY_TIMEOUT_SECONDS,
    STAGE_PROCESSING,
)
from .health import HealthState
from .qwen_client import (
    create_phrase,
    create_phrase_splice,
    create_profile,
    get_phrase_status,
    get_profile_status,
    health_check,
)
from .s3_utils import object_exists, read_json
from .task_api import (
    complete_task,
    #create_task,
    failed_task,
    list_tasks,
    task_id_from_record,
    task_params_from_record,
    update_progress,
)
from timing_utils import setup_timing_logger, timed_operation


logger = logging.getLogger("task_worker")
timing_logger = setup_timing_logger(
    logger_name="task_worker.timing",
    log_dir=LOG_DIR,
    filename="task_worker_timing.log",
    max_bytes=LOG_MAX_BYTES,
    backups=LOG_BACKUPS,
)
GREETING_RE = re.compile(
    r"^\s*(hi|hello)(?:\s+[!,.]?\s*|[!,.]\s*)([a-zA-Z][a-zA-Z'\-]*)\s*([!,.]?)\s*",
    re.IGNORECASE,
)
_RETRYABLE_QWEN_API_ERROR_MARKERS = (
    "device-side assert triggered",
    "cuda error",
    "cublas",
    "cudnn",
    "illegal memory access",
    "connection refused",
    "connection aborted",
    "connection reset by peer",
    "failed to establish a new connection",
    "remote end closed connection",
    "max retries exceeded",
    "502 server error",
    "503 server error",
    "504 server error",
)


def _sample_key(support_id: str, voice_id: str) -> str:
    return f"support/{support_id}/voices/{voice_id}/sample.wav"


def _voice_json_key(support_id: str, voice_id: str) -> str:
    return f"support/{support_id}/voices/{voice_id}/voice.json"


def _phrase_json_key(support_id: str, phrase_id: str) -> str:
    return f"support/{support_id}/phrases/{phrase_id}.json"


def _wait_profile(support_id: str, voice_id: str) -> Dict[str, Any]:
    with timed_operation(
        timing_logger,
        "task_worker.wait_profile",
        support_id=support_id,
        voice_id=voice_id,
        max_wait_seconds=MAX_WAIT_SECONDS,
        poll_interval_seconds=PROFILE_POLL_INTERVAL,
    ) as span:
        start = time.time()
        polls = 0
        while time.time() - start < MAX_WAIT_SECONDS:
            polls += 1
            data = get_profile_status(support_id, voice_id)
            status = data.get("status")
            span.set(polls=polls, final_status=status)
            if status in ("done", "failed"):
                return data
            time.sleep(PROFILE_POLL_INTERVAL)
        span.set(polls=polls, final_status="failed")
        return {"status": "failed", "error": "timeout waiting for profile"}


def _wait_phrase(support_id: str, phrase_id: str) -> Dict[str, Any]:
    with timed_operation(
        timing_logger,
        "task_worker.wait_phrase",
        support_id=support_id,
        phrase_id=phrase_id,
        max_wait_seconds=MAX_WAIT_SECONDS,
        poll_interval_seconds=PHRASE_POLL_INTERVAL,
    ) as span:
        start = time.time()
        polls = 0
        while time.time() - start < MAX_WAIT_SECONDS:
            polls += 1
            data = get_phrase_status(support_id, phrase_id)
            status = data.get("status")
            span.set(polls=polls, final_status=status)
            if status in ("done", "failed"):
                return data
            time.sleep(PHRASE_POLL_INTERVAL)
        span.set(polls=polls, final_status="failed")
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


def _wait_for_qwen_tts_ready(state: HealthState) -> None:
    with timed_operation(
        timing_logger,
        "task_worker.wait_ready",
        timeout_seconds=QWEN_TTS_READY_TIMEOUT_SECONDS,
        poll_interval_seconds=QWEN_TTS_READY_POLL_INTERVAL,
    ) as span:
        start = time.time()
        attempts = 0
        while True:
            attempts += 1
            try:
                payload = health_check()
                if payload.get("status") == "ok":
                    span.set(attempts=attempts, final_status="ok")
                    if attempts > 1:
                        logger.info("Qwen TTS API became ready after %s attempts.", attempts)
                    return
            except Exception as exc:
                state.set_error(str(exc))
                span.set(attempts=attempts, final_status="waiting", last_error=str(exc))
                elapsed = time.time() - start
                if elapsed >= QWEN_TTS_READY_TIMEOUT_SECONDS:
                    logger.warning(
                        "Qwen TTS API is still unavailable after %.1fs; worker will keep waiting.",
                        elapsed,
                    )
                    start = time.time()
                else:
                    logger.info("Waiting for Qwen TTS API readiness: %s", exc)
            time.sleep(max(1, QWEN_TTS_READY_POLL_INTERVAL))


def _normalize_body_for_grouping(text: str) -> str:
    value = text.strip().lower()
    value = value.replace("—", "-").replace("–", "-")
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _split_greeting_body(text: str) -> Optional[Tuple[str, str]]:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    m = GREETING_RE.match(raw)
    if not m:
        return None
    greeting = raw[: m.end()].strip()
    body = raw[m.end() :].strip()
    if not greeting or not body:
        return None
    return greeting, body


def _use_splice_grouping_for_support(support_id: str) -> bool:
    _ = support_id
    return ENABLE_PHRASE_SPLICE_GROUPING


def _is_retryable_qwen_api_error(exc_or_text: Any) -> bool:
    text = str(exc_or_text or "").lower()
    return any(marker in text for marker in _RETRYABLE_QWEN_API_ERROR_MARKERS)


def _submit_and_wait_full_phrase(
    task_id: str,
    support_id: str,
    voice_id: str,
    phrase_id: str,
    text: str,
    state: HealthState,
) -> Tuple[bool, str]:
    with timed_operation(
        timing_logger,
        "task_worker.full_phrase.total",
        task_id=task_id,
        support_id=support_id,
        voice_id=voice_id,
        phrase_id=phrase_id,
        text_chars=len(text or ""),
    ) as span:
        create_phrase(support_id, voice_id, phrase_id, text)
        update_progress(task_id, progress=10, stage=STAGE_PROCESSING, data="phrase submitted (full)")
        result = _wait_phrase(support_id, phrase_id)
        span.set(final_status=result.get("status"))
        if result.get("status") == "done":
            public_url = result.get("public_url", "")
            complete_task(task_id, data=public_url)
            state.inc_phrase_success()
            return True, ""
        return False, str(result.get("error", "phrase failed"))


def _submit_and_wait_splice_phrase(
    task_id: str,
    support_id: str,
    voice_id: str,
    phrase_id: str,
    greeting: str,
    body: str,
    state: HealthState,
) -> Tuple[bool, str]:
    with timed_operation(
        timing_logger,
        "task_worker.splice_phrase.total",
        task_id=task_id,
        support_id=support_id,
        voice_id=voice_id,
        phrase_id=phrase_id,
        greeting_chars=len(greeting or ""),
        body_chars=len(body or ""),
        pause_ms=PHRASE_SPLICE_PAUSE_MS,
        crossfade_ms=PHRASE_SPLICE_CROSSFADE_MS,
    ) as span:
        create_phrase_splice(
            support_id=support_id,
            voice_id=voice_id,
            phrase_id=phrase_id,
            greeting=greeting,
            body=body,
            pause_ms=PHRASE_SPLICE_PAUSE_MS,
            crossfade_ms=PHRASE_SPLICE_CROSSFADE_MS,
            content_aware=True,
            target_lufs=-16.0,
        )
        update_progress(task_id, progress=10, stage=STAGE_PROCESSING, data="phrase submitted (splice)")
        result = _wait_phrase(support_id, phrase_id)
        span.set(final_status=result.get("status"))
        if result.get("status") == "done":
            public_url = result.get("public_url", "")
            complete_task(task_id, data=public_url)
            state.inc_phrase_success()
            return True, ""
        return False, str(result.get("error", "phrase failed"))


def process_create_profiles(state: HealthState) -> None:
    with timed_operation(timing_logger, "task_worker.process_profiles.batch") as span:
        tasks = list_tasks(task_type="QWEN_TTS_CREATE_PROFILE", statuses=["NEW"], ignore_user_filter=True)
        span.set(listed_tasks=len(tasks))
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
                with timed_operation(
                    timing_logger,
                    "task_worker.process_profiles.task",
                    task_id=task_id,
                    support_id=support_id,
                    voice_id=voice_id,
                ):
                    create_profile(support_id, voice_id, voice_name, ref_text, xvector_only)
                    update_progress(task_id, progress=10, stage=STAGE_PROCESSING, data="profile submitted")
                    result = _wait_profile(support_id, voice_id)
                    if result.get("status") == "done":
                        complete_task(task_id, data="done")
                        state.inc_profile_success()
                    else:
                        failed_task(task_id, error=result.get("error", "profile failed"))
                        state.inc_profile_failed()
            except Exception as exc:
                failed_task(task_id, error=str(exc))
                state.inc_profile_failed()
                state.set_error(str(exc))


def process_phrases_batch(state: HealthState) -> None:
    with timed_operation(
        timing_logger,
        "task_worker.process_phrases.batch",
        page_size=PHRASE_PAGE_SIZE,
        splice_grouping=ENABLE_PHRASE_SPLICE_GROUPING,
    ) as span:
        tasks = list_tasks(
            task_type="QWEN_TTS_PHRASE",
            statuses=["NEW"],
            page_size=PHRASE_PAGE_SIZE,
            ignore_user_filter=True,
        )
        span.set(listed_tasks=len(tasks))
        state.mark_phrase_poll(len(tasks))
        if not tasks:
            return

        parsed: List[Dict[str, Any]] = []
        with timed_operation(
            timing_logger,
            "task_worker.process_phrases.prepare",
            listed_tasks=len(tasks),
        ) as prepare_span:
            profile_ready_checks = 0
            profile_not_ready = 0
            invalid_tasks = 0
            split_candidates = 0
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
                    invalid_tasks += 1
                    continue
                profile_ready_checks += 1
                if not _profile_ready_in_s3(support_id, voice_id):
                    profile_not_ready += 1
                    continue
                split = _split_greeting_body(text) if _use_splice_grouping_for_support(support_id) else None
                if split:
                    split_candidates += 1
                parsed.append(
                    {
                        "task_id": task_id,
                        "support_id": support_id,
                        "voice_id": voice_id,
                        "phrase_id": phrase_id,
                        "text": text,
                        "split": split,
                    }
                )

            groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
            groupable_items = 0
            for item in parsed:
                split = item.get("split")
                if not split:
                    continue
                greeting, body = split
                body_key = _normalize_body_for_grouping(body)
                if not body_key:
                    continue
                groupable_items += 1
                groups[(item["support_id"], item["voice_id"], body_key)].append(item)
            prepare_span.set(
                parsed_tasks=len(parsed),
                invalid_tasks=invalid_tasks,
                profile_ready_checks=profile_ready_checks,
                profile_not_ready=profile_not_ready,
                split_candidates=split_candidates,
                groupable_items=groupable_items,
                grouping_keys=len(groups),
            )

        grouped_task_ids = set()
        splice_group_count = 0
        for group_items in groups.values():
            if len(group_items) < 2:
                continue
            splice_group_count += 1
            state.inc_phrase_grouped(len(group_items))
            for item in group_items:
                grouped_task_ids.add(item["task_id"])
                try:
                    greeting, body = item["split"]
                    ok, err = _submit_and_wait_splice_phrase(
                        task_id=item["task_id"],
                        support_id=item["support_id"],
                        voice_id=item["voice_id"],
                        phrase_id=item["phrase_id"],
                        greeting=greeting,
                        body=body,
                        state=state,
                    )
                    if ok:
                        state.inc_phrase_splice_path()
                    else:
                        if _is_retryable_qwen_api_error(err):
                            state.inc_splice_failure()
                            raise RuntimeError(f"retryable splice failure: {err}")
                        state.inc_splice_failure()
                        state.inc_phrase_fallback_full()
                        full_ok, full_err = _submit_and_wait_full_phrase(
                            task_id=item["task_id"],
                            support_id=item["support_id"],
                            voice_id=item["voice_id"],
                            phrase_id=item["phrase_id"],
                            text=item["text"],
                            state=state,
                        )
                        if not full_ok:
                            if _is_retryable_qwen_api_error(full_err):
                                raise RuntimeError(f"retryable full fallback failure: {full_err}")
                            failed_task(
                                item["task_id"],
                                error=f"splice failed: {err}; full fallback failed: {full_err}",
                            )
                            state.inc_phrase_failed()
                except Exception as exc:
                    state.inc_splice_failure()
                    if _is_retryable_qwen_api_error(exc):
                        logger.warning(
                            "splice path hit retryable Qwen API error for task_id=%s phrase_id=%s: %s",
                            item["task_id"],
                            item["phrase_id"],
                            exc,
                        )
                        raise
                    logger.warning(
                        "splice path failed for task_id=%s phrase_id=%s, fallback to full phrase: %s",
                        item["task_id"],
                        item["phrase_id"],
                        exc,
                    )
                    try:
                        state.inc_phrase_fallback_full()
                        ok, full_err = _submit_and_wait_full_phrase(
                            task_id=item["task_id"],
                            support_id=item["support_id"],
                            voice_id=item["voice_id"],
                            phrase_id=item["phrase_id"],
                            text=item["text"],
                            state=state,
                        )
                        if not ok:
                            if _is_retryable_qwen_api_error(full_err):
                                raise RuntimeError(full_err)
                            failed_task(item["task_id"], error=f"splice exception and full fallback failed: {full_err}")
                            state.inc_phrase_failed()
                    except Exception as full_exc:
                        if _is_retryable_qwen_api_error(full_exc):
                            raise
                        failed_task(item["task_id"], error=str(full_exc))
                        state.inc_phrase_failed()
                        state.set_error(str(full_exc))

        for item in parsed:
            if item["task_id"] in grouped_task_ids:
                continue
            try:
                state.inc_phrase_fallback_full()
                ok, full_err = _submit_and_wait_full_phrase(
                    task_id=item["task_id"],
                    support_id=item["support_id"],
                    voice_id=item["voice_id"],
                    phrase_id=item["phrase_id"],
                    text=item["text"],
                    state=state,
                )
                if not ok:
                    if _is_retryable_qwen_api_error(full_err):
                        raise RuntimeError(full_err)
                    failed_task(item["task_id"], error=f"full phrase failed: {full_err}")
                    state.inc_phrase_failed()
            except Exception as exc:
                if _is_retryable_qwen_api_error(exc):
                    raise
                failed_task(item["task_id"], error=str(exc))
                state.inc_phrase_failed()
                state.set_error(str(exc))

        span.set(parsed_tasks=len(parsed), splice_groups=splice_group_count, grouped_tasks=len(grouped_task_ids))
        if ENABLE_PHRASE_SPLICE_GROUPING:
            logger.info(
                "phrase batch processed: total=%s parsed=%s splice_groups=%s grouped_tasks=%s",
                len(tasks),
                len(parsed),
                splice_group_count,
                len(grouped_task_ids),
            )


def run_loop(state: HealthState) -> None:
    backoff = 5
    _wait_for_qwen_tts_ready(state)
    while True:
        try:
            with timed_operation(timing_logger, "task_worker.run_loop.iteration", backoff_seconds=backoff):
                _wait_for_qwen_tts_ready(state)
                process_create_profiles(state)
                process_phrases_batch(state)
            backoff = 5
            time.sleep(15)
        except Exception as exc:
            logger.exception("worker loop error: %s", exc)
            state.set_error(str(exc))
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
