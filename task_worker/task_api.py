import json
from typing import Any, Dict, List, Optional

import requests

from timing_utils import setup_timing_logger, timed_operation

from .config import BASE_URL, FINGERPRINT, LOG_BACKUPS, LOG_DIR, LOG_MAX_BYTES, SYSTEM_TOKEN, USER_TOKEN


TIMING_LOGGER = setup_timing_logger(
    logger_name="task_worker.task_api.timing",
    log_dir=LOG_DIR,
    filename="task_worker_timing.log",
    max_bytes=LOG_MAX_BYTES,
    backups=LOG_BACKUPS,
)


def _request_json(method: str, url: str, *, operation: str, timeout: int, headers: Dict[str, str], payload: dict) -> dict:
    with timed_operation(
        TIMING_LOGGER,
        operation,
        method=method.upper(),
        url=url,
        timeout_seconds=timeout,
    ) as span:
        request = getattr(requests, method.lower())
        response = request(url, json=payload, headers=headers, timeout=timeout)
        span.set(status_code=response.status_code)
        response.raise_for_status()
        return response.json()


def _request_empty(method: str, url: str, *, operation: str, timeout: int, headers: Dict[str, str], payload: dict) -> None:
    with timed_operation(
        TIMING_LOGGER,
        operation,
        method=method.upper(),
        url=url,
        timeout_seconds=timeout,
    ) as span:
        request = getattr(requests, method.lower())
        response = request(url, json=payload, headers=headers, timeout=timeout)
        span.set(status_code=response.status_code)
        response.raise_for_status()


def create_task(task_type: str, file_id: Optional[str], params: dict) -> str:
    url = f"{BASE_URL}/Tasks/Create"
    headers = {"Token": USER_TOKEN}
    payload = {
        "Type": task_type,
        "FileId": file_id,
        "TaskParameters": json.dumps(params, ensure_ascii=False),
    }
    data = _request_json(
        "post",
        url,
        operation="task_worker.task_api.create_task",
        timeout=30,
        headers=headers,
        payload=payload,
    )
    if "id" not in data:
        raise RuntimeError(f"Bad response: {data}")
    return str(data["id"])


def list_tasks(task_type: str, statuses: List[str], page_size: int = 99999, ignore_user_filter: bool = False):
    url = f"{BASE_URL}/Tasks/List"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {
        "PageSize": page_size,
        "TypesFilter": [task_type],
        "IncludeStatusesFilter": statuses,
        "IgnoreUserFilter": ignore_user_filter,
    }
    data = _request_json(
        "post",
        url,
        operation="task_worker.task_api.list_tasks",
        timeout=30,
        headers=headers,
        payload=payload,
    )
    return data.get("records", [])


def update_progress(task_id: str, progress: int, stage: str = "", data: str = "") -> None:
    url = f"{BASE_URL}/process/ChangeTaskProgress"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {
        "Id": str(task_id),
        "Progress": progress,
        "Stage": stage,
        "Data": data,
    }
    _request_empty(
        "post",
        url,
        operation="task_worker.task_api.update_progress",
        timeout=30,
        headers=headers,
        payload=payload,
    )


def complete_task(task_id: str, data: str = "") -> None:
    url = f"{BASE_URL}/process/CompleteTask"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {"Id": str(task_id), "Data": data}
    _request_empty(
        "post",
        url,
        operation="task_worker.task_api.complete_task",
        timeout=30,
        headers=headers,
        payload=payload,
    )


def failed_task(task_id: str, error: str = "") -> None:
    url = f"{BASE_URL}/process/CompleteTask"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {"Id": str(task_id), "Error": error}
    _request_empty(
        "post",
        url,
        operation="task_worker.task_api.failed_task",
        timeout=30,
        headers=headers,
        payload=payload,
    )


def task_id_from_record(rec: Dict[str, Any]) -> str:
    for k in ("id", "Id", "ID"):
        if k in rec:
            return str(rec[k])
    raise KeyError("task id not found in record")


def task_params_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("taskParameters", "TaskParameters", "task_parameters"):
        if k in rec:
            raw = rec[k]
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    return {}
            if isinstance(raw, dict):
                return raw
    return {}
