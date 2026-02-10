import json
from typing import Any, Dict, List, Optional

import requests

from .config import BASE_URL, FINGERPRINT, SYSTEM_TOKEN, USER_TOKEN


def create_task(task_type: str, file_id: Optional[str], params: dict) -> str:
    url = f"{BASE_URL}/Tasks/Create"
    headers = {"Token": USER_TOKEN}
    payload = {
        "Type": task_type,
        "FileId": file_id,
        "TaskParameters": json.dumps(params, ensure_ascii=False),
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "id" not in data:
        raise RuntimeError(f"Bad response: {data}")
    return str(data["id"])


def list_tasks(task_type: str, statuses: List[str], page_size: int = 99999, ignore_user_filter: bool = False):
    url = f"{BASE_URL}/Tasks/List"
    headers = {"Token": USER_TOKEN}
    payload = {
        "PageSize": page_size,
        "TypesFilter": [task_type],
        "IncludeStatusesFilter": statuses,
        "IgnoreUserFilter": ignore_user_filter,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
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
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


def complete_task(task_id: str, data: str = "") -> None:
    url = f"{BASE_URL}/process/CompleteTask"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {"Id": str(task_id), "Data": data}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


def failed_task(task_id: str, error: str = "") -> None:
    url = f"{BASE_URL}/process/CompleteTask"
    headers = {"Token": SYSTEM_TOKEN, "FingerPrint": FINGERPRINT}
    payload = {"Id": str(task_id), "Error": error}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


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
