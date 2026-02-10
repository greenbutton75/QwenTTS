import json
from typing import List, Optional

import requests

from .config import BASE_URL, USER_TOKEN


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
