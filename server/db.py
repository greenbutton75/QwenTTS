import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TaskRecord:
    task_id: int
    task_type: str
    payload: Dict[str, Any]
    status: str
    attempts: int
    created_at: float
    updated_at: float
    next_run_at: float


class TaskDB:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    next_run_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, next_run_at)")
            conn.commit()
        finally:
            conn.close()

    def enqueue(self, task_type: str, payload: Dict[str, Any]) -> int:
        now = time.time()
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO tasks (task_type, payload_json, status, attempts, created_at, updated_at, next_run_at)
                VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (task_type, json.dumps(payload), now, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_next_due(self) -> Optional[TaskRecord]:
        now = time.time()
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'queued' AND next_run_at <= ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if not row:
                return None
            return TaskRecord(
                task_id=row["id"],
                task_type=row["task_type"],
                payload=json.loads(row["payload_json"]),
                status=row["status"],
                attempts=row["attempts"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                next_run_at=row["next_run_at"],
            )
        finally:
            conn.close()

    def mark_running(self, task_id: int) -> None:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET status='running', updated_at=? WHERE id=?",
                (now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_done(self, task_id: int) -> None:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET status='done', updated_at=? WHERE id=?",
                (now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, task_id: int) -> None:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET status='failed', updated_at=? WHERE id=?",
                (now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def requeue_with_backoff(self, task_id: int, attempts: int, delay_seconds: int) -> None:
        now = time.time()
        next_run = now + delay_seconds
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE tasks
                SET status='queued', attempts=?, updated_at=?, next_run_at=?
                WHERE id=?
                """,
                (attempts, now, next_run, task_id),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> Dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(1) AS cnt FROM tasks GROUP BY status"
            ).fetchall()
            out = {"queued": 0, "running": 0, "done": 0, "failed": 0}
            for r in rows:
                out[r["status"]] = int(r["cnt"])
            return out
        finally:
            conn.close()

    def list_recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

