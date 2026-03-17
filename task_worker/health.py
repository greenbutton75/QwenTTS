import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict


class HealthState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.last_profile_poll = 0.0
        self.last_phrase_poll = 0.0
        self.profile_seen = 0
        self.profile_success = 0
        self.profile_failed = 0
        self.phrase_seen = 0
        self.phrase_success = 0
        self.phrase_failed = 0
        self.phrase_grouped = 0
        self.phrase_splice_path = 0
        self.phrase_fallback_full = 0
        self.splice_failures = 0
        self.last_error = ""

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "last_profile_poll": self.last_profile_poll,
                "last_phrase_poll": self.last_phrase_poll,
                "profile_seen": self.profile_seen,
                "profile_success": self.profile_success,
                "profile_failed": self.profile_failed,
                "phrase_seen": self.phrase_seen,
                "phrase_success": self.phrase_success,
                "phrase_failed": self.phrase_failed,
                "phrase_grouped": self.phrase_grouped,
                "phrase_splice_path": self.phrase_splice_path,
                "phrase_fallback_full": self.phrase_fallback_full,
                "splice_failures": self.splice_failures,
                "last_error": self.last_error,
            }

    def mark_profile_poll(self, seen: int) -> None:
        with self._lock:
            self.last_profile_poll = time.time()
            self.profile_seen += seen

    def mark_phrase_poll(self, seen: int) -> None:
        with self._lock:
            self.last_phrase_poll = time.time()
            self.phrase_seen += seen

    def inc_profile_success(self) -> None:
        with self._lock:
            self.profile_success += 1

    def inc_profile_failed(self) -> None:
        with self._lock:
            self.profile_failed += 1

    def inc_phrase_success(self) -> None:
        with self._lock:
            self.phrase_success += 1

    def inc_phrase_failed(self) -> None:
        with self._lock:
            self.phrase_failed += 1

    def inc_phrase_grouped(self, count: int = 1) -> None:
        with self._lock:
            self.phrase_grouped += int(count)

    def inc_phrase_splice_path(self, count: int = 1) -> None:
        with self._lock:
            self.phrase_splice_path += int(count)

    def inc_phrase_fallback_full(self, count: int = 1) -> None:
        with self._lock:
            self.phrase_fallback_full += int(count)

    def inc_splice_failure(self, count: int = 1) -> None:
        with self._lock:
            self.splice_failures += int(count)

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.last_error = msg


def run_health_server(state: HealthState, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(state.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # noqa: N802
            return

    server = HTTPServer((host, port), Handler)
    server.serve_forever()
