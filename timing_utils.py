import json
import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterator, Optional


def setup_timing_logger(
    logger_name: str,
    log_dir: str,
    filename: str,
    max_bytes: int,
    backups: int,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target_path = _resolve_timing_log_path(log_dir, filename)
    if target_path is None:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == target_path:
            return logger

    fmt = logging.Formatter("%(asctime)s %(message)s")
    handler = RotatingFileHandler(
        target_path,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def _resolve_timing_log_path(log_dir: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.abspath(os.path.join(log_dir, filename)),
        os.path.abspath(os.path.join(os.getcwd(), "logs", filename)),
    ]
    for candidate in candidates:
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            return candidate
        except OSError:
            continue
    return None


class TimingSpan:
    def __init__(self, logger: logging.Logger, operation: str, **fields: Any) -> None:
        self.logger = logger
        self.operation = operation
        self.fields: Dict[str, Any] = dict(fields)
        self._start = time.perf_counter()

    def set(self, **fields: Any) -> None:
        self.fields.update(fields)

    def log(self, status: str = "ok", error: Optional[str] = None) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        payload: Dict[str, Any] = {
            "op": self.operation,
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }
        for key, value in self.fields.items():
            if value is not None:
                payload[key] = value
        if error:
            payload["error"] = error
        self.logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


@contextmanager
def timed_operation(logger: logging.Logger, operation: str, **fields: Any) -> Iterator[TimingSpan]:
    span = TimingSpan(logger, operation, **fields)
    try:
        yield span
    except Exception as exc:
        span.log(status="error", error=str(exc))
        raise
    else:
        span.log(status="ok")
