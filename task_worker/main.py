import logging
import os
import threading
from logging.handlers import RotatingFileHandler

from .config import HEALTH_PORT, LOG_BACKUPS, LOG_DIR, LOG_MAX_BYTES
from .health import HealthState, run_health_server
from .worker import run_loop


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "task_worker.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    state = HealthState()
    thread = threading.Thread(target=run_health_server, args=(state, "0.0.0.0", HEALTH_PORT), daemon=True)
    thread.start()
    run_loop(state)


if __name__ == "__main__":
    main()
