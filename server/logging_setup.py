import logging
import os
from logging.handlers import RotatingFileHandler

from .config import LOG_BACKUPS, LOG_DIR, LOG_MAX_BYTES


def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("qwentts")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "server.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger

