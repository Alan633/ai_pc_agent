from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


class SecretRedactionFilter(logging.Filter):
    _TOKEN_PATTERNS = (
        re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
        re.compile(r"\bAA[A-Za-z0-9_-]{30,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    )

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "***")
        for pattern in self._TOKEN_PATTERNS:
            message = pattern.sub("***", message)
        record.msg = message
        record.args = ()
        return True


def configure_logging(log_dir: Path, secrets: list[str]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = SecretRedactionFilter(secrets)
    file_handler = RotatingFileHandler(
        log_dir / "agent.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
