from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class MaxLevelFilter(logging.Filter):
    """只放行低于指定级别的日志，避免 ERROR 同时出现在 stdout 和 stderr。"""

    def __init__(self, max_exclusive: int) -> None:
        super().__init__()
        self.max_exclusive = max_exclusive

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_exclusive


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
    # IDE（包括 PyCharm）通常把 stderr 整体显示成红色。logging.StreamHandler()
    # 默认就是 stderr，因此之前 INFO/httpx 等普通日志也全是红色。
    # 现在普通日志走 stdout（白色），只有 ERROR/CRITICAL 走 stderr（红色）。
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)

    for handler in (file_handler, stdout_handler, stderr_handler):
        handler.setFormatter(formatter)
        handler.addFilter(redactor)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
