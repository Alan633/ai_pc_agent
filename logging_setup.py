from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class StdoutConsoleFilter(logging.Filter):
    """普通日志走 stdout；被标记为 console_red 的 INFO 不在这里重复输出。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR and not bool(
            getattr(record, "console_red", False)
        )


class StderrConsoleFilter(logging.Filter):
    """ERROR/CRITICAL 和显式标红日志走 stderr。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR or bool(
            getattr(record, "console_red", False)
        )


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
    # IDE（包括 PyCharm）通常把 stderr 整体显示成红色。
    # 普通 INFO/WARNING 走 stdout；ERROR/CRITICAL 走 stderr。
    # 特定重要 INFO（如“收到用户命令”）可通过 extra={"console_red": True}
    # 单独走 stderr 显示红色，但文件日志中的级别仍保持 INFO。
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(StdoutConsoleFilter())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.addFilter(StderrConsoleFilter())

    for handler in (file_handler, stdout_handler, stderr_handler):
        handler.setFormatter(formatter)
        handler.addFilter(redactor)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
