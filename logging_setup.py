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


_RED = "[91m"
_RESET = "[0m"


def _enable_windows_ansi(stream: object) -> bool:
    """Windows 控制台默认不解析 ANSI 颜色码，需要为该输出句柄打开 VT 模式。"""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(mode.value & enable_vt or kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except (AttributeError, OSError, ValueError):
        return False


class RedConsoleFormatter(logging.Formatter):
    """在真正的控制台窗口（cmd / Windows Terminal）里把 stderr 日志染成红色。

    IDE 运行窗口不是 tty，本来就把 stderr 显示成红色，这里不加颜色码，
    避免出现乱码；文件日志使用另一个 formatter，永远不含颜色码。
    """

    def __init__(self, stream: object, fmt: str, datefmt: str) -> None:
        super().__init__(fmt, datefmt=datefmt)
        isatty = getattr(stream, "isatty", None)
        self.colorize = bool(callable(isatty) and isatty() and _enable_windows_ansi(stream))

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return f"{_RED}{text}{_RESET}" if self.colorize else text


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
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
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

    for handler in (file_handler, stdout_handler):
        handler.setFormatter(formatter)
    # 命令行窗口不会自动把 stderr 显示成红色，这里显式加 ANSI 红色。
    stderr_handler.setFormatter(RedConsoleFormatter(sys.stderr, log_format, date_format))
    for handler in (file_handler, stdout_handler, stderr_handler):
        handler.addFilter(redactor)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
