from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

from config import Settings
from logging_setup import configure_logging


def test_logs_directory_is_uppercase_logs(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        telegram_bot_token="token",
        allowed_telegram_user_id=1,
        openai_api_key="",
        openai_model="model",
        playwright_headless=True,
        max_steps=15,
        confirm_timeout_seconds=120,
        allowed_roots=(tmp_path,),
        allowed_executables=frozenset(),
    )
    assert settings.logs_dir == tmp_path / "LOGS"



def test_console_only_errors_use_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        configure_logging(tmp_path / "LOGS", [])
        logger = logging.getLogger("tests.console_color")
        logger.info("普通信息")
        logger.warning("普通警告")
        logger.info("用户命令", extra={"console_red": True})
        logger.error("重要错误")

        for handler in root.handlers:
            handler.flush()

        assert "普通信息" in stdout.getvalue()
        assert "普通警告" in stdout.getvalue()
        assert "用户命令" not in stdout.getvalue()
        assert "重要错误" not in stdout.getvalue()

        assert "用户命令" in stderr.getvalue()
        assert "重要错误" in stderr.getvalue()
        assert "普通信息" not in stderr.getvalue()
        assert "普通警告" not in stderr.getvalue()

        log_text = (tmp_path / "LOGS" / "agent.log").read_text(encoding="utf-8")
        assert "普通信息" in log_text
        assert "普通警告" in log_text
        assert "用户命令" in log_text
        assert "| INFO |" in log_text
        assert "重要错误" in log_text
    finally:
        for handler in list(root.handlers):
            handler.close()
        root.handlers.clear()
        for handler in old_handlers:
            root.addHandler(handler)
        root.setLevel(old_level)
