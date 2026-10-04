from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from telegram_bot.callbacks import ConfirmationStore
from telegram_bot.handlers import TelegramHandlers


class FakeMessage:
    def __init__(self, text: str | None = None) -> None:
        self.replies: list[str] = []
        self.text = text

    async def reply_text(self, text: str, **kwargs: object) -> None:
        self.replies.append(text)


@pytest.mark.asyncio
async def test_unauthorized_user_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        telegram_bot_token="token",
        allowed_telegram_user_id=42,
        openai_api_key="key",
        openai_model="model",
        playwright_headless=True,
        max_steps=20,
        confirm_timeout_seconds=120,
        allowed_roots=(tmp_path,),
        allowed_executables=frozenset(),
    )
    message = FakeMessage()
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=99),
        effective_message=message,
        callback_query=None,
    )
    handlers = TelegramHandlers(
        settings=settings,
        registry=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
        processes=None,  # type: ignore[arg-type]
        confirmations=ConfirmationStore(),
    )
    assert await handlers._authorized(update) is False
    assert message.replies and "未授权" in message.replies[0]



@pytest.mark.asyncio
async def test_every_text_command_is_logged(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        telegram_bot_token="token",
        allowed_telegram_user_id=42,
        openai_api_key="key",
        openai_model="model",
        playwright_headless=True,
        max_steps=20,
        confirm_timeout_seconds=120,
        allowed_roots=(tmp_path,),
        allowed_executables=frozenset(),
    )
    handlers = TelegramHandlers(
        settings=settings,
        registry=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
        processes=None,  # type: ignore[arg-type]
        confirmations=ConfirmationStore(),
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=42),
        effective_chat=SimpleNamespace(id=1001),
        effective_message=FakeMessage("打开微信\n然后切到前台"),
    )

    with caplog.at_level(logging.INFO, logger="telegram_bot.handlers"):
        await handlers.log_incoming_command(update, SimpleNamespace())

    assert any(
        "收到用户命令" in record.message
        and "user_id=42" in record.message
        and "chat_id=1001" in record.message
        and r"打开微信\n然后切到前台" in record.message
        for record in caplog.records
    )
