from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from telegram_bot.callbacks import ConfirmationStore
from telegram_bot.handlers import TelegramHandlers


class FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

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
