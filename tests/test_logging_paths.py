from __future__ import annotations

from pathlib import Path

from config import Settings


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
