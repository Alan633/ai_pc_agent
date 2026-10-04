from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    telegram_bot_token: str
    allowed_telegram_user_id: int
    openai_api_key: str
    openai_model: str
    playwright_headless: bool
    max_steps: int
    confirm_timeout_seconds: int
    allowed_roots: tuple[Path, ...]
    allowed_executables: frozenset[str]

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def screenshots_dir(self) -> Path:
        return self.base_dir / "screenshots"

    def prepare_directories(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)


def load_settings(base_dir: Path | None = None, *, require_secrets: bool = True) -> Settings:
    root = (base_dir or Path(__file__).resolve().parent).resolve()
    load_dotenv(root / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    user_id_text = os.getenv("ALLOWED_TELEGRAM_USER_ID", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if require_secrets:
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", token),
                ("ALLOWED_TELEGRAM_USER_ID", user_id_text),
            )
            if not value
        ]
        if missing:
            raise ValueError(f".env 缺少配置：{', '.join(missing)}")
    try:
        allowed_user_id = int(user_id_text or "0")
    except ValueError as exc:
        raise ValueError("ALLOWED_TELEGRAM_USER_ID 必须是整数") from exc

    roots_text = os.getenv("AGENT_ALLOWED_ROOTS", "").strip()
    roots = tuple(
        Path(item.strip()).expanduser().resolve()
        for item in roots_text.split(";")
        if item.strip()
    ) or (root,)
    executables = frozenset(
        item.strip().lower()
        for item in os.getenv(
            "AGENT_ALLOWED_EXECUTABLES", "python,python.exe,pythonw.exe,git,git.exe"
        ).split(",")
        if item.strip()
    )
    max_steps = min(max(int(os.getenv("MAX_STEPS", "15")), 1), 15)
    settings = Settings(
        base_dir=root,
        telegram_bot_token=token,
        allowed_telegram_user_id=allowed_user_id,
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        playwright_headless=os.getenv("PLAYWRIGHT_HEADLESS", "false").strip().lower() in {"1", "true", "yes"},
        max_steps=max_steps,
        confirm_timeout_seconds=max(int(os.getenv("CONFIRM_TIMEOUT_SECONDS", "120")), 10),
        allowed_roots=roots,
        allowed_executables=executables,
    )
    settings.prepare_directories()
    return settings
