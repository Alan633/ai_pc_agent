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
    game_vision_model: str = "gpt-4.1-mini"
    game_window_title_keyword: str = "欢乐斗地主"
    game_action_interval_seconds: float = 1.5
    game_min_card_confidence: float = 0.72
    game_auto_next_round: bool = False
    game_record_fps: float = 6.0
    game_record_archive_fps: float = 2.0
    game_record_diff_threshold: float = 0.028
    game_record_jpeg_quality: int = 78
    openai_base_url: str = ""
    game_vision_base_url: str = ""
    game_min_event_confidence: float = 0.80

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def screenshots_dir(self) -> Path:
        return self.base_dir / "screenshots"

    @property
    def game_recordings_dir(self) -> Path:
        return self.base_dir / "game_recordings"

    def prepare_directories(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.game_recordings_dir.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


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
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    game_vision_base_url = os.getenv("GAME_VISION_BASE_URL", openai_base_url).strip().rstrip("/")
    game_confidence = min(max(float(os.getenv("GAME_MIN_CARD_CONFIDENCE", "0.72")), 0.0), 1.0)
    event_confidence = min(max(float(os.getenv("GAME_MIN_EVENT_CONFIDENCE", "0.80")), 0.0), 1.0)
    settings = Settings(
        base_dir=root,
        telegram_bot_token=token,
        allowed_telegram_user_id=allowed_user_id,
        openai_api_key=api_key,
        openai_model=openai_model,
        playwright_headless=_env_bool("PLAYWRIGHT_HEADLESS"),
        max_steps=max_steps,
        confirm_timeout_seconds=max(int(os.getenv("CONFIRM_TIMEOUT_SECONDS", "120")), 10),
        allowed_roots=roots,
        allowed_executables=executables,
        game_vision_model=os.getenv("GAME_VISION_MODEL", openai_model).strip(),
        game_window_title_keyword=os.getenv("GAME_WINDOW_TITLE", "欢乐斗地主").strip(),
        game_action_interval_seconds=max(float(os.getenv("GAME_ACTION_INTERVAL_SECONDS", "1.5")), 0.5),
        game_min_card_confidence=game_confidence,
        game_auto_next_round=_env_bool("GAME_AUTO_NEXT_ROUND"),
        game_record_fps=min(max(float(os.getenv("GAME_RECORD_FPS", "6")), 1.0), 12.0),
        game_record_archive_fps=max(float(os.getenv("GAME_RECORD_ARCHIVE_FPS", "2")), 0.2),
        game_record_diff_threshold=min(max(float(os.getenv("GAME_RECORD_DIFF_THRESHOLD", "0.028")), 0.001), 0.30),
        game_record_jpeg_quality=min(max(int(os.getenv("GAME_RECORD_JPEG_QUALITY", "78")), 40), 95),
        openai_base_url=openai_base_url,
        game_vision_base_url=game_vision_base_url,
        game_min_event_confidence=event_confidence,
    )
    settings.prepare_directories()
    return settings
