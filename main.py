from __future__ import annotations

import asyncio
import logging

# 必须在任何会间接导入 PyAutoGUI 的模块之前启用 DPI Awareness，
# 否则高缩放 Windows 上截图像素与点击坐标可能不一致。
from tools.screen import enable_high_dpi_awareness

enable_high_dpi_awareness()

from telegram.ext import Application

from agent.executor import AgentExecutor
from agent.planner import OpenAIPlanner
from agent.security import SecurityPolicy
from agent.session import AgentSession
from config import load_settings
from emergency import EmergencyHotkey
from game_baque.controller import BaqueGameController
from game_baque.event_vision import OpponentEventVision
from game_baque.observer import OpponentObserver
from game_baque.recorder import GameRecorder
from game_baque.vision import BaqueVision
from logging_setup import configure_logging
from telegram_bot.callbacks import ConfirmationStore
from telegram_bot.handlers import TelegramHandlers
from tools.apps import AppCatalog
from tools.factory import build_registry
from tools.process import ProcessManager
from tools.screen import get_screen_size

LOGGER = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    configure_logging(settings.logs_dir, [settings.telegram_bot_token, settings.openai_api_key])
    security = SecurityPolicy(settings)
    session = AgentSession()
    catalog = AppCatalog(settings.base_dir / "config" / "apps.yaml")
    processes = ProcessManager(security, catalog)
    registry = build_registry(settings, security, processes, session)
    planner = OpenAIPlanner(
        settings.openai_api_key,
        settings.openai_model,
        registry.schemas(),
        settings.openai_base_url,
    )
    executor = AgentExecutor(planner, registry, security, session, settings.max_steps)
    confirmations = ConfirmationStore()
    emergency = EmergencyHotkey(session)
    game_recorder = GameRecorder(
        recordings_dir=settings.game_recordings_dir,
        window_title_keyword=settings.game_window_title_keyword,
        fps=settings.game_record_fps,
        archive_fps=settings.game_record_archive_fps,
        change_threshold=settings.game_record_diff_threshold,
        jpeg_quality=settings.game_record_jpeg_quality,
    )

    game: BaqueGameController | None = None
    if settings.openai_api_key or settings.game_vision_base_url:
        game = BaqueGameController(
            vision=BaqueVision(
                settings.openai_api_key,
                settings.game_vision_model,
                settings.game_vision_base_url,
            ),
            screenshots_dir=settings.screenshots_dir,
            session=session,
            window_title_keyword=settings.game_window_title_keyword,
            action_interval_seconds=settings.game_action_interval_seconds,
            min_card_confidence=settings.game_min_card_confidence,
            min_event_confidence=settings.game_min_event_confidence,
            auto_next_round=settings.game_auto_next_round,
        )

    game_observer: OpponentObserver | None = None
    if game is not None:
        game_observer = OpponentObserver(
            vision=OpponentEventVision(
                settings.openai_api_key,
                settings.game_vision_model,
                base_url=settings.game_vision_base_url,
            ),
            output_root=settings.game_recordings_dir,
            on_observation=game.apply_opponent_observation,
            window_title_keyword=settings.game_window_title_keyword,
            fps=settings.game_record_fps,
            min_confidence=settings.game_min_event_confidence,
        )

    async def post_init(application: Application) -> None:
        session.attach_loop(asyncio.get_running_loop())
        emergency.start()
        bot = await application.bot.get_me()
        width, height = get_screen_size()
        masked = str(settings.allowed_telegram_user_id)
        masked = "*" * max(0, len(masked) - 4) + masked[-4:]
        print(
            "\n".join(
                [
                    "================================",
                    "AI PC Agent",
                    "================================",
                    f"Telegram: Connected (@{bot.username})",
                    f"LLM: {'Ready' if settings.openai_api_key or settings.openai_base_url else 'Not configured (commands only)'}",
                    f"Authorized User: {masked}",
                    f"Screen: {width}x{height}",
                    f"点字成金: {'Ready' if game else '需要 OPENAI_API_KEY 或 GAME_VISION_BASE_URL'}",
                    "牌局采集器: Ready",
                    f"对手观察器: {'Ready' if game_observer else 'Unavailable'}",
                    "Emergency Stop: Ctrl+Alt+F12",
                    "Agent: READY",
                    "================================",
                ]
            )
        )

    async def post_shutdown(application: Application) -> None:
        confirmations.cancel_all()
        session.request_stop()
        if game_recorder.running:
            await game_recorder.stop()
        if game_observer is not None and game_observer.running:
            await game_observer.stop()
        emergency.stop()

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    TelegramHandlers(
        settings,
        registry,
        executor,
        session,
        processes,
        confirmations,
        game=game,
        game_recorder=game_recorder,
        game_observer=game_observer,
    ).install(application)
    LOGGER.info("启动 Telegram polling")
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
