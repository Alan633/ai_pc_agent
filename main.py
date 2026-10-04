from __future__ import annotations

import asyncio
import logging

from telegram.ext import Application

from agent.executor import AgentExecutor
from agent.planner import OpenAIPlanner
from agent.security import SecurityPolicy
from agent.session import AgentSession
from config import load_settings
from emergency import EmergencyHotkey
from game_baque.controller import BaqueGameController
from game_baque.vision import BaqueVision
from logging_setup import configure_logging
from telegram_bot.callbacks import ConfirmationStore
from telegram_bot.handlers import TelegramHandlers
from tools.apps import AppCatalog
from tools.factory import build_registry
from tools.process import ProcessManager
from tools.screen import enable_high_dpi_awareness, get_screen_size

LOGGER = logging.getLogger(__name__)


def main() -> None:
    enable_high_dpi_awareness()
    settings = load_settings()
    configure_logging(settings.logs_dir, [settings.telegram_bot_token, settings.openai_api_key])
    security = SecurityPolicy(settings)
    session = AgentSession()
    catalog = AppCatalog(settings.base_dir / "config" / "apps.yaml")
    processes = ProcessManager(security, catalog)
    registry = build_registry(settings, security, processes, session)
    planner = OpenAIPlanner(settings.openai_api_key, settings.openai_model, registry.schemas())
    executor = AgentExecutor(planner, registry, security, session, settings.max_steps)
    confirmations = ConfirmationStore()
    emergency = EmergencyHotkey(session)

    game: BaqueGameController | None = None
    if settings.openai_api_key:
        game = BaqueGameController(
            vision=BaqueVision(settings.openai_api_key, settings.game_vision_model),
            screenshots_dir=settings.screenshots_dir,
            session=session,
            window_title_keyword=settings.game_window_title_keyword,
            action_interval_seconds=settings.game_action_interval_seconds,
            min_card_confidence=settings.game_min_card_confidence,
            auto_next_round=settings.game_auto_next_round,
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
                    f"OpenAI: {'Ready' if settings.openai_api_key else 'Not configured (commands only)'}",
                    f"Authorized User: {masked}",
                    f"Screen: {width}x{height}",
                    f"点字成金: {'Ready' if game else '需要 OPENAI_API_KEY'}",
                    "Emergency Stop: Ctrl+Alt+F12",
                    "Agent: READY",
                    "================================",
                ]
            )
        )

    async def post_shutdown(application: Application) -> None:
        confirmations.cancel_all()
        session.request_stop()
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
    ).install(application)
    LOGGER.info("启动 Telegram polling")
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
