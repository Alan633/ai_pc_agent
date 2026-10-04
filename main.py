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
from logging_setup import configure_logging
from plugins.baque import BaquePlugin
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

    # 通用 Agent 核心：模型 -> 工具路由 -> Windows。
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

    # 专项能力全部走插件层。点字成金只是 Agent 的一个插件，不再作为主程序核心。
    baque = BaquePlugin.build(settings, session)

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
                    f"LLM API: {'Chat Completions / local compatible' if settings.openai_base_url else 'OpenAI Responses'}",
                    f"Authorized User: {masked}",
                    f"Screen: {width}x{height}",
                    f"Tools: {len(registry.names())} registered",
                    f"Apps: {', '.join(catalog.canonical_names())}",
                    f"点字成金插件: {'Ready' if baque.game else '需要 OPENAI_API_KEY 或 GAME_VISION_BASE_URL'}",
                    "牌局采集器: Ready",
                    f"对手观察器: {'Ready' if baque.observer else 'Unavailable'}",
                    "Emergency Stop: Ctrl+Alt+F12",
                    "Agent: READY",
                    "================================",
                ]
            )
        )

    async def post_shutdown(application: Application) -> None:
        confirmations.cancel_all()
        session.request_stop()
        await baque.shutdown()
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
        game=baque.game,
        game_recorder=baque.recorder,
        game_observer=baque.observer,
    ).install(application)
    LOGGER.info("启动 Telegram polling")
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
