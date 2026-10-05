from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
from ctypes import wintypes


def _relaunch_as_admin_if_needed() -> None:
    """以管理员身份运行本程序。

    游戏若以管理员权限运行，非管理员进程发出的模拟点击会被 Windows（UIPI）
    拦截。非管理员启动时弹出 UAC，用同一个 Python 以管理员身份重新运行本脚本，
    当前进程退出。设置环境变量 AGENT_NO_ELEVATE=1 可跳过（例如 PyCharm 调试）；
    用户取消 UAC 时以普通权限继续运行。
    """
    if os.name != "nt" or os.environ.get("AGENT_NO_ELEVATE") == "1":
        return
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
    except OSError:
        return
    script = os.path.abspath(sys.argv[0])
    # 提权后的新进程不会继承当前进程的环境变量（如 start.bat 设置的
    # PYTHONUTF8），用 -X utf8 保证中文日志输出正常。
    params = subprocess.list2cmdline(["-X", "utf8", script, *sys.argv[1:]])
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        params,
        os.path.dirname(script),
        1,
    )
    if result > 32:
        sys.exit(0)
    print("[WARN] 未获得管理员权限（UAC 已取消），以普通权限继续运行；若游戏以管理员身份运行，模拟点击可能无效。")


if __name__ == "__main__":
    # 必须在启动 Telegram、热键等任何资源之前完成，避免普通权限进程先占用。
    _relaunch_as_admin_if_needed()

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
_AGENT_MUTEX: int | None = None


def acquire_single_instance() -> None:
    """阻止同一 Windows 会话重复启动 Telegram polling。"""
    global _AGENT_MUTEX
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    create_mutex.restype = wintypes.HANDLE
    ctypes.set_last_error(0)
    handle = create_mutex(None, False, "Local\\AIPCAgentTelegramPolling")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        raise RuntimeError("AI PC Agent 已在运行，请不要重复启动 start.bat")
    _AGENT_MUTEX = int(handle)


def disable_console_quick_edit() -> None:
    """防止误点 CMD 进入“选择”模式后暂停整个 Telegram Agent。"""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        stdin_handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint()
        if stdin_handle in {0, -1} or not kernel32.GetConsoleMode(
            stdin_handle,
            ctypes.byref(mode),
        ):
            return
        enable_extended_flags = 0x0080
        enable_quick_edit_mode = 0x0040
        new_mode = (mode.value | enable_extended_flags) & ~enable_quick_edit_mode
        kernel32.SetConsoleMode(stdin_handle, new_mode)
    except (AttributeError, OSError):
        # 例如没有附加控制台时无需处理，不能因此阻止 Agent 启动。
        LOGGER.debug("无法关闭控制台快速编辑模式", exc_info=True)


def main() -> None:
    try:
        acquire_single_instance()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return
    disable_console_quick_edit()
    settings = load_settings()
    configure_logging(settings.logs_dir, [settings.telegram_bot_token, settings.openai_api_key])

    # 通用 Agent 核心：模型 -> 工具路由 -> Windows。
    security = SecurityPolicy(settings)
    session = AgentSession()
    catalog = AppCatalog(settings.base_dir / "config" / "apps.yaml")
    processes = ProcessManager(security, catalog)

    # 专项能力全部走插件层。必须在 Planner 读取 schemas() 前注册，
    # 这样本地 Qwen 才能直接调用“观察八雀/自动玩/验收报告”等能力。
    baque = BaquePlugin.build(settings, session)
    registry = build_registry(settings, security, processes, session)
    baque.register_tools(registry)

    planner = OpenAIPlanner(
        settings.openai_api_key,
        settings.openai_model,
        registry.schemas(),
        settings.openai_base_url,
    )
    executor = AgentExecutor(planner, registry, security, session, settings.max_steps)

    confirmations = ConfirmationStore()
    emergency = EmergencyHotkey(session)

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
                    f"八雀决策: {baque.game.decision_mode.upper() if baque.game else 'Unavailable'}",
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
        baque_plugin=baque,
    ).install(application)
    LOGGER.info("启动 Telegram polling")
    application.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
