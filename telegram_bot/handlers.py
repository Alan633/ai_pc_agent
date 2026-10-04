from __future__ import annotations

import asyncio
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from agent.executor import AgentExecutor
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry
from config import Settings
from telegram_bot.callbacks import ConfirmationStore
from telegram_bot.messages import BUSY, NO_TASK, UNAUTHORIZED, command_usage
from tools.process import ProcessManager

LOGGER = logging.getLogger(__name__)


class TelegramHandlers:
    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        executor: AgentExecutor,
        session: AgentSession,
        processes: ProcessManager,
        confirmations: ConfirmationStore,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.executor = executor
        self.session = session
        self.processes = processes
        self.confirmations = confirmations

    def install(self, application: Application) -> None:
        commands = {
            "start": self.start,
            "status": self.status,
            "screen": self.screen,
            "mouse": self.mouse,
            "click": self.click,
            "doubleclick": self.doubleclick,
            "move": self.move,
            "scroll": self.scroll,
            "type": self.type_text,
            "hotkey": self.hotkey,
            "process": self.process,
            "stop": self.stop,
        }
        for name, callback in commands.items():
            application.add_handler(CommandHandler(name, callback))
        application.add_handler(CallbackQueryHandler(self.confirm_callback, pattern=r"^confirm:"))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.natural_language))
        application.add_error_handler(self.on_error)

    async def _authorized(self, update: Update) -> bool:
        user = update.effective_user
        if user is None or user.id != self.settings.allowed_telegram_user_id:
            LOGGER.warning("拒绝未授权用户 user_id=%s", getattr(user, "id", None))
            if update.callback_query:
                await update.callback_query.answer(UNAUTHORIZED, show_alert=True)
            elif update.effective_message:
                await update.effective_message.reply_text(UNAUTHORIZED)
            return False
        return True

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        state = self.session.status().state.value
        await update.effective_message.reply_text(
            f"电脑在线\n主机名：{platform.node()}\nAgent 状态：{state}"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        memory = psutil.virtual_memory()
        session = self.session.status()
        active = await asyncio.to_thread(self.registry.execute, "get_active_window", {})
        title = active.data.get("title", "未知") if active.ok else f"获取失败：{active.error}"
        uptime = datetime.now(UTC) - session.started_at
        await update.effective_message.reply_text(
            "\n".join(
                [
                    f"CPU：{psutil.cpu_percent(interval=None):.1f}%",
                    f"内存：{memory.percent:.1f}%（{memory.used // 1024**2} / {memory.total // 1024**2} MB）",
                    "Python Agent：正常",
                    f"当前前台窗口：{title}",
                    f"Agent 已运行：{str(uptime).split('.')[0]}",
                    f"任务状态：{session.state.value}",
                ]
            )
        )

    async def screen(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        result = await asyncio.to_thread(self.registry.execute, "take_screenshot", {})
        if not result.ok or not result.artifact:
            await update.effective_message.reply_text(f"截图失败：{result.error}")
            return
        with result.artifact.open("rb") as image:
            await update.effective_message.reply_photo(image, caption=f"屏幕 {result.data['width']}x{result.data['height']}")

    async def mouse(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._simple(update, "get_mouse_position", {}, lambda data: f"鼠标坐标：({data['x']}, {data['y']})")

    async def click(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._point_command(update, context, "click", "click")

    async def doubleclick(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._point_command(update, context, "double_click", "doubleclick")

    async def move(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._point_command(update, context, "move_mouse", "move")

    async def _point_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, tool: str, command: str) -> None:
        if not await self._authorized(update):
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        try:
            x, y = map(int, context.args)
        except (ValueError, TypeError):
            await update.effective_message.reply_text(command_usage(command, "800 500"))
            return
        await self._simple_authorized(update, tool, {"x": x, "y": y})

    async def scroll(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        try:
            amount = int(context.args[0])
        except (ValueError, IndexError):
            await update.effective_message.reply_text(command_usage("scroll", "-5"))
            return
        await self._simple_authorized(update, "scroll", {"amount": amount})

    async def type_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        text = " ".join(context.args)
        if not text:
            await update.effective_message.reply_text(command_usage("type", "要输入的文本"))
            return
        await self._simple_authorized(update, "type_text", {"text": text, "interval": 0.02})

    async def hotkey(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        if len(context.args) < 2:
            await update.effective_message.reply_text(command_usage("hotkey", "ctrl c"))
            return
        await self._simple_authorized(update, "hotkey", {"keys": context.args})

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        data = await asyncio.to_thread(self.processes.list_managed)
        if not data["processes"]:
            await update.effective_message.reply_text("Agent 当前没有管理中的后台程序。")
            return
        lines = [f"PID {item['pid']}：{item['command']}" for item in data["processes"]]
        await update.effective_message.reply_text("\n".join(lines))

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        self.confirmations.cancel_all()
        stopped = self.session.request_stop()
        await update.effective_message.reply_text("已发送停止信号。" if stopped else NO_TASK)

    async def natural_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.session.status().busy or self.session.task_lock.locked():
            await update.effective_message.reply_text(BUSY)
            return
        text = (update.effective_message.text or "").strip()
        if not text:
            return
        task = context.application.create_task(self._run_agent(update, text), update=update)
        self.session.begin(task, text[:200])
        await update.effective_message.reply_text("已收到，开始执行。需要高风险操作时我会再次确认。")

    async def _run_agent(self, update: Update, text: str) -> None:
        user_id = update.effective_user.id

        async def confirm(description: str, reason: str) -> bool:
            token, future = self.confirmations.create(user_id, self.settings.confirm_timeout_seconds)
            keyboard = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("确认执行", callback_data=f"confirm:{token}:yes"),
                    InlineKeyboardButton("取消", callback_data=f"confirm:{token}:no"),
                ]]
            )
            await update.effective_message.reply_text(
                f"即将执行：\n{description}\n\n原因：{reason}\n两分钟内有效。", reply_markup=keyboard
            )
            try:
                return await asyncio.wait_for(future, timeout=self.settings.confirm_timeout_seconds)
            except TimeoutError:
                return False
            finally:
                self.confirmations.discard(token)

        try:
            async with self.session.task_lock:
                outcome = await self.executor.run(text, confirm)
            for artifact in dict.fromkeys(outcome.artifacts):
                if artifact.exists() and artifact.suffix.lower() == ".png":
                    with artifact.open("rb") as image:
                        await update.effective_message.reply_photo(image)
            await update.effective_message.reply_text(outcome.message)
        finally:
            self.session.finish()

    async def confirm_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        query = update.callback_query
        _, token, decision = query.data.split(":", 2)
        accepted = self.confirmations.resolve(token, update.effective_user.id, decision == "yes")
        await query.answer("已记录" if accepted else "确认请求已失效", show_alert=not accepted)
        if accepted:
            await query.edit_message_reply_markup(reply_markup=None)

    async def _simple(self, update: Update, tool: str, args: dict[str, Any], formatter: Any) -> None:
        if not await self._authorized(update):
            return
        result = await asyncio.to_thread(self.registry.execute, tool, args)
        await update.effective_message.reply_text(formatter(result.data) if result.ok else f"执行失败：{result.error}")

    async def _simple_authorized(self, update: Update, tool: str, args: dict[str, Any]) -> None:
        result = await asyncio.to_thread(self.registry.execute, tool, args)
        await update.effective_message.reply_text("执行成功。" if result.ok else f"执行失败：{result.error}")

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        LOGGER.exception("Telegram 更新处理失败", exc_info=context.error)
