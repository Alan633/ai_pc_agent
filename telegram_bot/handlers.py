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
from game_baque.controller import BaqueGameController
from game_baque.observer import OpponentObserver
from game_baque.recorder import GameRecorder
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
        game: BaqueGameController | None = None,
        game_recorder: GameRecorder | None = None,
        game_observer: OpponentObserver | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.executor = executor
        self.session = session
        self.processes = processes
        self.confirmations = confirmations
        self.game = game
        self.game_recorder = game_recorder
        self.game_observer = game_observer

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
            "game_state": self.game_state,
            "game_step": self.game_step,
            "game_start": self.game_start,
            "game_stop": self.game_stop,
            "game_record_start": self.game_record_start,
            "game_record_status": self.game_record_status,
            "game_record_stop": self.game_record_stop,
            "game_watch_start": self.game_watch_start,
            "game_watch_status": self.game_watch_status,
            "game_watch_stop": self.game_watch_stop,
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
                    f"牌局采集：{'RUNNING' if self.game_recorder and self.game_recorder.running else 'IDLE'}",
                    f"对手观察：{'RUNNING' if self.game_observer and self.game_observer.running else 'IDLE'}",
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
        stopped_agent = self.session.request_stop()
        stopped_recording = False
        stopped_observer = False
        if self.game_recorder is not None and self.game_recorder.running:
            await self.game_recorder.stop()
            stopped_recording = True
        if self.game_observer is not None and self.game_observer.running:
            await self.game_observer.stop()
            stopped_observer = True
            if self.game is not None:
                self.game.set_external_observer_active(False)
        if stopped_agent or stopped_recording or stopped_observer:
            parts = []
            if stopped_agent:
                parts.append("Agent 任务")
            if stopped_recording:
                parts.append("牌局采集")
            if stopped_observer:
                parts.append("对手观察")
            await update.effective_message.reply_text("已停止：" + "、".join(parts))
        else:
            await update.effective_message.reply_text(NO_TASK)

    async def game_record_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game_recorder is None:
            await update.effective_message.reply_text("牌局采集器未初始化。")
            return
        if self.game_recorder.running:
            await update.effective_message.reply_text(
                "牌局采集已经在运行。\n" + self.game_recorder.status().to_text()
            )
            return
        try:
            status = await self.game_recorder.start()
            await update.effective_message.reply_text(
                status.to_text()
                + "\n\n请保持欢乐斗地主窗口在前台，正常手动玩完整一局。"
                + "\n结束后发送 /game_record_stop。"
            )
        except Exception as exc:
            LOGGER.exception("启动牌局采集失败")
            await update.effective_message.reply_text(
                f"启动采集失败：{type(exc).__name__}: {exc}"
            )

    async def game_record_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game_recorder is None:
            await update.effective_message.reply_text("牌局采集器未初始化。")
            return
        await update.effective_message.reply_text(self.game_recorder.status().to_text())

    async def game_record_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game_recorder is None:
            await update.effective_message.reply_text("牌局采集器未初始化。")
            return
        if not self.game_recorder.running:
            await update.effective_message.reply_text(
                "当前没有正在进行的牌局采集。\n" + self.game_recorder.status().to_text()
            )
            return
        status = await self.game_recorder.stop()
        await update.effective_message.reply_text(
            status.to_text()
            + "\n\n本局数据已保存。后续可根据 events.jsonl 和关键帧训练/调试三家动作识别器。"
        )

    async def game_watch_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game is None or self.game_observer is None:
            await update.effective_message.reply_text("对手观察器需要可用的游戏视觉模型。")
            return
        if self.game_observer.running:
            await update.effective_message.reply_text(self.game_observer.status().to_text())
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return

        try:
            # 先关闭旧的“两帧差分事件”路径，避免上一张历史截图被误当成本局事件。
            self.game.set_external_observer_active(True)
            async with self.session.task_lock:
                # 只读取当前帧，建立自己的 7/8 张手牌、金牌和牌堆基线。
                await self.game.inspect()
            status = await self.game_observer.start()
            await update.effective_message.reply_text(
                status.to_text()
                + "\n\n现在可以继续手动玩。程序会后台识别三家完整回合，"
                + "高置信事件会自动写入记牌器。"
            )
        except Exception as exc:
            self.game.set_external_observer_active(False)
            LOGGER.exception("启动对手观察器失败")
            await update.effective_message.reply_text(
                f"启动对手观察失败：{type(exc).__name__}: {exc}"
            )

    async def game_watch_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game_observer is None:
            await update.effective_message.reply_text("对手观察器未初始化。")
            return
        parts = [self.game_observer.status().to_text()]
        if self.game is not None:
            parts.append(self.game.opponent_state_text())
            parts.append(self.game.recent_opponent_events_text())
        await update.effective_message.reply_text("\n\n".join(parts))

    async def game_watch_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game_observer is None or not self.game_observer.running:
            await update.effective_message.reply_text("当前没有运行中的对手观察器。")
            return
        status = await self.game_observer.stop()
        if self.game is not None:
            self.game.set_external_observer_active(False)
        await update.effective_message.reply_text(status.to_text())

    async def game_state(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game is None:
            await update.effective_message.reply_text("点字成金模块需要配置 OPENAI_API_KEY 或本地 GAME_VISION_BASE_URL。")
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        try:
            async with self.session.task_lock:
                result = await self.game.step(execute=False)
            await update.effective_message.reply_text(
                self.game.state_text()
                + "\n"
                + self.game.opponent_state_text()
                + "\n策略预览："
                + result.message
            )
            if result.frame.path.exists():
                with result.frame.path.open("rb") as image:
                    await update.effective_message.reply_photo(image, caption="点字成金识别截图")
        except Exception as exc:
            LOGGER.exception("读取点字成金状态失败")
            await update.effective_message.reply_text(f"识别失败：{type(exc).__name__}: {exc}")

    async def game_step(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game is None:
            await update.effective_message.reply_text("点字成金模块需要配置 OPENAI_API_KEY 或本地 GAME_VISION_BASE_URL。")
            return
        if self.session.status().busy:
            await update.effective_message.reply_text(BUSY)
            return
        try:
            async with self.session.task_lock:
                result = await self.game.step(execute=True)
            await update.effective_message.reply_text(result.message)
        except Exception as exc:
            LOGGER.exception("执行点字成金单步失败")
            await update.effective_message.reply_text(f"执行失败：{type(exc).__name__}: {exc}")

    async def game_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.game is None:
            await update.effective_message.reply_text("点字成金模块需要配置 OPENAI_API_KEY 或本地 GAME_VISION_BASE_URL。")
            return
        if self.session.status().busy or self.session.task_lock.locked():
            await update.effective_message.reply_text(BUSY)
            return
        task = context.application.create_task(self._run_game_loop(update), update=update)
        self.session.begin(task, "点字成金自动游戏")
        await update.effective_message.reply_text(
            "点字成金自动游戏已启动。请保持欢乐斗地主窗口在前台；/game_stop 或 /stop 可停止。"
        )

    async def game_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.stop(update, context)

    async def _run_game_loop(self, update: Update) -> None:
        observer_started_here = False
        try:
            async with self.session.task_lock:
                if self.game is None:
                    await update.effective_message.reply_text("游戏模块不可用。")
                    return

                if self.game_observer is not None and not self.game_observer.running:
                    # 先关闭旧两帧事件路径，再用单帧建立本局基线，避免历史截图污染记牌。
                    self.game.set_external_observer_active(True)
                    await self.game.inspect()
                    await self.game_observer.start()
                    observer_started_here = True

                message = await self.game.run_loop()
            await update.effective_message.reply_text(message)
        finally:
            if (
                observer_started_here
                and self.game_observer is not None
                and self.game_observer.running
            ):
                await self.game_observer.stop()
            if observer_started_here and self.game is not None:
                self.game.set_external_observer_active(False)
            self.session.finish()

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
