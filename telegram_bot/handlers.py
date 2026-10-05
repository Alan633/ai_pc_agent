from __future__ import annotations

import asyncio
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from agent.executor import AgentExecutor
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry
from config import Settings
from game_baque.controller import BaqueGameController
from game_baque.observer import OpponentObserver
from game_baque.recorder import GameRecorder
from plugins.baque import BaquePlugin
from plugins.balatro import BalatroPlugin
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
        baque_plugin: BaquePlugin | None = None,
        balatro_plugin: BalatroPlugin | None = None,
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
        self.baque_plugin = baque_plugin
        self.balatro_plugin = balatro_plugin

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
            "game_acceptance": self.game_acceptance,
            "game_accept": self.game_accept,
            "game_accept_reset": self.game_accept_reset,
            "game_rounds": self.game_rounds,
            "balatro_state": self.balatro_state,
            "balatro_step": self.balatro_step,
            "balatro_start": self.balatro_start,
            "balatro_stop": self.balatro_stop,
            "balatro_stats": self.balatro_stats,
            "stop": self.stop,
        }
        # group=-1 先统一记录所有 Telegram 文本输入；后续新增命令无需再单独补日志。
        application.add_handler(
            MessageHandler(filters.TEXT, self.log_incoming_command),
            group=-1,
        )
        for name, callback in commands.items():
            application.add_handler(CommandHandler(name, callback))
        application.add_handler(CallbackQueryHandler(self.confirm_callback, pattern=r"^confirm:"))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.natural_language))
        application.add_error_handler(self.on_error)

    async def log_incoming_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """把每一条 Telegram 文本命令/自然语言输入写入 LOGS/agent.log。"""
        message = update.effective_message
        if message is None:
            return
        raw_text = (message.text or "").strip()
        if not raw_text:
            return

        # 保持一条用户输入对应一行日志，便于 grep/排查；超长文本限制长度。
        log_text = raw_text.replace("\r", "\\r").replace("\n", "\\n")
        if len(log_text) > 4000:
            log_text = log_text[:4000] + "...<truncated>"

        user = update.effective_user
        chat = update.effective_chat
        LOGGER.info(
            "收到用户命令 source=telegram user_id=%s chat_id=%s text=%s",
            getattr(user, "id", None),
            getattr(chat, "id", None),
            log_text,
            extra={"console_red": True},
        )

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
                    f"八雀自动：{'RUNNING' if self.baque_plugin and self.baque_plugin.auto_running else 'IDLE'}",
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
        stopped_auto = False
        stopped_recording = False
        stopped_observer = False

        if self.baque_plugin is not None and self.baque_plugin.auto_running:
            result = await self.baque_plugin.auto_stop()
            stopped_auto = bool(result.get("stopped"))

        if self.game_recorder is not None and self.game_recorder.running:
            await self.game_recorder.stop()
            stopped_recording = True

        # 全局 /stop 会把手动观察模式也停掉。
        if self.baque_plugin is not None:
            if self.game_observer is not None and self.game_observer.running:
                try:
                    result = await self.baque_plugin.watch_stop()
                    stopped_observer = bool(result.get("stopped"))
                except RuntimeError:
                    pass
        elif self.game_observer is not None and self.game_observer.running:
            await self.game_observer.stop()
            stopped_observer = True
            if self.game is not None:
                self.game.set_external_observer_active(False)

        if stopped_agent or stopped_auto or stopped_recording or stopped_observer:
            parts = []
            if stopped_agent:
                parts.append("Agent 任务")
            if stopped_auto:
                parts.append("八雀自动游戏")
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
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        try:
            result = await self.baque_plugin.watch_start()
            await update.effective_message.reply_text(
                result.get("message", "观察器状态已更新。")
                + "\n\n"
                + result.get("observer", "")
            )
        except Exception as exc:
            LOGGER.exception("启动八雀观察器失败")
            await update.effective_message.reply_text(
                f"启动观察失败：{type(exc).__name__}: {exc}"
            )

    async def game_watch_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        data = self.baque_plugin.watch_status()
        await update.effective_message.reply_text(
            "\n\n".join(
                [
                    str(data.get("observer", "")),
                    str(data.get("opponents", "")),
                    str(data.get("recent_events", "")),
                ]
            )
        )

    async def game_watch_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        try:
            result = await self.baque_plugin.watch_stop()
            await update.effective_message.reply_text(
                str(result.get("observer") or result.get("message") or "观察器已停止。")
            )
        except Exception as exc:
            await update.effective_message.reply_text(
                f"停止观察失败：{type(exc).__name__}: {exc}"
            )

    async def game_state(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        try:
            result = await self.baque_plugin.get_state()
            if not result.ok:
                await update.effective_message.reply_text(result.error or "八雀状态读取失败")
                return
            data = result.data
            await update.effective_message.reply_text(
                "\n\n".join(
                    [
                        str(data.get("state", "")),
                        str(data.get("opponents", "")),
                        "策略预览：" + str(data.get("strategy_preview") or data.get("message") or ""),
                    ]
                )
            )
            if result.artifact and result.artifact.exists():
                with result.artifact.open("rb") as image:
                    await update.effective_message.reply_photo(image, caption="八雀识别截图")
        except Exception as exc:
            LOGGER.exception("读取八雀状态失败")
            await update.effective_message.reply_text(
                f"识别失败：{type(exc).__name__}: {exc}"
            )

    async def game_step(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        try:
            result = await self.baque_plugin.step_once()
            await update.effective_message.reply_text(str(result.get("message", "单步完成。")))
        except Exception as exc:
            LOGGER.exception("执行八雀单步失败")
            await update.effective_message.reply_text(
                f"执行失败：{type(exc).__name__}: {exc}"
            )

    async def game_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        try:
            result = await self.baque_plugin.auto_start()
            await update.effective_message.reply_text(str(result.get("message", "八雀自动游戏已启动。")))
        except Exception as exc:
            LOGGER.exception("启动八雀自动游戏失败")
            await update.effective_message.reply_text(
                f"启动失败：{type(exc).__name__}: {exc}"
            )

    async def game_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        parts: list[str] = []
        if self.baque_plugin.auto_running:
            result = await self.baque_plugin.auto_stop()
            parts.append(str(result.get("message", "自动游戏已停止。")))
        if self.game_observer is not None and self.game_observer.running:
            try:
                result = await self.baque_plugin.watch_stop()
                parts.append(str(result.get("message") or result.get("observer") or "观察器已停止。"))
            except RuntimeError:
                pass
        await update.effective_message.reply_text("\n".join(parts) if parts else "当前没有运行中的八雀任务。")


    async def balatro_state(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.balatro_plugin is None:
            await update.effective_message.reply_text("Balatro 插件未初始化。")
            return
        try:
            result = await self.balatro_plugin.get_state()
            data = result.data or {}
            state = str(data.get("state", "未读取到状态"))
            decision = data.get("decision")
            if isinstance(decision, dict):
                state += (
                    f"\n决策={decision.get('action')}："
                    f"{decision.get('reason')}"
                )
            await update.effective_message.reply_text(state)
        except Exception as exc:
            LOGGER.exception("读取Balatro状态失败")
            await update.effective_message.reply_text(
                f"读取失败：{type(exc).__name__}: {exc}"
            )

    async def balatro_step(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.balatro_plugin is None:
            await update.effective_message.reply_text("Balatro 插件未初始化。")
            return
        try:
            result = await self.balatro_plugin.step_once()
            await update.effective_message.reply_text(
                str(result.get("message", "Balatro 单步完成。"))
            )
        except Exception as exc:
            LOGGER.exception("执行Balatro单步失败")
            await update.effective_message.reply_text(
                f"执行失败：{type(exc).__name__}: {exc}"
            )

    async def balatro_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.balatro_plugin is None:
            await update.effective_message.reply_text("Balatro 插件未初始化。")
            return
        try:
            result = await self.balatro_plugin.auto_start()
            await update.effective_message.reply_text(
                str(result.get("message", "Balatro 自动游戏已启动。"))
            )
        except Exception as exc:
            LOGGER.exception("启动Balatro自动游戏失败")
            await update.effective_message.reply_text(
                f"启动失败：{type(exc).__name__}: {exc}"
            )

    async def balatro_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.balatro_plugin is None:
            await update.effective_message.reply_text("Balatro 插件未初始化。")
            return
        try:
            result = await self.balatro_plugin.auto_stop()
            await update.effective_message.reply_text(
                str(result.get("message", "Balatro 自动游戏已停止。"))
            )
        except Exception as exc:
            LOGGER.exception("停止Balatro自动游戏失败")
            await update.effective_message.reply_text(
                f"停止失败：{type(exc).__name__}: {exc}"
            )

    async def balatro_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.balatro_plugin is None:
            await update.effective_message.reply_text("Balatro 插件未初始化。")
            return
        try:
            stats = self.balatro_plugin.tool_stats()
            last = stats.get("last_decision") or {}
            text = (
                f"Balatro统计：启动{stats.get('runs_started', 0)}次，"
                f"识别{stats.get('inspections', 0)}帧，"
                f"执行{stats.get('actions', 0)}步，"
                f"unknown={stats.get('unknown', 0)}，"
                f"结束局={stats.get('game_overs', 0)}"
                f"\n决策模式={stats.get('decision_mode', 'python')}，"
                f"最近来源={stats.get('last_decision_source', 'none')}，"
                f"Qwen成功={stats.get('qwen_strategy_count', 0)}，"
                f"Qwen回退={stats.get('qwen_fallback_count', 0)}"
            )
            memory = stats.get("strategy_memory") or []
            if memory:
                text += f"\n当前构筑计划：{memory[-1]}"
            if last:
                text += (
                    f"\n最近决策={last.get('action')}："
                    f"{last.get('reason')}"
                )
            await update.effective_message.reply_text(text)
        except Exception as exc:
            LOGGER.exception("读取Balatro统计失败")
            await update.effective_message.reply_text(
                f"读取失败：{type(exc).__name__}: {exc}"
            )

    async def game_acceptance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        report = self.baque_plugin.acceptance_report()
        await update.effective_message.reply_text(self._format_acceptance(report))

    async def game_rounds(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/game_rounds [最近N局]：八雀自动游戏战绩。"""
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        last = None
        if context.args:
            try:
                last = max(1, int(context.args[0]))
            except ValueError:
                pass
        stats = self.baque_plugin.round_stats(last)
        await update.effective_message.reply_text(self._format_round_stats(stats, last))

    @staticmethod
    def _format_round_stats(stats: dict[str, Any], last: int | None) -> str:
        if not stats.get("rounds"):
            return "还没有八雀对局记录。自动游戏每打完一局会自动记录。"
        scope = f"最近 {last} 局" if last else "全部"
        lost_rate = stats.get("lost_turn_rate")
        lines = [
            f"八雀战绩（{scope}）：{stats['rounds']} 局，胜 {stats['wins']} 局，胜率 {stats['win_rate'] * 100:.1f}%",
            f"平均每局轮到自己 {stats['avg_turns']} 回合；丢失回合 {stats['lost_turns']} 次"
            + (f"（{lost_rate * 100:.1f}%）" if lost_rate is not None else ""),
            f"听过牌的局 {stats['tenpai_rounds']}（{stats['tenpai_rate'] * 100:.1f}%）；"
            f"出现过可胡的局 {stats['hu_available_rounds']}",
        ]
        reasons = stats.get("top_lost_reasons") or []
        if reasons:
            lines.append("丢失回合原因：" + "，".join(f"{name}×{count}" for name, count in reasons))
        recent = stats.get("recent") or []
        if recent:
            lines.append("最近几局：")
            lines.extend(f"- {line}" for line in recent)
        return "\n".join(lines)

    async def game_accept(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        if len(context.args) < 2:
            await update.effective_message.reply_text(
                "用法：/game_accept own|opponent|action ok|bad [备注]"
            )
            return

        category_map = {
            "own": "own_state",
            "自己": "own_state",
            "opponent": "opponent_event",
            "对手": "opponent_event",
            "action": "action",
            "动作": "action",
            "点击": "action",
        }
        result_map = {
            "ok": True,
            "对": True,
            "正确": True,
            "bad": False,
            "错": False,
            "错误": False,
        }
        category = category_map.get(context.args[0].lower())
        correct = result_map.get(context.args[1].lower())
        if category is None or correct is None:
            await update.effective_message.reply_text(
                "用法：/game_accept own|opponent|action ok|bad [备注]"
            )
            return
        note = " ".join(context.args[2:])
        result = self.baque_plugin.acceptance_mark(category, correct, note)
        accuracy = result.get("accuracy")
        accuracy_text = f"{float(accuracy) * 100:.1f}%" if accuracy is not None else "暂无"
        await update.effective_message.reply_text(
            f"已记录：{category} {'正确' if correct else '错误'}；"
            f"样本 {result['checks']}，当前准确率 {accuracy_text}。"
        )

    async def game_accept_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorized(update):
            return
        if self.baque_plugin is None:
            await update.effective_message.reply_text("八雀插件未初始化。")
            return
        result = self.baque_plugin.acceptance.reset_session()
        await update.effective_message.reply_text(
            "已开始新一轮八雀验收统计；历史标签仍保留。\n"
            + str(result.get("labels_path", ""))
        )

    @staticmethod
    def _format_acceptance(report: dict[str, Any]) -> str:
        manual = report.get("manual_accuracy", {})
        automatic = report.get("automatic_checks", {})

        def metric(name: str, label: str) -> str:
            item = manual.get(name, {})
            accuracy = item.get("accuracy")
            accuracy_text = "暂无" if accuracy is None else f"{float(accuracy) * 100:.1f}%"
            return (
                f"{label}：{accuracy_text} "
                f"({item.get('correct', 0)}/{item.get('checks', 0)})"
            )

        lines = [
            f"八雀验收：{report.get('verdict', 'UNKNOWN')}",
            metric("own_state", "自己牌面"),
            metric("opponent_event", "对手动作"),
            metric("action", "决策/点击"),
        ]
        own_auto = automatic.get("own_unknown_rate", {})
        observer_auto = automatic.get("observer_stability", {})
        tracker = automatic.get("tracker_consistency", {})
        unknown = own_auto.get("value")
        low = observer_auto.get("low_confidence_rate")
        lines.extend(
            [
                "自动指标：",
                f"- 牌池一致性：{tracker.get('status', 'UNKNOWN')}",
                f"- 画面 unknown 率：{'暂无' if unknown is None else f'{float(unknown) * 100:.1f}%'} "
                f"[{own_auto.get('status', 'UNKNOWN')}]",
                f"- 对手低置信率：{'暂无' if low is None else f'{float(low) * 100:.1f}%'} "
                f"[{observer_auto.get('status', 'UNKNOWN')}]",
            ]
        )
        if report.get("verdict") == "INSUFFICIENT_DATA":
            lines.append("还需要继续打真值标签，达到样本门槛后才会给 PASS/FAIL。")
        return "\n".join(lines)

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
        try:
            await update.effective_message.reply_text("已收到，开始执行。需要高风险操作时我会再次确认。")
        except NetworkError as exc:
            # 任务已在后台执行；回执只是提示，网络抖动时不应刷出整段异常栈。
            LOGGER.warning("Telegram 回执发送失败，任务仍在执行：%s", exc)

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
