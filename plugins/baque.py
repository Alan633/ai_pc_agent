from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from agent.models import ToolResult
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry, ToolSpec
from config import Settings
from game_baque.acceptance import BaqueAcceptanceTracker
from game_baque.controller import BaqueGameController, GameStepResult
from game_baque.event_vision import OpponentEventVision
from game_baque.local_decision import LocalDecisionReranker
from game_baque.observer import OpponentObserver
from game_baque.recorder import GameRecorder
from game_baque.vision import BaqueVision


LOGGER = logging.getLogger(__name__)


OBJECT = {"type": "object", "additionalProperties": False}


def _schema(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **OBJECT,
        "properties": properties or {},
        "required": required or [],
    }


@dataclass(slots=True)
class BaquePlugin:
    """百变八雀·点字成金插件。

    插件同时服务 Telegram 固定命令和自然语言 Agent Tool，确保两种入口共享
    同一套观察器、自动循环、验收统计和停止逻辑。
    """

    game: BaqueGameController | None
    recorder: GameRecorder
    observer: OpponentObserver | None
    session: AgentSession
    acceptance: BaqueAcceptanceTracker
    _auto_task: asyncio.Task[str] | None = field(default=None, init=False, repr=False)
    _auto_stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _auto_started_observer: bool = field(default=False, init=False, repr=False)
    _last_auto_message: str = field(default="尚未启动自动游戏", init=False)

    @classmethod
    def build(cls, settings: Settings, session: AgentSession) -> "BaquePlugin":
        recorder = GameRecorder(
            recordings_dir=settings.game_recordings_dir,
            window_title_keyword=settings.game_window_title_keyword,
            fps=settings.game_record_fps,
            archive_fps=settings.game_record_archive_fps,
            change_threshold=settings.game_record_diff_threshold,
            jpeg_quality=settings.game_record_jpeg_quality,
        )

        game: BaqueGameController | None = None
        if settings.openai_api_key or settings.game_vision_base_url:
            decision_reranker = None
            if (
                settings.game_decision_mode == "hybrid"
                and settings.game_decision_base_url
            ):
                decision_reranker = LocalDecisionReranker(
                    base_url=settings.game_decision_base_url,
                    model=settings.game_decision_model or settings.openai_model,
                    api_key=settings.openai_api_key,
                    timeout_seconds=settings.game_decision_timeout_seconds,
                    top_n=settings.game_decision_top_n,
                    min_confidence=settings.game_decision_min_confidence,
                    max_ev_drop=settings.game_hybrid_max_ev_drop,
                )
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
                fast_vision_enabled=settings.game_fast_vision,
                decision_mode=settings.game_decision_mode,
                decision_reranker=decision_reranker,
            )

        observer: OpponentObserver | None = None
        if game is not None:
            observer = OpponentObserver(
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

        return cls(
            game=game,
            recorder=recorder,
            observer=observer,
            session=session,
            acceptance=BaqueAcceptanceTracker(settings.logs_dir / "baque"),
        )

    @property
    def auto_running(self) -> bool:
        return self._auto_task is not None and not self._auto_task.done()

    def register_tools(self, registry: ToolRegistry) -> None:
        """向通用 Agent 注册八雀能力，必须在 Planner 获取 schemas() 前调用。"""

        def add(
            name: str,
            description: str,
            parameters: dict[str, Any],
            function: Any,
        ) -> None:
            registry.register(ToolSpec(name, description, parameters, function))

        add(
            "baque_get_state",
            "读取当前百变八雀/点字成金画面和策略预览，不点击。用户说“看看八雀状态/现在该怎么打/识别一下牌面”时使用。",
            _schema(),
            self.tool_get_state,
        )
        add(
            "baque_step",
            "让八雀只执行一个当前最佳动作，然后立即返回。适合调试和半自动玩法；不要用于连续自动游戏。",
            _schema(),
            self.tool_step,
        )
        add(
            "baque_watch_start",
            "启动八雀三家对手实时观察和记牌，但不自动点击自己的牌。用户说“只记牌/只观察/我自己玩你帮我算”时使用。",
            _schema(),
            self.tool_watch_start,
        )
        add(
            "baque_watch_status",
            "查看八雀观察器采样、动作片段、识别、写入记牌器、低置信事件以及三家可能牌型。",
            _schema(),
            self.tool_watch_status,
        )
        add(
            "baque_watch_stop",
            "停止八雀只观察/记牌模式。自动游戏运行中不能单独停观察器，应使用 baque_auto_stop。",
            _schema(),
            self.tool_watch_stop,
        )
        add(
            "baque_auto_start",
            "启动八雀连续自动游戏，同时自动启动三家对手观察器。只在用户明确要求“自动玩/开始八雀自动游戏”时使用。",
            _schema(),
            self.tool_auto_start,
        )
        add(
            "baque_auto_stop",
            "停止八雀连续自动游戏。若观察器是自动游戏启动的，也一并停止。",
            _schema(),
            self.tool_auto_stop,
        )
        add(
            "baque_acceptance_report",
            "生成八雀真实牌局验收报告：unknown率、低置信率、牌池一致性，以及人工真值准确率。",
            _schema(),
            self.tool_acceptance_report,
        )
        add(
            "baque_acceptance_mark",
            "给刚才的八雀结果打真值标签，用于真实准确率统计。category: own_state=自己牌面，opponent_event=对手动作，action=点击/决策。",
            _schema(
                {
                    "category": {
                        "type": "string",
                        "enum": ["own_state", "opponent_event", "action"],
                    },
                    "correct": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                ["category", "correct", "note"],
            ),
            self.tool_acceptance_mark,
        )
        add(
            "baque_acceptance_reset",
            "开始新一轮八雀验收统计；历史标签不会删除，只在日志中写入 reset 分界。",
            _schema(),
            self.tool_acceptance_reset,
        )

    def _require_game(self) -> BaqueGameController:
        if self.game is None:
            raise RuntimeError(
                "八雀插件未配置视觉模型，请检查 GAME_VISION_BASE_URL / GAME_VISION_MODEL"
            )
        return self.game

    def _run(self, coroutine: Any, *, timeout: float = 60.0) -> Any:
        return self.session.run_coroutine_threadsafe(coroutine, timeout=timeout)

    async def get_state(self) -> ToolResult:
        game = self._require_game()
        if self.auto_running:
            return ToolResult(
                True,
                "baque_get_state",
                data={
                    "auto_running": True,
                    "state": game.state_text(),
                    "opponents": game.opponent_state_text(),
                    "message": "自动游戏运行中，返回缓存状态，避免与实时点击并发识别。",
                },
            )

        result = await game.step(execute=False)
        data = {
            "auto_running": False,
            "state": game.state_text(),
            "opponents": game.opponent_state_text(),
            "strategy_preview": result.message,
            "screen_type": result.snapshot.screen_type.value,
        }
        return ToolResult(
            True,
            "baque_get_state",
            data=data,
            artifact=result.frame.path if result.frame.path.exists() else None,
        )

    async def step_once(self) -> dict[str, Any]:
        game = self._require_game()
        if self.auto_running:
            raise RuntimeError("八雀自动游戏正在运行，不能同时执行单步")
        result = await game.step(execute=True)
        return self._step_dict(result)

    async def watch_start(self) -> dict[str, Any]:
        game = self._require_game()
        observer = self.observer
        if observer is None:
            raise RuntimeError("八雀对手观察器不可用")
        if self.auto_running:
            return {
                "running": True,
                "message": "自动游戏正在运行，三家观察器已由自动模式管理。",
                "observer": observer.status().to_text(),
            }
        if observer.running:
            return {
                "running": True,
                "message": "三家观察器已经在运行。",
                "observer": observer.status().to_text(),
            }

        game.set_external_observer_active(True)
        try:
            await game.inspect()
            status = await observer.start()
        except Exception:
            game.set_external_observer_active(False)
            raise
        return {
            "running": True,
            "message": "已启动三家实时观察/记牌，不会自动点击。",
            "observer": status.to_text(),
        }

    def watch_status(self) -> dict[str, Any]:
        game = self._require_game()
        observer = self.observer
        status_text = observer.status().to_text() if observer is not None else "观察器不可用"
        return {
            "auto_running": self.auto_running,
            "observer_running": bool(observer and observer.running),
            "observer": status_text,
            "opponents": game.opponent_state_text(),
            "recent_events": game.recent_opponent_events_text(),
        }

    async def watch_stop(self) -> dict[str, Any]:
        game = self._require_game()
        observer = self.observer
        if self.auto_running:
            raise RuntimeError("自动游戏运行中，请使用 baque_auto_stop")
        if observer is None or not observer.running:
            game.set_external_observer_active(False)
            return {"stopped": False, "message": "当前没有运行中的三家观察器。"}
        status = await observer.stop()
        game.set_external_observer_active(False)
        return {"stopped": True, "observer": status.to_text()}

    async def auto_start(self) -> dict[str, Any]:
        game = self._require_game()
        if self.auto_running:
            return {
                "running": True,
                "message": "八雀自动游戏已经在运行。",
                "state": game.state_text(),
            }

        self._auto_stop_event = Event()
        self._auto_started_observer = False

        # 不在工具调用阶段直接 inspect：用户发送命令时前台可能是抖音/IDE。
        # 后台自动流程会先把欢乐斗地主切到前台，识别并点击“开始游戏”，
        # 等正式发牌完成后再启动三家观察器和 EV 循环。
        self._auto_task = asyncio.create_task(
            self._auto_loop(),
            name="baque-plugin-auto-game",
        )
        return {
            "running": True,
            "message": "八雀自动游戏启动中：将先切到欢乐斗地主并自动点击“开始游戏”，进入正式牌局后再启动三家记牌和EV。",
            "auto_next_round": game.auto_next_round,
        }

    async def _auto_loop(self) -> str:
        game = self._require_game()
        observer = self.observer
        try:
            await game.prepare_auto_round()

            if self._auto_stop_event.is_set():
                self._last_auto_message = "八雀自动游戏已在进入牌局前停止。"
                return self._last_auto_message

            if observer is not None and not game.vision.base_url:
                game.set_external_observer_active(True)
                if not observer.running:
                    await observer.start()
                    self._auto_started_observer = True
                    LOGGER.info("八雀启动流程：正式牌局已就绪，三家观察器已启动")
            elif game.vision.base_url:
                # Ollama 通常串行处理同一模型请求。并行启动对手视觉 worker 会
                # 让自己的回合识别排队几十秒，最终点击过期画面。本地模式优先
                # 实时操作，主循环使用单帧识别且暂不调用对手视觉模型。
                game.set_external_observer_active(True)
                LOGGER.info("本地视觉模式：暂停并行对手识别，优先保证回合内操作")

            message = await game.run_loop(self._auto_stop_event)
            self._last_auto_message = message
            return message
        except asyncio.CancelledError:
            self._last_auto_message = "八雀自动游戏已取消。"
            return self._last_auto_message
        except Exception as exc:
            LOGGER.exception("八雀自动启动/游戏循环失败")
            self._last_auto_message = f"八雀自动游戏启动失败：{type(exc).__name__}: {exc}"
            return self._last_auto_message
        finally:
            if self._auto_started_observer and observer is not None and observer.running:
                try:
                    await observer.stop()
                except Exception:
                    LOGGER.exception("停止八雀自动观察器失败")
            if observer is None or not observer.running:
                game.set_external_observer_active(False)
            self._auto_started_observer = False

    async def auto_stop(self) -> dict[str, Any]:
        task = self._auto_task
        if task is None or task.done():
            self._auto_task = None
            return {
                "stopped": False,
                "message": self._last_auto_message,
            }

        self._auto_stop_event.set()
        try:
            message = await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                message = await task
            except asyncio.CancelledError:
                message = "自动游戏任务已强制取消。"
        self._auto_task = None
        self._last_auto_message = message
        return {"stopped": True, "message": message}

    def acceptance_report(self) -> dict[str, Any]:
        return self.acceptance.report(self.game, self.observer)

    def acceptance_mark(
        self,
        category: str,
        correct: bool,
        note: str = "",
    ) -> dict[str, Any]:
        game = self._require_game()
        if category == "own_state":
            context = game.state_text()
        elif category == "opponent_event":
            context = game.recent_opponent_events_text()
        elif category == "action":
            context = game.last_message
        else:
            context = ""
        return self.acceptance.mark(
            category,
            correct,
            note=note,
            context=context,
        )

    # ---- ToolRegistry 同步入口：AgentExecutor 会在工作线程调用 ----

    def tool_get_state(self) -> ToolResult:
        return self._run(self.get_state(), timeout=90.0)

    def tool_step(self) -> dict[str, Any]:
        return self._run(self.step_once(), timeout=90.0)

    def tool_watch_start(self) -> dict[str, Any]:
        return self._run(self.watch_start(), timeout=90.0)

    def tool_watch_status(self) -> dict[str, Any]:
        return self.watch_status()

    def tool_watch_stop(self) -> dict[str, Any]:
        return self._run(self.watch_stop(), timeout=30.0)

    def tool_auto_start(self) -> dict[str, Any]:
        return self._run(self.auto_start(), timeout=90.0)

    def tool_auto_stop(self) -> dict[str, Any]:
        return self._run(self.auto_stop(), timeout=20.0)

    def tool_acceptance_report(self) -> dict[str, Any]:
        return self.acceptance_report()

    def tool_acceptance_mark(
        self,
        category: str,
        correct: bool,
        note: str,
    ) -> dict[str, Any]:
        return self.acceptance_mark(category, correct, note)

    def tool_acceptance_reset(self) -> dict[str, Any]:
        return self.acceptance.reset_session()

    async def shutdown(self) -> None:
        if self.auto_running:
            await self.auto_stop()
        if self.recorder.running:
            await self.recorder.stop()
        if self.observer is not None and self.observer.running:
            await self.observer.stop()
        if self.game is not None:
            self.game.set_external_observer_active(False)

    @staticmethod
    def _step_dict(result: GameStepResult) -> dict[str, Any]:
        return {
            "acted": result.acted,
            "message": result.message,
            "screen_type": result.snapshot.screen_type.value,
            "stop_loop": result.stop_loop,
        }
