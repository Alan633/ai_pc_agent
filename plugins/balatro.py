from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from agent.models import ToolResult
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry, ToolSpec
from config import Settings
from game_balatro.balatrobot import BalatroBotClient
from game_balatro.controller import BalatroController
from game_balatro.local_strategy import BalatroLocalStrategist
from game_balatro.solver import BalatroSolver
from game_balatro.vision import BalatroVision


LOGGER = logging.getLogger(__name__)
OBJECT = {
    "type": "object",
    "additionalProperties": False,
}


def _schema() -> dict[str, Any]:
    return {
        **OBJECT,
        "properties": {},
        "required": [],
    }


@dataclass(slots=True)
class BalatroPlugin:
    game: BalatroController | None
    session: AgentSession
    _auto_task: asyncio.Task[str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _auto_stop_event: Event = field(
        default_factory=Event,
        init=False,
        repr=False,
    )
    _last_auto_message: str = field(
        default="尚未启动 Balatro 自动游戏",
        init=False,
    )

    @classmethod
    def build(
        cls,
        settings: Settings,
        session: AgentSession,
    ) -> "BalatroPlugin":
        game: BalatroController | None = None
        if (
            settings.openai_api_key
            or settings.balatro_vision_base_url
        ):
            balatrobot: BalatroBotClient | None = None
            if (
                settings.balatro_control_mode != "mouse"
                and settings.balatrobot_url
            ):
                balatrobot = BalatroBotClient(
                    base_url=settings.balatrobot_url,
                    timeout_seconds=settings.balatrobot_timeout_seconds,
                    deck=settings.balatrobot_deck,
                    stake=settings.balatrobot_stake,
                )
            strategist: BalatroLocalStrategist | None = None
            if (
                settings.balatro_decision_mode == "hybrid"
                and settings.balatro_decision_base_url
            ):
                strategist = BalatroLocalStrategist(
                    base_url=settings.balatro_decision_base_url,
                    model=settings.balatro_decision_model,
                    api_key=settings.openai_api_key,
                    timeout_seconds=settings.balatro_decision_timeout_seconds,
                    top_n=settings.balatro_decision_top_n,
                    min_confidence=settings.balatro_decision_min_confidence,
                )
            game = BalatroController(
                vision=BalatroVision(
                    settings.openai_api_key,
                    settings.balatro_vision_model,
                    settings.balatro_vision_base_url,
                ),
                solver=BalatroSolver(
                    reserve_money=(
                        settings.balatro_reserve_money
                    )
                ),
                strategist=strategist,
                decision_mode=settings.balatro_decision_mode,
                screenshots_dir=(
                    settings.screenshots_dir
                ),
                logs_dir=settings.logs_dir,
                session=session,
                window_title_keyword=(
                    settings.balatro_window_title_keyword
                ),
                action_interval_seconds=(
                    settings.balatro_action_interval_seconds
                ),
                auto_next_run=(
                    settings.balatro_auto_next_run
                ),
                balatrobot=balatrobot,
                control_mode=settings.balatro_control_mode,
                balatrobot_auto_start=settings.balatrobot_auto_start,
                balatrobot_startup_timeout_seconds=(
                    settings.balatrobot_startup_timeout_seconds
                ),
            )
        return cls(
            game=game,
            session=session,
        )

    @property
    def auto_running(self) -> bool:
        return (
            self._auto_task is not None
            and not self._auto_task.done()
        )

    def _require_game(
        self,
    ) -> BalatroController:
        if self.game is None:
            raise RuntimeError(
                "Balatro 插件未配置视觉模型，"
                "请检查 BALATRO_VISION_BASE_URL / "
                "BALATRO_VISION_MODEL"
            )
        return self.game

    def register_tools(
        self,
        registry: ToolRegistry,
    ) -> None:
        def add(
            name: str,
            description: str,
            function: Any,
        ) -> None:
            registry.register(
                ToolSpec(
                    name,
                    description,
                    _schema(),
                    function,
                )
            )

        add(
            "balatro_get_state",
            "读取 Steam《小丑牌/Balatro》当前状态、"
            "手牌、Joker、Blind、商店和 Python Solver + 本地 Qwen "
            "混合决策预览。auto 模式优先使用 BalatroBot API，不点击。",
            self.tool_get_state,
        )
        add(
            "balatro_step",
            "让 Balatro 只执行一个当前最佳动作，然后返回。"
            "适合调试和半自动测试。",
            self.tool_step,
        )
        add(
            "balatro_auto_start",
            "启动 Balatro 连续自动游戏。会先检查 BalatroBot API；"
            "若游戏未启动且 API 不可用，则自动后台运行 uvx balatrobot serve；"
            "auto 模式失败时才回退视觉+安全鼠标。",
            self.tool_auto_start,
        )
        add(
            "balatro_auto_stop",
            "停止 Balatro 连续自动游戏。",
            self.tool_auto_stop,
        )
        add(
            "balatro_stats",
            "查看 Balatro 自动游戏统计和最近一次决策解释。",
            self.tool_stats,
        )

    async def get_state(
        self,
    ) -> ToolResult:
        game = self._require_game()
        if self.auto_running:
            return ToolResult(
                True,
                "balatro_get_state",
                data={
                    "auto_running": True,
                    "state": game.state_text(),
                    "stats": game.stats(),
                    "message": (
                        "自动游戏运行中，返回缓存状态，"
                        "避免和实时点击并发识别。"
                    ),
                },
            )

        result = await game.step(
            execute=False
        )
        return ToolResult(
            True,
            "balatro_get_state",
            data={
                "auto_running": False,
                "state": game.state_text(),
                "decision": {
                    "action": (
                        result.decision.action
                    ),
                    "reason": (
                        result.decision.reason
                    ),
                    "cards": list(
                        result.decision.card_indices
                    ),
                },
            },
            artifact=(
                result.frame.path
                if result.frame.path.exists()
                else None
            ),
        )

    async def step_once(
        self,
    ) -> dict[str, Any]:
        game = self._require_game()
        if self.auto_running:
            raise RuntimeError(
                "Balatro 自动游戏正在运行，"
                "不能同时执行单步"
            )
        result = await game.step(
            execute=True
        )
        return {
            "acted": result.acted,
            "message": result.message,
            "action": (
                result.decision.action
            ),
            "cards": list(
                result.decision.card_indices
            ),
        }

    async def auto_start(
        self,
    ) -> dict[str, Any]:
        self._require_game()
        if self.auto_running:
            return {
                "running": True,
                "message": (
                    "Balatro 自动游戏已经在运行。"
                ),
            }

        self._auto_stop_event = Event()
        self._auto_task = asyncio.create_task(
            self._auto_loop(),
            name="balatro-plugin-auto-game",
        )
        return {
            "running": True,
            "message": (
                "Balatro 自动游戏启动中：先检查/自动启动 BalatroBot，"
                "再由 Python 枚举合法候选，本地Qwen负责长期策略重排；"
                "必要时回退视觉+安全鼠标。"
            ),
        }

    async def _auto_loop(
        self,
    ) -> str:
        game = self._require_game()
        try:
            await game.prepare_auto()
            message = await game.run_loop(
                self._auto_stop_event
            )
            self._last_auto_message = message
            return message
        except asyncio.CancelledError:
            self._last_auto_message = (
                "Balatro 自动游戏已取消。"
            )
            return self._last_auto_message
        except Exception as exc:
            LOGGER.exception(
                "Balatro自动启动/循环失败"
            )
            self._last_auto_message = (
                "Balatro 自动游戏启动失败："
                f"{type(exc).__name__}: {exc}"
            )
            return self._last_auto_message

    async def auto_stop(
        self,
    ) -> dict[str, Any]:
        task = self._auto_task
        if (
            task is None
            or task.done()
        ):
            self._auto_task = None
            return {
                "stopped": False,
                "message": (
                    self._last_auto_message
                ),
            }

        self._auto_stop_event.set()
        try:
            message = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            task.cancel()
            try:
                message = await task
            except asyncio.CancelledError:
                message = (
                    "Balatro 自动任务已强制取消。"
                )

        self._auto_task = None
        self._last_auto_message = message
        return {
            "stopped": True,
            "message": message,
        }

    def _run(
        self,
        coroutine: Any,
        *,
        timeout: float = 60.0,
    ) -> Any:
        return (
            self.session.run_coroutine_threadsafe(
                coroutine,
                timeout=timeout,
            )
        )

    def tool_get_state(
        self,
    ) -> ToolResult:
        return self._run(
            self.get_state(),
            timeout=90.0,
        )

    def tool_step(
        self,
    ) -> dict[str, Any]:
        return self._run(
            self.step_once(),
            timeout=90.0,
        )

    def tool_auto_start(
        self,
    ) -> dict[str, Any]:
        return self._run(
            self.auto_start(),
            timeout=30.0,
        )

    def tool_auto_stop(
        self,
    ) -> dict[str, Any]:
        return self._run(
            self.auto_stop(),
            timeout=20.0,
        )

    def tool_stats(
        self,
    ) -> dict[str, Any]:
        game = self._require_game()
        return game.stats()

    async def shutdown(
        self,
    ) -> None:
        if self.auto_running:
            await self.auto_stop()
