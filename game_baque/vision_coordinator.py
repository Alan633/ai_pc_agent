from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


class LocalVisionCoordinator:
    """协调本地视觉模型的自己回合/对手回合请求。

    Ollama 上的本地视觉模型通常不适合同时处理多路截图请求。观察器进入
    自己回合时会先切换 generation 并取消在途对手任务；这里再用一把锁保证
    最终只有一个模型请求在运行，同时拒绝跨回合的旧对手请求。
    """

    def __init__(self) -> None:
        self._model_lock = asyncio.Lock()
        self._generation = 0
        self._phase = "idle"

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def phase(self) -> str:
        return self._phase

    async def enter_own_phase(self) -> int:
        """切到自己回合，使此前所有对手请求立即成为旧 generation。"""
        self._generation += 1
        self._phase = "own"
        return self._generation

    async def enter_opponent_phase(self) -> int:
        """切到对手回合并返回本阶段 generation。"""
        self._generation += 1
        self._phase = "opponent"
        return self._generation

    async def run_own(
        self,
        request: Callable[[], Awaitable[T]],
    ) -> T:
        """串行执行自己的视觉请求。

        对手观察器会在调用本方法前暂停并取消在途分类，因此这里只需等待
        已释放的模型锁，不再自行改 generation，避免一次牌面识别产生额外
        阶段切换。
        """
        async with self._model_lock:
            return await request()

    async def run_opponent(
        self,
        generation: int,
        request: Callable[[], Awaitable[T]],
    ) -> T:
        """仅在 generation 仍有效且处于对手阶段时执行请求。"""
        self._ensure_opponent_generation(generation)
        async with self._model_lock:
            # 排队期间可能已经轮到自己，拿到锁后必须再次校验。
            self._ensure_opponent_generation(generation)
            return await request()

    def _ensure_opponent_generation(self, generation: int) -> None:
        if (
            self._phase != "opponent"
            or generation != self._generation
        ):
            # Observer 专门把 CancelledError 解释为“自己回合抢占”，不会把
            # 旧结果计入识别失败，也不会继续写入 TableTracker。
            raise asyncio.CancelledError(
                "对手视觉请求已跨回合失效"
            )
