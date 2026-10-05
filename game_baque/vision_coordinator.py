from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


class StaleOpponentRequestError(asyncio.CancelledError):
    """对手视觉请求跨过了自己的回合边界，结果必须丢弃。"""


class LocalVisionCoordinator:
    """给单并发本地视觉模型提供“自己回合优先”的请求门控。"""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_kind: str | None = None
        self._active_opponent: asyncio.Task[object] | None = None
        self._own_waiters = 0
        self._opponent_allowed = False
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    async def enter_own_phase(self) -> int:
        task: asyncio.Task[object] | None
        async with self._condition:
            if self._opponent_allowed:
                self._generation += 1
            self._opponent_allowed = False
            task = self._active_opponent
            if task is not None and not task.done():
                task.cancel()
            self._condition.notify_all()
        return self._generation

    async def enter_opponent_phase(self) -> int:
        async with self._condition:
            self._opponent_allowed = True
            self._condition.notify_all()
            return self._generation

    async def run_own(self, request: Callable[[], Awaitable[T]]) -> T:
        async with self._condition:
            self._own_waiters += 1
            try:
                await self._condition.wait_for(lambda: self._active_kind is None)
                self._active_kind = "own"
            finally:
                self._own_waiters -= 1
        try:
            return await request()
        finally:
            async with self._condition:
                self._active_kind = None
                self._condition.notify_all()

    async def run_opponent(
        self,
        generation: int,
        request: Callable[[], Awaitable[T]],
    ) -> T:
        current = asyncio.current_task()
        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    not self._opponent_allowed
                    or generation != self._generation
                    or (self._active_kind is None and self._own_waiters == 0)
                )
            )
            if not self._opponent_allowed or generation != self._generation:
                raise StaleOpponentRequestError()
            self._active_kind = "opponent"
            self._active_opponent = current
        try:
            result = await request()
            async with self._condition:
                if not self._opponent_allowed or generation != self._generation:
                    raise StaleOpponentRequestError()
            return result
        finally:
            async with self._condition:
                if self._active_opponent is current:
                    self._active_opponent = None
                self._active_kind = None
                self._condition.notify_all()
