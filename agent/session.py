from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


T = TypeVar("T")


class TaskState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    STOPPING = "STOPPING"


@dataclass(slots=True)
class SessionStatus:
    busy: bool
    state: TaskState
    started_at: datetime
    current_description: str | None


class AgentSession:
    def __init__(self) -> None:
        self.started_at = datetime.now(UTC)
        self.task_lock = asyncio.Lock()
        self.stop_event = threading.Event()
        self.current_task: asyncio.Task[object] | None = None
        self.current_description: str | None = None
        self.state = TaskState.IDLE
        self.native_call_in_progress = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def run_coroutine_threadsafe(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        timeout: float = 30.0,
    ) -> T:
        """从 ToolRegistry 的工作线程安全调用主事件循环中的异步能力。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            raise RuntimeError("Agent 事件循环尚未初始化")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            coroutine.close()
            raise RuntimeError("不能在主事件循环线程同步等待异步工具")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result(timeout=max(1.0, timeout))

    def begin(self, task: asyncio.Task[object], description: str) -> None:
        self.stop_event.clear()
        self.current_task = task
        self.current_description = description
        self.state = TaskState.RUNNING

    def finish(self) -> None:
        self.current_task = None
        self.current_description = None
        self.stop_event.clear()
        self.native_call_in_progress = False
        self.state = TaskState.IDLE

    def set_waiting_confirmation(self, waiting: bool) -> None:
        if self.state != TaskState.STOPPING:
            self.state = TaskState.WAITING_CONFIRMATION if waiting else TaskState.RUNNING

    def set_native_call(self, running: bool) -> None:
        self.native_call_in_progress = running

    def request_stop(self) -> bool:
        self.stop_event.set()
        task = self.current_task
        if task and not task.done():
            self.state = TaskState.STOPPING
            # 模型网络请求可以立即取消；已经进入本地线程的 GUI 调用则必须安全收尾。
            if not self.native_call_in_progress:
                task.cancel()
            return True
        return False

    def request_stop_threadsafe(self) -> None:
        self.stop_event.set()
        if self._loop:
            self._loop.call_soon_threadsafe(self.request_stop)

    def status(self) -> SessionStatus:
        return SessionStatus(
            busy=self.current_task is not None and not self.current_task.done(),
            state=self.state,
            started_at=self.started_at,
            current_description=self.current_description,
        )
