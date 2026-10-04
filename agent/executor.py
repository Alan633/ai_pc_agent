from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from agent.models import TaskOutcome
from agent.planner import OpenAIPlanner
from agent.security import SecurityPolicy
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry

LOGGER = logging.getLogger(__name__)
ConfirmCallback = Callable[[str, str], Awaitable[bool]]


class AgentExecutor:
    def __init__(
        self,
        planner: OpenAIPlanner,
        registry: ToolRegistry,
        security: SecurityPolicy,
        session: AgentSession,
        max_steps: int = 15,
    ) -> None:
        self.planner = planner
        self.registry = registry
        self.security = security
        self.session = session
        self.max_steps = min(max(max_steps, 1), 15)

    async def run(self, user_text: str, confirm: ConfirmCallback) -> TaskOutcome:
        started = time.monotonic()
        artifacts = []
        steps = 0
        active_response_id: str | None = None
        try:
            plan = await self.planner.first(user_text)
            active_response_id = plan.response_id
            while steps < self.max_steps:
                if self.session.stop_event.is_set():
                    return TaskOutcome(False, "任务已停止。", steps, artifacts)
                if not plan.calls:
                    return TaskOutcome(True, plan.final_text or "任务已完成。", steps, artifacts)
                call = plan.calls[0]
                if self.registry.get(call.name) is None:
                    result = self.registry.execute(call.name, call.arguments)
                else:
                    risk = self.security.assess(call.name, call.arguments)
                    if risk.requires_confirmation:
                        self.session.set_waiting_confirmation(True)
                        try:
                            approved = await confirm(risk.description, risk.reason)
                        finally:
                            self.session.set_waiting_confirmation(False)
                        if not approved:
                            return TaskOutcome(False, "高风险操作已取消或确认超时。", steps, artifacts)
                    self.session.set_native_call(True)
                    try:
                        result = await asyncio.to_thread(self.registry.execute, call.name, call.arguments)
                    finally:
                        self.session.set_native_call(False)
                steps += 1
                if result.artifact:
                    artifacts.append(result.artifact)
                if self.session.stop_event.is_set():
                    return TaskOutcome(False, "任务已停止。", steps, artifacts)
                plan = await self.planner.next(plan.response_id, call.call_id, result)
                active_response_id = plan.response_id
            return TaskOutcome(False, f"已达到最大步骤数 {self.max_steps}，任务自动停止。", steps, artifacts)
        except asyncio.CancelledError:
            return TaskOutcome(False, "任务已停止。", steps, artifacts)
        except Exception as exc:
            LOGGER.exception("Agent 任务异常")
            return TaskOutcome(False, f"Agent 任务失败：{type(exc).__name__}: {exc}", steps, artifacts)
        finally:
            discard = getattr(self.planner, "discard", None)
            if callable(discard):
                discard(active_response_id)
            LOGGER.info("任务结束 steps=%s duration=%.2fs", steps, time.monotonic() - started)
