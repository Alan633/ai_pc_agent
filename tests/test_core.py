from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic

import pytest

from agent.executor import AgentExecutor
from agent.models import PlanResponse, ToolCall
from agent.security import SecurityPolicy
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry, ToolSpec
from config import Settings
from telegram_bot.callbacks import ConfirmationStore


def make_settings(root: Path) -> Settings:
    return Settings(
        base_dir=root,
        telegram_bot_token="test-token",
        allowed_telegram_user_id=12345,
        openai_api_key="test-key",
        openai_model="test-model",
        playwright_headless=True,
        max_steps=20,
        confirm_timeout_seconds=120,
        allowed_roots=(root.resolve(),),
        allowed_executables=frozenset({"python.exe", "git.exe", "git"}),
    )


def test_tool_registry_executes_and_catches_failure() -> None:
    registry = ToolRegistry()
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    registry.register(ToolSpec("ok", "ok", schema, lambda: {"value": 7}))

    def explode() -> None:
        raise RuntimeError("boom")

    registry.register(ToolSpec("explode", "explode", schema, explode))
    assert registry.execute("ok", {}).data["value"] == 7
    failed = registry.execute("explode", {})
    assert failed.ok is False
    assert "boom" in (failed.error or "")
    assert registry.execute("missing", {}).ok is False


def test_tool_registry_rejects_malformed_arguments() -> None:
    registry = ToolRegistry()
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    registry.register(ToolSpec("typed", "typed", schema, lambda value: {"value": value}))
    assert registry.execute("typed", {"value": "not-an-int"}).ok is False
    assert registry.execute("typed", {"value": 1, "extra": True}).ok is False


def test_security_requires_confirmation_for_high_risk_git(tmp_path: Path) -> None:
    policy = SecurityPolicy(make_settings(tmp_path))
    assert policy.assess("run_safe_command", {"command": ["git", "push"]}).requires_confirmation
    assert policy.assess("run_safe_command", {"command": ["git", "reset", "--hard"]}).requires_confirmation
    assert not policy.assess("run_safe_command", {"command": ["git", "status"]}).requires_confirmation
    with pytest.raises(PermissionError):
        policy.validate_command(["python.exe", "-c", "print(1)"], tmp_path)


def test_confirmation_is_one_time_and_bound_to_user() -> None:
    async def scenario() -> None:
        store = ConfirmationStore()
        token, future = store.create(12345, 120)
        assert not store.resolve(token, 999, True)
        assert await future is False
        token2, future2 = store.create(12345, 120)
        assert store.resolve(token2, 12345, True)
        assert await future2 is True
        assert not store.resolve(token2, 12345, True)

    asyncio.run(scenario())


class EndlessPlanner:
    async def first(self, user_text: str) -> PlanResponse:
        return PlanResponse("response-0", [ToolCall("call-0", "noop", {})])

    async def next(self, previous_response_id: str, call_id: str, result: object) -> PlanResponse:
        index = int(previous_response_id.split("-")[-1]) + 1
        return PlanResponse(f"response-{index}", [ToolCall(f"call-{index}", "noop", {})])


@pytest.mark.asyncio
async def test_max_steps_stops_executor(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "noop",
            "noop",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            lambda: {},
        )
    )
    executor = AgentExecutor(
        EndlessPlanner(), registry, SecurityPolicy(settings), AgentSession(), max_steps=3  # type: ignore[arg-type]
    )
    outcome = await executor.run("循环", lambda description, reason: asyncio.sleep(0, result=True))
    assert outcome.ok is False
    assert outcome.steps == 3
    assert "最大步骤数" in outcome.message


@pytest.mark.asyncio
async def test_stop_sets_cooperative_signal() -> None:
    session = AgentSession()
    blocker = asyncio.create_task(asyncio.sleep(10))
    session.begin(blocker, "test")
    assert session.request_stop() is True
    assert session.stop_event.is_set()
    assert not blocker.cancelled()
    blocker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocker
