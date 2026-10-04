from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.executor import AgentExecutor
from agent.models import PlanResponse, ToolCall, ToolResult
from agent.security import SecurityPolicy
from agent.session import AgentSession, TaskState
from agent.tool_registry import ToolRegistry, ToolSpec
from config import Settings
from tools.apps import AppCatalog
from tools.factory import build_registry
from tools.process import ProcessManager


def settings_for(root: Path) -> Settings:
    return Settings(
        base_dir=root,
        telegram_bot_token="token",
        allowed_telegram_user_id=1,
        openai_api_key="",
        openai_model="model",
        playwright_headless=True,
        max_steps=15,
        confirm_timeout_seconds=120,
        allowed_roots=(root,),
        allowed_executables=frozenset(),
    )


def write_catalog(tmp_path: Path) -> tuple[AppCatalog, Path]:
    executable = tmp_path / "safe.exe"
    executable.write_bytes(b"test")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "apps.yaml"
    config.write_text(
        "apps:\n"
        "  notepad:\n"
        f"    paths: ['{executable.as_posix()}']\n"
        "    allow_extra_args: false\n"
        "  cmd:\n"
        f"    paths: ['{executable.as_posix()}']\n"
        "    allow_extra_args: false\n",
        encoding="utf-8",
    )
    return AppCatalog(config), executable


def test_app_catalog_is_name_based_and_rejects_arguments(tmp_path: Path) -> None:
    catalog, executable = write_catalog(tmp_path)
    assert catalog.resolve("notepad", []).executable == executable
    with pytest.raises(PermissionError):
        catalog.resolve("unknown", [])
    with pytest.raises(PermissionError):
        catalog.resolve("cmd", ["/c", "whoami"])


def test_phase2_registry_exposes_only_requested_tools(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    catalog, _ = write_catalog(tmp_path)
    security = SecurityPolicy(settings)
    session = AgentSession()
    registry = build_registry(settings, security, ProcessManager(security, catalog), session)
    assert set(registry.names()) == {
        "take_screenshot",
        "get_system_status",
        "get_active_window",
        "get_mouse_position",
        "move_mouse",
        "click",
        "double_click",
        "scroll",
        "type_text",
        "hotkey",
        "launch_app",
        "open_url",
        "list_processes",
        "start_process",
        "stop_managed_process",
        "wait",
    }


@pytest.mark.asyncio
async def test_task_state_transitions_and_stop() -> None:
    session = AgentSession()
    sleeper = asyncio.create_task(asyncio.sleep(10))
    session.begin(sleeper, "测试")
    assert session.status().state is TaskState.RUNNING
    session.set_waiting_confirmation(True)
    assert session.status().state is TaskState.WAITING_CONFIRMATION
    session.set_native_call(True)
    assert session.request_stop()
    assert session.status().state is TaskState.STOPPING
    assert not sleeper.cancelled()
    sleeper.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sleeper
    session.finish()
    assert session.status().state is TaskState.IDLE


class ScreenshotPlanner:
    async def first(self, user_text: str) -> PlanResponse:
        return PlanResponse("r1", [ToolCall("c1", "take_screenshot", {})])

    async def next(self, previous_response_id: str, call_id: str, result: ToolResult) -> PlanResponse:
        return PlanResponse("r2", [], "截图完成")


@pytest.mark.asyncio
async def test_planned_screenshot_artifact_is_returned(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"png")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "take_screenshot",
            "screen",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            lambda: ToolResult(True, "take_screenshot", artifact=screenshot),
        )
    )
    session = AgentSession()
    executor = AgentExecutor(
        ScreenshotPlanner(), registry, SecurityPolicy(settings_for(tmp_path)), session, max_steps=15  # type: ignore[arg-type]
    )
    outcome = await executor.run("给我看看电脑屏幕", lambda *_: asyncio.sleep(0, result=True))
    assert outcome.ok
    assert outcome.message == "截图完成"
    assert outcome.artifacts == [screenshot]


def test_executor_hard_caps_at_fifteen(tmp_path: Path) -> None:
    registry = ToolRegistry()
    executor = AgentExecutor(
        ScreenshotPlanner(), registry, SecurityPolicy(settings_for(tmp_path)), AgentSession(), max_steps=999  # type: ignore[arg-type]
    )
    assert executor.max_steps == 15


def test_stop_managed_process_handles_already_exited(tmp_path: Path) -> None:
    catalog, _ = write_catalog(tmp_path)
    manager = ProcessManager(SecurityPolicy(settings_for(tmp_path)), catalog)

    class FinishedProcess:
        def poll(self) -> int:
            return 0

    manager._managed[123] = FinishedProcess()  # type: ignore[assignment]
    result = manager.stop_managed_process(123)
    assert result == {"stopped": False, "already_exited": True, "pid": 123}
