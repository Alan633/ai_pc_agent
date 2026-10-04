from __future__ import annotations

from functools import partial
from typing import Any

from agent.security import SecurityPolicy
from agent.session import AgentSession
from agent.tool_registry import ToolRegistry, ToolSpec
from config import Settings
from tools import browser, keyboard, mouse, windows_ui
from tools.process import ProcessManager, list_processes
from tools.screen import take_screenshot
from tools.system import get_system_status
from tools.wait import wait


OBJECT = {"type": "object", "additionalProperties": False}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {**OBJECT, "properties": properties, "required": required or []}


def build_registry(
    settings: Settings, security: SecurityPolicy, processes: ProcessManager, session: AgentSession
) -> ToolRegistry:
    registry = ToolRegistry()

    def add(name: str, description: str, parameters: dict[str, Any], function: Any) -> None:
        registry.register(ToolSpec(name, description, parameters, function))

    integer = {"type": "integer"}
    string = {"type": "string"}
    app_names = ", ".join(processes.catalog.names())
    add("launch_app", f"启动 apps.yaml 白名单应用。可用：{app_names}。cmd/powershell 只能打开空窗口。", _schema({"app_name": string}, ["app_name"]), processes.launch_app)
    add("open_url", "用系统默认浏览器打开 http/https URL。", _schema({"url": string}, ["url"]), browser.open_url)
    add("take_screenshot", "截取 Windows 全屏；返回原始像素宽高和 PNG。", _schema({}), partial(take_screenshot, settings.screenshots_dir))
    add(
        "get_system_status",
        "获取 CPU、内存、Windows/Agent 运行时间和 Agent 状态。",
        _schema({}),
        partial(
            get_system_status,
            lambda: {
                "state": session.status().state.value,
                "busy": session.status().busy,
                "task": session.status().current_description,
            },
            session.started_at,
        ),
    )
    add("get_mouse_position", "获取当前鼠标原生屏幕坐标。", _schema({}), mouse.get_mouse_position)
    add("click", "单击原生屏幕坐标。", _schema({"x": integer, "y": integer}, ["x", "y"]), mouse.click)
    add("double_click", "双击原生屏幕坐标。", _schema({"x": integer, "y": integer}, ["x", "y"]), mouse.double_click)
    add("move_mouse", "移动鼠标到原生屏幕坐标。", _schema({"x": integer, "y": integer}, ["x", "y"]), mouse.move_mouse)
    add("scroll", "滚动鼠标，正数向上、负数向下。", _schema({"amount": integer}, ["amount"]), mouse.scroll)
    add("type_text", "向当前焦点输入文本，支持中文。", _schema({"text": string}, ["text"]), keyboard.type_text)
    add("hotkey", "执行 2-5 键组合键。", _schema({"keys": {"type": "array", "items": string, "minItems": 2, "maxItems": 5}}, ["keys"]), keyboard.hotkey)
    add("get_active_window", "获取当前前台窗口标题和类名。", _schema({}), windows_ui.get_active_window)
    add("list_processes", "列出最多 120 个进程及简短命令行，可用于识别 Python 程序。", _schema({}), list_processes)
    add("start_process", "按 apps.yaml 中的 program 名启动白名单程序；args 只有配置明确允许时才可使用。", _schema({"program": string, "args": {"type": "array", "items": string}}, ["program", "args"]), processes.start_process)
    add("stop_managed_process", "仅停止本 Agent 通过 launch_app/start_process 启动并仍在管理的进程。", _schema({"pid": integer}, ["pid"]), processes.stop_managed_process)
    add("wait", "等待短时间，最多 30 秒；/stop 可立即中止等待。", _schema({"seconds": {"type": "number"}}, ["seconds"]), partial(wait, stop_event=session.stop_event))
    return registry
