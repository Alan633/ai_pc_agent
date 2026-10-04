from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from agent.models import PlanResponse, ToolCall, ToolResult


SYSTEM_INSTRUCTIONS = """你是 Windows 11 PC Agent 的规划器。你只能调用提供的白名单工具，绝不生成或请求执行 Python、PowerShell、CMD、eval、exec、Shell 命令或脚本。
优先级：本地系统 API/subprocess > Playwright > Windows UI Automation > PyAutoGUI。当前阶段不要进行视觉坐标推断。
意图映射：看屏幕用 take_screenshot；电脑状态用 get_system_status；打开已知应用用 launch_app；打开网站用 open_url；查看进程用 list_processes。
cmd 和 powershell 只能通过 launch_app 打开空窗口，禁止传入或输入命令。启动自定义程序时，program 必须是 apps.yaml 中的名称。
每次根据用户目标和工具结果选择最少且可验证的下一步。任务完成后用简短中文说明结果，不再调用工具。
禁止用 click、type_text、hotkey、UIA 或 Playwright 绕过安全确认去完成删除、付款、转账、真实下单、发送消息、上传敏感文件、修改生产数据库或系统配置等高风险目标；没有专用且可确认的工具时必须说明不支持。
失败时可换一种安全方法，但不要重复无效动作。涉及屏幕坐标时先截图，并使用截图的原始像素尺寸。"""


class OpenAIPlanner:
    def __init__(
        self,
        api_key: str,
        model: str,
        tool_schemas: list[dict[str, Any]],
        base_url: str = "",
    ) -> None:
        # Ollama and other OpenAI-compatible local servers usually do not verify
        # the key, but the OpenAI SDK still requires a non-empty value.
        client_key = api_key or ("local-model" if base_url else "")
        client_options: dict[str, Any] = {"api_key": client_key}
        if base_url:
            client_options["base_url"] = base_url
        self.client = AsyncOpenAI(**client_options) if client_key else None
        self.model = model
        self.tool_schemas = tool_schemas

    async def first(self, user_text: str) -> PlanResponse:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY 尚未配置；Telegram 基础指令仍可正常使用。")
        response = await self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=user_text,
            tools=self.tool_schemas,
            parallel_tool_calls=False,
        )
        return self._parse(response)

    async def next(
        self,
        previous_response_id: str,
        call_id: str,
        result: ToolResult,
    ) -> PlanResponse:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY 尚未配置")
        items: list[dict[str, Any]] = [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result.to_model_dict(), ensure_ascii=False, default=str),
            }
        ]
        response = await self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            previous_response_id=previous_response_id,
            input=items,
            tools=self.tool_schemas,
            parallel_tool_calls=False,
        )
        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> PlanResponse:
        calls: list[ToolCall] = []
        texts: list[str] = []
        for item in response.output:
            if item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                calls.append(ToolCall(item.call_id, item.name, arguments))
            elif item.type == "message":
                for content in item.content:
                    if getattr(content, "type", "") == "output_text":
                        texts.append(content.text)
        return PlanResponse(response.id, calls, "\n".join(texts).strip() or None)
