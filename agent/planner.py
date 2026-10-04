from __future__ import annotations

import json
import re
import uuid
from typing import Any

from openai import AsyncOpenAI

from agent.models import PlanResponse, ToolCall, ToolResult


SYSTEM_INSTRUCTIONS = """你是运行在用户 Windows 11 电脑上的本地 AI PC Agent 规划器，不是普通聊天机器人。

你已经通过宿主程序获得了真实的电脑工具能力。只要工具列表里存在对应能力，就必须优先调用工具验证或执行，不能回答“我无法查看设备状态”“我无法控制本地应用”“请你自己去任务管理器查看”等泛化拒绝。

核心原则：
1. 用户问电脑当前状态、应用是否运行、当前窗口等实时信息：先调用工具查询，再依据工具结果回答。
2. 用户要求打开应用：优先 is_app_running；若已运行则 focus_app；若未运行则 launch_app，必要时 wait 后再 focus_app。
3. 用户要求打开微信/WeChat/Weixin，统一优先使用 app_name="wechat"；抖音/Douyin 使用 app_name="douyin"。
4. 用户要求“切到/显示/置前”某应用：优先 focus_app，不要用鼠标坐标猜。
5. 用户要求看屏幕：使用 take_screenshot。用户明确给出坐标时才使用 click/move_mouse；不要自行从文字臆测坐标。
6. 优先级：Windows 系统/API 或进程工具 > Windows UI Automation > 浏览器自动化 > 鼠标键盘。
7. 每一步选择最少且可验证的下一步；工具失败后可以换一种安全方法，但不要重复无效动作。
8. 任务完成后用简短中文说明实际结果。没有成功执行时，不得声称已完成。
9. “八雀 / 百变八雀 / 点字成金”属于已注册插件能力：
   - 看状态、识别牌面、问现在怎么打：baque_get_state；
   - 只记牌、只观察、用户自己操作：baque_watch_start；
   - 查看三家记牌和观察质量：baque_watch_status；
   - 明确要求只走一步：baque_step；
   - 明确要求连续自动玩：baque_auto_start；
   - 停止自动游戏：baque_auto_stop；
   - “刚才识别对/错、刚才点击对/错”应调用 baque_acceptance_mark 记录真值；
   - 问准确率、稳定性、是否可以无人值守：先调用 baque_acceptance_report，再依据报告回答。
10. 八雀自动游戏已经有独立停止逻辑。启动后不要再用通用 click/type_text 干预牌桌，除非用户明确要求手动调试。

安全限制：
- 只能调用提供的白名单工具。
- 绝不生成或请求执行 Python、PowerShell、CMD、eval、exec、Shell 命令或脚本。
- cmd 和 powershell 只能通过 launch_app 打开空窗口，禁止传入或输入命令。
- 启动自定义程序时，program/app_name 必须是 apps.yaml 中允许的名称或别名。
- 禁止用 click、type_text、hotkey、UIA 或浏览器绕过安全确认去完成删除、付款、转账、真实下单、发送消息、上传敏感文件、修改生产数据库或系统配置等高风险目标；没有专用且可确认的工具时必须说明不支持。
"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpenAIPlanner:
    """同时兼容 OpenAI Responses API 与 Ollama/OpenAI-compatible Chat Completions。

    当配置 OPENAI_BASE_URL 时默认走 Chat Completions。这样 qwen3.5:9b 等本地模型
    可以稳定使用 function calling；若模型没有按原生 tool_calls 返回，而是输出单个
    JSON 工具请求，还会做一次受限解析兜底。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        tool_schemas: list[dict[str, Any]],
        base_url: str = "",
    ) -> None:
        client_key = api_key or ("local-model" if base_url else "")
        client_options: dict[str, Any] = {"api_key": client_key}
        if base_url:
            client_options["base_url"] = base_url
        self.client = AsyncOpenAI(**client_options) if client_key else None
        self.model = model
        self.tool_schemas = tool_schemas
        self.use_chat_completions = bool(base_url)
        self.chat_tools = [self._to_chat_tool(schema) for schema in tool_schemas]
        self._chat_histories: dict[str, list[dict[str, Any]]] = {}

    def discard(self, response_id: str | None) -> None:
        """丢弃未完成的本地模型会话，供任务取消和步数耗尽时清理。"""
        if response_id:
            self._chat_histories.pop(response_id, None)

    async def first(self, user_text: str) -> PlanResponse:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY 或 OPENAI_BASE_URL 尚未配置；Telegram 基础指令仍可正常使用。")
        if self.use_chat_completions:
            history: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_text},
            ]
            return await self._chat_complete(history)

        response = await self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=user_text,
            tools=self.tool_schemas,
            parallel_tool_calls=False,
        )
        return self._parse_response(response)

    async def next(
        self,
        previous_response_id: str,
        call_id: str,
        result: ToolResult,
    ) -> PlanResponse:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY 或 OPENAI_BASE_URL 尚未配置")

        if self.use_chat_completions:
            history = self._chat_histories.pop(previous_response_id, None)
            if history is None:
                raise RuntimeError("本地模型会话状态已丢失，请重新发送任务。")
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result.to_model_dict(), ensure_ascii=False, default=str),
                }
            )
            return await self._chat_complete(history)

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
        return self._parse_response(response)

    async def _chat_complete(self, history: list[dict[str, Any]]) -> PlanResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=history,
            tools=self.chat_tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0,
        )
        if not response.choices:
            raise RuntimeError("本地模型没有返回任何结果")
        message = response.choices[0].message
        response_id = getattr(response, "id", None) or f"chat-{uuid.uuid4().hex}"

        calls: list[ToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for raw_call in message.tool_calls or []:
            arguments = self._decode_arguments(raw_call.function.arguments)
            calls.append(ToolCall(raw_call.id, raw_call.function.name, arguments))
            assistant_tool_calls.append(
                {
                    "id": raw_call.id,
                    "type": "function",
                    "function": {
                        "name": raw_call.function.name,
                        "arguments": raw_call.function.arguments or "{}",
                    },
                }
            )

        final_text = (message.content or "").strip() or None
        if not calls and final_text:
            fallback = self._parse_text_tool_call(final_text)
            if fallback is not None:
                calls = [fallback]
                assistant_tool_calls = [
                    {
                        "id": fallback.call_id,
                        "type": "function",
                        "function": {
                            "name": fallback.name,
                            "arguments": json.dumps(fallback.arguments, ensure_ascii=False),
                        },
                    }
                ]
                final_text = None

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        history.append(assistant_message)
        # 只有仍需回传工具结果时才保留会话，避免已完成任务长期占用内存。
        if calls:
            self._chat_histories[response_id] = history
        return PlanResponse(response_id, calls, final_text)

    def _parse_text_tool_call(self, text: str) -> ToolCall | None:
        """只接受单个、已注册工具的 JSON 兜底，避免把普通文本当命令执行。"""
        match = _JSON_BLOCK.search(text)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("tool") or payload.get("name")
        arguments = payload.get("arguments", payload.get("args", {}))
        allowed = {schema.get("name") for schema in self.tool_schemas}
        if not isinstance(name, str) or name not in allowed or not isinstance(arguments, dict):
            return None
        return ToolCall(f"local-{uuid.uuid4().hex}", name, arguments)

    @staticmethod
    def _decode_arguments(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _to_chat_tool(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _parse_response(response: Any) -> PlanResponse:
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
