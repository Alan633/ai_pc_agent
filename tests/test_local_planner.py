from __future__ import annotations

from agent.planner import OpenAIPlanner


def _planner() -> OpenAIPlanner:
    return OpenAIPlanner(
        api_key="",
        model="qwen3.5:9b",
        base_url="http://127.0.0.1:11434/v1",
        tool_schemas=[
            {
                "type": "function",
                "name": "launch_app",
                "description": "启动应用",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    )


def test_local_json_tool_fallback_accepts_registered_tool() -> None:
    planner = _planner()
    call = planner._parse_text_tool_call(
        '我来执行：{"tool":"launch_app","arguments":{"app_name":"wechat"}}'
    )
    assert call is not None
    assert call.name == "launch_app"
    assert call.arguments == {"app_name": "wechat"}


def test_local_json_tool_fallback_rejects_unknown_tool() -> None:
    planner = _planner()
    call = planner._parse_text_tool_call(
        '{"tool":"run_shell","arguments":{"command":"whoami"}}'
    )
    assert call is None


def test_chat_tool_schema_conversion() -> None:
    planner = _planner()
    tool = planner.chat_tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "launch_app"
