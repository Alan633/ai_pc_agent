from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable

from agent.models import ToolResult

LOGGER = logging.getLogger(__name__)
ToolCallable = Callable[..., ToolResult | dict[str, Any] | Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolCallable

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册：{spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult(False, name, error="工具不存在或未获授权")
        try:
            _validate_arguments(arguments, spec.parameters)
            inspect.signature(spec.function).bind(**arguments)
            LOGGER.info("执行工具 tool=%s args=%s", name, _safe_arguments(arguments))
            value = spec.function(**arguments)
            if isinstance(value, ToolResult):
                result = value
            elif isinstance(value, dict):
                result = ToolResult(True, name, data=value)
            else:
                result = ToolResult(True, name, data={"value": value})
            LOGGER.info("工具结果 tool=%s ok=%s", name, result.ok)
            return result
        except Exception as exc:  # 单个工具失败不得使 Agent 退出
            LOGGER.exception("工具失败 tool=%s", name)
            return ToolResult(False, name, error=f"{type(exc).__name__}: {exc}")


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(arguments)
    for key in tuple(redacted):
        if any(word in key.lower() for word in ("token", "password", "secret", "api_key")):
            redacted[key] = "***"
        elif key.lower() in {"text", "content"}:
            redacted[key] = f"<文本长度 {len(str(redacted[key]))}>"
    return redacted


def _validate_arguments(arguments: Any, schema: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise TypeError("工具参数必须是 JSON 对象")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    extra = arguments.keys() - properties.keys()
    if missing:
        raise ValueError(f"缺少必需参数：{', '.join(sorted(missing))}")
    if schema.get("additionalProperties") is False and extra:
        raise ValueError(f"包含未声明参数：{', '.join(sorted(extra))}")
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        allowed = set(expected if isinstance(expected, list) else [expected])
        if value is None and "null" in allowed:
            continue
        valid = (
            ("string" in allowed and isinstance(value, str))
            or ("integer" in allowed and isinstance(value, int) and not isinstance(value, bool))
            or ("number" in allowed and isinstance(value, (int, float)) and not isinstance(value, bool))
            or ("boolean" in allowed and isinstance(value, bool))
            or ("array" in allowed and isinstance(value, list))
            or ("object" in allowed and isinstance(value, dict))
        )
        if expected and not valid:
            raise TypeError(f"参数 {key} 类型错误")
        if "array" in allowed and isinstance(value, list):
            item_type = properties[key].get("items", {}).get("type")
            if item_type == "string" and any(not isinstance(item, str) for item in value):
                raise TypeError(f"参数 {key} 的数组元素必须是字符串")
            minimum = properties[key].get("minItems")
            maximum = properties[key].get("maxItems")
            if minimum is not None and len(value) < minimum:
                raise ValueError(f"参数 {key} 元素过少")
            if maximum is not None and len(value) > maximum:
                raise ValueError(f"参数 {key} 元素过多")
