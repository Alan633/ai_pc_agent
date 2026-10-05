from __future__ import annotations

from agent.session import AgentSession
from agent.tool_registry import ToolRegistry
from plugins.balatro import BalatroPlugin


def test_balatro_plugin_registers_agent_tools() -> None:
    plugin = BalatroPlugin(
        game=None,
        session=AgentSession(),
    )
    registry = ToolRegistry()
    plugin.register_tools(registry)
    assert set(registry.names()) == {
        "balatro_get_state",
        "balatro_step",
        "balatro_auto_start",
        "balatro_auto_stop",
        "balatro_stats",
    }
