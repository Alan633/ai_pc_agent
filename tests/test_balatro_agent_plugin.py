from __future__ import annotations

from pathlib import Path

from agent.session import AgentSession
from agent.tool_registry import ToolRegistry
from config import Settings
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


def test_balatro_plugin_builds_local_hybrid_strategist(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        telegram_bot_token="",
        allowed_telegram_user_id=0,
        openai_api_key="",
        openai_model="local",
        playwright_headless=True,
        max_steps=5,
        confirm_timeout_seconds=30,
        allowed_roots=(tmp_path,),
        allowed_executables=frozenset(),
        balatro_vision_model="qwen3.5-9b-16k",
        balatro_vision_base_url="http://127.0.0.1:11434/v1",
        balatro_decision_mode="hybrid",
        balatro_decision_model="qwen3.5-9b-16k",
        balatro_decision_base_url="http://127.0.0.1:11434/v1",
    )

    plugin = BalatroPlugin.build(settings, AgentSession())

    assert plugin.game is not None
    assert plugin.game.decision_mode == "hybrid"
    assert plugin.game.strategist is not None
    assert plugin.game.control_mode == "auto"
    assert plugin.game.balatrobot is not None
