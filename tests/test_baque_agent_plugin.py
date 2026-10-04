from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agent.session import AgentSession
from agent.tool_registry import ToolRegistry
from game_baque.acceptance import BaqueAcceptanceTracker
from plugins.baque import BaquePlugin


class FakeTracker:
    def unseen_counts(self) -> Counter[str]:
        return Counter({"6:spade": 1, "7:heart": 2})


class FakeGame:
    action_count = 20
    opponent_event_count = 30
    tracker = FakeTracker()
    last_message = "已出牌"

    def quality_metrics(self) -> dict[str, object]:
        return {
            "inspect_count": 40,
            "unknown_snapshot_count": 1,
            "step_count": 30,
            "validation_rejection_count": 1,
            "action_count": 20,
            "opponent_event_count": 30,
        }

    def state_text(self) -> str:
        return "牌面正常"


@dataclass
class FakeObserverStatus:
    running: bool = True
    directory: str | None = "x"
    sampled_frames: int = 500
    episodes_detected: int = 40
    global_effects_skipped: int = 5
    classified: int = 35
    applied: int = 32
    low_confidence: int = 3
    queue_size: int = 0
    paused_reason: str | None = None
    recent: tuple[str, ...] = ()


class FakeObserver:
    def status(self) -> FakeObserverStatus:
        return FakeObserverStatus()


def test_baque_plugin_registers_agent_tools(tmp_path: Path) -> None:
    plugin = BaquePlugin(
        game=None,
        recorder=SimpleNamespace(running=False),
        observer=None,
        session=AgentSession(),
        acceptance=BaqueAcceptanceTracker(tmp_path),
    )
    registry = ToolRegistry()
    plugin.register_tools(registry)
    assert set(registry.names()) == {
        "baque_get_state",
        "baque_step",
        "baque_watch_start",
        "baque_watch_status",
        "baque_watch_stop",
        "baque_auto_start",
        "baque_auto_stop",
        "baque_acceptance_report",
        "baque_acceptance_mark",
        "baque_acceptance_reset",
    }


def test_acceptance_report_reaches_pass_with_ground_truth(tmp_path: Path) -> None:
    tracker = BaqueAcceptanceTracker(tmp_path)
    for _ in range(20):
        tracker.mark("own_state", True)
    for _ in range(30):
        tracker.mark("opponent_event", True)
    for _ in range(20):
        tracker.mark("action", True)

    report = tracker.report(FakeGame(), FakeObserver())
    assert report["verdict"] == "PASS"
    assert report["manual_accuracy"]["own_state"]["accuracy"] == 1.0
    assert report["automatic_checks"]["tracker_consistency"]["status"] == "PASS"


def test_acceptance_report_requires_real_labels(tmp_path: Path) -> None:
    tracker = BaqueAcceptanceTracker(tmp_path)
    report = tracker.report(FakeGame(), FakeObserver())
    assert report["verdict"] == "INSUFFICIENT_DATA"
