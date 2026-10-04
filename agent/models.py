from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    artifact: Path | None = None

    def to_model_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact"] = str(self.artifact) if self.artifact else None
        return value


@dataclass(slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class PlanResponse:
    response_id: str
    calls: list[ToolCall]
    final_text: str | None = None


@dataclass(slots=True)
class TaskOutcome:
    ok: bool
    message: str
    steps: int
    artifacts: list[Path] = field(default_factory=list)

