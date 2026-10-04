from __future__ import annotations

import platform
from datetime import UTC, datetime
from typing import Any, Callable

import psutil


def get_system_status(
    agent_status: Callable[[], dict[str, Any]], started_at: datetime
) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)
    now = datetime.now(UTC)
    return {
        "hostname": platform.node(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": memory.percent,
        "memory_used_mb": round(memory.used / 1024**2),
        "memory_total_mb": round(memory.total / 1024**2),
        "windows_uptime_seconds": round((now - boot_time).total_seconds()),
        "agent_uptime_seconds": round((now - started_at).total_seconds()),
        "agent": agent_status(),
    }
