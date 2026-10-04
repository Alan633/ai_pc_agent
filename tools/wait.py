from __future__ import annotations

import time
import threading


def wait(seconds: float, stop_event: threading.Event | None = None) -> dict[str, float | bool]:
    duration = max(0.0, min(float(seconds), 30.0))
    started = time.monotonic()
    stopped = stop_event.wait(duration) if stop_event else False
    return {"waited_seconds": round(time.monotonic() - started, 3), "stopped": stopped}
