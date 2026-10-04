from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.models import ToolResult


def enable_high_dpi_awareness() -> None:
    """尽早启用每显示器 DPI 感知，避免截图坐标与鼠标坐标不一致。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def get_screen_size() -> tuple[int, int]:
    import pyautogui

    size = pyautogui.size()
    return int(size.width), int(size.height)


def take_screenshot(output_dir: Path) -> ToolResult:
    import pyautogui

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screen_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    image = pyautogui.screenshot()
    image.save(path, "PNG")
    width, height = image.size
    return ToolResult(
        True,
        "take_screenshot",
        data={"path": str(path), "width": width, "height": height, "coordinate_space": "native_pixels"},
        artifact=path,
    )


def map_image_coordinates(
    x: float, y: float, image_width: int, image_height: int, screen_width: int, screen_height: int
) -> tuple[int, int]:
    if min(image_width, image_height, screen_width, screen_height) <= 0:
        raise ValueError("屏幕和图片尺寸必须大于 0")
    mapped_x = round(x * screen_width / image_width)
    mapped_y = round(y * screen_height / image_height)
    return (
        max(0, min(mapped_x, screen_width - 1)),
        max(0, min(mapped_y, screen_height - 1)),
    )

