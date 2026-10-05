from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from game_baque.capture import (
    click_screen_native,
    focus_game_window,
    get_game_window,
)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    path: Path
    origin_x: int
    origin_y: int
    width: int
    height: int
    window_title: str


def focus_balatro_window(title_keyword: str) -> str:
    return focus_game_window(title_keyword)


def capture_balatro_window(
    output_dir: Path,
    title_keyword: str,
) -> CapturedFrame:
    import pyautogui

    window = get_game_window(title_keyword)
    image: Any = pyautogui.screenshot(
        region=(
            window.origin_x,
            window.origin_y,
            window.width,
            window.height,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"balatro_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    image.save(path, "PNG")
    return CapturedFrame(
        path=path,
        origin_x=window.origin_x,
        origin_y=window.origin_y,
        width=window.width,
        height=window.height,
        window_title=window.title,
    )


def click_frame_point(
    frame: CapturedFrame,
    x: int,
    y: int,
) -> None:
    if not (0 <= x < frame.width and 0 <= y < frame.height):
        raise RuntimeError(
            f"Balatro 点击坐标超出窗口：({x},{y}) / "
            f"{frame.width}x{frame.height}"
        )
    click_screen_native(
        frame.origin_x + x,
        frame.origin_y + y,
    )
