from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GameWindow:
    origin_x: int
    origin_y: int
    width: int
    height: int
    title: str


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    path: Path
    origin_x: int
    origin_y: int
    width: int
    height: int
    window_title: str


def _foreground_window() -> tuple[int, str, tuple[int, int, int, int]]:
    user32 = ctypes.windll.user32
    hwnd = int(user32.GetForegroundWindow())
    if not hwnd:
        raise RuntimeError("没有可用的前台窗口")

    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    title = buffer.value

    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("无法获取前台窗口坐标")
    return hwnd, title, (rect.left, rect.top, rect.right, rect.bottom)


def get_game_window(title_keyword: str) -> GameWindow:
    """返回当前前台游戏窗口的位置，并统一做窗口/主屏范围校验。"""
    import pyautogui

    _, title, (left, top, right, bottom) = _foreground_window()
    if title_keyword.lower() not in title.lower():
        raise RuntimeError(
            f"前台窗口不是目标游戏：{title!r}；请先把包含“{title_keyword}”的游戏窗口切到前台"
        )

    width = right - left
    height = bottom - top
    if width < 800 or height < 500:
        raise RuntimeError(f"游戏窗口尺寸异常：{width}x{height}")
    if left < 0 or top < 0:
        raise RuntimeError("第一版暂不支持位于主屏负坐标区域的游戏窗口，请移到主屏")

    screen = pyautogui.size()
    if right > screen.width or bottom > screen.height:
        raise RuntimeError(
            f"游戏窗口超出主屏边界：window=({left},{top},{right},{bottom}) "
            f"screen={screen.width}x{screen.height}"
        )
    return GameWindow(left, top, width, height, title)


def grab_game_window(title_keyword: str) -> tuple[GameWindow, Any]:
    """抓取当前前台游戏窗口，返回窗口元数据与 PIL.Image。

    该函数不落盘，供高频录像/差分检测使用。
    """
    import pyautogui

    window = get_game_window(title_keyword)
    image = pyautogui.screenshot(
        region=(window.origin_x, window.origin_y, window.width, window.height)
    )
    return window, image


def assert_game_foreground(title_keyword: str) -> str:
    return get_game_window(title_keyword).title


def capture_game_window(output_dir: Path, title_keyword: str) -> CapturedFrame:
    window, image = grab_game_window(title_keyword)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baque_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    image.save(path, "PNG")
    return CapturedFrame(
        path,
        window.origin_x,
        window.origin_y,
        window.width,
        window.height,
        window.title,
    )
