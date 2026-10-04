from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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


def assert_game_foreground(title_keyword: str) -> str:
    _, title, _ = _foreground_window()
    if title_keyword.lower() not in title.lower():
        raise RuntimeError(
            f"前台窗口已切换：{title!r}；为避免误点，已停止本次游戏操作"
        )
    return title


def capture_game_window(output_dir: Path, title_keyword: str) -> CapturedFrame:
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
            f"游戏窗口超出主屏边界：window=({left},{top},{right},{bottom}) screen={screen.width}x{screen.height}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baque_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    image = pyautogui.screenshot(region=(left, top, width, height))
    image.save(path, "PNG")
    return CapturedFrame(path, left, top, width, height, title)
