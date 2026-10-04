from __future__ import annotations

import pyautogui


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


def _validated_point(x: int, y: int) -> tuple[int, int]:
    width, height = pyautogui.size()
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"坐标超出屏幕范围：({x}, {y})，屏幕为 {width}x{height}")
    return x, y


def get_mouse_position() -> dict[str, int]:
    point = pyautogui.position()
    return {"x": int(point.x), "y": int(point.y)}


def click(x: int, y: int) -> dict[str, object]:
    pyautogui.click(*_validated_point(x, y))
    return {"clicked": True, "x": x, "y": y}


def double_click(x: int, y: int) -> dict[str, object]:
    pyautogui.doubleClick(*_validated_point(x, y), interval=0.12)
    return {"double_clicked": True, "x": x, "y": y}


def move_mouse(x: int, y: int) -> dict[str, object]:
    pyautogui.moveTo(*_validated_point(x, y), duration=0.2)
    return {"moved": True, "x": x, "y": y}


def scroll(amount: int) -> dict[str, int]:
    pyautogui.scroll(amount)
    return {"scrolled": amount}

