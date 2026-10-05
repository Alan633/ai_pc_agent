from __future__ import annotations

import ctypes
import time
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


@dataclass(frozen=True, slots=True)
class TurnProbe:
    action_state: str | None
    round_in_progress: bool
    frame: CapturedFrame | None


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("type", wintypes.DWORD), ("value", _InputUnion)]


_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


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


def focus_game_window(title_keyword: str) -> str:
    """按标题找到欢乐斗地主窗口并切到前台。

    Telegram/本地模型命令可能在抖音、IDE 等窗口前台发出；八雀自动模式不能
    要求用户手动切窗，因此启动时先主动恢复并聚焦目标游戏窗口。
    """
    from pywinauto import Desktop

    keyword = title_keyword.strip().casefold()
    if not keyword:
        raise ValueError("游戏窗口标题关键字不能为空")

    matches = []
    for window in Desktop(backend="win32").windows():
        try:
            title = window.window_text().strip()
            if not title or keyword not in title.casefold() or not window.is_visible():
                continue
            rect = window.rectangle()
            area = max(0, rect.width()) * max(0, rect.height())
            matches.append((area, window, title))
        except Exception:
            continue

    if not matches:
        raise RuntimeError(f"没有找到包含“{title_keyword}”的游戏窗口")

    _, window, title = max(matches, key=lambda item: item[0])
    try:
        if window.is_minimized():
            window.restore()
    except Exception:
        pass
    window.set_focus()
    return title


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


def click_screen_native(x: int, y: int) -> None:
    """使用 Win32 SendInput 模拟一次真实鼠标点击。

    腾讯游戏窗口可能忽略 PyAutoGUI 使用的旧式 mouse_event；SendInput 对
    DirectX/游戏窗口更可靠，并支持多显示器虚拟桌面的绝对坐标。
    """
    user32 = ctypes.windll.user32
    virtual_x = int(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN))
    virtual_y = int(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN))
    virtual_width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    virtual_height = int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    if virtual_width <= 1 or virtual_height <= 1:
        raise RuntimeError("无法读取 Windows 虚拟桌面尺寸")
    if not (
        virtual_x <= x < virtual_x + virtual_width
        and virtual_y <= y < virtual_y + virtual_height
    ):
        raise RuntimeError(
            f"点击坐标超出虚拟桌面：({x},{y}) / "
            f"({virtual_x},{virtual_y},{virtual_width},{virtual_height})"
        )

    normalized_x = round((x - virtual_x) * 65535 / (virtual_width - 1))
    normalized_y = round((y - virtual_y) * 65535 / (virtual_height - 1))

    def send(flags: int) -> None:
        value = _Input(
            type=_INPUT_MOUSE,
            value=_InputUnion(
                mi=_MouseInput(
                    dx=normalized_x,
                    dy=normalized_y,
                    mouseData=0,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        sent = int(user32.SendInput(1, ctypes.byref(value), ctypes.sizeof(_Input)))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    absolute = _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK
    send(absolute | _MOUSEEVENTF_MOVE)
    time.sleep(0.08)
    send(absolute | _MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.12)
    send(absolute | _MOUSEEVENTF_LEFTUP)


def _save_frame(output_dir: Path, window: GameWindow, image: Any) -> CapturedFrame:
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


def capture_game_window(output_dir: Path, title_keyword: str) -> CapturedFrame:
    window, image = grab_game_window(title_keyword)
    return _save_frame(output_dir, window, image)


def capture_game_window_if_swap(
    output_dir: Path,
    title_keyword: str,
) -> CapturedFrame | None:
    """快速检测换牌页，仅命中时保存截图。

    换牌倒计时很短，而本地视觉模型一次识别可能需要数秒。这里利用固定游戏
    布局中灰色“换牌”按钮的中性灰像素占比，在发牌动画期间只做廉价截图轮询，
    避免把模型时间浪费在不可操作的动画帧上。
    """
    window, image = grab_game_window(title_keyword)
    _, gray_ratio = _action_button_ratios(image)
    if gray_ratio < BUTTON_GRAY_RATIO:
        return None
    return _save_frame(output_dir, window, image)


# 只取“换牌/出牌”按钮最右侧。摸牌阶段倒计时居中，实测占 x=0.475~0.522；
# 倒计时有多种皮肤（小鸭子、橙色闹钟等），不能依赖其外观——橙色闹钟皮肤
# 曾让整块按钮区橙色占比达 0.21 而被误判为“出牌”。这里与倒计时留出约
# 0.04 屏宽余量；实测按钮橙 ≥0.85、灰 ≥0.90，倒计时 0。
ACTION_BUTTON_BOX = (0.56, 0.66, 0.58, 0.71)
BUTTON_ORANGE_RATIO = 0.40
BUTTON_GRAY_RATIO = 0.50


def _action_button_ratios(image: Any) -> tuple[float, float]:
    """返回按钮区域的（橙色占比, 中性灰占比）。"""
    image = image.convert("RGB")
    width, height = image.size
    left, top, right, bottom = ACTION_BUTTON_BOX
    pixels = list(
        image.crop(
            (
                round(width * left),
                round(height * top),
                round(width * right),
                round(height * bottom),
            )
        ).getdata()
    )
    if not pixels:
        return 0.0, 0.0
    orange = sum(
        1
        for red, green, blue in pixels
        if red > 180 and 60 < green < 210 and blue < 100
    )
    gray = sum(
        1
        for red, green, blue in pixels
        if 80 <= red <= 235
        and abs(red - green) < 18
        and abs(green - blue) < 18
    )
    return orange / len(pixels), gray / len(pixels)


def detect_action_state(image: Any) -> str | None:
    """用固定 UI 色块快速判断当前玩家是否有操作权。"""
    image = image.convert("RGB")
    width, height = image.size
    banner = list(
        image.crop(
            (
                round(width * 0.30),
                round(height * 0.55),
                round(width * 0.70),
                round(height * 0.62),
            )
        ).getdata()
    )
    if not banner:
        return None

    blue_ratio = sum(
        1
        for red, green, blue in banner
        if blue > 150 and blue > red * 1.25 and blue > green * 1.05
    ) / len(banner)
    if blue_ratio < 0.25:
        # 没有“请选择……”蓝色横幅时，不是自己的操作回合。
        return None

    orange_ratio, gray_ratio = _action_button_ratios(image)
    if orange_ratio >= BUTTON_ORANGE_RATIO:
        return "discard"
    if gray_ratio >= BUTTON_GRAY_RATIO:
        return "swap"
    return "draw"


def current_game_action_state(title_keyword: str) -> str | None:
    """读取当前游戏帧；None 表示不是自己的操作回合。"""
    _, image = grab_game_window(title_keyword)
    return detect_action_state(image)


def detect_round_in_progress(image: Any) -> bool:
    """中央蓝色背面牌堆可见即视为牌局进行中。

    结算页、开始页、豆子弹窗遮罩下牌堆都不可见（实测蓝色占比 0~0.46，
    牌局中稳定约 0.87），可据此决定是否需要调用视觉模型识别非回合画面。
    """
    image = image.convert("RGB")
    width, height = image.size
    pixels = list(
        image.crop(
            (
                round(width * 0.368),
                round(height * 0.34),
                round(width * 0.40),
                round(height * 0.43),
            )
        ).getdata()
    )
    if not pixels:
        return False
    blue_ratio = sum(
        1
        for red, green, blue in pixels
        if blue > 170 and blue > red * 1.3 and blue > green * 1.1
    ) / len(pixels)
    return blue_ratio >= 0.70


def probe_game_turn(output_dir: Path, title_keyword: str) -> TurnProbe:
    """廉价探测一帧：只在轮到自己操作时才落盘，供视觉模型识别。"""
    window, image = grab_game_window(title_keyword)
    action_state = detect_action_state(image)
    frame = (
        _save_frame(output_dir, window, image)
        if action_state is not None
        else None
    )
    return TurnProbe(action_state, detect_round_in_progress(image), frame)
