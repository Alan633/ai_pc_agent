from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from game_baque.models import Point

from game_baque.fast_vision import (
    detect_action_state as cv_detect_action_state,
    detect_round_in_progress as cv_detect_round_in_progress,
    detect_hu_settlement,
)


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
    opportunity_card: bool = False
    opportunity_point: Point | None = None
    # 牌堆 + 亮牌行区域的缩略灰度图，用来判断对手回合里亮牌是否变化、是否已稳定。
    table_signature: bytes | None = None
    # 胡牌结算横幅：赢家座位（right/left/top/self/unknown），没有结算时为 None。
    settlement: str | None = None


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
_DWMWA_EXTENDED_FRAME_BOUNDS = 9


def _visible_window_rect(
    hwnd: int,
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """返回 DWM 实际绘制的窗口边界。

    GetWindowRect 会把 Windows 的不可见缩放边框算进去。窗口贴边或最大化时，
    这些边框通常会越过显示器约 8~12 像素，不能拿它们判断画面是否越界。
    """
    try:
        rect = wintypes.RECT()
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd,
            _DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return rect.left, rect.top, rect.right, rect.bottom
    except (AttributeError, OSError):
        pass
    return fallback


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
    bounds = (rect.left, rect.top, rect.right, rect.bottom)
    return hwnd, title, _visible_window_rect(hwnd, bounds)


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
    """返回当前前台游戏窗口的可见位置，并校验虚拟桌面范围。"""
    _, title, (left, top, right, bottom) = _foreground_window()
    if title_keyword.lower() not in title.lower():
        raise RuntimeError(
            f"前台窗口不是目标游戏：{title!r}；请先把包含“{title_keyword}”的游戏窗口切到前台"
        )

    width = right - left
    height = bottom - top
    if width < 800 or height < 500:
        raise RuntimeError(f"游戏窗口尺寸异常：{width}x{height}")
    user32 = ctypes.windll.user32
    virtual_left = int(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN))
    virtual_top = int(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN))
    virtual_width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    virtual_height = int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    virtual_right = virtual_left + virtual_width
    virtual_bottom = virtual_top + virtual_height
    if (
        left < virtual_left
        or top < virtual_top
        or right > virtual_right
        or bottom > virtual_bottom
    ):
        raise RuntimeError(
            f"游戏窗口超出虚拟桌面边界：window=({left},{top},{right},{bottom}) "
            f"desktop=({virtual_left},{virtual_top},{virtual_right},{virtual_bottom})"
        )
    return GameWindow(left, top, width, height, title)


def grab_window_image(window: GameWindow) -> Any:
    """按屏幕坐标抓取窗口可见区域，兼容负坐标副屏。"""
    from PIL import ImageGrab

    return ImageGrab.grab(
        bbox=(
            window.origin_x,
            window.origin_y,
            window.origin_x + window.width,
            window.origin_y + window.height,
        ),
        all_screens=True,
    )


def grab_game_window(title_keyword: str) -> tuple[GameWindow, Any]:
    """抓取当前前台游戏窗口，返回窗口元数据与 PIL.Image。

    该函数不落盘，供高频录像/差分检测使用。
    """
    window = get_game_window(title_keyword)
    image = grab_window_image(window)
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
    if cv_detect_action_state(image) != "swap":
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
    """OpenCV 快速判断当前玩家是否有操作权。"""
    return cv_detect_action_state(image)


def detect_table_felt_visible(image: Any) -> bool:
    """牌桌两侧绿色台面是否正常可见（未被弹窗遮罩压暗或覆盖）。

    “豆子不足啦”购买弹窗、送豆弹窗等会压暗整张牌桌，其上的黄色“💎30”
    购买按钮形状颜色都接近“出牌”按钮。取样区在手牌两侧（0.64~0.74 屏高）：
    听牌提示面板在听口多时几乎横贯全屏（实测到 0.585 屏高），曾盖住更高处
    的取样区而误判为遮罩；而购买/送豆弹窗在此处仍会覆盖或压暗台面。
    """
    from PIL import ImageStat

    image = image.convert("RGB")
    width, height = image.size
    for left, top, right, bottom in ((0.14, 0.64, 0.22, 0.74), (0.78, 0.64, 0.86, 0.74)):
        red, green, _ = ImageStat.Stat(
            image.crop(
                (
                    round(width * left),
                    round(height * top),
                    round(width * right),
                    round(height * bottom),
                )
            )
        ).mean
        if not (95 <= green <= 140 and red < green - 40):
            return False
    return True


def find_action_buttons(image: Any) -> dict[str, Point]:
    """按颜色+形状在操作区找“出牌”（橙）与“胡”（红）按钮，返回中心像素坐标。

    按钮位置不固定：能胡时布局变为 [倒计时][出牌][胡]，出牌左移、胡占据
    原出牌位置。药丸形宽高比 1.5~4 可排除圆形倒计时（含橙色闹钟皮肤）；
    宽度 0.07~0.12 屏宽可排除更宽的“开始游戏”(0.15)、“确定”(0.13)。
    实测出牌 (255,158~210,57~77)，胡 (247~255,85~154,77~123)。
    """
    import cv2
    import numpy as np

    image = image.convert("RGB")
    if not detect_table_felt_visible(image):
        # 安全：牌桌被弹窗遮罩时绝不返回任何可点击按钮（可能是购买按钮）。
        return {}
    width, height = image.size
    left, top = round(width * 0.36), round(height * 0.60)
    right, bottom = round(width * 0.84), round(height * 0.77)
    pixels = np.asarray(image.crop((left, top, right, bottom))).astype(np.int16)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    masks = {
        "discard": (red >= 235) & (green >= 135) & (green <= 225) & (blue <= 110) & (green - blue >= 60),
        "hu": (
            (red >= 230)
            & (green >= 70)
            & (green <= 170)
            & (blue >= 55)
            & (blue <= 140)
            & (red - green >= 80)
            & (np.abs(green - blue) <= 45)
        ),
    }
    found: dict[str, Point] = {}
    kernel = np.ones((9, 9), np.uint8)
    for name, mask in masks.items():
        closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(closed)
        best: tuple[int, Point] | None = None
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            # 实测出牌/胡宽 0.091~0.094 屏宽；“💎30”购买钮约 0.13、开始游戏 0.15。
            if not (width * 0.075 <= w <= width * 0.11) or h < height * 0.04:
                continue
            if not 1.5 <= w / h <= 4.0 or area < 0.5 * w * h:
                continue
            if best is None or area > best[0]:
                best = (area, Point(left + x + w // 2, top + y + h // 2))
        if best is not None:
            found[name] = best[1]
    return found


def detect_discard_button(image: Any) -> bool:
    """“出牌”按钮是否仍在，不看蓝色横幅、也不要求看到牌堆。

    选中一张牌后蓝色横幅会消失；若选完剩下的牌已听牌，游戏还会弹出
    “胡：10♣ 90分 32倍…”提示面板盖住牌堆。两种情况下按钮都还在。
    """
    return "discard" in find_action_buttons(image)


def is_start_page_button(image: Any, point: Point) -> bool:
    """安全闸：只有真正的开始页才允许点击“开始游戏”。

    曾把“豆子不足啦”弹窗里的“💎250”购买按钮当成开始按钮点击。真开始页
    实测：按钮恰在水平正中（x=0.50 屏宽，y≈0.69 屏高）、台面未被遮罩压暗、
    中央没有牌堆。购买弹窗台面被压暗；出牌/胡按钮所在画面有牌堆。
    """
    width, height = image.size
    return (
        detect_table_felt_visible(image)
        and not detect_round_in_progress(image)
        and abs(point.x / width - 0.5) <= 0.03
        and 0.64 <= point.y / height <= 0.74
    )


def current_start_page_check(title_keyword: str, point: Point) -> bool:
    _, image = grab_game_window(title_keyword)
    return is_start_page_button(image, point)


def current_action_buttons(title_keyword: str) -> dict[str, Point]:
    _, image = grab_game_window(title_keyword)
    return find_action_buttons(image)


def current_game_action_state(title_keyword: str) -> str | None:
    """读取当前游戏帧；None 表示不是自己的操作回合。

    与回合探测同一套判定：从选牌弹窗拿牌后出牌阶段没有蓝色横幅，只剩
    “出牌”按钮，只看横幅会把这种回合误判为过期而放弃出牌。
    """
    _, image = grab_game_window(title_keyword)
    return classify_turn_image(image)


def detect_round_in_progress(image: Any) -> bool:
    """OpenCV 检测中央蓝色牌堆，判断牌局是否进行中。"""
    return cv_detect_round_in_progress(image)


def detect_choice_popup(image: Any) -> bool:
    """检测使用“?”选牌卡后弹出的“请选择想要的牌”面板。

    面板为浅蓝白底，盖住中央牌堆且没有蓝色操作横幅，像素门原本会把它当成
    “非自己回合”，转去慢速完整识别直到倒计时结束。取面板标题两侧和底边
    三条浅色带判断；实测弹窗 1.0/1.0/0.94，其余所有截图均低于 0.3。
    """
    image = image.convert("RGB")
    width, height = image.size
    for left, top, right, bottom in (
        (0.28, 0.262, 0.40, 0.29),
        (0.60, 0.262, 0.72, 0.29),
        # 面板底边留白：在“吃”标签（四张候选都可能带）下方。
        (0.30, 0.550, 0.70, 0.562),
    ):
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
            return False
        light = sum(
            1
            for red, green, blue in pixels
            if red >= 200 and green >= 218 and blue >= 235 and blue >= red
        )
        if light / len(pixels) < 0.80:
            return False
    return True


def find_opportunity_card(image: Any) -> Point | None:
    """扫描出牌阶段全部手牌位，返回“?”选牌卡的点击中心。

    “?”必须先摸牌、到出牌阶段才能打出，所以只扫出牌阶段的 8 张布局。

    双问号牌局完成第一张四选一后，第二张可能仍留在后续牌位，不能只检查
    最左侧。实测：选牌卡橙色约 0.15、白底 ≥0.63；癞子（桃色底）橙色
    ≤0.066、白底约 0.01；普通牌橙色 ≤0.006。
    """
    image = image.convert("RGB")
    width, height = image.size
    # 问号牌固定排在手牌最左侧，规则上最多 4 张；只扫左侧四格可减少误判。
    slot_xs = tuple(round(282 + index * 53.4) for index in range(4))
    for slot_x in slot_xs:
        center = slot_x / 1000
        pixels = list(
            image.crop(
                (
                    round(width * (center - 0.0195)),
                    round(height * 0.79),
                    round(width * (center + 0.0195)),
                    round(height * 0.92),
                )
            ).getdata()
        )
        if not pixels:
            continue
        orange = sum(
            1
            for red, green, blue in pixels
            if red > 215 and 110 < green < 200 and blue < 150 and red - blue > 80
        )
        neutral = sum(
            1
            for red, green, blue in pixels
            if red > 195 and abs(red - green) < 12 and abs(green - blue) < 12
        )
        if orange / len(pixels) >= 0.10 and neutral / len(pixels) >= 0.40:
            return Point(round(width * center), round(height * 0.85))
    return None


def detect_opportunity_card(image: Any) -> bool:
    return find_opportunity_card(image) is not None


def classify_turn_image(image: Any) -> str | None:
    """swap/draw/discard/choose；None 表示不是自己可操作的画面。"""
    action_state = detect_action_state(image)
    if (
        action_state is None
        and detect_round_in_progress(image)
        and detect_discard_button(image)
    ):
        # 选中一张牌、或从“请选择想要的牌”弹窗拿牌后，蓝色横幅会消失，
        # 但“出牌”按钮仍在：这仍是自己的出牌阶段（实测漏掉它会被系统代打）。
        action_state = "discard"
    if action_state is None and detect_choice_popup(image):
        action_state = "choose"
    return action_state


# 与 vision.TABLE_CROP_BOX 一致：牌堆数字 + 4 个亮牌位 + 最右侧小牌。
TABLE_SIGNATURE_BOX = (0.355, 0.32, 0.67, 0.45)
TABLE_SIGNATURE_SIZE = (40, 10)


def table_signature(image: Any) -> bytes:
    width, height = image.size
    left, top, right, bottom = TABLE_SIGNATURE_BOX
    crop = image.crop(
        (round(width * left), round(height * top), round(width * right), round(height * bottom))
    )
    return crop.convert("L").resize(TABLE_SIGNATURE_SIZE).tobytes()


def table_signature_changed(first: bytes | None, second: bytes | None, threshold: float = 6.0) -> bool:
    """两次亮牌区缩略图的平均灰度差是否超过阈值（缺任一方视为变化）。"""
    if first is None or second is None or len(first) != len(second):
        return True
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first) > threshold


def probe_game_turn(output_dir: Path, title_keyword: str) -> TurnProbe:
    """廉价探测一帧：只在轮到自己操作时才落盘，供视觉模型识别。

    action_state 额外可能为 "choose"，表示“请选择想要的牌”弹窗。
    """
    window, image = grab_game_window(title_keyword)
    action_state = classify_turn_image(image)
    opportunity_point = (
        find_opportunity_card(image) if action_state == "discard" else None
    )
    frame = (
        _save_frame(output_dir, window, image)
        if action_state is not None
        else None
    )
    return TurnProbe(
        action_state,
        detect_round_in_progress(image),
        frame,
        opportunity_card=opportunity_point is not None,
        opportunity_point=opportunity_point,
        table_signature=table_signature(image),
        settlement=detect_hu_settlement(image),
    )
