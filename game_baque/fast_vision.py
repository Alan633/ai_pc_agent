from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from game_baque.models import Point, ScreenType


@dataclass(frozen=True, slots=True)
class FastUIState:
    screen_type: ScreenType
    confidence: float
    start_button: Point | None
    action_state: str | None
    round_in_progress: bool
    message: str


def _to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _crop(array: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int]:
    height, width = array.shape[:2]
    left = max(0, min(width - 1, round(width * box[0])))
    top = max(0, min(height - 1, round(height * box[1])))
    right = max(left + 1, min(width, round(width * box[2])))
    bottom = max(top + 1, min(height, round(height * box[3])))
    return array[top:bottom, left:right], left, top


def _orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 腾讯游戏按钮偏橙黄；保留较宽 hue，文字白洞由闭运算补齐。
    lower = np.array([4, 105, 150], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _blue_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([88, 75, 110], dtype=np.uint8),
        np.array([138, 255, 255], dtype=np.uint8),
    )
    return float(np.count_nonzero(mask)) / float(mask.size)


def _neutral_gray_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    channels = bgr.astype(np.int16)
    maximum = channels.max(axis=2)
    minimum = channels.min(axis=2)
    neutral = (maximum - minimum <= 20) & (maximum >= 75) & (maximum <= 240)
    return float(np.count_nonzero(neutral)) / float(neutral.size)


def detect_start_button(image: Image.Image) -> tuple[Point | None, float]:
    """通过橙色圆角按钮轮廓识别“开始游戏”。

    不识别文字，避免依赖 OCR；只在游戏中央候选区寻找横向橙色大按钮，
    可规避右上角商城等小橙色图标和回合倒计时。
    """
    bgr = _to_bgr(image)
    height, width = bgr.shape[:2]
    roi, offset_x, offset_y = _crop(bgr, (0.28, 0.40, 0.72, 0.82))
    mask = _orange_mask(roi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, Point] | None = None
    full_area = float(width * height)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue
        aspect = w / h
        area = float(cv2.contourArea(contour))
        area_ratio = area / full_area
        if not (1.8 <= aspect <= 6.5):
            continue
        if not (0.0025 <= area_ratio <= 0.04):
            continue
        if w < width * 0.07 or h < height * 0.025:
            continue

        candidate_mask = mask[y : y + h, x : x + w]
        fill_ratio = (
            float(np.count_nonzero(candidate_mask)) / float(candidate_mask.size)
            if candidate_mask.size
            else 0.0
        )
        if fill_ratio < 0.36:
            continue

        center_x = offset_x + x + w // 2
        center_y = offset_y + y + h // 2
        if not (width * 0.34 <= center_x <= width * 0.66):
            continue
        if not (height * 0.46 <= center_y <= height * 0.78):
            continue

        # 面积、橙色填充率、宽高比都越接近真实大按钮，分数越高。
        aspect_score = max(0.0, 1.0 - abs(aspect - 3.1) / 3.1)
        area_score = min(1.0, area_ratio / 0.012)
        score = min(0.99, 0.52 + fill_ratio * 0.24 + aspect_score * 0.12 + area_score * 0.12)
        point = Point(center_x, center_y)
        if best is None or score > best[0]:
            best = (score, point)

    if best is None:
        return None, 0.0
    return best[1], best[0]


def detect_action_state(image: Image.Image) -> str | None:
    """OpenCV 快速判断 swap/draw/discard，不读取牌面。"""
    bgr = _to_bgr(image)

    banner, _, _ = _crop(bgr, (0.30, 0.55, 0.70, 0.62))
    if _blue_ratio(banner) < 0.22:
        return None

    button, _, _ = _crop(bgr, (0.555, 0.655, 0.595, 0.715))
    orange_ratio = (
        float(np.count_nonzero(_orange_mask(button))) / float(button.shape[0] * button.shape[1])
        if button.size
        else 0.0
    )
    gray_ratio = _neutral_gray_ratio(button)

    if orange_ratio >= 0.28:
        return "discard"
    if gray_ratio >= 0.38:
        return "swap"
    return "draw"


def detect_round_in_progress(image: Image.Image) -> bool:
    """检测中央蓝色背面牌堆。"""
    bgr = _to_bgr(image)
    deck, _, _ = _crop(bgr, (0.365, 0.335, 0.405, 0.435))
    return _blue_ratio(deck) >= 0.55


def detect_room_lobby(image: Image.Image) -> bool:
    """检测新手/普通/高级/顶级四个大色块组成的场次选择页。"""
    hsv = cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2HSV)

    def median_hsv(box: tuple[float, float, float, float]) -> tuple[float, float, float]:
        patch, _, _ = _crop(hsv, box)
        values = np.median(patch.reshape(-1, 3), axis=0)
        return float(values[0]), float(values[1]), float(values[2])

    cyan = median_hsv((0.18, 0.31, 0.32, 0.45))
    purple = median_hsv((0.39, 0.31, 0.50, 0.45))
    orange = median_hsv((0.60, 0.31, 0.72, 0.45))
    red = median_hsv((0.80, 0.31, 0.91, 0.45))
    return (
        70 <= cyan[0] <= 100 and cyan[1] >= 55 and cyan[2] >= 150
        and 110 <= purple[0] <= 145 and purple[1] >= 45 and purple[2] >= 150
        and 8 <= orange[0] <= 30 and orange[1] >= 70 and orange[2] >= 150
        and (red[0] <= 10 or red[0] >= 170) and red[1] >= 60 and red[2] >= 150
    )


# 胡牌结算时各家头像旁弹出的输赢横幅位置（相对窗口）：赢家红橙底“+X万”，输家蓝灰底“-X万”。
SETTLEMENT_SEATS: dict[str, tuple[float, float, float, float]] = {
    "right": (0.865, 0.45, 0.995, 0.505),
    "left": (0.005, 0.45, 0.135, 0.505),
    "top": (0.43, 0.185, 0.57, 0.24),
    "self": (0.44, 0.87, 0.56, 0.93),
}


def detect_hu_settlement(image: Image.Image) -> str | None:
    """检测胡牌结算横幅，返回赢家座位（right/left/top/self），不确定赢家时返回 "unknown"。

    实测 1996 帧八雀画面：横幅处红/蓝灰像素占比要么接近 0，要么 ≥0.3，49 帧结算全部
    命中、无误判。某家红色 = 他胡了；两家以上蓝灰 = 结算画面，唯一没有横幅的一家是赢家。
    有人胡牌后本局可能继续，所以这只用来停止误归属，不当作本局结束。
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = rgb.shape[:2]
    reds: list[str] = []
    blues: list[str] = []
    for seat, (left, top, right, bottom) in SETTLEMENT_SEATS.items():
        patch = rgb[round(height * top) : round(height * bottom), round(width * left) : round(width * right)]
        if patch.size == 0:
            continue
        red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
        red_ratio = float(
            ((red >= 200) & (green >= 60) & (green <= 130) & (blue >= 40) & (blue <= 120) & (red - green >= 90)).mean()
        )
        blue_ratio = float(
            (
                (red >= 90) & (red <= 140) & (green >= 125) & (green <= 170)
                & (blue >= 160) & (blue <= 205) & (blue - red >= 40)
            ).mean()
        )
        if red_ratio >= 0.25:
            reds.append(seat)
        elif blue_ratio >= 0.2:
            blues.append(seat)
    if len(reds) == 1:
        return reds[0]
    if len(blues) >= 2:
        silent = [seat for seat in SETTLEMENT_SEATS if seat not in blues and seat not in reds]
        return silent[0] if len(silent) == 1 else "unknown"
    return None


# 亮牌行 4 个牌位（0-1000 横坐标，与 vision.TURN_TABLE_SLOTS 一致）。
TABLE_SLOT_XS = (438, 492, 546, 600)


def detect_table_hu_badges(image: Image.Image) -> dict[int, bool]:
    """亮牌下沿的红色“胡 ×N”角标：游戏标出“拿这张就能胡”的牌。返回 {牌位横坐标: 是否有角标}。

    实测角标处红色像素占比约 0.29，其余牌（含红色花色、橙色“吃”角标）为 0。
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = rgb.shape[:2]
    result: dict[int, bool] = {}
    for slot in TABLE_SLOT_XS:
        center = slot / 1000
        patch = rgb[
            round(height * 0.405) : round(height * 0.445),
            round(width * (center - 0.022)) : round(width * (center + 0.022)),
        ]
        if patch.size == 0:
            result[slot] = False
            continue
        red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
        result[slot] = float(((red >= 200) & (green <= 110) & (blue <= 110)).mean()) >= 0.15
    return result


def detect_recharge_popup(image: Image.Image) -> bool:
    """检测“豆子不足啦”购买弹窗（输光豆子后出现，代表本局游戏结束）。

    特征：浅蓝白购买面板底边那一行（“需充值 X 元”）+ 面板下方深蓝的
    “认输倒计时”条。实测 453 张截图中 10 张真弹窗面板 ≥0.84、倒计时条 ≥0.71，
    其余截图两项不会同时 ≥0.4/0.3。只用于识别，绝不点击弹窗上的任何按钮。
    """
    bgr = _to_bgr(image)
    panel, _, _ = _crop(bgr, (0.30, 0.775, 0.70, 0.805))
    bar, _, _ = _crop(bgr, (0.40, 0.845, 0.62, 0.875))
    if panel.size == 0 or bar.size == 0:
        return False
    p = panel.astype(np.int16)
    blue, green, red = p[..., 0], p[..., 1], p[..., 2]
    light = (red >= 200) & (green >= 210) & (blue >= 230) & (blue >= red)
    b = bar.astype(np.int16)
    b_blue, b_green, b_red = b[..., 0], b[..., 1], b[..., 2]
    navy = (b_blue >= 90) & (b_blue - b_red >= 40) & (b_red <= 110) & (b_green <= 130)
    return float(light.mean()) >= 0.6 and float(navy.mean()) >= 0.5


def analyze_game_ui(image: Image.Image) -> FastUIState:
    """快速识别固定 UI；不能确定的页面交给 Qwen 兜底。"""
    if detect_recharge_popup(image):
        return FastUIState(
            screen_type=ScreenType.RECHARGE,
            confidence=0.97,
            start_button=None,
            action_state=None,
            round_in_progress=False,
            message="OpenCV 快速识别：豆子不足购买弹窗，本局游戏结束",
        )

    if detect_room_lobby(image):
        return FastUIState(
            screen_type=ScreenType.LOBBY,
            confidence=0.99,
            start_button=None,
            action_state=None,
            round_in_progress=False,
            message="OpenCV 快速识别：已返回场次选择页",
        )

    start_button, start_confidence = detect_start_button(image)
    if start_button is not None and start_confidence >= 0.72:
        return FastUIState(
            screen_type=ScreenType.START,
            confidence=start_confidence,
            start_button=start_button,
            action_state=None,
            round_in_progress=False,
            message="OpenCV 快速识别：开始游戏页面",
        )

    action_state = detect_action_state(image)
    if action_state is not None:
        screen_type = {
            "swap": ScreenType.SWAP,
            "draw": ScreenType.DRAW,
            "discard": ScreenType.DISCARD,
        }[action_state]
        return FastUIState(
            screen_type=screen_type,
            confidence=0.93,
            start_button=None,
            action_state=action_state,
            round_in_progress=True,
            message=f"OpenCV 快速识别：{action_state} 回合",
        )

    round_in_progress = detect_round_in_progress(image)
    if round_in_progress:
        return FastUIState(
            screen_type=ScreenType.WAITING,
            confidence=0.90,
            start_button=None,
            action_state=None,
            round_in_progress=True,
            message="OpenCV 快速识别：牌局进行中，当前不是自己的操作回合",
        )

    return FastUIState(
        screen_type=ScreenType.UNKNOWN,
        confidence=0.0,
        start_button=None,
        action_state=None,
        round_in_progress=False,
        message="OpenCV 无法可靠判断，交给 Qwen 兜底",
    )


def analyze_image_path(path: Path) -> FastUIState:
    with Image.open(path) as image:
        return analyze_game_ui(image)


def fast_state_to_dict(state: FastUIState) -> dict[str, Any]:
    return {
        "screen_type": state.screen_type.value,
        "confidence": round(state.confidence, 4),
        "start_button": (
            None
            if state.start_button is None
            else {"x": state.start_button.x, "y": state.start_button.y}
        ),
        "action_state": state.action_state,
        "round_in_progress": state.round_in_progress,
        "message": state.message,
    }
