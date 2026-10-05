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


def analyze_game_ui(image: Image.Image) -> FastUIState:
    """快速识别固定 UI；不能确定的页面交给 Qwen 兜底。"""
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
