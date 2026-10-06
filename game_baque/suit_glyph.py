"""手牌花色的像素校验：按点数下方花色图标的颜色与形状判断 ♠♣♥♦。

Qwen 能读准点数，但同色花色常互认（J♠→J♣、Q♦→Q♥），放大拼图后仍会错；
图标本身很规整，用形状特征即可稳定区分（实测 6 帧 38 张普通牌全部正确）：
- 黑色：♠ 实心度 0.945~0.96；♣ 三个圆瓣之间有缺口，实心度 0.913~0.931。
- 红色：♥ 实心度 0.977~0.997；♦ 菱形几乎填满凸包，实心度 1.015~1.045。
（实心度 = 像素面积 / 凸包面积；像素凸包略小，菱形会略大于 1。）
拿不准（离分界太近、没找到图标）时返回 None，由调用方保留 Qwen 的读数。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# 与 vision.TURN_HAND_SLOTS 相同的牌位（0-1000 横坐标）。
HAND_SLOTS: dict[int, tuple[int, ...]] = {
    7: tuple(283 + index * 62 for index in range(7)),
    8: tuple(round(282 + index * 53.4) for index in range(8)),
}
# 每张牌左上角“点数+花色”所在区域（相对牌位中心 / 窗口高度）。抬起的牌和扇形
# 最左侧偏低的牌都要覆盖，所以纵向放宽，再按尺寸挑出最像小花色图标的色块。
_CELL_X = (-0.022, 0.028)
_CELL_Y = (0.80, 0.975)
_REFERENCE_HEIGHT = 1080
_BLACK_SPLIT = 0.938  # ♠/♣ 实心度分界
_RED_SPLIT = 1.006  # ♥/♦ 实心度分界
_MARGIN = 0.004  # 离分界太近时不下结论
# 点数下方小花色图标的标准尺寸（1080 高画面）；牌底中央的大花色、点数字形都更大或被截断。
_GLYPH_SIZE = (39, 37)


def _glyph_candidates(cell: np.ndarray) -> list[tuple[int, str, float]]:
    """返回 (与标准尺寸的偏差, 颜色, 实心度)：符合花色图标大小与形状的色块。"""
    red, green, blue = cell[..., 0], cell[..., 1], cell[..., 2]
    masks = {
        "black": (red < 90) & (green < 90) & (blue < 110),
        "red": (red > 150) & (green < 90) & (blue < 90),
    }
    found: list[tuple[int, str, float]] = []
    for color, mask in masks.items():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if not (33 <= w <= 44 and 28 <= h <= 42 and 650 <= area <= 1000):
                continue
            points = cv2.findNonZero((labels[y : y + h, x : x + w] == index).astype(np.uint8))
            hull_area = float(cv2.contourArea(cv2.convexHull(points)))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if 0.88 <= solidity <= 1.07:
                deviation = abs(w - _GLYPH_SIZE[0]) + abs(h - _GLYPH_SIZE[1])
                found.append((deviation, color, solidity))
    return found


def classify_suit(cell: np.ndarray) -> str | None:
    """单张牌左上角区域 → "S"/"C"/"H"/"D"；没找到图标或拿不准时返回 None。"""
    candidates = _glyph_candidates(cell)
    if not candidates:
        return None
    _, color, solidity = min(candidates)  # 取尺寸最接近标准小图标的那个
    split = _BLACK_SPLIT if color == "black" else _RED_SPLIT
    if abs(solidity - split) < _MARGIN:
        return None
    if color == "black":
        return "S" if solidity > split else "C"
    return "D" if solidity > split else "H"


def classify_hand_suits(image: Any, slot_count: int) -> list[str | None]:
    """按牌位从左到右给出每张手牌的花色（问号、癞子、拿不准的位置为 None）。"""
    slots = HAND_SLOTS.get(slot_count)
    if slots is None:
        return []
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = _REFERENCE_HEIGHT / height
    suits: list[str | None] = []
    for slot in slots:
        center = slot / 1000
        box = (
            round(width * (center + _CELL_X[0])),
            round(height * _CELL_Y[0]),
            round(width * (center + _CELL_X[1])),
            round(height * _CELL_Y[1]),
        )
        cell = rgb.crop(box)
        if abs(scale - 1.0) > 0.02:
            cell = cell.resize((max(1, round(cell.width * scale)), max(1, round(cell.height * scale))))
        suits.append(classify_suit(np.asarray(cell).astype(np.int16)))
    return suits
