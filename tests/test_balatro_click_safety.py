from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from game_balatro.controller import BalatroController
from game_balatro.models import BalatroSnapshot, PlayingCard, ScreenType
from game_balatro.vision import BalatroVision


def _card(rank: str, x: int, y: int = 700) -> PlayingCard:
    return PlayingCard(
        rank=rank,
        suit="S",
        x=x,
        y=y,
    )


def test_safe_card_click_moves_into_exposed_left_area() -> None:
    snapshot = BalatroSnapshot(
        screen_type=ScreenType.PLAY,
        hand=(
            _card("A", 500),
            _card("K", 590),
            _card("Q", 680),
            _card("J", 770),
        ),
    )

    x0, y0 = BalatroController._safe_card_click_point(snapshot, 0)
    x2, y2 = BalatroController._safe_card_click_point(snapshot, 2)
    x3, y3 = BalatroController._safe_card_click_point(snapshot, 3)

    assert x0 < snapshot.hand[0].x
    assert x2 < snapshot.hand[2].x
    assert x3 < snapshot.hand[3].x
    assert 30 <= snapshot.hand[0].x - x0 <= 50
    assert y0 == y2 == y3 == 700


def test_safe_card_click_handles_single_card() -> None:
    snapshot = BalatroSnapshot(
        screen_type=ScreenType.PLAY,
        hand=(_card("A", 600, 720),),
    )

    assert BalatroController._safe_card_click_point(snapshot, 0) == (600, 720)


def test_local_play_control_detector_finds_blue_and_red_buttons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "balatro_controls.png"
    image = Image.new("RGB", (1600, 1000), (35, 105, 85))
    draw = ImageDraw.Draw(image)
    # 下半区的“出牌”蓝按钮与“弃牌”红按钮。
    draw.rounded_rectangle(
        (520, 610, 740, 690),
        radius=18,
        fill=(20, 135, 245),
    )
    draw.rounded_rectangle(
        (810, 610, 1030, 690),
        radius=18,
        fill=(235, 65, 70),
    )
    # 左侧干扰按钮：应被 x 区域过滤。
    draw.rounded_rectangle(
        (40, 610, 220, 690),
        radius=18,
        fill=(235, 65, 70),
    )
    image.save(path)

    controls = BalatroVision._detect_play_controls(path)

    assert "play" in controls
    assert "discard" in controls
    assert abs(controls["play"].x - 630) <= 10
    assert abs(controls["play"].y - 650) <= 10
    assert abs(controls["discard"].x - 920) <= 10
    assert abs(controls["discard"].y - 650) <= 10
