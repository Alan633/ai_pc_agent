from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from agent.session import AgentSession
from game_baque.capture import CapturedFrame
from game_baque.controller import BaqueGameController
from game_baque.fast_vision import (
    analyze_game_ui,
    detect_action_state,
    detect_room_lobby,
    detect_round_in_progress,
    detect_start_button,
)
from game_baque.models import ScreenType


def _canvas() -> Image.Image:
    return Image.new("RGB", (1000, 600), (45, 125, 100))


def test_opencv_detects_start_button() -> None:
    image = _canvas()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((400, 310, 600, 365), radius=24, fill=(255, 160, 28))

    point, confidence = detect_start_button(image)
    state = analyze_game_ui(image)

    assert point is not None
    assert abs(point.x - 500) <= 5
    assert abs(point.y - 337) <= 5
    assert confidence >= 0.72
    assert state.screen_type == ScreenType.START
    assert state.start_button is not None


def test_opencv_detects_discard_turn() -> None:
    image = _canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 330, 700, 372), fill=(35, 110, 230))
    draw.rectangle((555, 393, 595, 429), fill=(245, 145, 25))

    assert detect_action_state(image) == "discard"
    assert analyze_game_ui(image).screen_type == ScreenType.DISCARD


def test_opencv_detects_swap_turn() -> None:
    image = _canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 330, 700, 372), fill=(35, 110, 230))
    draw.rectangle((555, 393, 595, 429), fill=(165, 165, 165))

    assert detect_action_state(image) == "swap"
    assert analyze_game_ui(image).screen_type == ScreenType.SWAP


def test_opencv_detects_waiting_round() -> None:
    image = _canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((365, 201, 405, 261), fill=(20, 90, 235))

    assert detect_round_in_progress(image) is True
    state = analyze_game_ui(image)
    assert state.screen_type == ScreenType.WAITING
    assert state.round_in_progress is True


def test_opencv_detects_room_lobby_as_terminal_state() -> None:
    image = Image.new("RGB", (1000, 600), (120, 160, 225))
    draw = ImageDraw.Draw(image)
    panels = (
        ((180, 186, 320, 270), (65, 195, 180)),
        ((390, 186, 500, 270), (150, 135, 250)),
        ((600, 186, 720, 270), (240, 150, 65)),
        ((800, 186, 910, 270), (245, 105, 95)),
    )
    for box, color in panels:
        draw.rectangle(box, fill=color)

    assert detect_room_lobby(image)
    assert analyze_game_ui(image).screen_type == ScreenType.LOBBY


@pytest.mark.asyncio
async def test_controller_start_page_does_not_call_qwen(tmp_path: Path) -> None:
    image_path = tmp_path / "start.png"
    image = _canvas()
    draw = ImageDraw.Draw(image)
    # 真开始页按钮位于水平正中、约 0.69 屏高（安全闸只放行这个位置）。
    draw.rounded_rectangle((400, 387, 600, 442), radius=24, fill=(255, 160, 28))
    image.save(image_path)

    class NeverCallVision:
        async def parse(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("开始页面不应调用 Qwen")

    controller = BaqueGameController(
        vision=NeverCallVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    frame = CapturedFrame(
        path=image_path,
        origin_x=0,
        origin_y=0,
        width=1000,
        height=600,
        window_title="欢乐斗地主",
    )

    snapshot = await controller._parse_captured_frame(frame, include_previous=False)

    assert snapshot.screen_type == ScreenType.START
    assert snapshot.start_button is not None
    assert controller.fast_ui_count == 1
    assert controller.qwen_vision_count == 0


@pytest.mark.asyncio
async def test_controller_room_lobby_stops_without_qwen(tmp_path: Path) -> None:
    image_path = tmp_path / "lobby.png"
    image = Image.new("RGB", (1000, 600), (120, 160, 225))
    draw = ImageDraw.Draw(image)
    for box, color in (
        ((180, 186, 320, 270), (65, 195, 180)),
        ((390, 186, 500, 270), (150, 135, 250)),
        ((600, 186, 720, 270), (240, 150, 65)),
        ((800, 186, 910, 270), (245, 105, 95)),
    ):
        draw.rectangle(box, fill=color)
    image.save(image_path)

    class NeverCallVision:
        async def parse(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("场次选择页不应调用 Qwen")

    controller = BaqueGameController(
        vision=NeverCallVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    frame = CapturedFrame(image_path, 0, 0, 1000, 600, "欢乐斗地主")

    snapshot = await controller._parse_captured_frame(frame, include_previous=False)
    result = await controller.step(prefetched=(frame, snapshot))

    assert snapshot.screen_type == ScreenType.LOBBY
    assert result.stop_loop is True
    assert "已经终止" in result.message
    assert controller.qwen_vision_count == 0
