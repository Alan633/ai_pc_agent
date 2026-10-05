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


@pytest.mark.asyncio
async def test_controller_start_page_does_not_call_qwen(tmp_path: Path) -> None:
    image_path = tmp_path / "start.png"
    image = _canvas()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((400, 310, 600, 365), radius=24, fill=(255, 160, 28))
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
