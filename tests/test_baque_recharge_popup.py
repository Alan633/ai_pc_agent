from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from agent.session import AgentSession
from game_baque.capture import CapturedFrame
from game_baque.controller import BaqueGameController
from game_baque.fast_vision import analyze_game_ui, detect_recharge_popup
from game_baque.models import ScreenType

W, H = 1920, 1080
DIM_FELT = (20, 60, 50)
PANEL = (225, 232, 252)  # 购买面板浅蓝白
BAR = (45, 60, 140)  # “认输倒计时”深蓝条


def _box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    return round(W * x0), round(H * y0), round(W * x1), round(H * y1)


def _recharge_image() -> Image.Image:
    image = Image.new("RGB", (W, H), DIM_FELT)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_box(0.22, 0.26, 0.80, 0.83), fill=PANEL)
    draw.rectangle(_box(0.39, 0.84, 0.63, 0.88), fill=BAR)
    return image


def test_recharge_popup_is_recognized_as_game_over() -> None:
    state = analyze_game_ui(_recharge_image())
    assert state.screen_type == ScreenType.RECHARGE
    assert state.start_button is None and state.action_state is None


def test_choice_popup_and_plain_table_are_not_recharge() -> None:
    # “请选择想要的牌”面板也是浅蓝白，但只到约 0.56 屏高，下方没有认输倒计时条。
    choice = Image.new("RGB", (W, H), (30, 120, 100))
    ImageDraw.Draw(choice).rectangle(_box(0.25, 0.25, 0.75, 0.56), fill=PANEL)
    assert not detect_recharge_popup(choice)
    assert not detect_recharge_popup(Image.new("RGB", (W, H), (30, 120, 100)))
    # 只有倒计时条、没有面板（例如其他提示条）也不算。
    bar_only = Image.new("RGB", (W, H), DIM_FELT)
    ImageDraw.Draw(bar_only).rectangle(_box(0.39, 0.84, 0.63, 0.88), fill=BAR)
    assert not detect_recharge_popup(bar_only)


@pytest.mark.asyncio
async def test_recharge_frame_stops_without_calling_qwen(tmp_path: Path) -> None:
    path = tmp_path / "recharge.png"
    _recharge_image().save(path)

    class NoQwen:
        async def parse(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("购买弹窗应由 OpenCV 直接识别，不必等 Qwen")

    controller = BaqueGameController(
        vision=NoQwen(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    snapshot = await controller._parse_captured_frame(
        CapturedFrame(path, 0, 0, W, H, "欢乐斗地主"), include_previous=False
    )
    assert snapshot.screen_type == ScreenType.RECHARGE

