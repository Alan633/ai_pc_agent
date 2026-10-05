from __future__ import annotations

from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.capture import CapturedFrame
from game_baque.controller import BaqueGameController
from game_baque.models import Card, GameSnapshot, Point, ScreenType


def _card(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


@pytest.mark.asyncio
async def test_prepare_auto_round_clicks_start_once_then_waits_for_real_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )

    focused: list[str] = []
    monkeypatch.setattr(
        "game_baque.controller.focus_game_window",
        lambda keyword: focused.append(keyword) or "欢乐斗地主",
    )

    frame = CapturedFrame(
        path=tmp_path / "dummy.png",
        origin_x=100,
        origin_y=100,
        width=1000,
        height=600,
        window_title="欢乐斗地主",
    )
    snapshots = iter(
        [
            GameSnapshot(
                screen_type=ScreenType.START,
                start_button=Point(500, 400),
                message="开始页面",
            ),
            GameSnapshot(
                screen_type=ScreenType.WAITING,
                message="发牌动画",
            ),
            GameSnapshot(
                screen_type=ScreenType.SWAP,
                gold_ranks=("6", "10", "K"),
                hand=[
                    _card("6"),
                    _card("7", "H"),
                    _card("8", "C"),
                    _card("9", "D"),
                    _card("10"),
                    _card("J", "H"),
                ],
                deck_count=56,
            ),
        ]
    )

    async def fake_inspect(*, include_previous: bool = True) -> tuple[CapturedFrame, GameSnapshot]:
        assert include_previous is False
        return frame, next(snapshots)

    clicks: list[Point] = []

    async def fake_click(_frame: CapturedFrame, point: Point) -> None:
        clicks.append(point)

    controller.inspect = fake_inspect  # type: ignore[method-assign]
    controller._click = fake_click  # type: ignore[method-assign]

    async def start_page_ok(*_args: object, **_kwargs: object) -> bool:
        return True

    # 占位截图不是真实开始页；安全闸在 test_baque_turn_gate 中单独覆盖。
    controller._start_page_ok = start_page_ok  # type: ignore[method-assign]

    async def no_fast_swap_frame(timeout_seconds: float = 25.0) -> None:
        return None

    controller._wait_for_swap_frame = no_fast_swap_frame  # type: ignore[method-assign]

    result = await controller.prepare_auto_round(timeout_seconds=5)

    assert focused == ["欢乐斗地主"]
    assert result.screen_type == ScreenType.SWAP
    assert clicks == [Point(500, 400)]


@pytest.mark.asyncio
async def test_prepare_auto_round_refuses_start_without_reliable_button(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    monkeypatch.setattr(
        "game_baque.controller.focus_game_window",
        lambda keyword: "欢乐斗地主",
    )

    frame = CapturedFrame(
        path=tmp_path / "dummy.png",
        origin_x=0,
        origin_y=0,
        width=1000,
        height=600,
        window_title="欢乐斗地主",
    )

    async def fake_inspect(*, include_previous: bool = True) -> tuple[CapturedFrame, GameSnapshot]:
        assert include_previous is False
        return frame, GameSnapshot(screen_type=ScreenType.START)

    controller.inspect = fake_inspect  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="开始游戏"):
        await controller.prepare_auto_round(timeout_seconds=5)


@pytest.mark.asyncio
async def test_prepare_auto_round_refocuses_when_another_window_steals_foreground(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    focused: list[str] = []
    monkeypatch.setattr(
        "game_baque.controller.focus_game_window",
        lambda keyword: focused.append(keyword) or "欢乐斗地主",
    )
    frame = CapturedFrame(tmp_path / "dummy.png", 0, 0, 1936, 1088, "欢乐斗地主")
    ready = GameSnapshot(
        screen_type=ScreenType.DRAW,
        gold_ranks=("9", "J", "K"),
        hand=[_card(rank) for rank in ("6", "7", "8", "9", "10", "J", "Q")],
        deck_count=55,
    )
    calls = 0

    async def flaky_inspect(*, include_previous: bool = True) -> tuple[CapturedFrame, GameSnapshot]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("前台窗口不是目标游戏：'Weixin'；请先把包含“欢乐斗地主”的游戏窗口切到前台")
        return frame, ready

    controller.inspect = flaky_inspect  # type: ignore[method-assign]

    result = await controller.prepare_auto_round(timeout_seconds=5)

    assert result is ready
    # 启动时切一次，被微信抢走后再切一次。
    assert len(focused) == 2
