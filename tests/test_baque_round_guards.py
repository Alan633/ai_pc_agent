from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from agent.session import AgentSession
from game_baque.capture import CapturedFrame, TurnProbe
from game_baque.controller import BaqueGameController
from game_baque.models import Card, GameSnapshot, ScreenType


@pytest.fixture(autouse=True)
def _no_real_screen_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("测试环境不截屏")

    monkeypatch.setattr("game_baque.controller.capture_game_window", refuse)


def c(rank: str, suit: str) -> Card:
    return Card(rank, suit)


HAND = [c("A", "D"), c("Q", "D"), c("J", "D"), c("8", "S"), c("8", "C"), c("8", "D"), c("J", "H"), c("9", "S")]


def _controller(tmp_path: Path) -> BaqueGameController:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        rounds_log_path=tmp_path / "rounds.jsonl",
    )
    controller._round_info_attempts = 2  # 测试中不读取本局信息
    return controller


def _snapshot(gold: tuple[str, ...], deck: int | None) -> GameSnapshot:
    return GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=gold, hand=list(HAND), deck_count=deck)


def _start_round(controller: BaqueGameController) -> None:
    controller._sync_tracker(_snapshot(("8", "J", "A"), 56))
    assert controller._round_gold_ranks == ("8", "J", "A")
    controller.opponent_event_count = 5  # 用来判断记牌是否被清空


def test_single_gold_misread_mid_round_keeps_tracker_and_round_gold(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _start_round(controller)

    misread = _snapshot(("6", "J", "A"), 31)
    controller._sync_tracker(misread)

    assert misread.gold_ranks == ("8", "J", "A"), "决策必须按本局真实金牌计算"
    assert controller._round_gold_ranks == ("8", "J", "A")
    assert controller._last_gold_ranks == ("8", "J", "A"), "回合内快速识别不能复用误读金牌"
    assert controller.opponent_event_count == 5


def test_alternating_misreads_never_accumulate(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _start_round(controller)

    for _ in range(3):
        controller._sync_tracker(_snapshot(("6", "J", "A"), 31))
        controller._sync_tracker(_snapshot(("8", "J", "A"), 30))

    assert controller._round_gold_ranks == ("8", "J", "A")
    assert controller.opponent_event_count == 5


def test_repeated_new_gold_is_accepted_as_new_round(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _start_round(controller)

    controller._sync_tracker(_snapshot(("6", "9", "K"), None))
    assert controller._round_gold_ranks == ("8", "J", "A")
    controller._sync_tracker(_snapshot(("6", "9", "K"), None))

    assert controller._round_gold_ranks == ("6", "9", "K")
    assert controller.opponent_event_count == 0


def test_new_gold_with_near_full_deck_resets_immediately(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _start_round(controller)

    controller._sync_tracker(_snapshot(("6", "9", "K"), 54))

    assert controller._round_gold_ranks == ("6", "9", "K")
    assert controller.opponent_event_count == 0


def _frame(tmp_path: Path) -> CapturedFrame:
    return CapturedFrame(tmp_path / "turn.png", 0, 0, 1936, 1088, "欢乐斗地主")


@pytest.mark.asyncio
async def test_transient_swap_probe_mid_round_does_not_end_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    controller._swap_phase_active = False
    controller.round_report.turns = 5  # type: ignore[union-attr]
    report = controller.round_report
    frame = _frame(tmp_path)
    # 摸牌飞牌动画盖住按钮区：像素门瞬间读成“换牌”，随后恢复为摸牌。
    probes = iter([TurnProbe("swap", True, frame), TurnProbe("draw", True, frame)])
    monkeypatch.setattr("game_baque.controller.probe_game_turn", lambda _d, _k: next(probes))
    states: list[str] = []

    class FakeVision:
        async def parse_turn(self, _path: Path, state: str, **_kwargs: object) -> GameSnapshot:
            states.append(state)
            return GameSnapshot(screen_type=ScreenType.DRAW)

    controller.vision = FakeVision()  # type: ignore[assignment]
    await controller._wait_for_turn(Event())

    assert states == ["draw"], "瞬时换牌误判不得交给视觉按换牌识别"
    assert controller.round_report is report
    assert controller.round_log.load() == []
    assert controller._swap_phase_active is False


@pytest.mark.asyncio
async def test_persistent_swap_probe_starts_new_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    controller._swap_phase_active = False
    controller.round_report.turns = 5  # type: ignore[union-attr]
    frame = _frame(tmp_path)
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _d, _k: TurnProbe("swap", True, frame),
    )
    states: list[str] = []

    class FakeVision:
        async def parse_turn(self, _path: Path, state: str, **_kwargs: object) -> GameSnapshot:
            states.append(state)
            return GameSnapshot(screen_type=ScreenType.SWAP)

    controller.vision = FakeVision()  # type: ignore[assignment]
    await controller._wait_for_turn(Event())

    assert states == ["swap"]
    assert controller._swap_phase_active is True
    rows = controller.round_log.load()
    assert len(rows) == 1 and rows[0]["end_reason"] == "下一局开始（新一局换牌）"
