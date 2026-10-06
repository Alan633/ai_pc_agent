from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.capture import CapturedFrame, TurnProbe
from game_baque.controller import BaqueGameController
from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card, GameSnapshot, ScreenType
from game_baque.plan_report import target_line
from game_baque.strategy import choose_swap_two

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
GOLD = ("6", "7", "A")


def cards(text: str) -> list[Card]:
    return [Card(wild=True) if item == "癞子" else Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def test_target_line_follows_mode() -> None:
    flush = ExpectedValueEngine()
    flush.prefer_flush = True
    assert "同花顺 ♠7~Q" in target_line(flush, cards("癞子 Q♠ J♠ 9♠ 8♠ 7♠ K♥"), GOLD)
    big = ExpectedValueEngine()
    big.chase_big = True
    text = target_line(big, cards("6♠ 6♥ 6♦ 7♠ 7♥ A♠ K♦"), GOLD)
    assert "金牌 6 张" in text and "六事兴旺" in text


def test_swap_uses_strategy_mode() -> None:
    # 金牌多、但同花顺子也不错的 7 张：两种模式换掉的牌不同。
    hand = cards("6♠ 6♥ 7♠ 8♠ 9♠ 10♠ A♦")
    flush = choose_swap_two(hand, GOLD, prefer_flush=True).card_indexes
    big = choose_swap_two(hand, GOLD, chase_big=True).card_indexes
    kept_flush = [hand[i] for i in range(7) if i not in flush]
    kept_big = [hand[i] for i in range(7) if i not in big]
    assert {"7♠", "8♠", "9♠", "10♠"} <= {c.label for c in kept_flush}  # 同花顺优先：保住黑桃顺子
    gold_count = lambda kept: sum(c.rank in GOLD for c in kept)  # noqa: E731
    assert gold_count(kept_big) > gold_count(kept_flush)  # 追大牌：留下更多金牌


def test_controller_swap_passes_current_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_swap(cards_, gold_, *, prefer_flush=False, chase_big=False, prefer_pinghu=False):
        seen.update(prefer_flush=prefer_flush, chase_big=chase_big)
        return choose_swap_two(cards_, gold_)

    monkeypatch.setattr("game_baque.controller.choose_swap_two", fake_swap)
    controller = BaqueGameController(vision=object(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller.ev.chase_big = True
    snapshot = GameSnapshot(screen_type=ScreenType.SWAP, gold_ranks=GOLD, hand=cards("6♠ 6♥ 7♠ 8♠ 9♠ 10♠ A♦"))
    controller._decide(snapshot, controller.tracker)
    assert seen == {"prefer_flush": False, "chase_big": True}


@pytest.mark.asyncio
async def test_target_announced_from_idle_hand_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frame = CapturedFrame(tmp_path / "f.png", 0, 0, 1920, 1080, "欢乐斗地主")
    monkeypatch.setattr("game_baque.controller.capture_game_window", lambda *_a: frame)

    class FakeVision:
        async def parse_hand(self, _path: Path) -> GameSnapshot:
            return GameSnapshot(screen_type=ScreenType.DRAW, hand=cards("癞子 Q♠ J♠ 9♠ 8♠ 7♠ K♥"))

    controller = BaqueGameController(vision=FakeVision(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller._round_gold_ranks = GOLD
    controller.ev.prefer_flush = True
    controller._target_pending = True
    with caplog.at_level(logging.INFO, logger="game_baque.controller"):
        await controller._announce_target_from_idle()
    lines = [r.getMessage() for r in caplog.records if "【本局目标】" in r.getMessage()]
    assert len(lines) == 1 and "同花顺 ♠7~Q" in lines[0]
    assert controller._target_pending is False


@pytest.mark.asyncio
async def test_round_mode_read_before_first_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = CapturedFrame(tmp_path / "f.png", 0, 0, 1920, 1080, "欢乐斗地主")
    monkeypatch.setattr("game_baque.controller.capture_game_window", lambda *_a: frame)

    class FakeVision:
        async def parse_round_info(self, _path: Path):
            return 240.0, [5_000_000.0, 5_000_000.0, 5_000_000.0], 3_000_000.0

        async def parse_own_beans(self, _path: Path):
            return 9_000_000.0

    controller = BaqueGameController(vision=FakeVision(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    await controller._ensure_round_mode()
    assert controller.ev.chase_big is True  # 高级场、自己豆够多：换牌就按追大牌
    await controller._ensure_round_mode()  # 已就绪不再重读
