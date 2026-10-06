from __future__ import annotations

import io
import logging
import random
from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.controller import BaqueGameController
from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card, GameDecision, GameSnapshot, ScreenType
from game_baque.plan_report import opponent_plan_lines, own_plan_line
from game_baque.tracker import TableTracker
from logging_setup import RedConsoleFormatter

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
GOLD = ("6", "7", "8")


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def _tracker(hand: list[Card], deck: int = 40) -> TableTracker:
    tracker = TableTracker()
    tracker.reset_round(hand, deck)
    return tracker


def test_own_plan_lists_waits_when_tenpai() -> None:
    kept = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣")
    line = own_plan_line(ExpectedValueEngine(), kept, GOLD, _tracker(kept))
    assert "已听牌" in line
    assert "A♦" in line  # A♣ 配 A♦ 成对即可胡


def test_own_plan_estimates_patterns_when_not_tenpai() -> None:
    kept = cards("6♥ 9♣ J♦ Q♠ A♣ 10♥ K♦")
    line = own_plan_line(
        ExpectedValueEngine(), kept, GOLD, _tracker(kept), samples=40, rng=random.Random(1)
    )
    assert "未听牌" in line


def test_opponent_plan_names_seats_in_turn_order() -> None:
    kept = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣")
    lines = opponent_plan_lines(ExpectedValueEngine(simulations=60, seed=1), GOLD, _tracker(kept))
    assert [line.split("(")[0] for line in lines] == ["上家", "对家", "下家"]


def test_red_formatter_colors_only_real_consoles() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "牌型计划", (), None)

    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    plain = RedConsoleFormatter(io.StringIO(), "%(message)s", "%H")
    assert plain.format(record) == "牌型计划"

    tty = RedConsoleFormatter(FakeTty(), "%(message)s", "%H")
    tty.colorize = True  # 测试环境没有真实控制台句柄
    assert tty.format(record) == "\x1b[91m牌型计划\x1b[0m"


@pytest.mark.asyncio
async def test_plan_is_logged_red_after_discard(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    hand = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ 9♣")
    snapshot = GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=GOLD, hand=hand)

    with caplog.at_level(logging.INFO, logger="game_baque.controller"):
        await controller._log_plan(snapshot, GameDecision("discard", card_indexes=(7,)), _tracker(hand))

    plan = [record for record in caplog.records if "【牌型计划】" in record.getMessage()]
    assert len(plan) == 5  # 目标进展 + 自己 + 三家
    assert all(getattr(record, "console_red", False) for record in plan)
    assert "目标进展" in plan[0].getMessage()
    assert "自己" in plan[1].getMessage() and "已听牌" in plan[1].getMessage()
