from __future__ import annotations

from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.controller import BaqueGameController
from game_baque.models import Card
from game_baque.strategy import hand_potential

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


@pytest.mark.parametrize(
    ("base", "cap", "own", "expected"),
    [
        (10.0, 20_000.0, 10_000_000.0, True),  # 新手场：一律同花顺优先
        (35.0, 300_000.0, 10_000_000.0, True),  # 普通场：一律同花顺优先
        (240.0, 3_000_000.0, 25_000.0, True),  # 高级场、自己豆少：同花顺优先
        (800.0, 8_000_000.0, 1_000_000.0, True),  # 顶级场、自己豆不够让对手付满
        (240.0, 3_000_000.0, 8_000_000.0, False),  # 高级场、自己豆够多：可考虑特殊大牌型
        (800.0, 8_000_000.0, 9_000_000.0, False),  # 顶级场、自己豆够多
        (None, None, None, True),  # 底分未知：按同花顺优先
    ],
)
def test_prefer_flush_by_room_and_own_beans(tmp_path: Path, base, cap, own, expected) -> None:
    controller = BaqueGameController(vision=object(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(base, [50_000_000.0] * 3, cap, own_balance=own)
    assert controller.ev.prefer_flush is expected


def test_prefer_flush_widens_gap_toward_straight_flush() -> None:
    gold = ("6", "7", "A")
    run = cards("8♠ 9♠ 10♠ J♠ Q♠ K♥ K♦")
    sets = cards("8♠ 8♥ 8♦ K♥ K♦ K♣ Q♠")
    plain_gap = hand_potential(run, gold) - hand_potential(sets, gold)
    flush_gap = hand_potential(run, gold, prefer_flush=True) - hand_potential(sets, gold, prefer_flush=True)
    assert flush_gap > plain_gap > 0


def test_chase_big_steers_toward_gold_and_special_patterns_only_when_rich(tmp_path: Path) -> None:
    gold = ("6", "7", "A")
    golds = cards("6♠ 6♥ 6♦ 7♠ 7♥ A♠ K♦")  # 5 张金牌：往全金牌、六事兴旺上靠
    run = cards("8♠ 9♠ 10♠ J♠ Q♠ K♥ K♦")  # 无金牌的同花顺子
    assert hand_potential(golds, gold, prefer_flush=True) < hand_potential(run, gold, prefer_flush=True)
    assert hand_potential(golds, gold, chase_big=True) > hand_potential(run, gold, chase_big=True)

    controller = BaqueGameController(vision=object(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(240.0, [50_000_000.0] * 3, 3_000_000.0, own_balance=8_000_000.0)
    assert controller.ev.chase_big and not controller.ev.prefer_flush
    controller._apply_payout_cap(240.0, [50_000_000.0] * 3, 3_000_000.0, own_balance=25_000.0)
    assert not controller.ev.chase_big and controller.ev.prefer_flush


@pytest.mark.parametrize(
    ("opponents", "expected"),
    [
        ([700.0, 0.0, 2_000.0], True),  # 新手场对手都没什么豆：一个平胡就能收满
        ([700.0, 0.0, 39_700.0], False),  # 有一家还有豆（可付满 2 万封顶）：不必只求平胡
    ],
)
def test_pinghu_priority_when_opponents_are_broke(tmp_path: Path, opponents, expected) -> None:
    controller = BaqueGameController(vision=object(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(10.0, opponents, 20_000.0, own_balance=5_700.0)
    assert controller.ev.prefer_pinghu is expected
    if expected:
        assert not controller.ev.prefer_flush and not controller.ev.chase_big


def test_own_beans_low_alone_does_not_trigger_pinghu(tmp_path: Path) -> None:
    # 高级场自己只有 2.5 万、对手都富：按用户规则仍是同花顺优先。
    controller = BaqueGameController(vision=object(), screenshots_dir=tmp_path, session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(240.0, [50_000_000.0] * 3, 3_000_000.0, own_balance=25_000.0)
    assert controller.ev.prefer_flush and not controller.ev.prefer_pinghu


def test_pinghu_mode_favors_pairs_and_triples_without_wild() -> None:
    gold = ("6", "7", "A")
    sets = cards("9♠ 9♥ 9♦ J♠ J♥ J♦ Q♣")  # 两个三张：差一个对子就是 3+3+2
    run = cards("8♠ 9♠ 10♠ J♠ Q♠ K♥ A♦")  # 同花顺子
    assert hand_potential(sets, gold, prefer_pinghu=True) - hand_potential(run, gold, prefer_pinghu=True) > (
        hand_potential(sets, gold, prefer_flush=True) - hand_potential(run, gold, prefer_flush=True)
    )
    with_wild = [Card(wild=True)] + cards("9♠ 9♥ 9♦ J♠ J♥ J♦")
    without = cards("Q♣ 9♠ 9♥ 9♦ J♠ J♥ J♦")
    # 平胡不能带癞子：癞子不再比普通牌更值钱。
    assert hand_potential(with_wild, gold, prefer_pinghu=True) <= hand_potential(without, gold, prefer_pinghu=True) + 30
