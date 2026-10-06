from __future__ import annotations

import pytest

from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card
from game_baque.opponent_model import capped_payment
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
GOLD = ("6", "7", "A")


def cards(text: str) -> list[Card]:
    return [Card(wild=True) if item == "癞子" else Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


HU = evaluate_hu(cards("癞子 Q♠ J♠ 10♠ 9♠ 8♠ 7♠ K♥"), GOLD)


def test_self_draw_doubles_value_before_caps() -> None:
    assert HU is not None
    engine = ExpectedValueEngine()
    assert engine.hu_value(HU, self_draw=True) == 2 * engine.hu_value(HU)
    # 封顶在加倍之后才生效：封顶很低时自摸也收不到更多。
    engine.score_caps = (10.0, 10.0, 10.0)
    assert engine.hu_value(HU, self_draw=True) == engine.hu_value(HU) == 10.0


def test_capped_payment_self_draw_for_opponent_hu() -> None:
    assert HU is not None
    assert capped_payment(HU, None, self_draw=True) == 2 * capped_payment(HU, None)


def test_hu_now_from_deck_is_worth_double() -> None:
    hand = cards("癞子 Q♠ J♠ 10♠ 9♠ 8♠ 7♠ K♥")
    tracker = TableTracker()
    tracker.reset_round(hand, 40)
    engine = ExpectedValueEngine(simulations=30, seed=1)
    engine.repeat_hu_value = lambda *_a, **_k: 0.0  # type: ignore[method-assign]  # 只比较这一次胡
    _, from_visible = engine.choose_8_card_action(hand, GOLD, tracker, time_budget_seconds=0.5, self_draw=False)
    _, from_deck = engine.choose_8_card_action(hand, GOLD, tracker, time_budget_seconds=0.5, self_draw=True)
    hu_visible = next(item for item in from_visible if item.action == "hu")
    hu_deck = next(item for item in from_deck if item.action == "hu")
    assert hu_deck.immediate_hu_value == pytest.approx(2 * hu_visible.immediate_hu_value)


def test_repeat_value_counts_own_draws_double() -> None:
    locked = cards("癞子 Q♠ J♠ 10♠ 9♠ 8♠ 7♠")
    tracker = TableTracker()
    tracker.reset_round(locked, 40)
    engine = ExpectedValueEngine()
    with_double = engine.repeat_hu_value(locked, GOLD, tracker, 40)
    engine._repeat_cache.clear()
    real = engine.hu_value
    engine.hu_value = lambda hu, self_draw=False: real(hu)  # type: ignore[method-assign]  # 假设自摸不加倍
    without = engine.repeat_hu_value(locked, GOLD, tracker, 40)
    assert with_double > without
