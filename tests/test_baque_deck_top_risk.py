from __future__ import annotations

from collections import Counter

from game_baque.ev import DECK_TOP_ORDER, ExpectedValueEngine
from game_baque.models import Card
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker, card_key

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
GOLD = ("6", "7", "8")


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


HAND = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣")
RIGHT_HAND = cards("9♠ 9♥ 9♣ 10♠ 10♥ 10♣ 10♦")  # 下家：再来一张 9♦ 就胡
OPTIONS = cards("9♦ A♦ K♦ 6♠")


def _tracker() -> TableTracker:
    tracker = TableTracker()
    tracker.reset_round(HAND, 40)
    right = tracker.opponents["right"]
    right.known_hand = Counter(card_key(card) for card in RIGHT_HAND)
    right.unknown_hand_count = 0
    return tracker


def test_deck_top_order_is_next_three_players() -> None:
    assert DECK_TOP_ORDER == ("right", "top", "left")


def test_choice_avoids_leaving_opponent_winning_card_on_deck_top() -> None:
    assert evaluate_hu(RIGHT_HAND + cards("9♦"), GOLD) is not None
    engine = ExpectedValueEngine(simulations=200, seed=1)

    plain_index, _ = engine.choose_option_card(HAND, OPTIONS, GOLD)
    risky_returned_first = [j for j in range(4) if j != plain_index][0]
    # 前提：只看自己时不选 9♦，于是 9♦ 回到牌堆顶第一张，正好被下家摸到。
    assert plain_index != 0 and OPTIONS[risky_returned_first].label == "9♦"

    index, _ = engine.choose_option_card(HAND, OPTIONS, GOLD, _tracker(), time_budget_seconds=3.0)

    returned = [OPTIONS[j].label for j in range(4) if j != index]
    assert returned[0] != "9♦", "不能把下家要胡的牌留在牌堆顶第一张"
    assert "牌堆顶送胡风险" in engine.last_option_notes


def test_choice_without_tracker_ignores_deck_top_risk() -> None:
    engine = ExpectedValueEngine()
    engine.choose_option_card(HAND, OPTIONS, GOLD)
    assert "牌堆顶送胡风险" not in engine.last_option_notes
