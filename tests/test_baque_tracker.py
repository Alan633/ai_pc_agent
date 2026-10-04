from game_baque.models import Card
from game_baque.tracker import TableTracker, initial_deck_counts


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def test_deck_is_84_cards() -> None:
    counts = initial_deck_counts()
    assert sum(counts.values()) == 84
    assert counts["8:S"] == 3
    assert counts["K:S"] == 2
    assert counts["WILD"] == 4
    assert counts["OPPORTUNITY"] == 4


def test_known_visible_take_then_discard_is_tracked() -> None:
    tracker = TableTracker()
    tracker.reset_round([c("K"), c("K", "H"), c("10"), c("9"), c("8"), c("7"), c("6")])
    tracker.record_opponent_take_visible("top", c("A", "D"))
    assert tracker.opponents["top"].known_hand["A:D"] == 1
    assert tracker.opponents["top"].hand_count == 8

    tracker.record_opponent_discard("top", c("A", "D"))
    assert tracker.opponents["top"].known_hand["A:D"] == 0
    assert tracker.opponents["top"].hand_count == 7
    assert tracker.public_discards["A:D"] == 1


def test_unknown_deck_draw_and_discard() -> None:
    tracker = TableTracker()
    tracker.reset_round([c("K"), c("K", "H"), c("10"), c("9"), c("8"), c("7"), c("6")])
    tracker.record_opponent_draw_deck("left")
    assert tracker.opponents["left"].unknown_hand_count == 8
    tracker.record_opponent_discard("left", c("Q", "D"))
    assert tracker.opponents["left"].unknown_hand_count == 7
    assert tracker.public_discards["Q:D"] == 1


def test_opportunity_returns_three_known_cards_to_top() -> None:
    tracker = TableTracker()
    tracker.reset_round([c("K"), c("K", "H"), c("10"), c("9"), c("8"), c("7"), c("6")])
    options = [c("A", "S"), c("8", "H"), c("Q", "C"), c("10", "D")]
    tracker.record_opportunity_reveal("right", options, 1)
    assert tracker.opponents["right"].known_hand["8:H"] == 1
    assert tracker.known_deck_top == ["A:S", "Q:C", "10:D"]
    assert tracker.draw_distribution() == {"A:S": 1.0}
