from game_baque.models import Card
from game_baque.tracker import TableTracker


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def starting_hand() -> list[Card]:
    return [
        c("6", "S"),
        c("7", "H"),
        c("8", "C"),
        c("9", "D"),
        c("10", "S"),
        c("J", "H"),
        c("Q", "C"),
    ]


def test_unknown_draw_then_discard_keeps_opponent_at_seven() -> None:
    tracker = TableTracker()
    tracker.reset_round(starting_hand())
    tracker.record_opponent_turn(
        "left",
        draw_source="unknown",
        discard=c("K", "D"),
    )
    assert tracker.opponents["left"].hand_count == 7
    assert tracker.public_discards["K:D"] == 1


def test_visible_draw_then_discard_preserves_known_information() -> None:
    tracker = TableTracker()
    tracker.reset_round(starting_hand())
    tracker.record_opponent_turn(
        "top",
        draw_source="visible",
        drawn_card=c("A", "D"),
        discard=c("6", "D"),
    )
    assert tracker.opponents["top"].hand_count == 7
    assert tracker.opponents["top"].known_hand["A:D"] == 1
    assert tracker.public_discards["6:D"] == 1
