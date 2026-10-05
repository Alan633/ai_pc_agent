from __future__ import annotations

from game_balatro.models import (
    BalatroSnapshot,
    PlayingCard,
    Point,
    ScreenType,
)
from game_balatro.solver import (
    BalatroSolver,
    evaluate_hand,
)


def card(
    rank: str,
    suit: str,
    x: int = 0,
) -> PlayingCard:
    return PlayingCard(
        rank=rank,
        suit=suit,
        x=x,
        y=800,
    )


def test_evaluate_straight_flush() -> None:
    value = evaluate_hand(
        (
            card("10", "S"),
            card("J", "S"),
            card("Q", "S"),
            card("K", "S"),
            card("A", "S"),
        )
    )
    assert value.category == "Straight Flush"
    assert value.category_rank == 9


def test_solver_enumerates_best_five_cards() -> None:
    hand = (
        card("10", "S"),
        card("J", "S"),
        card("Q", "S"),
        card("K", "S"),
        card("A", "S"),
        card("2", "H"),
        card("2", "D"),
        card("7", "C"),
    )
    indices, value = BalatroSolver().best_play(
        hand
    )
    assert indices == (
        0,
        1,
        2,
        3,
        4,
    )
    assert value.category == "Straight Flush"


def test_weak_hand_uses_discard_when_available() -> None:
    # 刻意避开 A2345 等隐藏顺子，确保这是纯 High Card 弱牌。
    hand = (
        card("A", "S"),
        card("K", "H"),
        card("9", "D"),
        card("7", "C"),
        card("6", "S"),
        card("4", "H"),
        card("3", "D"),
        card("2", "C"),
    )
    snapshot = BalatroSnapshot(
        screen_type=ScreenType.PLAY,
        target_score=300,
        current_score=0,
        hands_left=4,
        discards_left=3,
        hand=hand,
        play_button=Point(
            800,
            650,
        ),
        discard_button=Point(
            650,
            650,
        ),
    )
    decision = BalatroSolver().decide(
        snapshot
    )
    assert decision.action == "discard"
    assert (
        decision.target
        == snapshot.discard_button
    )
    assert (
        1
        <= len(
            decision.card_indices
        )
        <= 5
    )
