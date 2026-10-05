from game_baque.models import Card
from game_baque.ev import ExpectedValueEngine
from game_baque.strategy import choose_discard, choose_draw, choose_swap_two
from game_baque.tracker import TableTracker


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def w() -> Card:
    return Card(wild=True)


def test_swap_does_not_throw_away_wildcard_in_normal_hand() -> None:
    cards = [w(), c("K", "S"), c("K", "H"), c("10", "D"), c("9", "C"), c("7", "S")]
    decision = choose_swap_two(cards, ("6", "10", "K"))
    assert 0 not in decision.card_indexes
    assert len(decision.card_indexes) == 2


def test_discard_detects_immediate_hu() -> None:
    cards = [
        c("K", "S"), c("K", "H"), c("K", "C"), c("K", "D"), c("K", "S"), c("K", "H"),
        c("A", "S"), c("A", "H"),
    ]
    decision = choose_discard(cards, ("6", "10", "K"))
    assert decision.action == "hu"
    assert decision.hu is not None
    assert decision.hu.pattern == "顶峰相见"


def test_draw_takes_visible_winning_card() -> None:
    hand = [
        c("K", "S"), c("K", "H"), c("K", "C"), c("K", "D"), c("K", "S"), c("K", "H"),
        c("A", "S"),
    ]
    visible = [c("A", "H"), c("7", "D")]
    decision = choose_draw(hand, visible, ("6", "10", "K"))
    assert decision.action == "draw_visible"
    assert decision.visible_index == 0


def test_ev_question_card_selects_target_before_question_confirmation() -> None:
    hand = [
        c("A", "H"),
        c("A", "D"),
        c("Q", "D"),
        c("J", "S"),
        c("10", "D"),
        c("7", "H"),
    ]
    tracker = TableTracker()
    tracker.reset_round(hand, deck_count=55)

    decision = ExpectedValueEngine(simulations=20, seed=7).choose_draw_action(
        hand,
        [],
        ("7", "K", "A"),
        tracker,
        opportunity_available=True,
    )

    assert decision.action == "draw_opportunity"
    assert decision.visible_index is None
