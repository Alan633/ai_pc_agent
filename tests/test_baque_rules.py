from game_baque.models import Card
from game_baque.rules import evaluate_hu, gold_multiplier


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def w() -> Card:
    return Card(wild=True)


def test_gold_multiplier_table() -> None:
    assert [gold_multiplier(i) for i in range(8)] == [1, 1, 2, 4, 8, 16, 32, 32]


def test_top_meeting_6_plus_2() -> None:
    cards = [
        c("K", "S"), c("K", "H"), c("K", "C"), c("K", "D"), c("K", "S"), c("K", "H"),
        c("A", "S"), c("A", "H"),
    ]
    result = evaluate_hu(cards, ("6", "10", "K"))
    assert result is not None
    assert result.pattern == "顶峰相见"
    assert result.pattern_multiplier == 32
    assert result.gold_count == 6


def test_eight_card_straight() -> None:
    cards = [c(rank, "S") for rank in ("6", "7", "8", "9", "10", "J", "Q", "K")]
    result = evaluate_hu(cards, ("6", "10", "K"))
    assert result is not None
    assert result.pattern == "八方来贺"
    assert result.pattern_multiplier == 50


def test_wildcard_can_complete_gold_rank() -> None:
    cards = [
        c("K", "S"), c("K", "H"), c("K", "C"), c("K", "D"), c("K", "S"), w(),
        c("A", "S"), c("A", "H"),
    ]
    result = evaluate_hu(cards, ("6", "10", "K"))
    assert result is not None
    assert result.pattern == "顶峰相见"
    assert result.hard is False
    assert result.gold_count == 6
    assert any(item.endswith(":K") for item in result.wildcard_assignments)


def test_ping_hu_3_3_2() -> None:
    cards = [
        c("6", "S"), c("6", "H"), c("6", "C"),
        c("9", "S"), c("9", "H"), c("9", "C"),
        c("A", "S"), c("A", "H"),
    ]
    result = evaluate_hu(cards, ("6", "10", "K"))
    assert result is not None
    assert result.pattern == "平胡"
