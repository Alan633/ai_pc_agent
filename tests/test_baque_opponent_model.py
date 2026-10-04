from game_baque.models import Card
from game_baque.opponent_model import OpponentPatternModel
from game_baque.tracker import TableTracker


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def test_known_six_k_makes_a_dangerous_for_opponent() -> None:
    tracker = TableTracker()
    tracker.reset_round([
        c("6", "S"), c("7", "S"), c("8", "S"), c("9", "S"),
        c("10", "S"), c("J", "S"), c("Q", "S"),
    ])

    top = tracker.opponents["top"]
    top.known_hand.clear()
    top.unknown_hand_count = 0
    for suit in ("S", "H", "C", "D"):
        top.known_hand[f"K:{suit}"] += 1
    top.known_hand["K:S"] += 1
    top.known_hand["K:H"] += 1
    top.known_hand["A:S"] += 1

    model = OpponentPatternModel(samples=120, seed=7)
    risk = model.release_risk(c("A", "D"), tracker, ("6", "10", "K"), samples=120)
    top_risk = next(item for item in risk if item.seat == "top")
    assert top_risk.hu_probability > 0.90
    assert any(name == "顶峰相见" for name, _ in top_risk.dominant_patterns)


def test_inference_returns_pattern_probabilities() -> None:
    tracker = TableTracker()
    tracker.reset_round([
        c("6", "S"), c("7", "H"), c("8", "C"), c("9", "D"),
        c("10", "S"), c("J", "H"), c("Q", "C"),
    ])
    tracker.record_opponent_take_visible("right", c("K", "D"))
    tracker.record_opponent_take_visible("right", c("K", "C"))
    tracker.record_opponent_discard("right", c("6", "D"))
    model = OpponentPatternModel(samples=100, seed=11)
    inferred = model.infer_all(tracker, ("6", "10", "K"), samples=100)
    assert len(inferred) == 3
    right = next(item for item in inferred if item.seat == "right")
    assert right.known_cards >= 2
    assert 0.0 <= right.any_hu_probability_next_draw <= 1.0
