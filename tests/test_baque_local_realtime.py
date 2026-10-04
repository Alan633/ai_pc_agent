from pathlib import Path

from agent.session import AgentSession
from game_baque.controller import BaqueGameController
from game_baque.models import Card, GameSnapshot, OpponentEvent, ScreenType
from game_baque.vision import BaqueVision


def c(rank: str, suit: str = "S") -> Card:
    return Card(rank=rank, suit=suit)


def base_snapshot(events: list[OpponentEvent] | None = None) -> GameSnapshot:
    return GameSnapshot(
        screen_type=ScreenType.WAITING,
        gold_ranks=("6", "10", "K"),
        hand=[
            c("6", "S"),
            c("7", "H"),
            c("8", "C"),
            c("9", "D"),
            c("10", "S"),
            c("J", "H"),
            c("Q", "C"),
        ],
        opponent_events=events or [],
        deck_count=56,
    )


def test_local_vision_payload_parses_opponent_discard() -> None:
    payload = {
        "screen_type": "waiting",
        "gold_ranks": ["6", "10", "K"],
        "hand": [],
        "visible_cards": [],
        "opponent_events": [
            {
                "seat": "left",
                "action": "discard",
                "card_rank": "A",
                "card_suit": "D",
                "card_wild": False,
                "confidence": 0.96,
                "detail": "左家新打出一张A方片",
            }
        ],
        "deck_click": {"x": 0, "y": 0, "visible": False},
        "swap_button": {"x": 0, "y": 0, "visible": False},
        "discard_button": {"x": 0, "y": 0, "visible": False},
        "hu_button": {"x": 0, "y": 0, "visible": False},
        "continue_button": {"x": 0, "y": 0, "visible": False},
        "close_popup": {"x": 0, "y": 0, "visible": False},
        "deck_count": 50,
        "message": "",
    }

    snapshot = BaqueVision._to_snapshot(payload)
    assert len(snapshot.opponent_events) == 1
    event = snapshot.opponent_events[0]
    assert event.seat == "left"
    assert event.action == "discard"
    assert event.card is not None
    assert event.card.rank == "A"
    assert event.card.suit == "D"
    assert event.confidence == 0.96


def test_controller_writes_and_deduplicates_opponent_events(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        min_event_confidence=0.80,
    )
    controller._sync_tracker(base_snapshot())

    events = [
        OpponentEvent("top", "draw_deck", confidence=0.95),
        OpponentEvent("top", "discard", c("A", "D"), confidence=0.97),
    ]
    snapshot = base_snapshot(events)
    controller._apply_opponent_events(snapshot)

    top = controller.tracker.opponents["top"]
    assert top.unknown_hand_count == 7
    assert top.deck_draws == 1
    assert top.discards == ["A:D"]
    assert controller.tracker.public_discards["A:D"] == 1
    assert controller.opponent_event_count == 2

    # 同一视觉事件可能连续出现多帧；短时间重复不得再次扣牌。
    controller._apply_opponent_events(snapshot)
    assert top.unknown_hand_count == 7
    assert top.deck_draws == 1
    assert top.discards == ["A:D"]
    assert controller.tracker.public_discards["A:D"] == 1
    assert controller.opponent_event_count == 2


def test_controller_ignores_low_confidence_opponent_event(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        min_event_confidence=0.80,
    )
    controller._sync_tracker(base_snapshot())
    snapshot = base_snapshot([
        OpponentEvent("right", "discard", c("K", "H"), confidence=0.60)
    ])

    controller._apply_opponent_events(snapshot)

    assert controller.tracker.opponents["right"].discards == []
    assert controller.opponent_event_count == 0
