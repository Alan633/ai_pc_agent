from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from game_baque.event_vision import LOCAL_EVENT_PROMPT, _local_event_image_url
from game_baque.models import Card, GameSnapshot, Point, ScreenType
from game_baque.vision import (
    LOCAL_VISION_PROMPT,
    LOCAL_VISION_SIZE,
    SNAPSHOT_SCHEMA,
    BaqueVision,
)


def test_local_vision_prompt_is_compact() -> None:
    # 本地 4096 ctx 还要给图片留大量 token；禁止再次把完整 schema 塞进 prompt。
    assert len(LOCAL_VISION_PROMPT) < 2600
    assert "additionalProperties" not in LOCAL_VISION_PROMPT
    assert "start_button" in LOCAL_VISION_PROMPT
    assert len(str(SNAPSHOT_SCHEMA)) > 1000


def test_local_event_prompt_is_compact() -> None:
    assert len(LOCAL_EVENT_PROMPT) < 1800
    assert "additionalProperties" not in LOCAL_EVENT_PROMPT
    assert "BEFORE" in LOCAL_EVENT_PROMPT


def test_context_length_error_detection() -> None:
    error = RuntimeError(
        'request (4123 tokens) exceeds the available context size (4096 tokens)'
    )
    assert BaqueVision._is_context_length_error(error) is True
    assert BaqueVision._is_context_length_error(RuntimeError("connection refused")) is False


def test_local_event_image_is_cropped(tmp_path: Path) -> None:
    source = tmp_path / "frame.jpg"
    Image.new("RGB", (1000, 600), "white").save(source, "JPEG")

    url = _local_event_image_url(source, "left")

    assert url.startswith("data:image/jpeg;base64,")
    # helper 能成功生成裁剪后的 data URL 即可；具体 JPEG 字节会因 Pillow 版本不同而变化。
    assert len(url) > 100


def test_local_vision_image_uses_fixed_coordinate_canvas(tmp_path: Path) -> None:
    source = tmp_path / "frame.png"
    Image.new("RGB", (1936, 1080), "white").save(source, "PNG")

    url, original_size = BaqueVision._local_data_url(source)
    encoded = url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == LOCAL_VISION_SIZE
    assert original_size == (1936, 1080)


def test_local_vision_coordinates_are_scaled_back_to_window_pixels() -> None:
    snapshot = GameSnapshot(
        screen_type=ScreenType.START,
        hand=[Card(rank="6", suit="S", x=250, y=900)],
        start_button=Point(500, 700),
    )

    scaled = BaqueVision._scale_local_snapshot(snapshot, (1936, 1080))

    assert scaled.start_button == Point(968, 756)
    assert scaled.hand[0].x == 484
    assert scaled.hand[0].y == 972


def test_start_without_button_is_normalized_to_waiting() -> None:
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "start",
            "gold_ranks": [],
            "hand": [],
            "visible_cards": [],
            "opponent_events": [],
            "deck_click": {"x": 0, "y": 0, "visible": False},
            "swap_button": {"x": 0, "y": 0, "visible": False},
            "discard_button": {"x": 0, "y": 0, "visible": False},
            "hu_button": {"x": 0, "y": 0, "visible": False},
            "start_button": {"x": 0, "y": 0, "visible": False},
            "continue_button": {"x": 0, "y": 0, "visible": False},
            "close_popup": {"x": 0, "y": 0, "visible": False},
            "deck_count": -1,
            "message": "",
        }
    )

    assert snapshot.screen_type is ScreenType.WAITING
    assert snapshot.start_button is None


def test_dirty_local_card_fields_are_normalized_without_crashing() -> None:
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "swap",
            "gold_ranks": [
                {"rank": "6"},
                {"rank": "8"},
                {"rank": "9"},
            ],
            "hand": [
                {
                    "rank": "K",
                    "suit": "S",
                    "wild": False,
                    "x": 300,
                    "y": 800,
                    "confidence": 0.99,
                }
            ],
            "visible_cards": [
                {
                    "rank": "56",
                    "suit": "NONE",
                    "wild": False,
                    "x": 500,
                    "y": 400,
                    "confidence": 0.99,
                }
            ],
            "opponent_events": [],
        }
    )

    assert snapshot.gold_ranks == ("6", "8", "9")
    assert [card.rank for card in snapshot.hand] == ["K"]
    assert snapshot.visible_cards == []


def test_hand_count_overrides_false_start_before_visible_cards_are_filtered() -> None:
    hand = [
        {
            "rank": rank,
            "suit": "S",
            "wild": False,
            "x": 200 + index * 80,
            "y": 850,
            "confidence": 0.99,
            "selected": False,
        }
        for index, rank in enumerate(["6", "7", "8", "9", "10", "J", "Q"])
    ]
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "start",
            "gold_ranks": ["6", "8", "9"],
            "hand": hand,
            "visible_cards": [
                {
                    "rank": "K",
                    "suit": "H",
                    "wild": False,
                    "x": 600,
                    "y": 400,
                    "confidence": 0.99,
                    "selected": False,
                }
            ],
            "start_button": {"x": 500, "y": 200, "visible": True},
        }
    )

    assert snapshot.screen_type is ScreenType.DRAW
    assert len(snapshot.hand) == 7
    assert [card.rank for card in snapshot.visible_cards] == ["K"]
    assert snapshot.start_button is None


def test_six_cards_override_false_start_as_swap() -> None:
    hand = [
        {
            "rank": rank,
            "suit": "H",
            "wild": False,
            "x": 250 + index * 90,
            "y": 850,
            "confidence": 0.99,
            "selected": False,
        }
        for index, rank in enumerate(["6", "7", "8", "9", "10", "J"])
    ]
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "start",
            "gold_ranks": ["6", "8", "9"],
            "hand": hand,
            "visible_cards": [],
            "swap_button": {"x": 500, "y": 700, "visible": True},
            "start_button": {"x": 500, "y": 200, "visible": True},
        }
    )

    assert snapshot.screen_type is ScreenType.SWAP
    assert len(snapshot.hand) == 6
    assert snapshot.swap_button == Point(500, 700)
    assert snapshot.start_button is None


def test_swap_cards_raised_higher_are_treated_as_selected() -> None:
    hand = [
        {
            "rank": rank,
            "suit": "H",
            "wild": False,
            "x": 250 + index * 90,
            "y": 790 if index in {1, 4} else 850,
            "confidence": 0.99,
            "selected": False,
        }
        for index, rank in enumerate(["6", "7", "8", "9", "10", "J"])
    ]
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "swap",
            "gold_ranks": ["6", "8", "9"],
            "hand": hand,
            "visible_cards": [],
        }
    )

    assert [index for index, card in enumerate(snapshot.hand) if card.selected] == [1, 4]


def test_visible_swap_button_overrides_seven_card_draw_inference() -> None:
    hand = [
        {
            "rank": rank,
            "suit": "S",
            "wild": False,
            "x": 200 + index * 80,
            "y": 850,
            "confidence": 0.99,
            "selected": False,
        }
        for index, rank in enumerate(["6", "7", "8", "9", "10", "J", "Q"])
    ]
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "draw",
            "gold_ranks": ["6", "8", "9"],
            "hand": hand,
            "visible_cards": [],
            "swap_button": {"x": 520, "y": 680, "visible": True},
        }
    )

    assert snapshot.screen_type is ScreenType.SWAP
    assert len(snapshot.hand) == 7
    assert snapshot.swap_button == Point(520, 680)


def test_question_opportunity_is_not_kept_as_wild_visible_card() -> None:
    snapshot = BaqueVision._to_snapshot(
        {
            "screen_type": "draw",
            "gold_ranks": ["7", "K", "A"],
            "hand": [
                {
                    "rank": rank,
                    "suit": "H",
                    "wild": False,
                    "x": 200 + index * 80,
                    "y": 850,
                    "confidence": 0.99,
                    "selected": False,
                }
                for index, rank in enumerate(["6", "7", "8", "9", "10", "J", "Q"])
            ],
            "visible_cards": [
                {
                    "rank": "WILD",
                    "suit": "NONE",
                    "wild": True,
                    "x": 430,
                    "y": 380,
                    "confidence": 0.99,
                    "selected": False,
                }
            ],
            "opportunity_click": {"x": 430, "y": 380, "visible": True},
        }
    )

    assert snapshot.opportunity_click == Point(430, 380)
    assert snapshot.visible_cards == []
