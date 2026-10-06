from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from game_baque.suit_glyph import classify_suit
from game_baque.vision import BaqueVision

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "suit_glyphs").glob("*.png"))


def test_fixtures_cover_all_suits() -> None:
    suits = {path.stem.split("_")[0][-1] for path in FIXTURES}
    assert suits == {"S", "C", "H", "D"}


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_real_card_corners_are_classified(path: Path) -> None:
    # 素材取自实战截图（已逐张核对），包含 Qwen 曾读错的 J♠/Q♣/Q♦/9♣ 等。
    cell = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    assert classify_suit(cell) == path.stem.split("_")[0][-1]


def test_empty_cell_is_undecided() -> None:
    assert classify_suit(np.full((190, 96, 3), 225, dtype=np.int16)) is None


def test_pixel_suits_override_model_and_detect_impossible_hands() -> None:
    items = [{"rank": "J", "suit": "C", "up": False}] * 3 + [{"rank": "WILD", "suit": "NONE", "up": False}]
    assert BaqueVision._exceeds_deck(items)
    fixed = BaqueVision._apply_pixel_suits(items, ["S", None, None, "H"])
    assert [item["suit"] for item in fixed] == ["S", "C", "C", "NONE"]
    assert not BaqueVision._exceeds_deck(fixed)
    # 张数对不上时不能按位置覆盖。
    assert BaqueVision._apply_pixel_suits(items, ["S"]) == items


def test_rank_only_tokens_are_filled_by_pixel_suits() -> None:
    # Qwen 偶尔整手只给点数（"K20"、"1010"）：由像素花色补齐，补不上的丢弃以触发重读。
    items = BaqueVision._compact_hand_items(["K20", "1010", "WILD", "QH"])
    assert [(item["rank"], item["suit"]) for item in items] == [
        ("K", ""), ("10", ""), ("WILD", "NONE"), ("Q", "H")
    ]
    filled = BaqueVision._apply_pixel_suits(items, ["H", "D", None, "D"])
    assert [(item["rank"], item["suit"]) for item in filled] == [
        ("K", "H"), ("10", "D"), ("WILD", "NONE"), ("Q", "D")
    ]
    partial = BaqueVision._apply_pixel_suits(items, ["H", None, None, None])
    assert len(partial) == 3  # 10 没有花色，被丢弃


@pytest.mark.asyncio
async def test_action_recheck_survives_brief_overlay(tmp_path: Path) -> None:
    from agent.session import AgentSession
    from game_baque.controller import BaqueGameController

    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    states = iter([None, None, "discard"])  # 互动表情飘过：横幅短暂消失

    async def current_state() -> str | None:
        return next(states)

    controller._current_action_state = current_state  # type: ignore[method-assign]
    await controller._require_action_state("discard")
