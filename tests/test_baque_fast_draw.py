from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from agent.session import AgentSession
from game_baque.capture import CapturedFrame
from game_baque.controller import BaqueGameController
from game_baque.models import Card, GameDecision, GameSnapshot, ScreenType
from game_baque.vision import BaqueVision

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(wild=True) if item == "癞子" else Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


HAND = cards("癞子 J♥ 10♥ 9♥ 8♥ 7♥ Q♦")
ROW = [Card("7", "D", x=438, y=386), Card("K", "C", x=492, y=386), Card("K", "H", x=546, y=386)]
PIXELS = [None, "H", "H", "H", "H", "H", "D"]


class TableOnlyVision(BaqueVision):
    """真实的快照组装，亮牌行读数用替身（记录是否被调用）。"""

    def __init__(self, table: list[Card], deck: int) -> None:
        super().__init__("", "local", "http://127.0.0.1:11434/v1")
        self.table, self.deck, self.table_reads = table, deck, 0

    async def parse_table(self, _path: Path):
        self.table_reads += 1
        return list(self.table), self.deck


def _controller(tmp_path: Path, vision: BaqueVision) -> tuple[BaqueGameController, CapturedFrame]:
    path = tmp_path / "draw.png"
    Image.new("RGB", (1920, 1080), (30, 120, 100)).save(path)
    controller = BaqueGameController(vision=vision, screenshots_dir=tmp_path, session=AgentSession())
    controller._round_gold_ranks = ("6", "8", "10")
    controller._tracker_started = True
    controller.tracker.reset_round(HAND, 44)
    controller._known_hand = (list(HAND), 0)
    controller._last_row_local = (list(ROW), 43)
    controller._last_row_hu = (False, False, False)
    controller._row_diff.cycle_complete = True
    controller._row_prev_signature = controller._row_read_signature = bytes([10]) * 400
    return controller, CapturedFrame(path, 0, 0, 1920, 1080, "欢乐斗地主")


def _pixels(monkeypatch: pytest.MonkeyPatch, suits: list[str | None]) -> None:
    monkeypatch.setattr("game_baque.controller.classify_hand_suits", lambda _image, _n: list(suits))


@pytest.mark.asyncio
async def test_fast_draw_reuses_known_hand_and_row_without_any_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vision = TableOnlyVision([], 0)
    controller, frame = _controller(tmp_path, vision)
    _pixels(monkeypatch, PIXELS)
    snapshot = await controller._fast_draw_snapshot(frame)
    assert snapshot is not None and snapshot.screen_type == ScreenType.DRAW
    assert [card.label for card in snapshot.hand] == [card.label for card in HAND]
    assert [card.label for card in snapshot.visible_cards] == ["7♦", "K♣", "K♥"]
    assert snapshot.deck_count == 43 and vision.table_reads == 0


@pytest.mark.asyncio
async def test_fast_draw_reads_only_the_table_when_left_just_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 最常见：上家刚打出 10♦ 就轮到自己，对手回合没来得及读。手牌沿用，只读亮牌行。
    new_row = [Card("10", "D", x=438, y=386)] + [Card(c.rank, c.suit, x=x, y=386) for c, x in zip(ROW, (492, 546, 600))]
    vision = TableOnlyVision(new_row, 42)
    controller, frame = _controller(tmp_path, vision)
    controller._row_diff.cycle_complete = False
    _pixels(monkeypatch, PIXELS)
    snapshot = await controller._fast_draw_snapshot(frame)
    assert snapshot is not None and vision.table_reads == 1
    assert [card.label for card in snapshot.hand] == [card.label for card in HAND]
    assert [card.label for card in snapshot.visible_cards] == ["10♦", "7♦", "K♣", "K♥"]
    assert snapshot.deck_count == 42


@pytest.mark.asyncio
async def test_fast_draw_falls_back_on_suit_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, frame = _controller(tmp_path, TableOnlyVision([], 0))
    _pixels(monkeypatch, [None, "H", "H", "S", "H", "H", "D"])  # 第 4 张对不上
    assert await controller._fast_draw_snapshot(frame) is None
    assert controller._known_hand is None


@pytest.mark.asyncio
async def test_precomputed_draw_decision_is_used_only_for_same_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, frame = _controller(tmp_path, TableOnlyVision([], 0))
    _pixels(monkeypatch, PIXELS)
    calls: list[int] = []

    def fake_choose(*_args: object, **_kwargs: object) -> GameDecision:
        calls.append(1)
        return GameDecision("draw_visible", visible_index=2, reason="预算：取 K♥")

    monkeypatch.setattr(controller.ev, "choose_draw_action", fake_choose)
    controller._start_draw_precompute()
    snapshot = await controller._fast_draw_snapshot(frame)
    assert snapshot is not None
    decision = await controller._take_draw_precompute(snapshot)
    assert decision is not None and decision.visible_index == 2 and calls == [1]

    controller._start_draw_precompute()
    controller._last_row_local = (list(ROW), 42)  # 局面变了（牌堆数不同）
    other = await controller._fast_draw_snapshot(frame)
    assert other is not None
    assert await controller._take_draw_precompute(other) is None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_discard_decision_precomputed_after_taking_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _frame = _controller(tmp_path, TableOnlyVision([], 0))
    hand7 = cards("癞子 Q♠ J♠ 9♠ 8♠ 7♠ K♥")
    taken = cards("10♠")[0]
    seen: dict[str, object] = {}

    def fake_choose(hand8, *_args, **kwargs):
        seen.update(hand8=[c.label for c in hand8], **kwargs)
        return GameDecision("discard", card_indexes=(6,), reason="预算：打 K♥"), []

    monkeypatch.setattr(controller.ev, "choose_8_card_action", fake_choose)
    controller._round_gold_ranks = ("6", "7", "A")
    controller._start_discard_precompute(hand7, taken)
    # 出牌阶段识别到同样 8 张，但显示顺序不同：按牌对应到当前位置。
    shown = cards("癞子 K♥ Q♠ J♠ 10♠ 9♠ 8♠ 7♠")
    # 这 8 张能胡（六六大顺），游戏会给胡按钮；预算时也按“可胡”算的，两边一致才采用。
    snapshot = GameSnapshot(screen_type=ScreenType.HU, gold_ranks=("6", "7", "A"), hand=shown)
    decision = await controller._take_discard_precompute(snapshot)
    assert decision is not None and shown[decision.card_indexes[0]].label == "K♥"
    assert seen["locked_hand"] is not None and seen["self_draw"] is False

    # 出牌阶段看到的 8 张与拿牌时不同：不用预算结果。
    controller._start_discard_precompute(hand7, taken)
    other = GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=("6", "7", "A"), hand=cards("癞子 K♥ Q♠ J♠ 10♠ 9♠ 8♠ 6♠"))
    assert await controller._take_discard_precompute(other) is None
