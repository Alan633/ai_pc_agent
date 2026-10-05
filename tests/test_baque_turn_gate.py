from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import pytest
from PIL import Image, ImageDraw

from agent.session import AgentSession
from game_baque.capture import (
    CapturedFrame,
    TurnProbe,
    detect_action_state,
    detect_choice_popup,
    detect_discard_button,
    detect_opportunity_card,
    detect_round_in_progress,
)
from game_baque.controller import BaqueGameController
from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card, GameDecision, GameSnapshot, Point, ScreenType
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker
from game_baque.vision import (
    TURN_DECK_POINT,
    TURN_DISCARD_BUTTON,
    TURN_HAND_SLOTS,
    TURN_HAND_Y,
    BaqueVision,
)

@pytest.fixture(autouse=True)
def _no_real_screen_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """对手回合会顺带读取本局信息；测试里绝不能真的截屏。"""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("测试环境不截屏")

    monkeypatch.setattr("game_baque.controller.capture_game_window", refuse)


TABLE_GREEN = (30, 120, 100)
CARD_BACK_BLUE = (70, 100, 220)
BANNER_BLUE = (60, 120, 220)


def _table_image(*, deck: bool, banner: bool) -> Image.Image:
    width, height = 1936, 1088
    image = Image.new("RGB", (width, height), TABLE_GREEN)
    draw = ImageDraw.Draw(image)
    if deck:
        draw.rectangle((700, 358, 790, 476), fill=CARD_BACK_BLUE)
    if banner:
        draw.rectangle((580, 600, 1360, 674), fill=BANNER_BLUE)
    return image


def test_round_in_progress_requires_visible_deck_pile() -> None:
    assert detect_round_in_progress(_table_image(deck=True, banner=False))
    assert not detect_round_in_progress(_table_image(deck=False, banner=False))


def test_action_state_is_none_without_blue_prompt_banner() -> None:
    assert detect_action_state(_table_image(deck=True, banner=False)) is None
    assert detect_action_state(_table_image(deck=True, banner=True)) == "draw"


def test_center_countdown_clock_is_not_mistaken_for_discard_button() -> None:
    clock = _table_image(deck=True, banner=True)
    # 摸牌阶段的橙色闹钟倒计时位于画面正中（实图约 x915-1020, y680-780）。
    ImageDraw.Draw(clock).ellipse((915, 680, 1020, 780), fill=(245, 150, 40))
    assert detect_action_state(clock) == "draw"

    discard = _table_image(deck=True, banner=True)
    ImageDraw.Draw(discard).rounded_rectangle((945, 695, 1130, 780), 40, fill=(245, 150, 40))
    assert detect_action_state(discard) == "discard"

    swap = _table_image(deck=True, banner=True)
    ImageDraw.Draw(swap).rounded_rectangle((945, 695, 1130, 780), 40, fill=(150, 150, 150))
    assert detect_action_state(swap) == "swap"


def test_turn_snapshot_draw_uses_fixed_slots_and_opportunity_card() -> None:
    payload = {
        "gold_ranks": ["7", "10", "Q"],
        "hand": [
            {"rank": "?", "suit": "NONE"},
            {"rank": "A", "suit": "H"},
            {"rank": "9", "suit": "H"},
            {"rank": "9", "suit": "H"},
            {"rank": "K", "suit": "C"},
            {"rank": "10", "suit": "D"},
            {"rank": "7", "suit": "D"},
        ],
        "table": [{"rank": "J", "suit": "D", "x": 440}, {"rank": "8", "suit": "S", "x": 495}],
        "deck_count": 54,
        "hu_button": None,
    }

    snapshot = BaqueVision._turn_snapshot(payload, "draw")

    slots = TURN_HAND_SLOTS[7]
    assert snapshot.screen_type == ScreenType.DRAW
    assert snapshot.gold_ranks == ("7", "10", "Q")
    assert snapshot.opportunity_click == Point(slots[0], TURN_HAND_Y)
    assert [card.label for card in snapshot.hand] == ["A♥", "9♥", "9♥", "K♣", "10♦", "7♦"]
    assert [card.x for card in snapshot.hand] == list(slots[1:])
    assert [(card.label, card.x) for card in snapshot.visible_cards] == [("J♦", 438), ("8♠", 492)]
    assert snapshot.deck_click == TURN_DECK_POINT
    assert snapshot.discard_button is None
    assert snapshot.deck_count == 54


def test_turn_snapshot_discard_with_eight_cards_and_hu_button() -> None:
    payload = {
        "gold_ranks": ["7", "K", "A"],
        "hand": [{"rank": rank, "suit": "D"} for rank in ("A", "A", "Q", "Q", "10", "8", "7", "6")],
        "table": [{"rank": "6", "suit": "D", "x": 492}],
        "deck_count": 50,
        "hu_button": {"x": 700, "y": 677},
    }
    payload["hand"][2]["up"] = True

    snapshot = BaqueVision._turn_snapshot(payload, "discard")

    assert snapshot.screen_type == ScreenType.HU
    assert snapshot.hu_button == Point(700, 677)
    assert snapshot.discard_button == TURN_DISCARD_BUTTON
    assert [card.x for card in snapshot.hand] == list(TURN_HAND_SLOTS[8])
    assert [card.selected for card in snapshot.hand].count(True) == 1
    assert snapshot.hand[2].selected
    # 出牌阶段桌面亮牌不可选，不能进入决策。
    assert snapshot.visible_cards == []


def test_turn_snapshot_unknown_layout_leaves_no_usable_coordinates() -> None:
    payload = {
        "gold_ranks": ["7", "K", "A"],
        "hand": [{"rank": "9", "suit": "D"}] * 5,
        "deck_count": 50,
    }

    snapshot = BaqueVision._turn_snapshot(payload, "discard")

    assert len(snapshot.hand) == 5
    assert all(card.x == 0 for card in snapshot.hand)


def _frame(tmp_path: Path) -> CapturedFrame:
    return CapturedFrame(tmp_path / "turn.png", 0, 0, 1936, 1088, "欢乐斗地主")


@pytest.mark.asyncio
async def test_wait_for_turn_skips_opponent_frames_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    probes = iter(
        [
            TurnProbe(None, True, None),
            TurnProbe(None, True, None),
            TurnProbe("draw", True, frame),
        ]
    )
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: next(probes),
    )

    calls: list[tuple[Path, str]] = []

    class FakeVision:
        base_url = "http://127.0.0.1:11434/v1"

        async def parse_turn(self, image_path: Path, action_state: str, **_kwargs: object) -> GameSnapshot:
            calls.append((image_path, action_state))
            return GameSnapshot(screen_type=ScreenType.DRAW, deck_count=52)

        async def parse(self, *_args: object, **_kwargs: object) -> GameSnapshot:
            raise AssertionError("对手回合不得调用完整视觉识别")

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )

    result = await controller._wait_for_turn(Event())

    assert result is not None
    assert result[0] == frame
    assert result[1].screen_type == ScreenType.DRAW
    assert calls == [(frame.path, "draw")]
    assert controller.last_snapshot is result[1]


@pytest.mark.asyncio
async def test_wait_for_turn_hands_off_to_full_inspect_outside_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: TurnProbe(None, False, None),
    )
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )

    assert await controller._wait_for_turn(Event()) is None


@pytest.mark.asyncio
async def test_wait_for_turn_retries_after_malformed_model_reply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: TurnProbe("discard", True, frame),
    )
    replies = iter([ValueError("bad json"), GameSnapshot(screen_type=ScreenType.DISCARD)])

    class FakeVision:
        async def parse_turn(self, _path: Path, _state: str, **_kwargs: object) -> GameSnapshot:
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )

    result = await controller._wait_for_turn(Event())

    assert result is not None
    assert result[1].screen_type == ScreenType.DISCARD


@pytest.mark.asyncio
async def test_parse_swap_keeps_wild_card(tmp_path: Path) -> None:
    image_path = tmp_path / "swap.png"
    Image.new("RGB", (1936, 1088), TABLE_GREEN).save(image_path)
    vision = BaqueVision("", "local", "http://127.0.0.1:11434/v1")

    async def fake_quick_json(_prompt: str, _url: str, _label: str) -> dict[str, object]:
        return {
            "gold_ranks": ["8", "9", "10"],
            "hand": [
                {"rank": "?", "suit": "NONE"},
                {"rank": "WILD", "suit": "NONE"},
                {"rank": "A", "suit": "S"},
                {"rank": "A", "suit": "H"},
                {"rank": "J", "suit": "H"},
                {"rank": "Q", "suit": "D"},
                {"rank": "癞子", "suit": "NONE"},
            ],
            "swap_button": {"x": 536, "y": 677},
        }

    vision._quick_json = fake_quick_json  # type: ignore[method-assign]

    snapshot = await vision.parse_swap(image_path)

    assert len(snapshot.hand) == 6
    assert [card.wild for card in snapshot.hand] == [True, False, False, False, False, True]


def test_turn_snapshot_keeps_wild_card() -> None:
    payload = {
        "gold_ranks": ["8", "9", "10"],
        "hand": [{"rank": "癞子", "suit": "NONE"}] + [{"rank": "A", "suit": "S"}] * 6,
        "deck_count": 50,
    }

    snapshot = BaqueVision._turn_snapshot(payload, "draw")

    assert len(snapshot.hand) == 7
    assert snapshot.hand[0].wild


def _two_wild_hand() -> list[Card]:
    return [
        Card(wild=True),
        Card(rank="10", suit="S"),
        Card(wild=True),
        Card(rank="10", suit="D"),
        Card(rank="8", suit="S"),
        Card(rank="8", suit="H"),
        Card(rank="A", suit="S"),
        Card(rank="6", suit="S"),
    ]


def test_eight_card_decision_respects_time_budget() -> None:
    hand = _two_wild_hand()
    tracker = TableTracker()
    tracker.reset_round(hand[:7], 53)
    engine = ExpectedValueEngine(simulations=600, seed=1)

    started = time.monotonic()
    decision, values = engine.choose_8_card_action(
        hand,
        ("8", "J", "Q"),
        tracker,
        time_budget_seconds=1.0,
    )

    # 实测两张癞子全量采样曾超过 2 分钟；预算必须把它压在回合倒计时内。
    assert time.monotonic() - started < 2.5
    assert decision.action == "discard"
    assert len(values) == 8


def test_evaluate_hu_maps_wild_assignments_back_to_original_order() -> None:
    hand = [
        Card(rank="9", suit="H"),
        Card(wild=True),
        Card(rank="9", suit="S"),
        Card(rank="9", suit="C"),
        Card(rank="9", suit="D"),
        Card(rank="10", suit="H"),
        Card(rank="10", suit="S"),
        Card(rank="10", suit="C"),
    ]

    result = evaluate_hu(hand, ("9", "10", "K"))

    assert result is not None
    indexes = [int(item.split(":", 1)[0]) for item in result.wildcard_assignments]
    assert indexes == [1]
    # 与牌面顺序无关：打乱后结果分值一致。
    shuffled = evaluate_hu(list(reversed(hand)), ("9", "10", "K"))
    assert shuffled is not None
    assert shuffled.theoretical_score == result.theoretical_score
    assert [int(item.split(":", 1)[0]) for item in shuffled.wildcard_assignments] == [6]


def _choice_popup_image() -> Image.Image:
    image = _table_image(deck=False, banner=False)
    ImageDraw.Draw(image).rectangle((496, 266, 1438, 616), fill=(232, 242, 252))
    return image


def test_choice_popup_is_detected_and_probed_as_choose() -> None:
    assert detect_choice_popup(_choice_popup_image())
    assert not detect_choice_popup(_table_image(deck=True, banner=True))


def test_choice_cards_use_fixed_slots_for_four_options() -> None:
    options, hand = BaqueVision._choice_cards(
        {
            "options": [
                {"rank": "6", "suit": "D", "x": 300},
                {"rank": "Q", "suit": "D", "x": 450},
                {"rank": "10", "suit": "D", "x": 560},
                {"rank": "8", "suit": "S", "x": 690},
            ],
            "hand": [{"rank": "J", "suit": "S"}, {"rank": "?", "suit": "NONE"}],
        }
    )
    assert [(card.label, card.x) for card in options] == [
        ("6♦", 327),
        ("Q♦", 441),
        ("10♦", 556),
        ("8♠", 671),
    ]
    assert [card.label for card in hand] == ["J♠"]


def test_choose_option_card_prefers_the_card_that_completes_hu() -> None:
    hand = [Card(rank="9", suit=suit) for suit in ("S", "H", "C", "D")] + [
        Card(rank="10", suit=suit) for suit in ("S", "H", "C")
    ]
    options = [Card(rank="6", suit="D"), Card(rank="10", suit="D"), Card(rank="K", suit="S")]

    index, value = ExpectedValueEngine().choose_option_card(hand, options, ("9", "10", "K"))

    assert index == 1
    assert value > 0


@pytest.mark.asyncio
async def test_wait_for_turn_clicks_choice_popup_then_continues_to_discard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    probes = iter([TurnProbe("choose", False, frame), TurnProbe("discard", True, frame)])
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: next(probes),
    )

    class FakeVision:
        async def parse_choice(self, _path: Path) -> tuple[list[Card], list[Card]]:
            return (
                [Card(rank="6", suit="D", x=633, y=453), Card(rank="10", suit="D", x=1076, y=453)],
                [Card(rank="10", suit="S"), Card(rank="10", suit="H")],
            )

        async def parse_turn(self, _path: Path, state: str, **_kwargs: object) -> GameSnapshot:
            assert state == "discard"
            return GameSnapshot(screen_type=ScreenType.DISCARD)

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    clicks: list[Point] = []

    async def fake_click(_frame: CapturedFrame, point: Point) -> None:
        clicks.append(point)

    controller._click = fake_click  # type: ignore[method-assign]

    result = await controller._wait_for_turn(Event())

    # 10♦ 与两张 10 组成同点，结构价值更高。
    assert clicks == [Point(1076, 453)]
    assert result is not None
    assert result[1].screen_type == ScreenType.DISCARD


def test_hybrid_skips_local_model_when_turn_time_is_short(tmp_path: Path) -> None:
    class FailingReranker:
        timeout_seconds = 4.0

        def rerank_8_card_action(self, **_kwargs: object) -> object:
            raise AssertionError("回合剩余时间不足时不得调用本地模型")

    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        decision_mode="hybrid",
        decision_reranker=FailingReranker(),  # type: ignore[arg-type]
    )
    hand = _two_wild_hand()
    tracker = TableTracker()
    tracker.reset_round(hand[:7], 53)
    snapshot = GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=("8", "J", "Q"), hand=hand)

    started = time.monotonic()
    decision = controller._decide(snapshot, tracker, time.monotonic() + 1.0)

    assert decision.action == "discard"
    assert controller.last_decision_source == "hybrid_ev_fallback"
    assert time.monotonic() - started < 2.5


def test_opportunity_card_detector_ignores_wild_and_normal_cards() -> None:
    image = _table_image(deck=True, banner=True)
    draw = ImageDraw.Draw(image)
    # 白底 + 橙色问号：选牌卡。
    draw.rectangle((505, 860, 580, 1000), fill=(230, 229, 229))
    draw.rectangle((520, 880, 545, 930), fill=(245, 160, 90))
    assert detect_opportunity_card(image)

    wild = _table_image(deck=True, banner=True)
    # 桃色底 + 少量橙色字：癞子。
    ImageDraw.Draw(wild).rectangle((505, 860, 580, 1000), fill=(237, 200, 168))
    ImageDraw.Draw(wild).rectangle((520, 880, 530, 900), fill=(245, 150, 60))
    assert not detect_opportunity_card(wild)


def _discard_probe(frame: CapturedFrame) -> TurnProbe:
    return TurnProbe("discard", True, frame, opportunity_card=True)


@pytest.mark.asyncio
async def test_discard_phase_plays_question_card_then_discard_button(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    probes = iter(
        [
            TurnProbe(None, True, None),  # 对手回合：换牌阶段结束
            _discard_probe(frame),
            TurnProbe("discard", True, frame),
        ]
    )
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: next(probes),
    )

    class FakeVision:
        async def parse_turn(self, _path: Path, state: str, **_kwargs: object) -> GameSnapshot:
            return GameSnapshot(screen_type=ScreenType.DISCARD)

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    clicks: list[Point] = []

    async def fake_click(_frame: CapturedFrame, point: Point) -> None:
        clicks.append(point)

    async def has_discard_turn() -> str:
        return "discard"

    controller._click = fake_click  # type: ignore[method-assign]
    controller._current_action_state = has_discard_turn  # type: ignore[method-assign]
    monkeypatch.setattr(
        "game_baque.controller.current_action_buttons",
        lambda _keyword: {"discard": Point(1038, 737)},
    )

    await controller._wait_for_turn(Event())

    # 先点最左侧“?”（8 张布局 x=282/1000），再点“出牌”（536,677）。
    assert clicks == [Point(546, 925), Point(1038, 737)]


@pytest.mark.asyncio
async def test_question_card_is_not_played_during_opening_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    # 换牌选满两张后按钮变橙，像素门读成 discard；此时不能打出选牌卡。
    probes = iter([_discard_probe(frame)])
    monkeypatch.setattr(
        "game_baque.controller.probe_game_turn",
        lambda _dir, _keyword: next(probes),
    )

    class FakeVision:
        async def parse_turn(self, _path: Path, state: str, **_kwargs: object) -> GameSnapshot:
            return GameSnapshot(screen_type=ScreenType.DISCARD)

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )

    async def no_click(_frame: CapturedFrame, _point: Point) -> None:
        raise AssertionError("换牌阶段不得点击选牌卡")

    controller._click = no_click  # type: ignore[method-assign]

    result = await controller._wait_for_turn(Event())

    assert result is not None


DISCARD_ORANGE = (255, 180, 66)
HU_RED = (250, 110, 95)


def test_discard_button_check_survives_banner_hiding_after_card_selection() -> None:
    # 选中一张牌后：蓝色横幅消失；若已听牌还会弹出提示面板盖住牌堆。按钮仍在。
    selected = _table_image(deck=False, banner=False)
    ImageDraw.Draw(selected).rounded_rectangle((945, 695, 1130, 780), 38, fill=DISCARD_ORANGE)
    assert detect_action_state(selected) is None
    assert detect_discard_button(selected)

    # 开始页的“开始游戏”更宽（约 0.15 屏宽），不能当成出牌按钮。
    start_page = _table_image(deck=False, banner=False)
    ImageDraw.Draw(start_page).rounded_rectangle((823, 714, 1113, 791), 38, fill=DISCARD_ORANGE)
    assert not detect_discard_button(start_page)


def test_find_action_buttons_handles_hu_layout_and_ignores_round_timer() -> None:
    from game_baque.capture import find_action_buttons

    image = _table_image(deck=True, banner=True)
    draw = ImageDraw.Draw(image)
    draw.ellipse((700, 680, 800, 780), fill=(241, 174, 23))  # 橙色闹钟皮肤倒计时（实图在出牌左侧约 40px）
    draw.rounded_rectangle((838, 695, 1015, 772), 38, fill=DISCARD_ORANGE)  # 出牌（左移）
    draw.rounded_rectangle((1055, 695, 1231, 772), 38, fill=HU_RED)  # 胡
    buttons = find_action_buttons(image)
    assert set(buttons) == {"discard", "hu"}
    assert abs(buttons["discard"].x - 926) <= 3
    assert abs(buttons["hu"].x - 1143) <= 3


@pytest.mark.asyncio
async def test_discard_clicks_button_after_selection_even_though_banner_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    states = iter(["discard", None])

    async def banner_then_gone() -> str | None:
        return next(states)

    controller._current_action_state = banner_then_gone  # type: ignore[method-assign]
    monkeypatch.setattr(
        "game_baque.controller.current_action_buttons",
        lambda _keyword: {"discard": Point(1038, 737)},
    )
    clicks: list[Point] = []

    async def fake_click(_frame: CapturedFrame, point: Point) -> None:
        clicks.append(point)

    controller._click = fake_click  # type: ignore[method-assign]
    ranks = ("6", "7", "8", "9", "10", "J", "Q", "K")
    hand = [Card(rank=rank, suit="S", x=546 + index * 103, y=925) for index, rank in enumerate(ranks)]
    controller._sync_tracker(GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=("6", "K", "A"), hand=hand, deck_count=50))
    snapshot = GameSnapshot(
        screen_type=ScreenType.DISCARD,
        gold_ranks=("6", "K", "A"),
        hand=hand,
        discard_button=Point(1038, 737),
    )

    message = await controller._execute(
        _frame(tmp_path),
        snapshot,
        GameDecision("discard", card_indexes=(2,), reason="测试"),
    )

    assert clicks == [Point(752, 925), Point(1038, 737)]
    assert message.endswith("已出牌。")


def test_discard_phase_without_banner_after_choice_pick_is_still_our_turn() -> None:
    from game_baque.capture import classify_turn_image

    after_pick = _table_image(deck=True, banner=False)
    ImageDraw.Draw(after_pick).rounded_rectangle((945, 695, 1130, 780), 40, fill=(245, 150, 40))
    assert classify_turn_image(after_pick) == "discard"
    assert classify_turn_image(_choice_popup_image()) == "choose"
    assert classify_turn_image(_table_image(deck=True, banner=False)) is None


def test_compact_card_tokens_are_parsed() -> None:
    parse = BaqueVision._parse_card_token
    assert parse("10S") == {"rank": "10", "suit": "S", "up": False}
    assert parse("J D") == {"rank": "J", "suit": "D", "up": False}
    assert parse("AH^") == {"rank": "A", "suit": "H", "up": True}
    assert parse("Q♣") == {"rank": "Q", "suit": "C", "up": False}
    assert parse("WILD")["rank"] == "WILD"
    assert parse("?")["rank"] == "?"
    assert parse("J") is None
    assert parse("") is None
    # 牌角“30分/20分”被抄进牌面时仍能解析。
    assert parse("A30H") == {"rank": "A", "suit": "H", "up": False}
    assert parse("J20S") == {"rank": "J", "suit": "S", "up": False}
    assert parse("10D") == {"rank": "10", "suit": "D", "up": False}


def test_tracker_conflict_rebuilds_instead_of_crashing(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    ranks = ("6", "7", "8", "9", "10", "J", "Q")
    hand = [Card(rank=rank, suit="C") for rank in ranks]
    controller._sync_tracker(GameSnapshot(screen_type=ScreenType.DRAW, gold_ranks=("6", "10", "A"), hand=hand, deck_count=51))
    controller._record(controller.tracker.record_own_discard, hand[4])  # 打出 10♣
    controller._record(controller.tracker.record_own_discard, hand[4])  # 视觉误认后再次记 10♣
    # 再“看到”一张 10♣（实际是 10♠ 被认错）——旧实现在这里抛异常终止整局。
    misread = hand[:4] + [Card(rank="10", suit="C")] + hand[5:] + [Card(rank="K", suit="S")]
    controller._sync_tracker(GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=("6", "10", "A"), hand=misread, deck_count=50))
    assert controller._tracker_started


def test_buttons_are_never_found_on_a_dimmed_purchase_popup() -> None:
    from game_baque.capture import detect_table_felt_visible, find_action_buttons

    normal = _table_image(deck=True, banner=True)
    ImageDraw.Draw(normal).rounded_rectangle((945, 695, 1130, 780), 38, fill=DISCARD_ORANGE)
    assert detect_table_felt_visible(normal)
    assert "discard" in find_action_buttons(normal)

    # “豆子不足啦”：整张牌桌被压暗，中间黄色“💎30”购买按钮形似出牌按钮。
    dimmed = Image.new("RGB", (1936, 1088), (10, 40, 34))
    ImageDraw.Draw(dimmed).rounded_rectangle((945, 695, 1130, 780), 38, fill=DISCARD_ORANGE)
    assert not detect_table_felt_visible(dimmed)
    assert find_action_buttons(dimmed) == {}


def _hu_hand() -> list[Card]:
    return [Card(rank="9", suit=suit) for suit in ("S", "H", "C", "D")] + [
        Card(rank="10", suit=suit) for suit in ("S", "H", "C", "D")
    ]


@pytest.mark.parametrize(("future_ratio", "expected"), [(1.3, "hu"), (3.0, "discard")])
def test_sure_hu_is_taken_unless_continuing_is_much_better(future_ratio: float, expected: str) -> None:
    hand = _hu_hand()
    gold = ("6", "7", "8")
    hu_value = float(evaluate_hu(hand, gold).theoretical_score)  # type: ignore[union-attr]
    tracker = TableTracker()
    tracker.reset_round(hand[:7], 50)
    engine = ExpectedValueEngine(simulations=50, seed=3)
    engine._rollout_future_hu = lambda *_args, **_kwargs: hu_value * future_ratio  # type: ignore[method-assign]

    decision, _values = engine.choose_8_card_action(hand, gold, tracker, time_budget_seconds=1.0)

    assert decision.action == expected


def test_hybrid_reranker_is_not_consulted_when_ev_takes_the_hu(tmp_path: Path) -> None:
    class FailingReranker:
        timeout_seconds = 4.0

        def rerank_8_card_action(self, **_kwargs: object) -> object:
            raise AssertionError("确定胡牌时不得让本地模型改成弃牌")

    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        decision_mode="hybrid",
        decision_reranker=FailingReranker(),  # type: ignore[arg-type]
    )
    hand = _hu_hand()
    tracker = TableTracker()
    tracker.reset_round(hand[:7], 50)
    snapshot = GameSnapshot(screen_type=ScreenType.HU, gold_ranks=("6", "7", "8"), hand=hand)

    decision = controller._decide(snapshot, tracker, None)

    assert decision.action == "hu"



def test_payout_cap_uses_room_cap_and_opponent_beans(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    # 新手场每家封顶 2 万：豆少的两家按自己的豆付，豆多的一家最多付 2 万。
    controller._apply_payout_cap(10.0, [2700.0, 45_000.0, 1588.0], 20_000.0)
    assert controller.ev.score_caps == pytest.approx((270.0, 2000.0, 158.8))
    # 理论分 3840：三家实付 2700 + 20000 + 1588（豆）→ 平均到理论分单位。
    assert controller.ev.capped(3840) == pytest.approx((270.0 + 2000.0 + 158.8) / 3)
    # 三家豆都充足：每家 2 万，合计 6 万。
    controller._apply_payout_cap(10.0, [1e6, 1e6, 1e6], 20_000.0)
    assert controller.ev.score_caps == pytest.approx((2000.0, 2000.0, 2000.0))
    # 某家豆数没读到：该家按场次封顶估计。
    controller._apply_payout_cap(240.0, [28_970.0, 28_200.0], 3_000_000.0)
    assert controller.ev.score_caps == pytest.approx((28_970 / 240, 28_200 / 240, 12_500.0))
    # 读不到底分时不封顶。
    controller._apply_payout_cap(None, [2700.0], 20_000.0)
    assert controller.ev.score_caps is None
    assert controller.ev.capped(3840) == 3840


def test_capped_hu_is_taken_even_when_rollout_promises_more() -> None:
    hand = _hu_hand()
    gold = ("6", "7", "8")
    tracker = TableTracker()
    tracker.reset_round(hand[:7], 50)
    engine = ExpectedValueEngine(simulations=50, seed=3)
    engine.score_caps = (500.0, 500.0, 500.0)
    # 继续做牌的未来价值同样被封顶，不可能超过“立即胡”的 2 倍。
    decision, values = engine.choose_8_card_action(hand, gold, tracker, time_budget_seconds=1.0)
    assert decision.action == "hu"
    assert max(item.future_hu_value for item in values) <= 500.0


def test_bean_amounts_are_parsed() -> None:
    parse = BaqueVision._parse_beans
    assert parse("2700") == 2700
    assert parse("1.258万") == pytest.approx(12580)
    assert parse("10.04 万") == pytest.approx(100400)
    assert parse(240) == 240
    assert parse("") is None



def test_start_click_gate_refuses_purchase_popup_and_in_round_buttons() -> None:
    from game_baque.capture import is_start_page_button

    start_page = _table_image(deck=False, banner=False)
    assert is_start_page_button(start_page, Point(968, 752))
    # 偏离正中的按钮（“💎250” 购买钮在 x≈0.62 屏宽）。
    assert not is_start_page_button(start_page, Point(1205, 724))
    # 牌局中（有牌堆）的出牌按钮。
    assert not is_start_page_button(_table_image(deck=True, banner=False), Point(968, 752))
    # 被购买弹窗压暗的牌桌。
    dimmed = Image.new("RGB", (1936, 1088), (10, 40, 34))
    assert not is_start_page_button(dimmed, Point(968, 752))


def test_large_ting_hint_panel_does_not_hide_the_discard_button() -> None:
    from game_baque.capture import find_action_buttons

    # 听口很多时“胡：…”提示面板几乎横贯全屏（实测 x 208~1728、y 232~637）。
    image = _table_image(deck=False, banner=False)
    draw = ImageDraw.Draw(image)
    draw.rectangle((208, 232, 1728, 637), fill=(236, 244, 252))
    draw.rounded_rectangle((945, 695, 1130, 780), 38, fill=DISCARD_ORANGE)
    assert "discard" in find_action_buttons(image)
