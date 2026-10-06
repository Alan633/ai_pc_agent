from __future__ import annotations

from game_baque.models import Card
from game_baque.table_watch import RowDiffTracker

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()] if text else []


def labels(turns) -> list[str]:
    return [turn.describe() for turn in turns]


def _started(row: str = "J♥ 6♥ 6♦", deck: int = 50) -> RowDiffTracker:
    tracker = RowDiffTracker()
    # 自己摸牌堆后打出 9♣：9♣ 进最左边。
    tracker.start_after_own_turn(cards(row), deck + 1, taken=None, discarded=cards("9♣")[0], drew_deck=True)
    return tracker


def test_three_deck_turns_in_order_right_top_left() -> None:
    tracker = _started()  # 基准：9♣ J♥ 6♥ 6♦，牌堆 50
    assert labels(tracker.observe(cards("8♠ 9♣ J♥ 6♥"), 49)) == ["right 摸牌堆 → 打 8♠"]
    assert labels(tracker.observe(cards("K♦ 8♠ 9♣ J♥"), 48)) == ["top 摸牌堆 → 打 K♦"]
    assert labels(tracker.observe(cards("A♠ K♦ 8♠ 9♣"), 47)) == ["left 摸牌堆 → 打 A♠"]
    assert not tracker.active


def test_take_then_discard_seen_in_two_reads() -> None:
    tracker = _started()
    # 下家拿走 J♥（留空位），牌堆数不变。
    assert tracker.observe(cards("9♣ 6♥ 6♦"), 50) == []
    turns = tracker.observe(cards("Q♠ 9♣ 6♥ 6♦"), 50)
    assert labels(turns) == ["right 拿亮牌 J♥ → 打 Q♠"]


def test_take_and_discard_within_one_read() -> None:
    tracker = _started()
    turns = tracker.observe(cards("Q♠ 9♣ 6♥ 6♦"), 50)
    assert labels(turns) == ["right 拿亮牌 J♥ → 打 Q♠"]


def test_push_out_on_full_row_is_not_a_take() -> None:
    tracker = _started()
    # 满 4 张时摸牌堆再打出：最旧的 6♦ 被挤出，不是被拿走。
    turns = tracker.observe(cards("8♠ 9♣ J♥ 6♥"), 49)
    assert turns[0].draw_source == "deck"


def test_two_turns_between_reads_are_split_in_order() -> None:
    tracker = _started()
    turns = tracker.observe(cards("K♦ 8♠ 9♣ J♥"), 48)
    assert labels(turns) == ["right 摸牌堆 → 打 8♠", "top 摸牌堆 → 打 K♦"]


def test_nothing_attributed_before_own_first_discard() -> None:
    tracker = RowDiffTracker()
    assert tracker.observe(cards("J♥ 6♥"), 50) == []
    assert tracker.observe(cards("8♠ J♥ 6♥"), 49) == []
    assert not tracker.active


def test_own_take_leaves_gap_without_push() -> None:
    tracker = RowDiffTracker()
    # 自己拿走 6♥ 后打出 9♣：补进空位，不挤掉 6♦。
    tracker.start_after_own_turn(
        cards("J♥ 6♥ 6♦ K♠"), 50, taken=cards("6♥")[0], discarded=cards("9♣")[0], drew_deck=False
    )
    turns = tracker.observe(cards("8♠ 9♣ J♥ 6♦"), 49)
    assert labels(turns) == ["right 摸牌堆 → 打 8♠"]


# ---- 控制器集成：对手回合读亮牌行 → 写入记牌器 ----

import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from agent.session import AgentSession  # noqa: E402
from game_baque.capture import CapturedFrame, TurnProbe  # noqa: E402
from game_baque.controller import BaqueGameController  # noqa: E402
from game_baque.models import GameSnapshot, ScreenType  # noqa: E402


@pytest.mark.asyncio
async def test_controller_records_three_opponent_turns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frame = CapturedFrame(tmp_path / "t.png", 0, 0, 1936, 1088, "欢乐斗地主")
    monkeypatch.setattr("game_baque.controller.capture_game_window", lambda *_a: frame)
    reads = iter([(cards("8♠ 9♣ J♥ 6♥"), 49), (cards("K♦ 8♠ 9♣ J♥"), 48)])

    class FakeVision:
        async def parse_table(self, _path: Path):
            return next(reads)

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    controller.tracker.reset_round(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣"), 51)
    controller._tracker_started = True
    controller._own_draw_row = (cards("J♥ 6♥ 6♦"), 51, None, True)
    controller._start_row_watch(cards("9♣")[0])

    def probe(signature: bytes) -> TurnProbe:
        return TurnProbe(None, True, None, table_signature=signature)

    a, b = bytes([10]) * 400, bytes([90]) * 400
    for signature in (a, a, a, b, b):  # 每种画面稳定两帧才读一次，且同一画面只读一次
        await controller._watch_table(probe(signature))

    # 轮到自己摸牌：摸牌阶段读到的亮牌行补齐上家。
    controller._register_snapshot(
        frame,
        GameSnapshot(screen_type=ScreenType.DRAW, visible_cards=cards("A♠ K♦ 8♠ 9♣"), deck_count=47),
    )

    opponents = controller.tracker.opponents
    assert opponents["right"].discards == ["8:S"]
    assert opponents["top"].discards == ["K:D"]
    assert opponents["left"].discards == ["A:S"]
    assert controller.opponent_event_count == 3


def test_partial_misread_is_ignored_instead_of_creating_fake_discards() -> None:
    tracker = _started()  # 9♣ J♥ 6♥ 6♦
    assert tracker.observe(cards(""), 50) == []  # 整行读漏
    assert tracker.observe(cards("9♣"), 50) == []  # 读漏三张
    # 恢复正常读数后，不应把“重新出现”的 J♥ 6♥ 6♦ 当成新打出的牌。
    assert labels(tracker.observe(cards("8♠ 9♣ J♥ 6♥"), 49)) == ["right 摸牌堆 → 打 8♠"]


def test_more_new_cards_than_remaining_players_only_resyncs() -> None:
    tracker = _started()
    assert tracker.observe(cards("8♠ 9♣ J♥ 6♥"), 49)  # 下家
    # 只剩两家没动，却多出 3 张：读数对不上，不归属。
    assert tracker.observe(cards("A♠ K♦ Q♣ 8♠"), 46) == []


# ---- 已知牌堆顶：只决定前几张，过期即清 ----

from game_baque.tracker import TableTracker  # noqa: E402


def test_known_deck_top_only_decides_the_first_draws() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣"), 40)
    tracker.known_deck_top = ["J:D", "8:S"]
    assert tracker.draw_distribution() == {"J:D": 1.0}
    assert tracker.draw_distribution(skip=1) == {"8:S": 1.0}
    later = tracker.draw_distribution(skip=3)
    assert len(later) > 1 and "J:D" in later  # 牌池里还有另一张 J♦
    assert abs(sum(later.values()) - 1.0) < 1e-9


def test_unknown_opponent_source_clears_known_deck_top() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣"), 40)
    tracker.known_deck_top = ["J:D", "8:S"]
    tracker.record_opponent_turn("right", draw_source="unknown", discard=cards("6♠")[0])
    assert tracker.known_deck_top == []


def test_rollout_does_not_redraw_the_same_known_card() -> None:
    from game_baque.ev import ExpectedValueEngine

    kept = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣")
    tracker = TableTracker()
    tracker.reset_round(kept, 40)
    tracker.known_deck_top = ["9:C"]  # 出牌后会被下家摸走，与自己无关
    plain = TableTracker()
    plain.reset_round(kept, 40)
    gold = ("6", "7", "8")
    with_top = ExpectedValueEngine(simulations=300, seed=3)._rollout_future_hu(kept, gold, tracker)
    without = ExpectedValueEngine(simulations=300, seed=3)._rollout_future_hu(kept, gold, plain)
    # 旧实现会让三次摸牌都摸到 9♣（永远不胡，价值 0）。
    assert with_top > 0.5 * without > 0


# ---- 14:26 实战回归：别人打出的牌被另一家拿走，不能冲突清空记录 ----


def test_taking_another_players_discard_keeps_all_records() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("Q♥ J♥ J♦ 10♠ 8♦ 7♠ 9♣"), 38)
    tracker.record_opponent_turn("right", draw_source="visible", drawn_card=cards("A♥")[0], discard=cards("10♦")[0])
    tracker.record_opponent_turn("top", draw_source="visible", drawn_card=cards("Q♠")[0], discard=cards("A♣")[0])
    # 上家拿走下家刚打出的 10♦：旧实现这里 10♦ 计数为 -1，触发整盘重建。
    tracker.record_opponent_turn("left", draw_source="visible", drawn_card=cards("10♦")[0], discard=cards("J♥")[0])
    assert tracker.opponents["right"].known_hand["A:H"] == 1
    assert tracker.opponents["top"].known_hand["Q:S"] == 1
    assert tracker.opponents["left"].known_hand["10:D"] == 1
    assert tracker.public_discards["10:D"] == 0


def test_own_take_of_a_discard_does_not_double_count() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣"), 40)
    tracker.record_opponent_turn("right", draw_source="deck", discard=cards("9♥")[0])
    tracker.record_own_take_visible(cards("9♥")[0])
    tracker.update_own_hand(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣ 9♥"))  # 旧实现：9♥ 计数 -1


def test_repair_keeps_other_opponent_knowledge() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("A♣ A♦ K♥ K♠ Q♥ 10♦ 9♣"), 40)
    tracker.record_opponent_turn("right", draw_source="visible", drawn_card=cards("J♦")[0], discard=cards("8♠")[0])
    # 视觉把对家打出的 8♠ 又认成自己手里的牌之类的误读：直接制造 8♠ 超额。
    tracker.public_discards["8:S"] += 5
    assert tracker.repair_counts() == ["8:S"]
    assert tracker.opponents["right"].known_hand["J:D"] == 1


def test_animated_deck_misread_is_ignored() -> None:
    tracker = _started(deck=37)  # 基准：9♣ J♥ 6♥ 6♦，牌堆 37
    assert tracker.observe(cards("9♣ J♥ 6♥"), 52) == []  # 动画中读成 52，且 6♦ 被拿走
    turns = tracker.observe(cards("8♥ 9♣ J♥ 6♥"), 37)
    assert labels(turns) == ["right 拿亮牌 6♦ → 打 8♥"]


# ---- 胡牌结算：拿亮牌后直接胡，不能把结算时翻出的牌当成出牌 ----

from PIL import Image, ImageDraw  # noqa: E402

from game_baque.fast_vision import SETTLEMENT_SEATS, detect_hu_settlement  # noqa: E402

W, H = 1920, 1080
RED_BANNER = (232, 95, 68)
LOSER_BANNER = (116, 145, 179)


def _settlement_image(winner: str | None, losers: tuple[str, ...]) -> Image.Image:
    image = Image.new("RGB", (W, H), (30, 120, 100))
    draw = ImageDraw.Draw(image)
    for seat in losers + ((winner,) if winner else ()):
        left, top, right, bottom = SETTLEMENT_SEATS[seat]
        draw.rectangle(
            (round(W * left), round(H * top), round(W * right), round(H * bottom)),
            fill=RED_BANNER if seat == winner else LOSER_BANNER,
        )
    return image


def test_settlement_detector_names_the_winner() -> None:
    assert detect_hu_settlement(_settlement_image("right", ("left", "top", "self"))) == "right"
    # 赢家横幅还没出来：两家以上输家横幅，唯一空着的座位就是赢家。
    assert detect_hu_settlement(_settlement_image(None, ("left", "right", "self"))) == "top"
    assert detect_hu_settlement(Image.new("RGB", (W, H), (30, 120, 100))) is None


@pytest.mark.asyncio
async def test_take_then_hu_records_take_without_fake_discard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = CapturedFrame(tmp_path / "t.png", 0, 0, 1936, 1088, "欢乐斗地主")
    monkeypatch.setattr("game_baque.controller.capture_game_window", lambda *_a: frame)
    reads = iter([(cards("9♣ J♥ 8♣"), 37), (cards("8♥ 9♣ J♥ 8♣"), 36)])

    class FakeVision:
        async def parse_table(self, _path: Path):
            return next(reads)

    controller = BaqueGameController(
        vision=FakeVision(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    controller.tracker.reset_round(cards("Q♥ J♦ 10♦ 8♦ 7♠ 7♦ 9♣"), 38)
    controller._tracker_started = True
    controller._own_draw_row = (cards("J♥ A♣ 8♣ 7♠"), 37, cards("7♠")[0], False)
    controller._start_row_watch(cards("9♣")[0])

    a = bytes([10]) * 400
    for _ in range(2):  # 下家拿走 A♣（还没打出）
        await controller._watch_table(TurnProbe(None, True, None, table_signature=a))
    # 下家胡牌结算：横幅出现，随后亮牌行翻出 8♥。
    controller._on_settlement("right")
    b = bytes([90]) * 400
    for _ in range(2):
        await controller._watch_table(TurnProbe(None, True, None, table_signature=b))

    right = controller.tracker.opponents["right"]
    assert right.known_hand["A:C"] == 1
    assert right.discards == [], "结算时翻出的 8♥ 不是下家打出的牌"


def test_after_right_hu_next_turn_belongs_to_top() -> None:
    tracker = _started()  # 9♣ J♥ 6♥ 6♦
    assert tracker.observe(cards("9♣ 6♥ 6♦"), 50) == []  # 下家拿走 J♥
    assert tracker.continue_after_hu("right").label == "J♥"
    assert tracker.observe(cards("8♥ 9♣ 6♥ 6♦"), 49) == []  # 结算翻牌：只作基准
    turns = tracker.observe(cards("K♣ 8♥ 9♣ 6♥"), 48)
    assert labels(turns) == ["top 摸牌堆 → 打 K♣"]


def test_after_left_hu_it_is_our_turn() -> None:
    tracker = _started()
    tracker.continue_after_hu("left")
    assert not tracker.active


def test_repair_caps_impossible_own_hand() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("J♣ J♣ 10♠ 10♣ 7♥ 9♦ 6♠"), 48)
    tracker.own_hand["J:C"] = 3  # 视觉读出 3 张 J♣（牌池只有 2 张）
    assert "J:C" in tracker.repair_counts()
    tracker.unseen_counts(include_opportunity=True)  # 不再抛冲突


def test_reset_round_survives_impossible_hand() -> None:
    tracker = TableTracker()
    tracker.reset_round(cards("J♣ J♣ J♣ 10♠ 10♣ 7♥ 9♦"), 48)
    assert tracker.own_hand["J:C"] == 2


def test_decision_tracker_never_raises(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    controller.tracker.reset_round(cards("J♣ J♣ 10♠ 10♣ 7♥ 9♦ 6♠"), 48)
    controller.tracker.public_discards["J:C"] = 2  # 与手牌冲突
    snapshot = GameSnapshot(screen_type=ScreenType.DRAW, hand=cards("J♣ J♣ 10♠ 10♣ 7♥ 9♦ 6♠"), deck_count=48)
    tracker = controller._decision_tracker(snapshot)
    tracker.draw_distribution()  # 14:44 崩溃点：不能再抛出


# ---- 开局谁先出是随机的：暂存后按自己前面出了几手倒推座位 ----


def test_opening_turns_are_assigned_backwards_from_our_first_turn() -> None:
    tracker = RowDiffTracker()
    tracker.start_opening()
    assert tracker.active
    assert tracker.observe(cards("J♦"), 52) == []  # 有人摸牌堆打出 J♦（还不知道是谁）
    assert tracker.observe(cards("9♠ J♦"), 51) == []
    turns = tracker.finish_opening()  # 轮到自己：前面出了 2 手 → 对家、上家
    assert labels(turns) == ["top 来源不明 → 打 J♦", "left 摸牌堆 → 打 9♠"]
    assert not tracker.active


def test_opening_when_we_act_first_records_nothing() -> None:
    tracker = RowDiffTracker()
    tracker.start_opening()
    assert tracker.finish_opening() == []


def test_opening_with_three_players_before_us() -> None:
    tracker = RowDiffTracker()
    tracker.start_opening()
    tracker.observe(cards("A♠"), 53)
    tracker.observe(cards("K♦ A♠"), 52)
    tracker.observe(cards("Q♣ K♦ A♠"), 51)
    assert [turn.seat for turn in tracker.finish_opening()] == ["right", "top", "left"]
