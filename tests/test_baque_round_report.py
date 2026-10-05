from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from agent.session import AgentSession
from game_baque.capture import CapturedFrame, TurnProbe
from game_baque.controller import BaqueGameController
from game_baque.models import Card, GameDecision, GameSnapshot, Point, ScreenType
from game_baque.round_report import RoundLog, RoundReport
from game_baque.tracker import TableTracker


def c(rank: str, suit: str) -> Card:
    return Card(rank=rank, suit=suit)


HAND7 = [c("6", "S"), c("7", "H"), c("8", "C"), c("9", "D"), c("10", "S"), c("J", "H"), c("Q", "C")]


# ---------------- 记牌器：桌面亮牌 ----------------


def test_table_cards_are_removed_from_the_deck_pool() -> None:
    tracker = TableTracker()
    tracker.reset_round(HAND7, 50)
    assert tracker.unseen_counts()["A:D"] == 2

    # 对手打出两张 A♦ 摆在桌面：牌堆里不可能再摸到。
    tracker.sync_table([c("A", "D"), c("A", "D")])
    assert "A:D" not in tracker.unseen_counts()
    assert "A:D" not in tracker.draw_distribution()


def test_own_discard_on_table_is_not_counted_twice() -> None:
    tracker = TableTracker()
    tracker.reset_round(HAND7, 50)
    tracker.record_own_discard(c("6", "S"))
    tracker.sync_table([c("6", "S")])  # 自己刚打出的 6♠ 仍在桌面
    # 6♠ 共 2 张：一张自己打出（已计入），另一张仍未见。
    assert tracker.unseen_counts()["6:S"] == 1


def test_stale_table_never_raises_a_tracker_conflict() -> None:
    tracker = TableTracker()
    tracker.reset_round(HAND7 + [c("A", "D")], 50)
    tracker.record_own_take_visible(c("A", "D"))
    # 视觉误差：桌面仍显示两张 A♦，而自己手里已有一张——只能“更少”，不能报错。
    tracker.sync_table([c("A", "D"), c("A", "D")])
    assert tracker.unseen_counts().get("A:D", 0) == 0


def test_tracker_starts_with_question_card_plus_six(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    snapshot = GameSnapshot(
        screen_type=ScreenType.DRAW,
        gold_ranks=("6", "J", "A"),
        hand=HAND7[:6],
        visible_cards=[c("K", "S")],
        opportunity_click=Point(546, 925),
        deck_count=52,
    )
    controller._sync_tracker(snapshot)
    assert controller._tracker_started
    assert controller.tracker.table_visible["K:S"] == 1

    # 出牌阶段快速识别没有牌堆数：不能把已知的 52 覆盖成未知。
    discard = GameSnapshot(
        screen_type=ScreenType.DISCARD,
        gold_ranks=("6", "J", "A"),
        hand=HAND7 + [c("K", "S")],
        deck_count=None,
    )
    controller._sync_tracker(discard)
    assert controller.tracker.deck_count == 52


# ---------------- 每局结果记录 ----------------


def _controller(tmp_path: Path) -> BaqueGameController:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
        rounds_log_path=tmp_path / "rounds.jsonl",
    )
    controller._round_info_attempts = 2  # 测试中不读取本局信息
    return controller


def test_turn_without_discard_is_counted_as_lost_with_reason(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    controller._swap_phase_active = False

    controller._begin_turn()
    controller._count_action("draw")
    controller._note_turn_problem("discard 阶段应识别到 8 张手牌，当前为 7 张，暂不操作。", rejection=True)
    controller._end_turn()

    controller._begin_turn()
    controller._count_action("draw")
    controller._count_action("discard")
    controller._end_turn()

    controller._finish_round_report("结算页")

    rows = controller.round_log.load()
    assert len(rows) == 1
    row = rows[0]
    assert row["turns"] == 2
    assert row["draws"] == 2 and row["discards"] == 1
    assert row["lost_turns"] == 1
    assert row["lost_turn_reasons"] == {"手牌张数识别不符": 1}
    assert row["won"] is False
    assert row["end_reason"] == "结算页"


def test_hu_click_marks_the_round_as_won(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    controller._begin_turn()
    controller._count_action("hu")
    controller._finish_round_report("结算页")

    stats = controller.round_stats()
    assert stats["rounds"] == 1 and stats["wins"] == 1 and stats["win_rate"] == 1.0


def test_round_without_own_turns_is_not_recorded(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("自动开局")
    controller._finish_round_report("停止自动游戏")
    assert controller.round_log.load() == []


def test_tenpai_and_passed_hu_are_noted(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    hand = [c("9", s) for s in ("S", "H", "C", "D")] + [c("10", s) for s in ("S", "H", "C")] + [c("K", "S")]
    tracker = TableTracker()
    tracker.reset_round(hand, 50)
    snapshot = GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=("6", "7", "8"), hand=hand)

    # 打出 K♠ 后：四张 9 + 三张 10，再摸到 10♦ 或癞子即可胡。
    controller._note_eight_card_decision(snapshot, GameDecision("discard", card_indexes=(7,)), tracker)

    report = controller.round_report
    assert report is not None
    assert report.tenpai_turns == 1
    assert report.best_waits >= 1
    assert report.hu_available == 0


@pytest.mark.asyncio
async def test_turn_gate_counts_turns_and_lost_turns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._start_round_report("测试")
    controller._swap_phase_active = False
    frame = CapturedFrame(tmp_path / "f.png", 0, 0, 1936, 1088, "欢乐斗地主")
    probes = iter(
        [
            TurnProbe("draw", True, frame),  # 轮到自己
        ]
    )
    monkeypatch.setattr("game_baque.controller.probe_game_turn", lambda _d, _k: next(probes))

    class FakeVision:
        async def parse_turn(self, *_args: object, **_kwargs: object) -> GameSnapshot:
            return GameSnapshot(screen_type=ScreenType.DRAW)

    controller.vision = FakeVision()  # type: ignore[assignment]
    await controller._wait_for_turn(Event())
    assert controller.round_report is not None and controller.round_report.turns == 1

    # 对手回合到来时，本回合没有出牌 → 记一次丢失回合。
    probes = iter([TurnProbe(None, True, None), TurnProbe("draw", True, frame)])
    monkeypatch.setattr("game_baque.controller.probe_game_turn", lambda _d, _k: next(probes))
    await controller._wait_for_turn(Event())
    assert controller.round_report.lost_turns == 1
    assert controller.round_report.turns == 2


def test_round_log_stats_aggregate(tmp_path: Path) -> None:
    log = RoundLog(tmp_path / "r.jsonl")
    for won, waits in ((False, 0), (True, 3), (False, 5)):
        report = RoundReport(turns=4, best_waits=waits, hu_clicked=won)
        log.append(report)
    stats = log.stats()
    assert stats["rounds"] == 3
    assert stats["wins"] == 1
    assert stats["tenpai_rounds"] == 2
    assert len(stats["recent"]) == 3
    assert log.stats(last=1)["rounds"] == 1


def test_each_round_saves_screenshots_in_its_own_folder(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    controller._start_round_report("自动开局")
    first_dir = controller.screenshots_dir
    assert first_dir.parent == tmp_path / "rounds"
    assert first_dir.is_dir()
    assert controller.round_report is not None
    assert controller.round_report.screenshots_dir == str(first_dir)

    controller._begin_turn()
    controller._finish_round_report("结算页")
    # 结算后、下一局开始前的截图仍放在本局文件夹（结算页属于这一局）。
    assert controller.screenshots_dir == first_dir

    controller._start_round_report("新一局换牌")
    second_dir = controller.screenshots_dir
    assert second_dir != first_dir
    assert second_dir.parent == tmp_path / "rounds"

    rows = controller.round_log.load()
    assert rows[0]["screenshots_dir"] == str(first_dir)


# ---------------- 摸牌/出牌一致：拿进的亮牌不打回 ----------------


def test_draw_skips_a_visible_card_it_would_discard_right_back() -> None:
    from game_baque.ev import ExpectedValueEngine

    hand = [c("9", "S"), c("9", "H"), c("9", "C"), c("10", "S"), c("10", "H"), c("10", "C"), c("J", "S")]
    tracker = TableTracker()
    tracker.reset_round(hand, 50)
    engine = ExpectedValueEngine(simulations=60, seed=2)
    # 6♦ 与手牌毫无关系：拿进来最优就是马上打回，不能取。
    decision = engine.choose_draw_action(hand, [c("6", "D")], ("7", "8", "K"), tracker, time_budget_seconds=2.0)
    assert decision.action == "draw_deck"
    assert "不取 6♦" in decision.reason


def test_discard_never_throws_back_the_card_taken_this_turn() -> None:
    from game_baque.ev import ExpectedValueEngine

    hand = [c("9", "S"), c("9", "H"), c("9", "C"), c("10", "S"), c("10", "H"), c("10", "C"), c("J", "S"), c("6", "D")]
    tracker = TableTracker()
    tracker.reset_round(hand, 50)
    engine = ExpectedValueEngine(simulations=60, seed=2)
    decision, values = engine.choose_8_card_action(
        hand, ("7", "8", "K"), tracker, time_budget_seconds=2.0, avoid_discard=c("6", "D")
    )
    assert decision.action == "discard"
    assert hand[decision.card_indexes[0]].label != "6♦"
    assert all(v.card_index != 7 for v in values if v.action == "discard")
