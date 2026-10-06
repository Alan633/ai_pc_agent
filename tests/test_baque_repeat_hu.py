from __future__ import annotations

from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker, card_from_key, initial_deck_counts

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
GOLD = ("6", "7", "Q")


def cards(text: str) -> list[Card]:
    return [Card(wild=True) if item == "癞子" else Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def _waits(locked: list[Card]) -> list[str]:
    return sorted(
        card_from_key(key).label
        for key in initial_deck_counts()
        if key != "OPPORTUNITY" and evaluate_hu(locked + [card_from_key(key)], GOLD) is not None
    )


def test_locked_hand_waits_match_the_game() -> None:
    # 17:36 实战：胡六六大顺后锁定这 7 张，游戏能胡的是 9♠、7♥、10♥、癞子。
    # 旧引擎多算了 7♠、K♠（6 张顺子 + 8♥9♥ 两张连号——两张一组只能是对子）。
    assert _waits(cards("癞子 Q♠ J♠ 10♠ 8♠ 8♥ 9♥")) == sorted(["9♠", "7♥", "10♥", "癞子"])


def _tracker(hand: list[Card], deck: int = 40) -> TableTracker:
    tracker = TableTracker()
    tracker.reset_round(hand, deck)
    return tracker


def test_wider_locked_hand_is_worth_more() -> None:
    engine = ExpectedValueEngine()
    narrow = cards("癞子 Q♠ J♠ 10♠ 8♠ 8♥ 9♥")  # 实际锁定：4 种
    wide = cards("Q♠ J♠ 10♠ 8♠ 8♥ 9♥ 9♠")  # 若留 9♠ 去癞子：听得更宽
    assert len(_waits(wide)) > len(_waits(narrow))
    tracker = _tracker(narrow + cards("9♠"))
    assert engine.repeat_hu_value(wide, GOLD, tracker, 40) > engine.repeat_hu_value(narrow, GOLD, tracker, 40)


def test_no_repeat_value_when_deck_is_empty_and_more_deck_means_more() -> None:
    engine = ExpectedValueEngine()
    locked = cards("癞子 Q♠ J♠ 10♠ 8♠ 8♥ 9♥")
    tracker = _tracker(locked)
    assert engine.repeat_hu_value(locked, GOLD, tracker, 0) == 0.0
    assert engine.repeat_hu_value(locked, GOLD, tracker, 40) > engine.repeat_hu_value(locked, GOLD, tracker, 8) > 0


def test_repeat_value_is_capped_by_what_opponents_can_still_pay() -> None:
    engine = ExpectedValueEngine()
    locked = cards("Q♠ J♠ 10♠ 8♠ 8♥ 9♥ 9♠")
    tracker = _tracker(locked)
    engine.score_caps = (100.0, 100.0, 100.0)
    engine.opponent_beans_total = 50.0  # 三家对手总共只剩 50 分的豆
    assert engine.repeat_hu_value(locked, GOLD, tracker, 40) <= 50.0
    # 不能按“胡一次能收多少”封顶：每家每次只付 20 分，连胡多次的总额可以超过一次的 60 分。
    engine.score_caps = (20.0, 20.0, 20.0)
    engine.opponent_beans_total = 10_000.0
    engine._repeat_cache.clear()
    assert engine.repeat_hu_value(locked, GOLD, tracker, 40) > 60.0


def test_hu_now_includes_locked_repeat_value() -> None:
    hand = cards("癞子 Q♠ J♠ 10♠ 8♠ 8♥ 9♥ 9♠")
    locked = hand[:7]
    engine = ExpectedValueEngine(simulations=40, seed=1)
    decision, values = engine.choose_8_card_action(
        hand, GOLD, _tracker(locked, 40), time_budget_seconds=1.0, locked_hand=locked
    )
    hu = next(item for item in values if item.action == "hu")
    assert hu.expected_value > engine.hu_value(evaluate_hu(hand, GOLD))  # type: ignore[arg-type]
    assert "连胡" in hu.reason


def test_18_02_pass_small_hu_to_lock_near_universal_wait(tmp_path) -> None:
    # 18:02 实战：拿 10♠ 后可胡六六大顺；但打出 K♥ 留下 癞子+7♠~Q♠，6 张顺子已成，
    # 癞子与几乎任何一张牌都能配对（6+2），锁定后几乎摸什么都胡。旧版把连胡总额按
    # “胡一次能收多少”封顶，两条路被截到同一上限，于是错选了现在胡。
    from pathlib import Path

    from agent.session import AgentSession
    from game_baque.controller import BaqueGameController

    gold = ("6", "7", "A")
    controller = BaqueGameController(vision=object(), screenshots_dir=Path(tmp_path), session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(10.0, [700.0, 14_700.0, 39_700.0], 20_000.0, own_balance=5_700.0)
    assert controller.ev.opponent_beans_total == 5_510.0  # 三家共 55100 豆 / 底分 10

    hand = cards("癞子 Q♠ J♠ 9♠ 8♠ 7♠ K♥ 10♠")
    tracker = TableTracker()
    tracker.reset_round(hand, 51)
    decision, _values = controller.ev.choose_8_card_action(
        hand, gold, tracker, time_budget_seconds=3.0, hu_allowed=True, locked_hand=hand[:7]
    )
    assert decision.action == "discard"
    assert hand[decision.card_indexes[0]].label == "K♥"


import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("base", "cap", "opponents", "own"),
    [
        (10.0, 20_000.0, [700.0, 14_700.0, 39_700.0], 5_700.0),  # 新手场
        (240.0, 3_000_000.0, [21_000_000.0, 9_000_000.0, 6_000_000.0], 25_000.0),  # 高级场、自己豆少
        (240.0, 3_000_000.0, [21_000_000.0, 9_000_000.0, 6_000_000.0], 8_000_000.0),  # 高级场、自己豆多
    ],
)
def test_wide_wait_beats_narrow_big_pattern_in_every_room(tmp_path, base, cap, opponents, own) -> None:
    # 封顶/豆数限制下每胡一次的收入基本固定，能胡的牌越多越好；豆够多时宽听同样占优。
    from pathlib import Path

    from agent.session import AgentSession
    from game_baque.controller import BaqueGameController

    controller = BaqueGameController(vision=object(), screenshots_dir=Path(tmp_path), session=AgentSession())  # type: ignore[arg-type]
    controller._apply_payout_cap(base, opponents, cap, own_balance=own)
    hand = cards("癞子 Q♠ J♠ 9♠ 8♠ 7♠ K♥ 10♠")
    tracker = TableTracker()
    tracker.reset_round(hand, 51)
    decision, _ = controller.ev.choose_8_card_action(
        hand, ("6", "7", "A"), tracker, time_budget_seconds=3.0, hu_allowed=True, locked_hand=hand[:7]
    )
    assert decision.action == "discard" and hand[decision.card_indexes[0]].label == "K♥"


def _set_beans(engine: ExpectedValueEngine, base: float, cap: float, own: float, opponents: list[float]) -> None:
    engine.room_cap_theory = cap / base
    engine.opponent_balances_theory = tuple(b / base for b in opponents)
    engine.own_theory = own / base


def test_repeat_growth_lifts_own_bean_cap_after_each_hu() -> None:
    # 自己只有 8500 豆、下家 2714 万：第一次胡被自己的豆卡住，但赢了之后上限升高，
    # 后面几次大牌型能多收很多；平胡（160 分）一直收不满，增长有限。
    hu_big = evaluate_hu(cards("8♠ 8♥ 8♣ 8♦ 10♠ 10♥ 10♣ 10♦"), ("7", "Q", "K"))
    hu_small = evaluate_hu(cards("8♠ 8♥ 8♦ 10♠ 10♥ 10♦ 6♣ 6♠"), ("7", "Q", "K"))
    assert hu_big is not None and hu_small is not None
    assert hu_big.theoretical_score > hu_small.theoretical_score
    engine = ExpectedValueEngine()
    _set_beans(engine, 35.0, 300000.0, 8500.0, [370300.0, 8500.0, 27140000.0])
    big = engine._simulate_repeat_growth([(1, hu_big)], 1, 1.0, 3.0)
    small = engine._simulate_repeat_growth([(1, hu_small)], 1, 1.0, 3.0)
    assert big > 1.5 * small
    # 对比固定上限（不计增长）：每次每家最多收自己的 8500 豆。
    fixed_ceiling = 4 * 3 * (8500.0 / 35.0) / 3
    assert big > fixed_ceiling


def test_repeat_growth_never_takes_more_than_opponents_have() -> None:
    hu_big = evaluate_hu(cards("8♠ 8♥ 8♣ 8♦ 10♠ 10♥ 10♣ 10♦"), ("7", "Q", "K"))
    assert hu_big is not None
    engine = ExpectedValueEngine()
    opponents = [2000.0, 3000.0, 5000.0]
    _set_beans(engine, 35.0, 300000.0, 1_000_000.0, opponents)
    value = engine._simulate_repeat_growth([(1, hu_big)], 1, 5.0, 5.0)
    assert 0 <= value * 3 <= sum(opponents) / 35.0 + 1e-6



def test_waiting_while_opponents_drain_own_beans_lowers_income_only_when_poor() -> None:
    # 每轮三家预计从自己这儿胡走 drain：豆少时晚几轮胡，每家能收的上限（不超过自己的豆）变低；
    # 豆多时被扣的那点不影响上限。
    hu = evaluate_hu(cards("8♠ 8♥ 8♣ 8♦ 10♠ 10♥ 10♣ 10♦"), ("7", "Q", "K"))
    assert hu is not None
    poor = ExpectedValueEngine()
    _set_beans(poor, 35.0, 300000.0, 8500.0, [3e7, 3e7, 3e7])
    poor.score_caps = (8500.0 / 35.0,) * 3
    poor.seat_balances = (8500.0 / 35.0,) * 3
    now = poor.hu_value(hu, self_draw=True)
    later = poor.hu_value(hu, self_draw=True, delay_rounds=3, drain=40.0)
    assert later < now
    assert poor.hu_value(hu, self_draw=True, delay_rounds=10, drain=40.0) == 0.0  # 豆被扣光

    rich = ExpectedValueEngine()
    _set_beans(rich, 35.0, 300000.0, 5e7, [3e7, 3e7, 3e7])
    rich.score_caps = (300000.0 / 35.0,) * 3
    rich.seat_balances = (3e7 / 35.0,) * 3
    assert rich.hu_value(hu, self_draw=True, delay_rounds=3, drain=40.0) == rich.hu_value(hu, self_draw=True)


def test_repeat_growth_stops_once_own_beans_are_drained() -> None:
    hu = evaluate_hu(cards("8♠ 8♥ 8♦ 10♠ 10♥ 10♦ 6♣ 6♠"), ("7", "Q", "K"))
    assert hu is not None
    engine = ExpectedValueEngine()
    _set_beans(engine, 35.0, 300000.0, 8500.0, [3e7, 3e7, 3e7])
    calm = engine._simulate_repeat_growth([(1, hu)], 1, 1.0, 3.0, rate=0.5)
    drained = engine._simulate_repeat_growth([(1, hu)], 1, 1.0, 3.0, rate=0.5, delay_rounds=3, drain=60.0)
    wiped = engine._simulate_repeat_growth([(1, hu)], 1, 1.0, 3.0, rate=0.5, delay_rounds=10, drain=60.0)
    assert calm > drained > 0
    assert wiped == 0.0
