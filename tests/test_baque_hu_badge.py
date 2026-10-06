from __future__ import annotations

from PIL import Image, ImageDraw

from game_baque.ev import ExpectedValueEngine
from game_baque.fast_vision import TABLE_SLOT_XS, detect_table_hu_badges
from game_baque.models import Card
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(wild=True) if item == "癞子" else Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def _row(text: str) -> list[Card]:
    return [Card(card.rank, card.suit, x=x, y=386) for card, x in zip(cards(text), TABLE_SLOT_XS)]


def _tracker(hand: list[Card], row: list[Card]) -> TableTracker:
    tracker = TableTracker()
    tracker.reset_round(hand, 44)
    tracker.sync_table(row)
    return tracker


def test_badge_detector_marks_only_badged_slot() -> None:
    image = Image.new("RGB", (1920, 1080), (30, 120, 100))
    x = round(1920 * TABLE_SLOT_XS[3] / 1000)
    ImageDraw.Draw(image).rectangle((x - 40, 440, x + 40, 478), fill=(235, 70, 70))
    assert detect_table_hu_badges(image) == {438: False, 492: False, 546: False, 600: True}


def test_engine_hu_without_game_badge_is_not_taken_as_hu(monkeypatch) -> None:
    # 17:08 实战：旧规则引擎认为拿 K♣ 能胡（用癞子凑 3+3+2），游戏没标“胡”，连续三回合白拿白打。
    # 规则已修正（癞子不能参与 3+3+2）；这里用替身模拟“引擎仍有别的误判”，确认以角标为准。
    import game_baque.ev as ev_module
    from game_baque.models import HuResult

    hand = cards("癞子 K♥ K♦ Q♦ Q♦ 10♠ 10♦")
    row = _row("6♠ 9♠ K♣")
    gold = ("6", "7", "10")
    assert evaluate_hu(hand + cards("K♣"), gold) is None  # 规则修正后与游戏一致
    fake = HuResult("平胡", 1, 160, 1, 1, False, 1, 1, 160, ())
    real = ev_module.evaluate_hu
    monkeypatch.setattr(
        ev_module,
        "evaluate_hu",
        lambda cards_, gold_: fake if any(c.label == "K♣" for c in cards_) else real(cards_, gold_),
    )
    engine = ExpectedValueEngine(simulations=60, seed=1)
    decision = engine.choose_draw_action(
        hand, row, gold, _tracker(hand, row), time_budget_seconds=1.0, visible_hu=(False, False, False)
    )
    assert "可直接胡" not in decision.reason
    assert any("游戏未标胡" in text for text in engine.last_rule_mismatches)


def test_game_badge_is_trusted_even_if_engine_disagrees(monkeypatch) -> None:
    # 规则引擎还不完全等同游戏规则：游戏标了“胡”而引擎算不出时，以游戏为准去拿。
    import game_baque.ev as ev_module

    hand = cards("10♠ 10♣ 10♣ J♠ J♣ J♣ 7♥")
    row = _row("6♥ 9♥ J♥ 7♦")
    gold = ("8", "K", "A")
    real = ev_module.evaluate_hu
    monkeypatch.setattr(
        ev_module,
        "evaluate_hu",
        lambda cards_, gold_: None if any(c.label == "7♦" for c in cards_) else real(cards_, gold_),
    )
    engine = ExpectedValueEngine(simulations=60, seed=1)
    decision = engine.choose_draw_action(
        hand, row, gold, _tracker(hand, row), time_budget_seconds=1.0, visible_hu=(False, False, False, True)
    )
    assert decision.action == "draw_visible" and decision.visible_index == 3
    assert any("游戏标胡、引擎判不可胡" in text for text in engine.last_rule_mismatches)


def test_taking_a_duplicate_only_to_discard_its_twin_is_skipped() -> None:
    # 17:08:21 实战：“取亮牌 Q♦（拿进后计划打出 Q♦）”——拿进一模一样的牌再打出，白白浪费一次摸牌。
    hand = cards("癞子 K♦ Q♦ 10♦ K♥ 10♠ 9♠")
    row = _row("A♠ Q♦")
    engine = ExpectedValueEngine(simulations=80, seed=2)
    decision = engine.choose_draw_action(hand, row, ("6", "7", "10"), _tracker(hand, row), time_budget_seconds=1.5)
    assert "拿进后计划打出 Q♦" not in decision.reason
