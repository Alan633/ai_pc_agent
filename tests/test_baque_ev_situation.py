from __future__ import annotations

from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.controller import BaqueGameController
from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card
from game_baque.rules import evaluate_hu
from game_baque.strategy import hand_potential
from game_baque.tracker import TableTracker

GOLD = ("6", "7", "8")
SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def _tracker(kept: list[Card], deck: int | None) -> TableTracker:
    tracker = TableTracker()
    tracker.reset_round(kept, deck)
    return tracker


KEPT = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣")  # 差一张即可胡，进张很多


def test_own_draws_left_follows_deck_count() -> None:
    engine = ExpectedValueEngine(horizon_own_draws=3)
    assert engine.own_draws_left(None, draw_now=False) == 3
    assert engine.own_draws_left(40, draw_now=False) == 3
    # 出牌后三家先各摸一张：剩 7 张只够自己再摸 1 次，剩 3 张一次也摸不到。
    assert engine.own_draws_left(7, draw_now=False) == 1
    assert engine.own_draws_left(3, draw_now=False) == 0
    # 摸牌阶段马上就摸：剩 1 张也能摸到。
    assert engine.own_draws_left(1, draw_now=True) == 1
    assert engine.own_draws_left(0, draw_now=True) == 0


def test_no_future_value_when_deck_cannot_reach_own_next_draw() -> None:
    engine = ExpectedValueEngine(simulations=80, seed=1)
    tracker = _tracker(KEPT, 3)
    assert engine._rollout_future_hu(KEPT, GOLD, tracker) == 0.0
    assert engine._rollout_future_hu(KEPT, GOLD, tracker, draw_now=True) > 0.0


def test_opponent_threat_lowers_future_value() -> None:
    tracker = _tracker(KEPT, 40)
    safe = ExpectedValueEngine(simulations=200, seed=5)._rollout_future_hu(KEPT, GOLD, tracker, survival=1.0)
    risky = ExpectedValueEngine(simulations=200, seed=5)._rollout_future_hu(KEPT, GOLD, tracker, survival=0.5)
    assert 0.0 < risky <= safe * 0.5 + 1e-9


def test_payment_cap_applies_per_hu_and_doubles_for_hard_hu() -> None:
    from game_baque.opponent_model import capped_payment

    hard = evaluate_hu(cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦"), GOLD)
    soft = evaluate_hu(cards("6♥ 7♥ 8♥ 9♥ J♠ Q♠ K♠") + [Card(wild=True)], GOLD)
    assert hard is not None and hard.hard_multiplier == 2
    assert soft is not None and soft.hard_multiplier == 1
    cap = 1.0  # 远低于两手牌的理论分
    # 每胡一次每家最多付封顶；无癞子胡算两次，最多 2 倍封顶。
    assert capped_payment(soft, cap) == pytest.approx(cap)
    assert capped_payment(hard, cap) == pytest.approx(2 * cap)
    assert capped_payment(hard, None) == hard.theoretical_score


def test_setting_loss_cap_reaches_opponent_model_and_drops_stale_cache() -> None:
    engine = ExpectedValueEngine()
    engine._inference_cache[("stale",)] = object()
    engine.loss_cap = 2000.0
    assert engine.opponent_model.loss_cap == 2000.0
    assert engine._inference_cache == {}


def test_hu_is_taken_when_deck_is_nearly_empty() -> None:
    hand = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦")
    assert evaluate_hu(hand, GOLD) is not None
    engine = ExpectedValueEngine(simulations=60, seed=2, hu_pass_margin=1.0)
    decision, values = engine.choose_8_card_action(hand, GOLD, _tracker(hand[:7], 3), time_budget_seconds=2.0)
    assert decision.action == "hu"
    assert "还能摸 0 次" in decision.reason
    assert all(item.future_hu_value == 0.0 for item in values if item.action == "discard")


def test_controller_sets_loss_cap_from_room_and_base(tmp_path: Path) -> None:
    controller = BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )
    controller._apply_payout_cap(35.0, [272000.0, 0.0, 0.0], 300_000.0)
    assert controller.ev.loss_cap == pytest.approx(300_000.0 / 35.0)
    controller._apply_payout_cap(None, [], None)
    assert controller.ev.loss_cap is None


HU_SHAPE = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦")


def test_no_hu_without_game_hu_button() -> None:
    # 8 张满足牌型，但游戏没给“胡”按钮（如“?”选牌补成）：只能出牌。
    engine = ExpectedValueEngine(simulations=40, seed=4)
    decision, values = engine.choose_8_card_action(
        HU_SHAPE, GOLD, _tracker(HU_SHAPE[:7], 40), time_budget_seconds=2.0, hu_allowed=False
    )
    assert decision.action == "discard"
    assert all(item.action == "discard" for item in values)


def test_choice_popup_does_not_value_option_as_immediate_hu() -> None:
    engine = ExpectedValueEngine()
    hand = HU_SHAPE[:7]
    _index, value = engine.choose_option_card(hand, [HU_SHAPE[7]], GOLD)
    # 选牌补成的 8 张之后还要出一张：按打掉最差一张后 7 张的结构潜力计价，不按立即胡牌分。
    best7 = max(
        hand_potential([c for j, c in enumerate(HU_SHAPE) if j != drop], GOLD) for drop in range(8)
    )
    assert value == pytest.approx(best7 * 0.55)


@pytest.mark.asyncio
async def test_hu_shape_without_hu_button_skips_slow_full_parse(tmp_path: Path) -> None:
    from dataclasses import replace

    from PIL import Image

    from game_baque.models import GameSnapshot, ScreenType
    from game_baque.vision import BaqueVision

    image_path = tmp_path / "discard.png"
    Image.new("RGB", (1936, 1088), (30, 120, 100)).save(image_path)
    vision = BaqueVision("", "local", "http://127.0.0.1:11434/v1")
    crop = GameSnapshot(screen_type=ScreenType.DISCARD, gold_ranks=GOLD, hand=list(HU_SHAPE))

    async def fake_crops(*_args: object, **_kwargs: object) -> GameSnapshot:
        return replace(crop)

    async def no_full_parse(*_args: object, **_kwargs: object) -> GameSnapshot:
        raise AssertionError("没有胡按钮时不得再做约 4 秒的整图识别")

    vision._parse_turn_crops = fake_crops  # type: ignore[method-assign]
    vision._parse_turn_full = no_full_parse  # type: ignore[method-assign]

    snapshot = await vision.parse_turn(image_path, "discard", gold_ranks=GOLD)

    assert snapshot.screen_type == ScreenType.DISCARD
    assert snapshot.hu_button is None
    assert BaqueGameController._hu_offered(snapshot) is False
