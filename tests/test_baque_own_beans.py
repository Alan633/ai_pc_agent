from __future__ import annotations

from pathlib import Path

import pytest

from agent.session import AgentSession
from game_baque.controller import BaqueGameController
from game_baque.models import Card
from game_baque.opponent_model import capped_payment
from game_baque.rules import evaluate_hu

SUITS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}


def cards(text: str) -> list[Card]:
    return [Card(item[:-1], SUITS[item[-1]]) for item in text.split()]


def _controller(tmp_path: Path) -> BaqueGameController:
    return BaqueGameController(
        vision=object(),  # type: ignore[arg-type]
        screenshots_dir=tmp_path,
        session=AgentSession(),
    )


def test_own_beans_cap_what_each_opponent_pays(tmp_path: Path) -> None:
    # 高级场（每家封顶 300 万），自己只有 2.5 万豆：每家最多付 2.5 万，不可能一把赢 300 万。
    controller = _controller(tmp_path)
    controller._apply_payout_cap(240.0, [21_170_000.0, 68_360.0, 25_000.0], 3_000_000.0, own_balance=25_000.0)
    limit = 25_000.0 / 240.0
    assert controller.ev.score_caps == pytest.approx((limit, limit, limit))
    assert controller.ev.seat_balances == pytest.approx((limit, limit, limit))
    # 反过来对手胡牌，自己最多也只赔 2.5 万。
    assert controller.ev.loss_cap == pytest.approx(limit)
    assert controller.ev.loss_limit == pytest.approx(limit)

    # 无癞子大牌（两次支付）：每家两次合计也不超过 2.5 万。
    big = evaluate_hu(cards("6♥ 7♥ 8♥ 9♥ 10♥ J♥ Q♥ K♥"), ("6", "7", "8"))
    assert big is not None and big.theoretical_score > 10 * limit
    assert controller.ev.hu_value(big) == pytest.approx(limit)


def test_without_own_beans_room_cap_still_applies(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._apply_payout_cap(240.0, [21_170_000.0, 68_360.0, 25_000.0], 3_000_000.0)
    assert controller.ev.score_caps == pytest.approx((3_000_000 / 240, 68_360 / 240, 25_000 / 240))
    assert controller.ev.loss_limit is None


def test_capped_payment_respects_total_limit() -> None:
    hard = evaluate_hu(cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦"), ("6", "7", "8"))
    assert hard is not None and hard.hard_multiplier == 2
    assert capped_payment(hard, 100.0) == pytest.approx(min(hard.theoretical_score, 200.0))
    assert capped_payment(hard, 100.0, limit=50.0) == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_round_info_reads_own_beans_and_settlement_triggers_reread(tmp_path: Path, monkeypatch) -> None:
    from game_baque.capture import CapturedFrame

    frame = CapturedFrame(tmp_path / "f.png", 0, 0, 1920, 1080, "欢乐斗地主")
    monkeypatch.setattr("game_baque.controller.capture_game_window", lambda *_a: frame)

    class FakeVision:
        async def parse_round_info(self, _path: Path):
            return 240.0, [21_170_000.0, 68_360.0, 25_000.0], 3_000_000.0

        async def parse_own_beans(self, _path: Path):
            return 25_000.0

    controller = _controller(tmp_path)
    controller.vision = FakeVision()  # type: ignore[assignment]
    await controller._read_round_info()
    assert controller.own_balance == 25_000.0
    assert controller.ev.loss_limit == pytest.approx(25_000.0 / 240.0)
    assert controller._round_info_attempts == 2

    controller._on_settlement("right")
    assert controller._round_info_attempts == 0  # 有人胡牌后豆数变了，重新读


def test_rich_opponent_pays_full_cap_when_we_can_cover_it(tmp_path: Path) -> None:
    # 高级场每家封顶 300 万：对手豆 > 300 万且自己豆也够时，每家能打满 300 万。
    controller = _controller(tmp_path)
    controller._apply_payout_cap(
        240.0, [21_170_000.0, 5_000_000.0, 68_360.0], 3_000_000.0, own_balance=8_000_000.0
    )
    full = 3_000_000.0 / 240.0
    assert controller.ev.score_caps == pytest.approx((full, full, 68_360 / 240))
    # 无癞子胡两次支付，每次最多 300 万：2117 万的一家付满 600 万；500 万的一家只付得出 500 万。
    big = evaluate_hu(cards("6♥ 7♥ 8♥ 9♥ 10♥ J♥ Q♥ K♥"), ("6", "7", "8"))
    assert big is not None and big.hard_multiplier == 2 and big.theoretical_score >= 2 * full
    expected = (2 * full + 5_000_000 / 240 + 68_360 / 240) / 3
    assert controller.ev.hu_value(big) == pytest.approx(expected)
    # 对手胡牌时自己最多赔一次 300 万封顶（无癞子两次合计不超过自己 800 万）。
    assert controller.ev.loss_cap == pytest.approx(full)
    assert controller.ev.loss_limit == pytest.approx(8_000_000.0 / 240.0)
