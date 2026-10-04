from __future__ import annotations

import math
import random
from dataclasses import dataclass

from game_baque.models import Card, GameDecision
from game_baque.opponent_model import OpponentPatternModel
from game_baque.rules import evaluate_hu
from game_baque.strategy import hand_potential
from game_baque.tracker import TableTracker, card_from_key


@dataclass(frozen=True, slots=True)
class OpponentThreat:
    seat: str
    win_probability_next_draw: float
    mean_hu_value: float
    known_cards: int
    unknown_cards: int


@dataclass(frozen=True, slots=True)
class ActionValue:
    action: str
    card_index: int | None
    expected_value: float
    immediate_hu_value: float
    future_hu_value: float
    opponent_risk: float
    reason: str


class ExpectedValueEngine:
    """用公开记牌状态做蒙特卡洛收益比较。

    这是策略层，不负责视觉识别。所有未知对手手牌均从 TableTracker 的条件牌池采样。
    """

    def __init__(
        self,
        *,
        simulations: int = 600,
        horizon_own_draws: int = 3,
        opponent_risk_weight: float = 0.55,
        discount: float = 0.92,
        seed: int | None = None,
    ) -> None:
        self.simulations = max(50, int(simulations))
        self.horizon_own_draws = max(1, min(int(horizon_own_draws), 8))
        self.opponent_risk_weight = max(0.0, float(opponent_risk_weight))
        self.discount = min(max(float(discount), 0.1), 1.0)
        self.rng = random.Random(seed)
        self.opponent_model = OpponentPatternModel(samples=max(160, self.simulations // 2), seed=seed)

    def opponent_threats(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
    ) -> list[OpponentThreat]:
        samples = samples or max(120, self.simulations // 3)
        draw_dist = tracker.draw_distribution()
        if not draw_dist:
            return []

        accum = {
            seat: {"wins": 0.0, "value": 0.0}
            for seat in tracker.opponents
        }
        for _ in range(samples):
            hands = tracker.sample_opponent_hands(self.rng)
            for seat, hand in hands.items():
                if tracker.opponents[seat].finished or len(hand) != 7:
                    continue
                win_prob = 0.0
                win_value = 0.0
                for key, probability in draw_dist.items():
                    hu = evaluate_hu(hand + [card_from_key(key)], gold_ranks)
                    if hu is not None:
                        win_prob += probability
                        win_value += probability * hu.theoretical_score
                accum[seat]["wins"] += win_prob
                accum[seat]["value"] += win_value

        result: list[OpponentThreat] = []
        for seat, opp in tracker.opponents.items():
            result.append(
                OpponentThreat(
                    seat=seat,
                    win_probability_next_draw=accum[seat]["wins"] / samples,
                    mean_hu_value=accum[seat]["value"] / samples,
                    known_cards=sum(opp.known_hand.values()),
                    unknown_cards=opp.unknown_hand_count,
                )
            )
        return result

    def danger_of_releasing(
        self,
        card: Card,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
    ) -> float:
        """估计把某张牌暴露给对手后，直接帮助三家成牌的期望损失。"""
        risks = self.opponent_model.release_risk(
            card,
            tracker,
            gold_ranks,
            samples=samples or max(120, self.simulations // 3),
        )
        return sum(item.expected_hu_score for item in risks)

    def choose_8_card_action(
        self,
        hand: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
    ) -> tuple[GameDecision, list[ActionValue]]:
        if len(hand) != 8:
            raise ValueError("收益决策必须输入 8 张手牌")

        values: list[ActionValue] = []
        pattern_inference = self.opponent_model.infer_all(
            tracker,
            gold_ranks,
            samples=max(120, self.simulations // 3),
            top_patterns=4,
            top_cards=6,
        )
        table_risk = sum(item.expected_hu_score_next_draw for item in pattern_inference)
        immediate = evaluate_hu(hand, gold_ranks)
        if immediate is not None:
            values.append(
                ActionValue(
                    action="hu",
                    card_index=None,
                    expected_value=float(immediate.theoretical_score),
                    immediate_hu_value=float(immediate.theoretical_score),
                    future_hu_value=0.0,
                    opponent_risk=0.0,
                    reason=f"现在胡 {immediate.pattern}，理论分值 {immediate.theoretical_score}",
                )
            )

        for index, discarded in enumerate(hand):
            kept = [card for i, card in enumerate(hand) if i != index]
            future = self._rollout_future_hu(kept, gold_ranks, tracker)
            release_risks = self.opponent_model.release_risk(
                discarded,
                tracker,
                gold_ranks,
                samples=max(120, self.simulations // 3),
            )
            danger = sum(item.expected_hu_score for item in release_risks)
            # 小幅加入结构潜力，避免所有有限 horizon 未胡时全部变成 0。
            structure = hand_potential(kept, gold_ranks) * 0.08
            # table_risk 反映继续一轮时三家抢先成牌的总体威胁；
            # danger 反映当前弃牌本身给三家送胡的增量风险。
            ev = future + structure - self.opponent_risk_weight * danger - 0.22 * table_risk
            values.append(
                ActionValue(
                    action="discard",
                    card_index=index,
                    expected_value=ev,
                    immediate_hu_value=0.0,
                    future_hu_value=future,
                    opponent_risk=danger,
                    reason=(
                        f"打 {discarded.label}：未来胡牌EV {future:.1f}，"
                        f"结构价值 {structure:.1f}，放牌风险 {danger:.1f}，"
                        f"全桌下一轮威胁 {table_risk:.1f}"
                    ),
                )
            )

        best = max(values, key=lambda item: item.expected_value)
        if best.action == "hu":
            assert immediate is not None
            return (
                GameDecision(
                    "hu",
                    reason=(
                        f"即时胡牌EV {best.expected_value:.1f} 高于继续做牌。"
                        f"{immediate.pattern} ×{immediate.pattern_multiplier}，"
                        f"金牌 {immediate.gold_count} 张 ×{immediate.gold_multiplier}"
                    ),
                    hu=immediate,
                ),
                sorted(values, key=lambda item: item.expected_value, reverse=True),
            )

        assert best.card_index is not None
        return (
            GameDecision(
                "discard",
                card_indexes=(best.card_index,),
                reason=(
                    f"不立即胡/继续做牌更优；{best.reason}；综合EV {best.expected_value:.1f}"
                ),
            ),
            sorted(values, key=lambda item: item.expected_value, reverse=True),
        )

    def _rollout_future_hu(
        self,
        kept: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
    ) -> float:
        draw_dist = tracker.draw_distribution()
        if not draw_dist:
            return 0.0
        keys = list(draw_dist)
        weights = [draw_dist[key] for key in keys]
        total = 0.0

        for _ in range(self.simulations):
            hand = list(kept)
            best_value = 0.0
            for depth in range(self.horizon_own_draws):
                key = self.rng.choices(keys, weights=weights, k=1)[0]
                hand8 = hand + [card_from_key(key)]
                hu = evaluate_hu(hand8, gold_ranks)
                if hu is not None:
                    best_value = max(
                        best_value,
                        hu.theoretical_score * (self.discount ** depth),
                    )
                # 即使当前能胡，也允许比较“继续做更大牌”的路线。
                if depth + 1 < self.horizon_own_draws:
                    best_discard = max(
                        range(8),
                        key=lambda i: hand_potential(
                            [card for j, card in enumerate(hand8) if j != i],
                            gold_ranks,
                        ),
                    )
                    hand = [card for j, card in enumerate(hand8) if j != best_discard]
            total += best_value
        return total / self.simulations
