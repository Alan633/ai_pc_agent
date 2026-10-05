from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from game_baque.models import Card, GameDecision
from game_baque.opponent_model import OpponentPatternModel
from game_baque.rules import evaluate_hu
from game_baque.strategy import hand_potential
from game_baque.tracker import TableTracker, card_from_key, card_key


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
        hu_pass_margin: float = 2.0,
        seed: int | None = None,
    ) -> None:
        self.simulations = max(50, int(simulations))
        self.horizon_own_draws = max(1, min(int(horizon_own_draws), 8))
        self.opponent_risk_weight = max(0.0, float(opponent_risk_weight))
        self.discount = min(max(float(discount), 0.1), 1.0)
        # 能胡时，只有继续做牌的期望至少是“立即胡”的这么多倍才放弃胡牌。
        # 未来收益的蒙特卡洛假设自己还能摸到后续几张，几乎不计对手先胡
        # 终局（自己反而要赔分）的风险，天然偏乐观；实测曾为 +13% 放弃
        # 3840 分的胡，随后对手胡牌。
        self.hu_pass_margin = max(1.0, float(hu_pass_margin))
        # 三家各自最多能付的分数（理论分单位）= min(场次每家封顶, 该家欢乐豆) / 底分。
        # 胡牌时每家支付 min(理论分, 该家上限)；超过上限的牌型分毫无意义，
        # 追大牌只会徒增被对手先胡的风险。None 表示未知、不封顶。
        self.score_caps: tuple[float, ...] | None = None
        self.rng = random.Random(seed)
        self.opponent_model = OpponentPatternModel(samples=max(160, self.simulations // 2), seed=seed)
        self._inference_cache: dict[tuple[object, ...], object] = {}
        self._release_cache: dict[tuple[object, ...], object] = {}
        self._rollout_cache: dict[tuple[object, ...], float] = {}

    def capped(self, score: float) -> float:
        """按三家各自支付上限折算后的胡牌价值。

        取三家实付的平均值，保持与未封顶时同一量纲（不封顶时恰好等于理论分），
        以便继续和对手风险等其他按理论分计的项比较。
        """
        if not self.score_caps:
            return float(score)
        return sum(min(float(score), cap) for cap in self.score_caps) / len(self.score_caps)

    @staticmethod
    def _gold_key(gold_ranks: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(gold_ranks)

    @staticmethod
    def _bounded_put(cache: dict, key: tuple[object, ...], value: object, limit: int = 512) -> None:
        if len(cache) >= limit:
            cache.clear()
        cache[key] = value

    def _cached_infer_all(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int,
        top_patterns: int,
        top_cards: int,
        deadline: float | None = None,
    ):
        key = (
            "infer",
            tracker.state_fingerprint(),
            self._gold_key(gold_ranks),
            samples,
            top_patterns,
            top_cards,
        )
        cached = self._inference_cache.get(key)
        if cached is not None:
            return cached
        value = self.opponent_model.infer_all(
            tracker,
            gold_ranks,
            samples=samples,
            top_patterns=top_patterns,
            top_cards=top_cards,
            deadline=deadline,
        )
        if not self._expired(deadline):
            self._bounded_put(self._inference_cache, key, value)
        return value

    def _cached_release_risk(
        self,
        card: Card,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int,
        deadline: float | None = None,
    ):
        key = (
            "release",
            tracker.state_fingerprint(),
            self._gold_key(gold_ranks),
            card_key(card),
            samples,
        )
        cached = self._release_cache.get(key)
        if cached is not None:
            return cached
        value = self.opponent_model.release_risk(
            card,
            tracker,
            gold_ranks,
            samples=samples,
            deadline=deadline,
        )
        if not self._expired(deadline):
            self._bounded_put(self._release_cache, key, value)
        return value

    @staticmethod
    def _expired(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    @staticmethod
    def _slice_deadline(end: float | None, parts_left: int, share: float = 1.0) -> float | None:
        """把剩余预算平均分给剩下的候选，保证每个候选都能拿到相近的样本量。"""
        if end is None:
            return None
        now = time.monotonic()
        return now + max(0.0, end - now) / max(1, parts_left) * share


    def infer_opponents(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
        top_patterns: int = 5,
        top_cards: int = 8,
    ):
        return self._cached_infer_all(
            tracker,
            gold_ranks,
            samples=samples or max(120, self.simulations // 3),
            top_patterns=top_patterns,
            top_cards=top_cards,
        )

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
        deadline: float | None = None,
    ) -> float:
        """估计把某张牌暴露给对手后，直接帮助三家成牌的期望损失。"""
        risks = self._cached_release_risk(
            card,
            tracker,
            gold_ranks,
            samples=samples or max(120, self.simulations // 3),
            deadline=deadline,
        )
        return sum(item.expected_hu_score for item in risks)

    def choose_draw_action(
        self,
        hand: list[Card],
        visible_cards: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        *,
        opportunity_available: bool = False,
        time_budget_seconds: float | None = None,
    ) -> GameDecision:
        """在摸牌堆和桌面亮牌之间做“自己收益 + 阻断对手”的联合决策。

        手里有“?”选牌卡时照常摸牌：选牌卡要到出牌阶段选中并点“出牌”才会
        弹出四选一，摸牌阶段点它只会把它抬起。
        """
        end = (
            time.monotonic() + time_budget_seconds
            if time_budget_seconds is not None
            else None
        )
        expected_hand_size = 6 if opportunity_available else 7
        if len(hand) != expected_hand_size:
            raise ValueError(
                f"摸牌决策必须输入 {expected_hand_size} 张普通手牌"
                + ("（另有 1 张问号选牌卡）" if opportunity_available else "")
            )

        def self_value(card: Card) -> float:
            hand8 = hand + [card]
            # 问号卡会被所选目标替换，得到的是 7 张普通手牌；此时尚未进入
            # 八张牌胡/出牌阶段，只比较换入后的结构潜力。
            if opportunity_available:
                return hand_potential(hand8, gold_ranks) * 0.55
            hu = evaluate_hu(hand8, gold_ranks)
            if hu is not None:
                return self.capped(hu.theoretical_score)
            # 非立即胡时使用结构潜力，量纲压到与中低倍成牌收益可比较。
            return hand_potential(hand8, gold_ranks) * 0.55

        visible_values: list[tuple[float, int, float, float]] = []
        for index, card in enumerate(visible_cards):
            own = self_value(card)
            deny = self.danger_of_releasing(
                card,
                tracker,
                gold_ranks,
                samples=max(100, self.simulations // 4),
                # 留一份预算给牌堆期望计算。
                deadline=self._slice_deadline(end, len(visible_cards) - index + 1),
            )
            total = own + 0.35 * deny
            visible_values.append((total, index, own, deny))

        draw_dist = tracker.draw_distribution()
        deck_own = 0.0
        for key, probability in draw_dist.items():
            deck_own += probability * self_value(card_from_key(key))
        # 摸暗牌意味着当前所有亮牌仍可能被对手利用；作为风险扣分。
        table_exposure = sum(item[3] for item in visible_values)
        deck_total = deck_own - 0.18 * table_exposure

        if visible_values:
            best_total, best_index, own, deny = max(visible_values)
            if best_total > deck_total:
                card = visible_cards[best_index]
                return GameDecision(
                    "draw_visible",
                    visible_index=best_index,
                    reason=(
                        f"取亮牌 {card.label}：自己价值 {own:.1f}，"
                        f"同时阻断对手价值 {deny:.1f}；综合 {best_total:.1f} > "
                        f"摸牌堆 {deck_total:.1f}"
                    ),
                )

        return GameDecision(
            "draw_deck",
            reason=(
                f"摸牌堆，期望 {deck_own:.1f}，扣除亮牌留给对手的风险后 "
                f"综合 {deck_total:.1f}，仍优于当前亮牌"
            ),
        )

    def choose_option_card(
        self,
        hand: list[Card],
        options: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
    ) -> tuple[int, float]:
        """“请选择想要的牌”弹窗：按与摸牌相同的自身价值口径挑一张。

        补入后满 8 张且能胡时取理论分，否则取结构潜力（与 choose_draw_action
        的 self_value 一致）。返回（候选下标, 价值）。
        """
        if not options:
            raise ValueError("选牌弹窗没有候选牌")

        def value(card: Card) -> float:
            cards = hand + [card]
            if len(cards) == 8:
                hu = evaluate_hu(cards, gold_ranks)
                if hu is not None:
                    return self.capped(hu.theoretical_score)
            return hand_potential(cards, gold_ranks) * 0.55

        scored = [(value(card), index) for index, card in enumerate(options)]
        best_value, best_index = max(scored)
        return best_index, best_value

    def choose_8_card_action(
        self,
        hand: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        *,
        time_budget_seconds: float | None = None,
    ) -> tuple[GameDecision, list[ActionValue]]:
        """time_budget_seconds 给出整体时间上限；到点后各采样按已完成样本估计。

        回合倒计时约 20 秒且识别要占去数秒，带癞子的手牌全量采样可能远超
        倒计时，因此实时对局必须设预算，宁可少采样也要按时出牌。
        """
        if len(hand) != 8:
            raise ValueError("收益决策必须输入 8 张手牌")

        end = (
            time.monotonic() + time_budget_seconds
            if time_budget_seconds is not None
            else None
        )
        values: list[ActionValue] = []
        pattern_inference = self._cached_infer_all(
            tracker,
            gold_ranks,
            samples=max(120, self.simulations // 3),
            top_patterns=4,
            top_cards=6,
            # 全桌威胁只是一个整体扣分项，占预算的四分之一即可。
            deadline=self._slice_deadline(end, 1, share=0.25),
        )
        table_risk = sum(item.expected_hu_score_next_draw for item in pattern_inference)
        immediate = evaluate_hu(hand, gold_ranks)
        if immediate is not None:
            immediate_value = self.capped(immediate.theoretical_score)
            values.append(
                ActionValue(
                    action="hu",
                    card_index=None,
                    expected_value=immediate_value,
                    immediate_hu_value=immediate_value,
                    future_hu_value=0.0,
                    opponent_risk=0.0,
                    reason=f"现在胡 {immediate.pattern}，理论分值 {immediate.theoretical_score}",
                )
            )

        for index, discarded in enumerate(hand):
            kept = [card for i, card in enumerate(hand) if i != index]
            future = self._rollout_future_hu(
                kept,
                gold_ranks,
                tracker,
                deadline=self._slice_deadline(end, len(hand) - index, share=0.7),
            )
            release_risks = self._cached_release_risk(
                discarded,
                tracker,
                gold_ranks,
                samples=max(120, self.simulations // 3),
                deadline=self._slice_deadline(end, len(hand) - index),
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
        if immediate is not None and best.action != "hu":
            hu_value = self.capped(immediate.theoretical_score)
            if best.expected_value < hu_value * self.hu_pass_margin:
                best = next(item for item in values if item.action == "hu")
        if best.action == "hu":
            assert immediate is not None
            return (
                GameDecision(
                    "hu",
                    reason=(
                        f"立即胡牌 {best.expected_value:.0f} 分（继续做牌期望须达到 "
                        f"{self.hu_pass_margin:.1f} 倍才放弃）。"
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
        *,
        deadline: float | None = None,
    ) -> float:
        cache_key = (
            "rollout",
            tracker.state_fingerprint(),
            self._gold_key(gold_ranks),
            tuple(sorted(card_key(card) for card in kept)),
            self.simulations,
            self.horizon_own_draws,
            self.discount,
            self.score_caps,
        )
        cached = self._rollout_cache.get(cache_key)
        if cached is not None:
            return cached

        draw_dist = tracker.draw_distribution()
        if not draw_dist:
            return 0.0
        keys = list(draw_dist)
        weights = [draw_dist[key] for key in keys]
        total = 0.0
        completed = 0

        for _ in range(self.simulations):
            if completed and self._expired(deadline):
                break
            completed += 1
            hand = list(kept)
            best_value = 0.0
            for depth in range(self.horizon_own_draws):
                key = self.rng.choices(keys, weights=weights, k=1)[0]
                hand8 = hand + [card_from_key(key)]
                hu = evaluate_hu(hand8, gold_ranks)
                if hu is not None:
                    best_value = max(
                        best_value,
                        self.capped(hu.theoretical_score) * (self.discount ** depth),
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
        value = total / completed
        if completed == self.simulations:
            self._bounded_put(self._rollout_cache, cache_key, value)
        return value
