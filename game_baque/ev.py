from __future__ import annotations

import copy
from collections.abc import Callable
import math
import random
import time
from dataclasses import dataclass

from game_baque.models import Card, GameDecision, HuResult
from game_baque.opponent_model import OpponentPatternModel, capped_payment
from game_baque.rules import SELF_DRAW_MULTIPLIER, evaluate_hu
from game_baque.strategy import hand_potential
from game_baque.tracker import TableTracker, card_from_key, card_key


# 胡牌后锁定 7 张、自动摸打到牌堆摸完：自己摸到或别人打出能胡的牌都会再胡。
# 剩余牌堆里的牌大多会被自己摸到或被对手打出；对手会留下有用的牌，保守打八折。
REPEAT_HU_VISIBILITY = 0.8
# 选牌后未选中的三张按原顺序回到牌堆顶；之后依次摸牌的是下家、对家、上家。
DECK_TOP_ORDER = ("right", "top", "left")


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
        # 三家各自的欢乐豆（理论分单位，未知为 inf）：没有癞子胡 2 次时，两次
        # 支付合计不能超过对方余额。
        self.seat_balances: tuple[float, ...] | None = None
        self.rng = random.Random(seed)
        self.opponent_model = OpponentPatternModel(samples=max(160, self.simulations // 2), seed=seed)
        self._inference_cache: dict[tuple[object, ...], object] = {}
        self._release_cache: dict[tuple[object, ...], object] = {}
        self._rollout_cache: dict[tuple[object, ...], float] = {}
        self._repeat_cache: dict[tuple[object, ...], float] = {}
        # 摸牌决策逐选项估值缓存：对手回合里先算好现有亮牌和摸牌堆，轮到自己只补算
        # 上家新打出的那张。键不含完整记牌指纹（对手多打一张牌对估值影响很小）。
        self._draw_option_cache: dict[tuple[object, ...], tuple[object, ...]] = {}
        # 同花顺优先：新手/普通场，或高级/顶级场自己豆不够让对手付满封顶时开启（见 controller）。
        self.prefer_flush = False
        # 追大牌：高级/顶级场自己豆够多时开启，引导往全金牌、特殊牌型上靠（见 controller）。
        self.chase_big = False
        # 平胡优先：三家每人能付的都不超过一个普通平胡（对手豆很少），胡得快最重要。
        self.prefer_pinghu = False
        # 逐次模拟连胡用的原始量（理论分单位，未知为 None）：场次每家每次封顶、三家对手
        # 豆数、自己豆数。每胡一次自己的豆变多、下一次能收的上限随之升高，不能用固定上限。
        self.room_cap_theory: float | None = None
        self.opponent_balances_theory: tuple[float, ...] | None = None
        self.own_theory: float | None = None
        # 三家对手欢乐豆总数（理论分单位，未知为 None）：连胡总共最多只能赢走这么多。
        self.opponent_beans_total: float | None = None
        # 最近一次选牌弹窗的逐张估值说明，供日志输出。
        self.last_option_notes = ""
        # 最近一次摸牌决策里规则引擎与游戏“胡”角标不一致的记录，供修正规则用。
        self.last_rule_mismatches: list[str] = []

    def hu_value(
        self,
        hu: HuResult,
        *,
        self_draw: bool = False,
        delay_rounds: float = 0.0,
        drain: float = 0.0,
    ) -> float:
        """一次胡牌实际能赢到的价值（理论分单位，取三家平均）。

        没有癞子“胡 2 次”：每次各付 min(单次理论分, 该家单次上限)，两次合计
        再受该家余额限制。有癞子只付一次。self_draw：从牌堆摸到胡（自摸）倍数 ×2，
        封顶与余额限制在加倍之后才生效。

        delay_rounds / drain：这次胡要再等 delay_rounds 轮，每轮三家预计从自己这儿胡走
        drain（理论分）。每家每次最多付自己的豆数，豆被扣少了，到时能收的也跟着变少；
        自己豆多时不受影响。
        """
        if delay_rounds > 0 and drain > 0 and self._can_simulate_growth():
            assert self.own_theory is not None and self.opponent_balances_theory is not None
            own = max(0.0, self.own_theory - drain * delay_rounds)
            return sum(self._payment(hu, self_draw, own, list(self.opponent_balances_theory))) / 3
        payments = hu.hard_multiplier if hu.hard_multiplier > 1 else 1
        score = float(hu.theoretical_score) * (SELF_DRAW_MULTIPLIER if self_draw else 1)
        if not self.score_caps:
            return score
        per_payment = score / payments
        balances = self.seat_balances or tuple(math.inf for _ in self.score_caps)
        total = 0.0
        for cap, balance in zip(self.score_caps, balances):
            total += min(balance, payments * min(per_payment, cap))
        return total / len(self.score_caps)

    def capped(self, score: float) -> float:
        """按三家各自支付上限折算后的胡牌价值。

        取三家实付的平均值，保持与未封顶时同一量纲（不封顶时恰好等于理论分），
        以便继续和对手风险等其他按理论分计的项比较。
        """
        if not self.score_caps:
            return float(score)
        return sum(min(float(score), cap) for cap in self.score_caps) / len(self.score_caps)

    @property
    def loss_cap(self) -> float | None:
        """对手每胡一次自己每次最多赔付（理论分单位）= 场次每家封顶 / 底分。

        无癞子胡算两次，最多赔 2 倍。放铳/全桌威胁按此封顶，否则高级场里一个
        理论大牌会把风险项放大到不合理。None 表示未知、不封顶。
        """
        return self.opponent_model.loss_cap

    @loss_cap.setter
    def loss_cap(self, value: float | None) -> None:
        if value != self.opponent_model.loss_cap:
            self.opponent_model.loss_cap = value
            # 对手推断与放铳风险的缓存按旧封顶计算，必须作废。
            self._inference_cache.clear()
            self._release_cache.clear()

    @property
    def loss_limit(self) -> float | None:
        """对手胡一次自己合计最多赔多少（理论分单位）= 自己的豆数 / 底分。"""
        return self.opponent_model.loss_limit

    @loss_limit.setter
    def loss_limit(self, value: float | None) -> None:
        if value != self.opponent_model.loss_limit:
            self.opponent_model.loss_limit = value
            self._inference_cache.clear()
            self._release_cache.clear()

    def repeat_hu_value(
        self,
        locked: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        deck_left: int | None,
        *,
        delay_rounds: float = 0.0,
        drain: float = 0.0,
    ) -> float:
        """胡牌后锁定这 7 张的“连胡”价值（理论分单位，不含第一次胡）。

        锁定后自动摸打直到牌堆摸完，自己摸到或别人打出能胡的牌都会再胡一次。
        预计次数 ≈ 之后能见到的张数 × 能胡的牌占未见牌比例（不超过能胡的牌剩余张数），
        每次按该牌的胡牌价值计（已按每胡一次的封顶）；总额不超过三家对手的豆总数。
        注意不能按“胡一次能收多少”封顶：每赢一次自己豆变多，下一次能收的也更多。
        听得宽、无癞子的手牌价值高。
        """
        if deck_left is None or deck_left <= 0 or len(locked) != 7:
            return 0.0
        key = (
            "repeat",
            tuple(sorted(card_key(card) for card in locked)),
            self._gold_key(gold_ranks),
            tracker.state_fingerprint(),
            deck_left,
            self.score_caps,
            self.seat_balances,
            self.opponent_beans_total,
            self.room_cap_theory,
            self.opponent_balances_theory,
            self.own_theory,
            round(delay_rounds, 2),
            round(drain, 2),
        )
        cached = self._repeat_cache.get(key)
        if cached is not None:
            return cached
        try:
            unseen = tracker.unseen_counts()
        except RuntimeError:
            return 0.0
        total = sum(unseen.values())
        waits = 0
        self_sum = 0.0
        other_sum = 0.0
        wait_hus: list[tuple[int, HuResult]] = []
        for card_id, count in unseen.items():
            hu = evaluate_hu(locked + [card_from_key(card_id)], gold_ranks)
            if hu is not None:
                waits += count
                wait_hus.append((count, hu))
                self_sum += count * self.hu_value(hu, self_draw=True)
                other_sum += count * self.hu_value(hu)
        value = 0.0
        if waits and total > 0 and self._can_simulate_growth():
            own_hits = deck_left / 4 * waits / total
            other_hits = deck_left * 3 / 4 * REPEAT_HU_VISIBILITY * waits / total
            if own_hits + other_hits > waits:
                scale = waits / (own_hits + other_hits)
                own_hits, other_hits = own_hits * scale, other_hits * scale
            # 每轮（自己摸一次）能见到自己摸的 1 张 + 三家打出的约 3×0.8 张。
            rate = (1 + 3 * REPEAT_HU_VISIBILITY) * waits / total
            value = self._simulate_repeat_growth(
                wait_hus,
                waits,
                own_hits,
                other_hits,
                rate=rate,
                delay_rounds=delay_rounds,
                drain=drain,
            )
        elif waits and total > 0:
            # 每轮四家各摸一张：剩余牌堆约 1/4 由自己摸到（自摸 ×2），其余被三家摸走后
            # 大多会打出来（别人打出的胡 ×1，对手会留下有用的牌，打折计）。
            own_seen = deck_left / 4
            other_seen = deck_left * 3 / 4 * REPEAT_HU_VISIBILITY
            own_hits = own_seen * waits / total
            other_hits = other_seen * waits / total
            if own_hits + other_hits > waits:
                scale = waits / (own_hits + other_hits)
                own_hits, other_hits = own_hits * scale, other_hits * scale
            value = (own_hits * self_sum + other_hits * other_sum) / waits
            if self.opponent_beans_total is not None:
                value = min(value, self.opponent_beans_total)
        self._bounded_put(self._repeat_cache, key, value, limit=2048)
        return value

    def _can_simulate_growth(self) -> bool:
        return (
            self.own_theory is not None
            and self.opponent_balances_theory is not None
            and len(self.opponent_balances_theory) == 3
        )

    def _payment(
        self, hu: HuResult, self_draw: bool, own: float, balances: list[float]
    ) -> list[float]:
        """这一次胡牌三家各付多少（理论分）：每次支付不超过封顶与自己豆数，合计不超过对方与自己的豆。"""
        payments = max(1, hu.hard_multiplier)
        score = float(hu.theoretical_score) * (SELF_DRAW_MULTIPLIER if self_draw else 1)
        per_payment = score / payments
        cap = self.room_cap_theory if self.room_cap_theory is not None else math.inf
        return [
            max(0.0, min(payments * min(per_payment, cap, own), balance, own))
            for balance in balances
        ]

    def _simulate_repeat_growth(
        self,
        wait_hus: list[tuple[int, HuResult]],
        waits: int,
        own_hits: float,
        other_hits: float,
        *,
        rate: float | None = None,
        delay_rounds: float = 0.0,
        drain: float = 0.0,
    ) -> float:
        """逐次模拟锁牌后的连胡：每胡一次自己的豆加上赢到的、对手的豆扣掉付出的，
        下一次按新的上限算。自己豆少时第一把被卡住，但赢几把后上限升高，大牌型才收得满；
        固定上限会严重低估大牌型（19:08 实战：自己 8500 豆，平胡与大牌一样“被卡住”，
        下家 2700 万豆却只胡了平胡）。返回值与 hu_value 同口径（三家平均）。

        rate：每轮（自己摸一次）预计胡几次。第 k 次胡约在第 k/rate 轮，按每轮 discount 折扣：
        对手的豆有限时，窄听要很久才胡一次，期间豆会被别人胡走、牌堆会摸完；宽听收得快。

        delay_rounds / drain：锁牌那次胡要再等几轮；每轮三家预计从自己这儿胡走 drain。
        自己豆少时，对手胡一把大的就可能把豆扣光，之后胡什么都收不到钱——所以豆少要先胡
        一把垫底；豆多时被扣的那点不影响每次能收的上限，可以放心做大牌。"""
        assert self.own_theory is not None and self.opponent_balances_theory is not None
        hits = own_hits + other_hits
        if hits <= 0:
            return 0.0
        self_share = own_hits / hits
        own = max(0.0, float(self.own_theory) - max(0.0, drain) * max(0.0, delay_rounds))
        balances = list(self.opponent_balances_theory)

        def average_payment(own_now: float, balances_now: list[float]) -> list[float]:
            seats = [0.0, 0.0, 0.0]
            for count, hu in wait_hus:
                for self_draw, weight in ((True, self_share), (False, 1.0 - self_share)):
                    if weight <= 0:
                        continue
                    for index, paid in enumerate(self._payment(hu, self_draw, own_now, balances_now)):
                        seats[index] += count * weight * paid / waits
            return seats

        # 先计入锁牌的那一次胡（不计入返回值，第一次胡另算）：之后自己的豆已经变多。
        first = average_payment(own, balances)
        own += sum(first)
        balances = [max(0.0, b - p) for b, p in zip(balances, first)]
        value = 0.0
        remaining = hits
        done = 0.0
        while remaining > 1e-9 and any(b > 0 for b in balances):
            step = min(1.0, remaining)
            if drain > 0 and rate and rate > 0:
                # 两次胡之间约 step/rate 轮，这期间对手胡牌从自己这儿扣走的豆。
                own = max(0.0, own - drain * step / rate)
            if own <= 0:
                break
            seats = average_payment(own, balances)
            gained = step * sum(seats)
            weight = self.discount ** ((done + step / 2) / rate) if rate and rate > 0 else 1.0
            value += gained / 3 * weight
            done += step
            own += gained
            balances = [max(0.0, b - step * p) for b, p in zip(balances, seats)]
            remaining -= step
        return value

    def own_draws_left(self, deck_count: int | None, *, draw_now: bool) -> int:
        """自己还能从牌堆摸几次，上限 horizon_own_draws。

        每轮四家各摸一张：出牌后要先等三家各摸一张才轮到自己。对手取亮牌
        不消耗牌堆，按全摸牌堆估计略偏保守，正好抵消 rollout 的乐观。
        """
        if deck_count is None:
            return self.horizon_own_draws
        deck = max(0, int(deck_count))
        draws = (1 + (deck - 1) // 4 if deck > 0 else 0) if draw_now else deck // 4
        return min(self.horizon_own_draws, draws)

    def round_survival(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        deadline: float | None = None,
    ) -> float:
        """三家在下一轮都没有胡牌的概率；对手越接近成牌，继续做牌越不值。"""
        return self.round_risk(tracker, gold_ranks, deadline=deadline)[0]

    def round_risk(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        deadline: float | None = None,
    ) -> tuple[float, float]:
        """(三家下一轮都不胡的概率, 每轮三家预计从自己这儿胡走的分)。

        后者已按封顶与自己的豆数限制（理论分），用于估计等待期间自己的豆被扣掉多少。
        """
        survival = 1.0
        drain = 0.0
        for item in self._cached_infer_all(
            tracker,
            gold_ranks,
            samples=max(120, self.simulations // 3),
            top_patterns=4,
            top_cards=6,
            deadline=deadline,
        ):
            survival *= 1.0 - min(1.0, max(0.0, item.any_hu_probability_next_draw))
            drain += max(0.0, item.expected_hu_score_next_draw)
        return survival, drain

    def _potential(self, cards: list[Card], gold_ranks: tuple[str, ...] | list[str]) -> float:
        return hand_potential(
            cards,
            gold_ranks,
            prefer_flush=self.prefer_flush,
            chase_big=self.chase_big,
            prefer_pinghu=self.prefer_pinghu,
        )

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
        deadline: float | None = None,
    ):
        return self._cached_infer_all(
            tracker,
            gold_ranks,
            samples=samples or max(120, self.simulations // 3),
            top_patterns=top_patterns,
            top_cards=top_cards,
            deadline=deadline,
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
                        win_value += probability * capped_payment(
                            hu, self.loss_cap, self.loss_limit, self_draw=True
                        )
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
        opportunity_count: int | None = None,
        time_budget_seconds: float | None = None,
        visible_hu: tuple[bool, ...] | None = None,
    ) -> GameDecision:
        """在摸牌堆和桌面亮牌之间做“自己收益 + 阻断对手”的联合决策。

        visible_hu：游戏在每张亮牌上是否标了“胡”角标。规则引擎与游戏判定不一致时
        以游戏为准（引擎曾把要用癞子凑同点三张的牌判成可胡，连续三回合白拿白打）。

        手里有“?”选牌卡时照常摸牌：选牌卡要到出牌阶段选中并点“出牌”才会
        弹出四选一，摸牌阶段点它只会把它抬起。
        """
        end = (
            time.monotonic() + time_budget_seconds
            if time_budget_seconds is not None
            else None
        )
        count = (
            max(0, int(opportunity_count))
            if opportunity_count is not None
            else (1 if opportunity_available else 0)
        )
        expected_hand_size = 7 - count
        if len(hand) != expected_hand_size:
            raise ValueError(
                f"摸牌决策必须输入 {expected_hand_size} 张普通手牌"
                + (f"（另有 {count} 张问号选牌卡）" if count else "")
            )

        if count == 0:
            return self._choose_draw_by_rollout(
                hand, visible_cards, gold_ranks, tracker, end, visible_hu=visible_hu
            )

        def self_value(card: Card) -> float:
            # 问号卡会被所选目标替换，得到的是 7 张普通手牌；此时尚未进入
            # 八张牌胡/出牌阶段，只比较换入后的结构潜力。
            return self._potential(hand + [card], gold_ranks) * 0.55

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

    def _keep_value(
        self,
        kept: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        deadline: float | None,
        *,
        draw_now: bool = False,
        survival: float = 1.0,
        drain: float = 0.0,
    ) -> float:
        """保留这 7 张继续打的价值：与出牌决策同一口径（未来胡牌 + 结构价值）。"""
        future = self._rollout_future_hu(
            kept,
            gold_ranks,
            tracker,
            deadline=deadline,
            draw_now=draw_now,
            survival=survival,
            drain=drain,
        )
        return future + self._potential(kept, gold_ranks) * 0.08

    def _choose_draw_by_rollout(
        self,
        hand: list[Card],
        visible_cards: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        end: float | None,
        *,
        visible_hu: tuple[bool, ...] | None = None,
    ) -> GameDecision:
        """摸牌堆 vs 取亮牌，与出牌决策使用同一套价值。

        旧做法给亮牌打“8 张结构分”，没考虑拿进来后还要打出一张：实测 27 次
        取亮牌里有 5 次下一手又把同一张打回，白白浪费一次摸牌。现在：
        - 摸牌堆：从当前 7 张开始模拟“现在摸一张、继续打”的未来价值；
        - 取亮牌 V：能直接胡就取；否则在“拿进 V 后打哪张”里取最优保留价值，
          若最优就是把 V 打回去，则 V 对自己毫无用处、不取。下一次摸牌要等
          下一轮，因此再乘一次折扣。
        """
        base_key = (
            tuple(sorted(card_key(card) for card in hand)),
            self._gold_key(gold_ranks),
            self.prefer_flush,
            self.chase_big,
            self.prefer_pinghu,
            self.score_caps,
            self.seat_balances,
            self.opponent_beans_total,
            None if tracker.deck_count is None else tracker.deck_count // 4,
        )

        def option_key(card: Card | None, badge: bool | None) -> tuple[object, ...]:
            return (*base_key, "DECK" if card is None else card_key(card), badge)

        # 每张亮牌只细算：结构潜力最好的前 2 种打法 + 打回 V 本身。
        plans: list[tuple[int, list[int]]] = []
        for index, card in enumerate(visible_cards):
            badge = visible_hu[index] if visible_hu is not None and index < len(visible_hu) else None
            if option_key(card, badge) in self._draw_option_cache:
                plans.append((index, []))
                continue
            hand8 = hand + [card]
            order = sorted(
                range(len(hand)),
                key=lambda i: self._potential(
                    [c for j, c in enumerate(hand8) if j != i], gold_ranks
                ),
                reverse=True,
            )
            plans.append((index, order[:2] + [len(hand)]))
        tasks_left = 1 + sum(len(discards) + 1 for _, discards in plans if discards)
        # 只剩少数选项要现算时，按比例缩短时限：每个选项分到的时间与完整计算时相同，
        # 精度不变，但不再让一张新牌独占整个预算。
        options_total = 1 + len(visible_cards)
        options_todo = (0 if option_key(None, None) in self._draw_option_cache else 1) + sum(
            1 for _, discards in plans if discards
        )
        if end is not None and options_todo < options_total:
            now = time.monotonic()
            end = now + max(0.0, end - now) * options_todo / options_total
        risk_box: list[tuple[float, float]] = []

        def risk() -> tuple[float, float]:
            # 只有确实要现算的选项才需要对手推断（全部命中缓存时省掉这一步）。
            if not risk_box:
                risk_box.append(
                    self.round_risk(
                        tracker,
                        gold_ranks,
                        deadline=self._slice_deadline(end, tasks_left + 1, share=0.5),
                    )
                )
            return risk_box[0]

        def survival() -> float:
            return risk()[0]

        def drain() -> float:
            return risk()[1]

        # 摸牌堆是“现在就摸”，不用先熬过对手一轮。
        deck_cached = self._draw_option_cache.get(option_key(None, None))
        if deck_cached is not None:
            deck_value = float(deck_cached[0])  # type: ignore[arg-type]
        else:
            deck_value = self._keep_value(
                hand,
                gold_ranks,
                tracker,
                self._slice_deadline(end, tasks_left),
                draw_now=True,
                survival=survival(),
                drain=drain(),
            )
            self._bounded_put(self._draw_option_cache, option_key(None, None), (deck_value,), limit=4096)
        tasks_left -= 1

        visible_values: list[tuple[float, int, float, float, Card | None]] = []
        skipped: list[str] = []
        badge_hu: list[int] = []
        self.last_rule_mismatches = []
        for index, discards in plans:
            card = visible_cards[index]
            badge = (
                visible_hu[index]
                if visible_hu is not None and index < len(visible_hu)
                else None
            )
            cached = self._draw_option_cache.get(option_key(card, badge))
            if cached is None:
                cached = self._evaluate_visible_option(
                    hand, card, discards, badge, gold_ranks, tracker, end, tasks_left, survival, drain
                )
                self._bounded_put(self._draw_option_cache, option_key(card, badge), cached, limit=4096)
            tasks_left -= 1 + len(discards)
            kind, total, own, deny, planned, mismatch = cached
            if mismatch:
                self.last_rule_mismatches.append(str(mismatch))
            if kind == "badge":
                badge_hu.append(index)
            elif kind == "skip":
                skipped.append(card.label)
            else:
                visible_values.append(
                    (float(total), index, float(own), float(deny), planned)  # type: ignore[arg-type]
                )

        # 游戏标了“胡”但引擎算不出牌型：一定拿（价值取所有选项的上限之上）。
        for index in badge_hu:
            best_other = max([deck_value, *(item[2] for item in visible_values)], default=0.0)
            own = best_other * 1.5 + 100.0
            visible_values.append((own, index, own, 0.0, None))
        # 摸暗牌意味着当前亮牌仍可能被对手利用；作为风险扣分。
        table_exposure = sum(item[3] for item in visible_values)
        deck_total = deck_value - 0.18 * table_exposure
        skipped_text = f"；不取 {'、'.join(skipped)}（拿进后最优仍是打回）" if skipped else ""

        if visible_values:
            best_total, best_index, own, deny, planned_discard = max(
                visible_values, key=lambda item: item[0]
            )
            if best_total > deck_total:
                card = visible_cards[best_index]
                plan = (
                    "可直接胡"
                    if planned_discard is None
                    else f"拿进后计划打出 {planned_discard.label}"
                )
                return GameDecision(
                    "draw_visible",
                    visible_index=best_index,
                    reason=(
                        f"取亮牌 {card.label}（{plan}）：自己价值 {own:.1f}，"
                        f"同时阻断对手价值 {deny:.1f}；综合 {best_total:.1f} > "
                        f"摸牌堆 {deck_total:.1f}{skipped_text}"
                    ),
                )

        return GameDecision(
            "draw_deck",
            reason=(
                f"摸牌堆，期望 {deck_value:.1f}，扣除亮牌留给对手的风险后 "
                f"综合 {deck_total:.1f}，优于当前亮牌{skipped_text}"
            ),
        )

    def _evaluate_visible_option(
        self,
        hand: list[Card],
        card: Card,
        discards: list[int],
        badge: bool | None,
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        end: float | None,
        tasks_left: int,
        survival: Callable[[], float],
        drain: Callable[[], float] = lambda: 0.0,
    ) -> tuple[object, ...]:
        """拿进这张亮牌的价值：(种类, 综合, 自己价值, 阻断价值, 计划打出的牌, 规则不一致说明)。"""
        hand8 = hand + [card]
        deny = self.danger_of_releasing(
            card,
            tracker,
            gold_ranks,
            samples=max(100, self.simulations // 4),
            deadline=self._slice_deadline(end, max(1, tasks_left)),
        )
        hu = evaluate_hu(hand8, gold_ranks)
        mismatch = None
        if hu is not None and badge is False:
            mismatch = f"引擎判可胡、游戏未标胡：{' '.join(c.label for c in hand)} + {card.label}"
            hu = None
        if badge is True and hu is None:
            mismatch = f"游戏标胡、引擎判不可胡：{' '.join(c.label for c in hand)} + {card.label}"
            return ("badge", 0.0, 0.0, deny, None, mismatch)
        if hu is not None:
            single = self.hu_value(hu)  # 拿亮牌胡不算自摸，×1
            repeat = self.repeat_hu_value(hand, gold_ranks, tracker, tracker.deck_count)
            own = single + repeat
            # 与出牌阶段同一门槛：这一次胡的分乘 hu_pass_margin，锁牌连胡一比一。
            # 未来估值偏乐观、几乎不计对手先胡，摸牌时不设门槛会白白放走到手的胡。
            return ("value", single * self.hu_pass_margin + repeat + 0.35 * deny, own, deny, None, mismatch)
        best_discard = -1
        best_keep = float("-inf")
        left = tasks_left
        for discard_index in discards:
            kept = [c for j, c in enumerate(hand8) if j != discard_index]
            value = self._keep_value(
                kept,
                gold_ranks,
                tracker,
                self._slice_deadline(end, max(1, left)),
                survival=survival(),
                drain=drain(),
            )
            left -= 1
            if value > best_keep:
                best_keep, best_discard = value, discard_index
        if best_discard == len(hand) or card_key(hand8[best_discard]) == card_key(card):
            # 拿进来之后最该打掉的就是它自己（或手里一模一样的那张）：等于什么都没做，还错过一次摸牌。
            return ("skip", 0.0, 0.0, deny, None, mismatch)
        own = best_keep * self.discount
        return ("value", own + 0.35 * deny, own, deny, hand8[best_discard], mismatch)

    def choose_option_card(
        self,
        hand: list[Card],
        options: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker | None = None,
        *,
        time_budget_seconds: float | None = None,
    ) -> tuple[int, float]:
        """“请选择想要的牌”弹窗：挑一张。返回（候选下标, 价值）。

        “?”只能在摸牌后、出牌阶段打出：选到的牌补成 8 张后还要再打出一张，
        而且这 8 张即使满足牌型也不能胡（胡牌须 7 张再摸一张）。因此按“打掉
        最差一张后剩下 7 张”的结构潜力比较，而不是 8 张整体或立即胡牌分。

        牌堆顶风险：未选中的三张按原顺序回到牌堆顶，接下来下家、对家、上家
        依次摸牌堆就正好摸到它们。给了 tracker 时，扣除“这张被对应那家摸到后
        直接胡”的期望损失（已按每胡一次封顶）。说明写入 last_option_notes。
        """
        if not options:
            raise ValueError("选牌弹窗没有候选牌")
        end = time.monotonic() + time_budget_seconds if time_budget_seconds is not None else None

        def value(card: Card) -> tuple[float, float]:
            cards = hand + [card]
            whole = self._potential(cards, gold_ranks) * 0.55
            if len(cards) != 8:
                return whole, whole
            best7 = max(
                self._potential([c for j, c in enumerate(cards) if j != drop], gold_ranks)
                for drop in range(8)
            ) * 0.55
            # 打掉一张后的 7 张常常打平（如选中后又把它打掉），平手时取 8 张结构更好的。
            return best7, whole

        risks = (
            self._deck_top_risks(hand, options, gold_ranks, tracker, end)
            if tracker is not None and len(options) == 4
            else None
        )
        scored: list[tuple[tuple[float, float], int, float, float]] = []
        for index, card in enumerate(options):
            own, whole = value(card)
            risk = 0.0
            if risks is not None:
                returned = [j for j in range(len(options)) if j != index]
                risk = sum(risks[j].get(seat, 0.0) for seat, j in zip(DECK_TOP_ORDER, returned))
            total = own - self.opponent_risk_weight * risk
            scored.append(((total, whole), index, own, risk))
        (best_value, _), best_index, _own, _risk = max(scored)
        self.last_option_notes = "；".join(
            f"{options[index].label} 自己 {own:.1f}"
            + (f" − 牌堆顶送胡风险 {risk:.1f}" if risks is not None else "")
            + f" = {total:.1f}"
            for (total, _), index, own, risk in scored
        )
        return best_index, best_value

    def _deck_top_risks(
        self,
        hand: list[Card],
        options: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        end: float | None,
    ) -> list[dict[str, float]]:
        """每张候选若被某家从牌堆摸到、直接凑成胡牌的期望损失（按座位）。

        4 张候选都来自牌堆，不可能在对手手里：先在记牌副本里把它们标成
        “自己手牌 + 已知牌堆顶”，采样对手手牌时就会排除它们。
        """
        world = tracker
        try:
            candidate = copy.deepcopy(tracker)
            candidate.record_opportunity_reveal("self", options, 0)
            candidate.update_own_hand(hand + [options[0]])
            world = candidate
        except (RuntimeError, ValueError, IndexError):
            pass
        result: list[dict[str, float]] = []
        for index, card in enumerate(options):
            per_seat = self._cached_release_risk(
                card,
                world,
                gold_ranks,
                samples=max(100, self.simulations // 4),
                deadline=self._slice_deadline(end, len(options) - index),
            )
            result.append({item.seat: item.expected_hu_score for item in per_seat})
        return result

    def choose_8_card_action(
        self,
        hand: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        *,
        time_budget_seconds: float | None = None,
        avoid_discard: Card | None = None,
        hu_allowed: bool = True,
        locked_hand: list[Card] | None = None,
        self_draw: bool = False,
    ) -> tuple[GameDecision, list[ActionValue]]:
        """time_budget_seconds 给出整体时间上限；到点后各采样按已完成样本估计。

        avoid_discard：本回合刚从亮牌拿进的牌。摸牌决策已判定保留它更优，
        出牌时不再把它（或同点同花的另一张）打回去，否则等于白白放弃一次摸牌。

        回合倒计时约 20 秒且识别要占去数秒，带癞子的手牌全量采样可能远超
        倒计时，因此实时对局必须设预算，宁可少采样也要按时出牌。

        hu_allowed：游戏是否真的给出了“胡”按钮。只有 7 张手牌再摸进一张凑成
        牌型才能胡；8 张满足牌型但没有胡按钮时（如“?”选牌补成），只能出牌。
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
        survival = 1.0
        for item in pattern_inference:
            survival *= 1.0 - min(1.0, max(0.0, item.any_hu_probability_next_draw))
        draws_left = self.own_draws_left(tracker.deck_count, draw_now=False)
        situation = (
            f"局势：牌堆 {tracker.deck_count if tracker.deck_count is not None else '?'} 张，"
            f"出牌后还能摸 {draws_left} 次，三家下一轮都不胡 {survival:.0%}"
        )
        immediate = evaluate_hu(hand, gold_ranks) if hu_allowed else None
        if immediate is not None:
            # 这张是从牌堆摸来的（自摸）倍数 ×2；拿亮牌胡 ×1。
            single_value = self.hu_value(immediate, self_draw=self_draw)
            immediate_value = single_value
            # 胡后锁定的是摸进这张之前的 7 张；不知道是哪张摸进来的时，按最保守的取法。
            candidates = (
                [locked_hand]
                if locked_hand is not None and len(locked_hand) == 7
                else [[card for j, card in enumerate(hand) if j != i] for i in range(8)]
            )
            repeat = min(
                self.repeat_hu_value(lock, gold_ranks, tracker, tracker.deck_count)
                for lock in candidates
            )
            immediate_value += repeat
            values.append(
                ActionValue(
                    action="hu",
                    card_index=None,
                    expected_value=immediate_value,
                    immediate_hu_value=single_value,
                    future_hu_value=0.0,
                    opponent_risk=0.0,
                    reason=(
                        f"现在胡 {immediate.pattern}，理论分值 {immediate.theoretical_score}，"
                        f"锁牌后预计连胡价值 {repeat:.0f}"
                    ),
                )
            )

        avoid_key = card_key(avoid_discard) if avoid_discard is not None else None
        if avoid_key is not None and all(card_key(card) == avoid_key for card in hand):
            avoid_key = None
        for index, discarded in enumerate(hand):
            if avoid_key is not None and card_key(discarded) == avoid_key:
                continue
            kept = [card for i, card in enumerate(hand) if i != index]
            future = self._rollout_future_hu(
                kept,
                gold_ranks,
                tracker,
                deadline=self._slice_deadline(end, len(hand) - index, share=0.7),
                survival=survival,
                drain=table_risk,
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
            structure = self._potential(kept, gold_ranks) * 0.08
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
            hu_option = next(item for item in values if item.action == "hu")
            # 门槛只加在“这一次胡”的分上（对冲未来估值偏乐观）；锁牌连胡两边是同一种
            # 估算，一比一比较。若整体乘 2 倍，连胡价值一碰到上限，继续做牌永远达不到门槛
            # （18:02 实战因此没打 K♥，错过锁定“几乎什么都能胡”的牌）。
            single = hu_option.immediate_hu_value
            threshold = single * self.hu_pass_margin + (hu_option.expected_value - single)
            if best.expected_value < threshold:
                best = hu_option
        if best.action == "hu":
            assert immediate is not None
            return (
                GameDecision(
                    "hu",
                    reason=(
                        f"立即胡牌 {best.expected_value:.0f} 分（继续做牌期望须达到 "
                        f"{self.hu_pass_margin:.1f} 倍才放弃）。"
                        f"{immediate.pattern} ×{immediate.pattern_multiplier}，"
                        f"金牌 {immediate.gold_count} 张 ×{immediate.gold_multiplier}。{situation}"
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
                    f"不立即胡/继续做牌更优；{best.reason}；综合EV {best.expected_value:.1f}；{situation}"
                    + (
                        f"（{avoid_discard.label} 是本回合刚拿的亮牌，不打回）"
                        if avoid_key is not None and avoid_discard is not None
                        else ""
                    )
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
        draw_now: bool = False,
        survival: float = 1.0,
        drain: float = 0.0,
    ) -> float:
        """保留 kept 继续打的未来胡牌期望。

        draw_now=True 表示马上就摸（摸牌阶段）；否则先要熬过对手一轮。
        可摸次数受剩余牌堆限制；每熬过一轮都要乘一次 survival（三家都没胡的概率）。
        drain：每轮三家预计从自己这儿胡走的分。晚胡几轮，自己的豆先被扣掉这么多，
        到时每家能收的上限（不超过自己的豆）随之降低。
        """
        drain = max(0.0, float(drain))
        horizon = self.own_draws_left(tracker.deck_count, draw_now=draw_now)
        survival = min(1.0, max(0.0, float(survival)))
        cache_key = (
            "rollout",
            tracker.state_fingerprint(),
            self._gold_key(gold_ranks),
            tuple(sorted(card_key(card) for card in kept)),
            self.simulations,
            horizon,
            draw_now,
            round(survival, 4),
            round(drain, 2),
            self.prefer_flush,
            self.chase_big,
            self.prefer_pinghu,
            self.discount,
            self.score_caps,
            self.seat_balances,
        )
        cached = self._rollout_cache.get(cache_key)
        if cached is not None:
            return cached

        rounds_before_first = 0 if draw_now else 1
        # 自己第 depth 次摸牌前，牌堆已被摸走的张数：每轮四家各一张，出牌后先轮三家。
        depth_dists = [
            tracker.draw_distribution(skip=depth * 4 + (0 if draw_now else 3))
            for depth in range(max(0, horizon))
        ]
        if horizon <= 0 or not depth_dists[0]:
            return 0.0
        depth_keys = [list(dist) for dist in depth_dists]
        depth_weights = [[dist[key] for key in keys] for dist, keys in zip(depth_dists, depth_keys)]
        total = 0.0
        completed = 0

        for _ in range(self.simulations):
            if completed and self._expired(deadline):
                break
            completed += 1
            hand = list(kept)
            best_value = 0.0
            for depth in range(horizon):
                if not depth_keys[depth]:
                    break
                key = self.rng.choices(depth_keys[depth], weights=depth_weights[depth], k=1)[0]
                hand8 = hand + [card_from_key(key)]
                hu = evaluate_hu(hand8, gold_ranks)
                if hu is not None:
                    # 封顶按每胡一次计算，无癞子胡最多 2 倍封顶：未来胡牌与立即胡同一口径。
                    # 胡后锁定的是摸这张之前的 7 张，之后还能连胡到牌堆摸完。
                    deck_then = (
                        None
                        if tracker.deck_count is None
                        else tracker.deck_count - 4 * (depth + rounds_before_first)
                    )
                    delay = depth + rounds_before_first
                    future_value = self.hu_value(
                        hu, self_draw=True, delay_rounds=delay, drain=drain
                    ) + self.repeat_hu_value(
                        hand, gold_ranks, tracker, deck_then, delay_rounds=delay, drain=drain
                    )
                    weight = (self.discount ** depth) * (
                        survival ** (depth + rounds_before_first)
                    )
                    best_value = max(best_value, future_value * weight)
                # 即使当前能胡，也允许比较“继续做更大牌”的路线。
                if depth + 1 < horizon:
                    best_discard = max(
                        range(8),
                        key=lambda i: self._potential(
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
