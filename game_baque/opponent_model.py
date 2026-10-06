from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from game_baque.models import Card, HuResult
from game_baque.rules import SELF_DRAW_MULTIPLIER, evaluate_hu
from game_baque.tracker import TableTracker, card_from_key


def capped_payment(
    hu: HuResult,
    cap: float | None,
    limit: float | None = None,
    *,
    self_draw: bool = False,
) -> float:
    """一次胡牌某一家实际要付的分（理论分单位）。

    封顶按每胡一次、每家计算：有癞子付一次，最多 cap；无癞子胡算两次，
    每次最多 cap，合计最多 2×cap。limit 是这一家本次合计最多能付的（付款方
    的豆数），None 表示不封顶。
    """
    payments = max(1, hu.hard_multiplier)
    value = float(hu.theoretical_score) * (SELF_DRAW_MULTIPLIER if self_draw else 1)
    if cap is not None:
        value = payments * min(value / payments, cap)
    if limit is not None:
        value = min(value, limit)
    return value


@dataclass(frozen=True, slots=True)
class PatternHypothesis:
    pattern: str
    probability_next_draw: float
    expected_score_next_draw: float
    dangerous_cards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpponentInference:
    seat: str
    any_hu_probability_next_draw: float
    expected_hu_score_next_draw: float
    patterns: tuple[PatternHypothesis, ...]
    dangerous_cards: tuple[tuple[str, float], ...]
    known_cards: int
    unknown_cards: int


@dataclass(frozen=True, slots=True)
class ReleaseSeatRisk:
    seat: str
    hu_probability: float
    expected_hu_score: float
    dominant_patterns: tuple[tuple[str, float], ...]


class OpponentPatternModel:
    """基于记牌信息，对三家未知手牌做条件采样并推断可能成牌方向。

    这里输出的是概率，不把未知牌“猜成确定牌”。随着明牌拿牌、机会牌和出牌事件增加，
    TableTracker 的条件空间会收缩，推断自然会越来越集中。
    """

    def __init__(self, *, samples: int = 360, seed: int | None = None) -> None:
        self.samples = max(80, int(samples))
        self.rng = random.Random(seed)
        # 对手胡牌时自己每次最多赔付（理论分单位）；None 表示不封顶。
        self.loss_cap: float | None = None
        # 对手胡牌时自己本次合计最多赔付（理论分单位）= 自己的豆数 / 底分；None 表示未知。
        self.loss_limit: float | None = None

    def infer_all(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
        top_patterns: int = 5,
        top_cards: int = 8,
        deadline: float | None = None,
    ) -> list[OpponentInference]:
        """deadline 为 time.monotonic() 截止时刻；到点后按已完成样本数归一化。"""
        sample_count = max(40, samples or self.samples)
        if not tracker.draw_distribution():
            return [
                OpponentInference(
                    seat=seat,
                    any_hu_probability_next_draw=0.0,
                    expected_hu_score_next_draw=0.0,
                    patterns=(),
                    dangerous_cards=(),
                    known_cards=sum(opp.known_hand.values()),
                    unknown_cards=opp.unknown_hand_count,
                )
                for seat, opp in tracker.opponents.items()
            ]

        pattern_mass: dict[str, Counter[str]] = {
            seat: Counter() for seat in tracker.opponents
        }
        pattern_score: dict[str, Counter[str]] = {
            seat: Counter() for seat in tracker.opponents
        }
        pattern_cards: dict[str, dict[str, Counter[str]]] = {
            seat: defaultdict(Counter) for seat in tracker.opponents
        }
        dangerous_mass: dict[str, Counter[str]] = {
            seat: Counter() for seat in tracker.opponents
        }
        any_hu_mass = Counter()
        hu_score_mass = Counter()

        completed = 0
        for _ in range(sample_count):
            if completed and deadline is not None and time.monotonic() >= deadline:
                break
            completed += 1
            hands, remaining_pool = tracker.sample_world(self.rng)
            if tracker.known_deck_top:
                sample_draw_dist = {tracker.known_deck_top[0]: 1.0}
            else:
                remaining_total = sum(remaining_pool.values())
                sample_draw_dist = (
                    {key: count / remaining_total for key, count in remaining_pool.items()}
                    if remaining_total > 0
                    else {}
                )
            for seat, hand in hands.items():
                opp = tracker.opponents[seat]
                if opp.finished or len(hand) != 7:
                    continue

                sample_hu_probability = 0.0
                sample_score = 0.0
                for key, draw_probability in sample_draw_dist.items():
                    hu = evaluate_hu(hand + [card_from_key(key)], gold_ranks)
                    if hu is None:
                        continue
                    sample_hu_probability += draw_probability
                    # 对手下一摸从牌堆摸到胡：自摸 ×2。
                    weighted_score = draw_probability * capped_payment(
                        hu, self.loss_cap, self.loss_limit, self_draw=True
                    )
                    sample_score += weighted_score
                    pattern_mass[seat][hu.pattern] += draw_probability
                    pattern_score[seat][hu.pattern] += weighted_score
                    pattern_cards[seat][hu.pattern][key] += draw_probability
                    dangerous_mass[seat][key] += draw_probability

                any_hu_mass[seat] += sample_hu_probability
                hu_score_mass[seat] += sample_score

        sample_count = completed
        output: list[OpponentInference] = []
        for seat, opp in tracker.opponents.items():
            hypotheses: list[PatternHypothesis] = []
            for pattern, mass in pattern_mass[seat].most_common():
                probability = mass / sample_count
                if probability <= 0:
                    continue
                score = pattern_score[seat][pattern] / sample_count
                cards = tuple(
                    key
                    for key, _ in pattern_cards[seat][pattern].most_common(6)
                )
                hypotheses.append(
                    PatternHypothesis(
                        pattern=pattern,
                        probability_next_draw=probability,
                        expected_score_next_draw=score,
                        dangerous_cards=cards,
                    )
                )

            dangerous = tuple(
                (key, mass / sample_count)
                for key, mass in dangerous_mass[seat].most_common(top_cards)
            )
            output.append(
                OpponentInference(
                    seat=seat,
                    any_hu_probability_next_draw=any_hu_mass[seat] / sample_count,
                    expected_hu_score_next_draw=hu_score_mass[seat] / sample_count,
                    patterns=tuple(hypotheses[:top_patterns]),
                    dangerous_cards=dangerous,
                    known_cards=sum(opp.known_hand.values()),
                    unknown_cards=opp.unknown_hand_count,
                )
            )
        return output

    def release_risk(
        self,
        card: Card,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
        deadline: float | None = None,
    ) -> list[ReleaseSeatRisk]:
        """如果把这张牌放到桌面，估算每一家能直接利用它成牌的概率与牌型。"""
        sample_count = max(40, samples or self.samples)
        wins = Counter()
        scores = Counter()
        patterns: dict[str, Counter[str]] = {
            seat: Counter() for seat in tracker.opponents
        }

        completed = 0
        for _ in range(sample_count):
            if completed and deadline is not None and time.monotonic() >= deadline:
                break
            completed += 1
            hands = tracker.sample_opponent_hands(self.rng)
            for seat, hand in hands.items():
                opp = tracker.opponents[seat]
                if opp.finished or len(hand) != 7:
                    continue
                hu = evaluate_hu(hand + [card], gold_ranks)
                if hu is None:
                    continue
                wins[seat] += 1
                scores[seat] += capped_payment(hu, self.loss_cap, self.loss_limit)
                patterns[seat][hu.pattern] += 1

        sample_count = completed
        result: list[ReleaseSeatRisk] = []
        for seat in tracker.opponents:
            probability = wins[seat] / sample_count
            expected_score = scores[seat] / sample_count
            dominant = tuple(
                (pattern, count / sample_count)
                for pattern, count in patterns[seat].most_common(4)
            )
            result.append(
                ReleaseSeatRisk(
                    seat=seat,
                    hu_probability=probability,
                    expected_hu_score=expected_score,
                    dominant_patterns=dominant,
                )
            )
        return result
