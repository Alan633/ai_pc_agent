from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from game_baque.models import Card
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker, card_from_key


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

    def infer_all(
        self,
        tracker: TableTracker,
        gold_ranks: tuple[str, ...] | list[str],
        *,
        samples: int | None = None,
        top_patterns: int = 5,
        top_cards: int = 8,
    ) -> list[OpponentInference]:
        sample_count = max(40, samples or self.samples)
        draw_dist = tracker.draw_distribution()
        if not draw_dist:
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

        for _ in range(sample_count):
            hands = tracker.sample_opponent_hands(self.rng)
            for seat, hand in hands.items():
                opp = tracker.opponents[seat]
                if opp.finished or len(hand) != 7:
                    continue

                sample_hu_probability = 0.0
                sample_score = 0.0
                for key, draw_probability in draw_dist.items():
                    hu = evaluate_hu(hand + [card_from_key(key)], gold_ranks)
                    if hu is None:
                        continue
                    sample_hu_probability += draw_probability
                    weighted_score = draw_probability * hu.theoretical_score
                    sample_score += weighted_score
                    pattern_mass[seat][hu.pattern] += draw_probability
                    pattern_score[seat][hu.pattern] += weighted_score
                    pattern_cards[seat][hu.pattern][key] += draw_probability
                    dangerous_mass[seat][key] += draw_probability

                any_hu_mass[seat] += sample_hu_probability
                hu_score_mass[seat] += sample_score

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
    ) -> list[ReleaseSeatRisk]:
        """如果把这张牌放到桌面，估算每一家能直接利用它成牌的概率与牌型。"""
        sample_count = max(40, samples or self.samples)
        wins = Counter()
        scores = Counter()
        patterns: dict[str, Counter[str]] = {
            seat: Counter() for seat in tracker.opponents
        }

        for _ in range(sample_count):
            hands = tracker.sample_opponent_hands(self.rng)
            for seat, hand in hands.items():
                opp = tracker.opponents[seat]
                if opp.finished or len(hand) != 7:
                    continue
                hu = evaluate_hu(hand + [card], gold_ranks)
                if hu is None:
                    continue
                wins[seat] += 1
                scores[seat] += hu.theoretical_score
                patterns[seat][hu.pattern] += 1

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
