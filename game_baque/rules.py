from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from game_baque.models import Card, HuResult, RANKS, SUITS

BASE_POINTS = {
    "6": 10,
    "7": 10,
    "8": 10,
    "9": 10,
    "10": 10,
    "J": 20,
    "Q": 20,
    "K": 20,
    "A": 30,
}
WILD_POINTS = 20
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}
SPECIAL_6_2 = {
    frozenset(("K", "A")): ("顶峰相见", 32),
    frozenset(("J", "Q")): ("心心相连", 32),
    frozenset(("9", "10")): ("十拿九稳", 32),
    frozenset(("6", "7")): ("六事兴旺", 32),
}
EIGHT_SPECIAL = {
    "A": ("独一无二", 100),
    "K": ("君临天下", 100),
    "10": ("十全十美", 100),
    "8": ("八方来财", 100),
}


@dataclass(frozen=True, slots=True)
class GroupMatch:
    kind: str
    key: str
    wildcard_assignments: tuple[tuple[int, str], ...]
    suit: str | None = None

    @property
    def rank(self) -> str | None:
        return self.key if self.kind == "same_rank" else None


@dataclass(frozen=True, slots=True)
class Candidate:
    pattern: str
    multiplier: int
    assignments: tuple[tuple[int, str], ...]


def card_base_points(card: Card) -> int:
    if card.wild:
        return WILD_POINTS
    return BASE_POINTS[card.rank or "6"]


def gold_multiplier(gold_count: int) -> int:
    if gold_count <= 1:
        return 1
    if gold_count == 2:
        return 2
    if gold_count == 3:
        return 4
    if gold_count == 4:
        return 8
    if gold_count == 5:
        return 16
    return 32


def _same_rank_matches(cards: list[Card], indexes: tuple[int, ...]) -> list[GroupMatch]:
    ranks = {cards[index].rank for index in indexes if not cards[index].wild}
    if len(ranks) > 1:
        return []
    targets = tuple(ranks) if ranks else RANKS
    wild_indexes = tuple(index for index in indexes if cards[index].wild)
    return [
        GroupMatch(
            "same_rank",
            target,
            tuple((index, target) for index in wild_indexes),
        )
        for target in targets
    ]


def _straight_matches(cards: list[Card], indexes: tuple[int, ...]) -> list[GroupMatch]:
    size = len(indexes)
    if size < 2 or size > len(RANKS):
        return []
    nonwild = [cards[index] for index in indexes if not cards[index].wild]
    ranks = [card.rank for card in nonwild]
    if len(set(ranks)) != len(ranks):
        return []
    wild_indexes = tuple(index for index in indexes if cards[index].wild)
    result: list[GroupMatch] = []
    for suit in SUITS:
        if any(card.suit != suit for card in nonwild):
            continue
        for start in range(0, len(RANKS) - size + 1):
            sequence = RANKS[start : start + size]
            if any(rank not in sequence for rank in ranks):
                continue
            missing = [rank for rank in sequence if rank not in ranks]
            if len(missing) != len(wild_indexes):
                continue
            result.append(
                GroupMatch(
                    "straight",
                    f"{suit}:{'-'.join(sequence)}",
                    tuple(zip(wild_indexes, missing, strict=True)),
                    suit=suit,
                )
            )
    return result


def _generic_group_matches(cards: list[Card], indexes: tuple[int, ...]) -> list[GroupMatch]:
    seen: set[tuple[object, ...]] = set()
    result: list[GroupMatch] = []
    for match in _same_rank_matches(cards, indexes) + _straight_matches(cards, indexes):
        key = (match.kind, match.key, match.wildcard_assignments, match.suit)
        if key not in seen:
            seen.add(key)
            result.append(match)
    return result


def _all_same_color(cards: list[Card]) -> str | None:
    colors = {card.color for card in cards if not card.wild}
    if len(colors) > 1:
        return None
    if not colors:
        return "ANY"
    return next(iter(colors))


def _match_color_compatible(match: GroupMatch, required: str) -> bool:
    if required == "ANY" or match.kind == "same_rank":
        return True
    if match.suit is None:
        return True
    color = "R" if match.suit in {"H", "D"} else "B"
    return color == required


def _is_adjacent(rank_a: str | None, rank_b: str | None) -> bool:
    if rank_a is None or rank_b is None:
        return False
    return abs(RANK_INDEX[rank_a] - RANK_INDEX[rank_b]) == 1


def _append_candidate(
    output: list[Candidate],
    pattern: str,
    multiplier: int,
    matches: tuple[GroupMatch, ...] | list[GroupMatch],
) -> None:
    assignments: list[tuple[int, str]] = []
    for match in matches:
        assignments.extend(match.wildcard_assignments)
    output.append(Candidate(pattern, multiplier, tuple(sorted(assignments))))


def _partitions(cards: list[Card], first_size: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    indexes = tuple(range(len(cards)))
    result = []
    for first in combinations(indexes, first_size):
        first_set = set(first)
        second = tuple(index for index in indexes if index not in first_set)
        if first_size == len(cards) - first_size and first and second and first[0] > second[0]:
            continue
        result.append((first, second))
    return result


def _pattern_candidates(cards: list[Card]) -> list[Candidate]:
    if len(cards) != 8:
        return []
    candidates: list[Candidate] = []
    all_indexes = tuple(range(8))

    # 八张同点：四个特殊点数 x100，其余同点归入“八方来贺” x50。
    for match in _same_rank_matches(cards, all_indexes):
        rank = match.rank
        if rank in EIGHT_SPECIAL:
            name, multiplier = EIGHT_SPECIAL[rank]
            _append_candidate(candidates, name, multiplier, [match])
        _append_candidate(candidates, "八方来贺", 50, [match])

    # 完整八张同花顺。
    for match in _straight_matches(cards, all_indexes):
        _append_candidate(candidates, "八方来贺", 50, [match])

    color = _all_same_color(cards)

    # 6 + 2
    for six, two in _partitions(cards, 6):
        six_same = _same_rank_matches(cards, six)
        two_same = _same_rank_matches(cards, two)
        for a, b in product(six_same, two_same):
            key = frozenset((a.rank, b.rank))
            if a.rank != b.rank and key in SPECIAL_6_2:
                name, multiplier = SPECIAL_6_2[key]
                _append_candidate(candidates, name, multiplier, [a, b])
            if a.rank != b.rank and _is_adjacent(a.rank, b.rank):
                _append_candidate(candidates, "比翼为邻", 16, [a, b])

        six_generic = _generic_group_matches(cards, six)
        two_generic = _generic_group_matches(cards, two)
        for a, b in product(six_generic, two_generic):
            if color is not None and _match_color_compatible(a, color) and _match_color_compatible(b, color):
                _append_candidate(candidates, "六朝金粉", 16, [a, b])
            _append_candidate(candidates, "六六大顺", 8, [a, b])

    # 5 + 3
    for five, three in _partitions(cards, 5):
        five_same = _same_rank_matches(cards, five)
        three_same = _same_rank_matches(cards, three)
        for a, b in product(five_same, three_same):
            if a.rank != b.rank and _is_adjacent(a.rank, b.rank):
                _append_candidate(candidates, "永恒相随", 8, [a, b])

        five_generic = _generic_group_matches(cards, five)
        three_generic = _generic_group_matches(cards, three)
        for a, b in product(five_generic, three_generic):
            if color is not None and _match_color_compatible(a, color) and _match_color_compatible(b, color):
                _append_candidate(candidates, "五谷丰登", 8, [a, b])
            _append_candidate(candidates, "五福临门", 4, [a, b])

    # 4 + 4
    for left, right in _partitions(cards, 4):
        left_same = _same_rank_matches(cards, left)
        right_same = _same_rank_matches(cards, right)
        for a, b in product(left_same, right_same):
            if a.rank != b.rank and _is_adjacent(a.rank, b.rank):
                _append_candidate(candidates, "二龙腾飞", 4, [a, b])

        left_generic = _generic_group_matches(cards, left)
        right_generic = _generic_group_matches(cards, right)
        for a, b in product(left_generic, right_generic):
            if color is not None and _match_color_compatible(a, color) and _match_color_compatible(b, color):
                _append_candidate(candidates, "四季发财", 4, [a, b])
            _append_candidate(candidates, "四季如春", 2, [a, b])

    # 3 + 3 + 2；对子必须为同点。
    indexes = tuple(range(8))
    for pair in combinations(indexes, 2):
        pair_matches = _same_rank_matches(cards, pair)
        if not pair_matches:
            continue
        remaining = tuple(index for index in indexes if index not in pair)
        for tri_a in combinations(remaining, 3):
            tri_a_set = set(tri_a)
            tri_b = tuple(index for index in remaining if index not in tri_a_set)
            if tri_a[0] > tri_b[0]:
                continue
            matches_a = _generic_group_matches(cards, tri_a)
            matches_b = _generic_group_matches(cards, tri_b)
            for p, a, b in product(pair_matches, matches_a, matches_b):
                _append_candidate(candidates, "平胡", 1, [p, a, b])

    return candidates


def evaluate_hu(cards: list[Card], gold_ranks: tuple[str, ...] | list[str]) -> HuResult | None:
    if len(cards) != 8:
        return None
    gold = {rank for rank in gold_ranks if rank in RANKS}
    base_points = sum(card_base_points(card) for card in cards)
    fixed_gold = sum(1 for card in cards if not card.wild and card.rank in gold)
    hard = not any(card.wild for card in cards)
    hard_multiplier = 2 if hard else 1

    best: HuResult | None = None
    seen: set[tuple[object, ...]] = set()
    for candidate in _pattern_candidates(cards):
        assignment_map: dict[int, str] = {}
        consistent = True
        for index, rank in candidate.assignments:
            previous = assignment_map.get(index)
            if previous is not None and previous != rank:
                consistent = False
                break
            assignment_map[index] = rank
        if not consistent:
            continue

        wild_gold = sum(1 for rank in assignment_map.values() if rank in gold)
        gold_count = fixed_gold + wild_gold
        gold_mult = gold_multiplier(gold_count)
        total_mult = candidate.multiplier * gold_mult * hard_multiplier
        result = HuResult(
            pattern=candidate.pattern,
            pattern_multiplier=candidate.multiplier,
            base_points=base_points,
            gold_count=gold_count,
            gold_multiplier=gold_mult,
            hard=hard,
            hard_multiplier=hard_multiplier,
            total_multiplier=total_mult,
            theoretical_score=base_points * total_mult,
            wildcard_assignments=tuple(
                f"{index}:{rank}" for index, rank in sorted(assignment_map.items())
            ),
        )
        identity = (
            result.pattern,
            result.pattern_multiplier,
            result.gold_count,
            result.wildcard_assignments,
        )
        if identity in seen:
            continue
        seen.add(identity)
        if best is None or (
            result.theoretical_score,
            result.pattern_multiplier,
            result.gold_count,
        ) > (
            best.theoretical_score,
            best.pattern_multiplier,
            best.gold_count,
        ):
            best = result
    return best
