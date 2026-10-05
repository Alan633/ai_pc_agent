from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import combinations
from operator import itemgetter

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


def card_base_points(card: Card) -> int:
    if card.wild:
        return WILD_POINTS
    return BASE_POINTS[card.rank or "6"]


def gold_multiplier(gold_count: int) -> int:
    """官方说明：“胡牌倍数额外 × 金牌数量”（线性，不是翻倍）；没有金牌不额外加倍。

    结算页实例：被自摸（平胡 ×1）、3 张金牌 → “3倍”。旧实现按 2/4/8/16 翻倍，
    会把多金牌的牌型价值严重高估。
    """
    return max(1, gold_count)


def _all_same_color(cards: list[Card]) -> str | None:
    colors = {card.color for card in cards if not card.wild}
    if len(colors) > 1:
        return None
    if not colors:
        return "ANY"
    return next(iter(colors))


def _is_adjacent(rank_a: str | None, rank_b: str | None) -> bool:
    if rank_a is None or rank_b is None:
        return False
    return abs(RANK_INDEX[rank_a] - RANK_INDEX[rank_b]) == 1


# 每个拆分附带 itemgetter，按下标取子集时走 C 实现，避免热循环里构造生成器。
Part = tuple[tuple[int, ...], itemgetter]


def _part(indexes: tuple[int, ...]) -> Part:
    return indexes, itemgetter(*indexes)


def _partitions(size: int, first_size: int) -> tuple[tuple[Part, Part], ...]:
    indexes = tuple(range(size))
    result = []
    for first in combinations(indexes, first_size):
        first_set = set(first)
        second = tuple(index for index in indexes if index not in first_set)
        if first_size == size - first_size and first and second and first[0] > second[0]:
            continue
        result.append((_part(first), _part(second)))
    return tuple(result)


PARTITIONS_6_2 = _partitions(8, 6)
PARTITIONS_5_3 = _partitions(8, 5)
PARTITIONS_4_4 = _partitions(8, 4)


def _split_3_3_2() -> tuple[tuple[Part, tuple[tuple[Part, Part], ...]], ...]:
    indexes = tuple(range(8))
    result = []
    for pair in combinations(indexes, 2):
        remaining = tuple(index for index in indexes if index not in pair)
        triples = []
        for tri_a in combinations(remaining, 3):
            tri_a_set = set(tri_a)
            tri_b = tuple(index for index in remaining if index not in tri_a_set)
            if tri_a[0] > tri_b[0]:
                continue
            triples.append((_part(tri_a), _part(tri_b)))
        result.append((_part(pair), tuple(triples)))
    return tuple(result)


SPLITS_3_3_2 = _split_3_3_2()


CanonicalCard = tuple[bool, str, str]
# (kind, key, suit, 按子集内癞子顺序给出的替代点数)
Shape = tuple[str, str, str | None, tuple[str, ...]]


def _canonical_card(card: Card) -> CanonicalCard:
    return (card.wild, "" if card.wild else card.rank or "", "" if card.wild else card.suit or "")


def _card_from_canonical(value: CanonicalCard) -> Card:
    wild, rank, suit = value
    return Card(wild=True) if wild else Card(rank=rank, suit=suit)


@lru_cache(maxsize=1_048_576)
def _same_rank_shapes(subset: tuple[CanonicalCard, ...]) -> tuple[Shape, ...]:
    ranks = {rank for wild, rank, _ in subset if not wild}
    if len(ranks) > 1:
        return ()
    wild_count = sum(1 for wild, _, _ in subset if wild)
    targets = tuple(ranks) if ranks else RANKS
    return tuple(("same_rank", target, None, (target,) * wild_count) for target in targets)


def _straight_shapes(subset: tuple[CanonicalCard, ...]) -> list[Shape]:
    size = len(subset)
    if size < 2 or size > len(RANKS):
        return []
    nonwild = [(rank, suit) for wild, rank, suit in subset if not wild]
    ranks = [rank for rank, _ in nonwild]
    if len(set(ranks)) != len(ranks):
        return []
    wild_count = size - len(nonwild)
    result: list[Shape] = []
    for suit in SUITS:
        if any(card_suit != suit for _, card_suit in nonwild):
            continue
        for start in range(0, len(RANKS) - size + 1):
            sequence = RANKS[start : start + size]
            if any(rank not in sequence for rank in ranks):
                continue
            missing = tuple(rank for rank in sequence if rank not in ranks)
            if len(missing) != wild_count:
                continue
            result.append(("straight", f"{suit}:{'-'.join(sequence)}", suit, missing))
    return result


@lru_cache(maxsize=1_048_576)
def _generic_shapes(subset: tuple[CanonicalCard, ...]) -> tuple[Shape, ...]:
    """同点或同花顺组。癞子在子集内可互换，结果只取决于子集牌面，可跨手牌复用。"""
    seen: set[Shape] = set()
    result: list[Shape] = []
    for shape in (*_same_rank_shapes(subset), *_straight_shapes(subset)):
        if shape not in seen:
            seen.add(shape)
            result.append(shape)
    return tuple(result)


def _shape_color_compatible(shape: Shape, required: str) -> bool:
    kind, _, suit, _ = shape
    if required == "ANY" or kind == "same_rank" or suit is None:
        return True
    color = "R" if suit in {"H", "D"} else "B"
    return color == required


def evaluate_hu(cards: list[Card], gold_ranks: tuple[str, ...] | list[str]) -> HuResult | None:
    """按牌面多重集合缓存胡牌判定。

    带癞子时组合数很大，而 EV 蒙特卡洛会对同一手牌反复判定；结果只与
    牌面有关、与顺序无关，命中缓存后只需把癞子下标映射回原顺序。
    """
    if len(cards) != 8:
        return None
    order = sorted(range(8), key=lambda index: _canonical_card(cards[index]))
    canonical = tuple(_canonical_card(cards[index]) for index in order)
    gold = tuple(sorted({rank for rank in gold_ranks if rank in RANKS}))
    result = _evaluate_canonical(canonical, gold)
    if result is None or not result.wildcard_assignments:
        return result
    remapped = []
    for item in result.wildcard_assignments:
        index, rank = item.split(":", 1)
        remapped.append((order[int(index)], rank))
    return replace(
        result,
        wildcard_assignments=tuple(f"{index}:{rank}" for index, rank in sorted(remapped)),
    )


@lru_cache(maxsize=131072)
def _evaluate_canonical(
    canonical: tuple[CanonicalCard, ...],
    gold_ranks: tuple[str, ...],
) -> HuResult | None:
    """穷举所有牌型拆分，边枚举边保留最优结果。

    旧实现先把每个(拆分 × 癞子取值)组合物化成候选再比较，两张癞子时一手
    牌要生成数十万个对象。这里按相同顺序枚举、只在严格更优时替换，结果
    （含并列时取首个）与旧实现一致。
    """
    cards = [_card_from_canonical(value) for value in canonical]
    gold = set(gold_ranks)
    base_points = sum(card_base_points(card) for card in cards)
    fixed_gold = sum(1 for card in cards if not card.wild and card.rank in gold)
    hard = not any(card.wild for card in cards)
    # 没有癞子时“能胡 2 次”：结算页出现两行相同的“被自摸(平胡) 230分 3倍”，
    # 分两次支付。理论分按两次合计（×2）；封顶与对手余额在 EV 层按每次支付计算。
    hard_multiplier = 2 if hard else 1
    wild_flags = tuple(value[0] for value in canonical)

    best_rank: tuple[int, int, int] | None = None
    best_parts: tuple[object, ...] = ()
    best_pattern = ""

    def consider(pattern: str, multiplier: int, *parts: tuple[Shape, tuple[int, ...]]) -> None:
        nonlocal best_rank, best_parts, best_pattern
        wild_gold = 0
        for shape, _ in parts:
            for rank in shape[3]:
                if rank in gold:
                    wild_gold += 1
        gold_count = fixed_gold + wild_gold
        total = multiplier * gold_multiplier(gold_count) * hard_multiplier
        rank_key = (base_points * total, multiplier, gold_count)
        if best_rank is None or rank_key > best_rank:
            best_rank = rank_key
            best_parts = parts
            best_pattern = pattern

    all_indexes = tuple(range(8))

    # 八张同点：四个特殊点数 x100，其余同点归入“八方来贺” x50。
    for shape in _same_rank_shapes(canonical):
        rank = shape[1]
        if rank in EIGHT_SPECIAL:
            name, multiplier = EIGHT_SPECIAL[rank]
            consider(name, multiplier, (shape, all_indexes))
        consider("八方来贺", 50, (shape, all_indexes))

    # 完整八张同花顺。
    for shape in _generic_shapes(canonical):
        if shape[0] == "straight":
            consider("八方来贺", 50, (shape, all_indexes))

    color = _all_same_color(cards)

    def same_rank_then_generic(
        partitions: tuple[tuple[Part, Part], ...],
        adjacent_name: str,
        adjacent_multiplier: int,
        color_name: str,
        color_multiplier: int,
        plain_name: str,
        plain_multiplier: int,
        *,
        special_6_2: bool = False,
    ) -> None:
        for (first, first_get), (second, second_get) in partitions:
            first_subset = first_get(canonical)
            second_subset = second_get(canonical)
            first_same = _same_rank_shapes(first_subset)
            second_same = _same_rank_shapes(second_subset) if first_same else ()
            for a in first_same:
                for b in second_same:
                    if a[1] == b[1]:
                        continue
                    if special_6_2:
                        special = SPECIAL_6_2.get(frozenset((a[1], b[1])))
                        if special is not None:
                            consider(special[0], special[1], (a, first), (b, second))
                    if _is_adjacent(a[1], b[1]):
                        consider(adjacent_name, adjacent_multiplier, (a, first), (b, second))

            first_generic = _generic_shapes(first_subset)
            if not first_generic:
                continue
            second_generic = _generic_shapes(second_subset)
            for a in first_generic:
                a_color = color is not None and _shape_color_compatible(a, color)
                for b in second_generic:
                    if a_color and _shape_color_compatible(b, color):
                        consider(color_name, color_multiplier, (a, first), (b, second))
                    consider(plain_name, plain_multiplier, (a, first), (b, second))

    same_rank_then_generic(
        PARTITIONS_6_2, "比翼为邻", 16, "六朝金粉", 16, "六六大顺", 8, special_6_2=True
    )
    same_rank_then_generic(PARTITIONS_5_3, "永恒相随", 8, "五谷丰登", 8, "五福临门", 4)
    same_rank_then_generic(PARTITIONS_4_4, "二龙腾飞", 4, "四季发财", 4, "四季如春", 2)

    # 3 + 3 + 2；对子必须为同点。
    for (pair, pair_get), triples in SPLITS_3_3_2:
        pair_shapes = _same_rank_shapes(pair_get(canonical))
        if not pair_shapes:
            continue
        for (tri_a, tri_a_get), (tri_b, tri_b_get) in triples:
            matches_a = _generic_shapes(tri_a_get(canonical))
            if not matches_a:
                continue
            matches_b = _generic_shapes(tri_b_get(canonical))
            if not matches_b:
                continue
            for p in pair_shapes:
                for a in matches_a:
                    for b in matches_b:
                        consider("平胡", 1, (p, pair), (a, tri_a), (b, tri_b))

    if best_rank is None:
        return None
    score, multiplier, gold_count = best_rank
    gold_mult = gold_multiplier(gold_count)
    assignments = sorted(
        (index, rank)
        for shape, indexes in best_parts  # type: ignore[misc]
        for index, rank in zip(
            (index for index in indexes if wild_flags[index]),
            shape[3],
            strict=True,
        )
    )
    return HuResult(
        pattern=best_pattern,
        pattern_multiplier=multiplier,
        base_points=base_points,
        gold_count=gold_count,
        gold_multiplier=gold_mult,
        hard=hard,
        hard_multiplier=hard_multiplier,
        total_multiplier=multiplier * gold_mult * hard_multiplier,
        theoretical_score=score,
        wildcard_assignments=tuple(f"{index}:{rank}" for index, rank in assignments),
    )
