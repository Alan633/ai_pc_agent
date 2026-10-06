from __future__ import annotations

from collections import Counter
from itertools import combinations

from game_baque.models import Card, GameDecision, RANKS, SUITS
from game_baque.rules import EIGHT_SPECIAL, SPECIAL_6_2, BASE_POINTS, evaluate_hu

RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}


def hand_potential(
    cards: list[Card],
    gold_ranks: tuple[str, ...] | list[str],
    *,
    prefer_flush: bool = False,
    chase_big: bool = False,
    prefer_pinghu: bool = False,
) -> float:
    """非胡状态启发式评分。分数只用于比较同一决策点，不代表游戏结算分。

    prefer_flush：同花顺优先（同花顺子加分 ×2，同点组合 ×0.6）。每胡一次的收入被
    封顶/自己豆数卡住时，大牌型多收不到钱，听得宽的同花顺（无癞子更好）才划算。

    chase_big：追大牌（高级/顶级场自己豆够多时）。结算按金牌张数线性加倍，评分乘以
    (1 + 0.5×金牌张数)；八张特殊点数（A/K/10/8）与 6+2 特殊组合（6-7、9-10、J-Q、K-A）
    的进展另加分，引导往全金牌、特殊牌型上靠。
    """
    # prefer_pinghu：平胡（3+3+2）优先。对手豆很少、一个平胡就能收满时，胡得快最重要；
    # 3+3+2 最容易凑（三张同点或三张同花顺 + 对子），但不能带癞子，所以癞子不再加分。
    same_weight = 1.5 if prefer_pinghu else 0.6 if prefer_flush else 1.0
    run_weight = 2.0 if prefer_flush and not prefer_pinghu else 1.0
    wild_weight = 0.0 if prefer_pinghu else 1.0
    if len(cards) == 8:
        hu = evaluate_hu(cards, gold_ranks)
        if hu is not None:
            return 1_000_000.0 + hu.theoretical_score

    gold = set(gold_ranks)
    nonwild = [card for card in cards if not card.wild]
    wild_count = sum(card.wild for card in cards)
    rank_counts = Counter(card.rank for card in nonwild)
    score = 0.0

    score += wild_count * 110.0 * wild_weight

    for rank, count in rank_counts.items():
        score += count * count * 12.0 * same_weight
        if count >= 2:
            score += (count - 1) * 18.0 * same_weight
        if rank in gold:
            score += count * 20.0

    for card in nonwild:
        score += BASE_POINTS[card.rank or "6"] * 0.2

    for suit in SUITS:
        held = {RANK_INDEX[card.rank] for card in nonwild if card.suit == suit and card.rank is not None}
        for length in range(2, 7):
            for start in range(0, len(RANKS) - length + 1):
                covered = len(held.intersection(range(start, start + length)))
                if covered < 2:
                    continue
                missing = length - covered
                usable_wild = 0 if prefer_pinghu else min(wild_count, missing)  # 平胡不能用癞子
                effective = covered + usable_wild
                score += effective * effective * (1.5 + length * 0.25) * run_weight
                if effective == length:
                    score += (20.0 + length * 4.0) * run_weight

    for left, right in zip(RANKS, RANKS[1:]):
        a, b = rank_counts.get(left, 0), rank_counts.get(right, 0)
        if a and b:
            score += min(a, b) * 8.0 + max(a, b) * 2.0

    fixed_gold = sum(1 for card in nonwild if card.rank in gold)
    score += fixed_gold * 24.0
    if fixed_gold >= 2:
        score += (2 ** min(fixed_gold, 6)) * 2.5

    if chase_big:
        for rank in EIGHT_SPECIAL:
            count = rank_counts.get(rank, 0) + (wild_count if rank_counts.get(rank, 0) >= 3 else 0)
            if count >= 3:
                score += count * count * 15.0
        for pair in SPECIAL_6_2:
            first, second = tuple(pair)
            a, b = rank_counts.get(first, 0), rank_counts.get(second, 0)
            if a and b and max(a, b) >= 2:
                score += (a + b) ** 2 * 6.0
        score *= 1.0 + 0.5 * fixed_gold

    return score


def choose_swap_two(
    cards: list[Card],
    gold_ranks: tuple[str, ...] | list[str],
    *,
    prefer_flush: bool = False,
    chase_big: bool = False,
    prefer_pinghu: bool = False,
) -> GameDecision:
    if len(cards) not in {3, 4, 5, 6, 7}:
        raise ValueError(f"换牌阶段应识别到 3 至 7 张可操作手牌，实际 {len(cards)} 张")
    best_pair: tuple[int, int] | None = None
    best_score = float("-inf")
    for pair in combinations(range(len(cards)), 2):
        kept = [card for index, card in enumerate(cards) if index not in pair]
        score = hand_potential(
            kept,
            gold_ranks,
            prefer_flush=prefer_flush,
            chase_big=chase_big,
            prefer_pinghu=prefer_pinghu,
        )
        swapped = [cards[index] for index in pair]
        score += sum(
            0.0
            if prefer_pinghu  # 平胡优先：癞子用不上，金牌多收不到钱，不必刻意保留
            else -120.0 if card.wild else -35.0 if card.rank in set(gold_ranks) else 0.0
            for card in swapped
        )
        if score > best_score:
            best_score = score
            best_pair = pair
    assert best_pair is not None
    labels = "、".join(cards[index].label for index in best_pair)
    return GameDecision("swap", card_indexes=best_pair, reason=f"换掉 {labels}，保留组合潜力最高")


def choose_discard(cards: list[Card], gold_ranks: tuple[str, ...] | list[str]) -> GameDecision:
    if len(cards) != 8:
        raise ValueError(f"出牌阶段应识别到 8 张手牌，实际 {len(cards)} 张")
    hu = evaluate_hu(cards, gold_ranks)
    if hu is not None:
        return GameDecision(
            "hu",
            reason=(
                f"已成 {hu.pattern} ×{hu.pattern_multiplier}，金牌 {hu.gold_count} 张 ×{hu.gold_multiplier}"
                + ("，硬气胡" if hu.hard else "")
            ),
            hu=hu,
        )

    best_index = 0
    best_score = float("-inf")
    gold = set(gold_ranks)
    for index, card in enumerate(cards):
        kept = [value for i, value in enumerate(cards) if i != index]
        score = hand_potential(kept, gold_ranks)
        if card.wild:
            score -= 500.0
        elif card.rank in gold:
            score -= 90.0
        else:
            score += (30 - BASE_POINTS[card.rank or "6"]) * 0.1
        if score > best_score:
            best_score = score
            best_index = index

    return GameDecision(
        "discard",
        card_indexes=(best_index,),
        reason=f"打出 {cards[best_index].label} 后，7 张牌的后续组合潜力最高",
    )


def _expected_random_draw_score(cards: list[Card], gold_ranks: tuple[str, ...] | list[str]) -> float:
    values: list[float] = []
    for rank in RANKS:
        for suit in SUITS:
            values.append(hand_potential(cards + [Card(rank=rank, suit=suit)], gold_ranks))
    return sum(values) / len(values)


def choose_draw(
    cards: list[Card],
    visible_cards: list[Card],
    gold_ranks: tuple[str, ...] | list[str],
    *,
    visible_margin: float = 4.0,
) -> GameDecision:
    if len(cards) != 7:
        raise ValueError(f"摸牌阶段应识别到 7 张手牌，实际 {len(cards)} 张")

    winning: list[tuple[int, object]] = []
    for index, card in enumerate(visible_cards):
        hu = evaluate_hu(cards + [card], gold_ranks)
        if hu is not None:
            winning.append((index, hu))
    if winning:
        index, hu = max(winning, key=lambda item: item[1].theoretical_score)
        return GameDecision(
            "draw_visible",
            visible_index=index,
            reason=f"取亮牌 {visible_cards[index].label} 可立即成 {hu.pattern}",
            hu=hu,
        )

    if not visible_cards:
        return GameDecision("draw_deck", reason="当前没有可靠识别到可取亮牌，选择牌堆")

    expected = _expected_random_draw_score(cards, gold_ranks)
    scored = [
        (hand_potential(cards + [card], gold_ranks), index)
        for index, card in enumerate(visible_cards)
    ]
    best_score, best_index = max(scored)
    if best_score >= expected + visible_margin:
        return GameDecision(
            "draw_visible",
            visible_index=best_index,
            reason=(
                f"亮牌 {visible_cards[best_index].label} 的组合价值 "
                f"{best_score:.1f} 高于随机摸牌期望 {expected:.1f}"
            ),
        )
    return GameDecision(
        "draw_deck",
        reason=f"最佳亮牌价值 {best_score:.1f} 未明显超过随机摸牌期望 {expected:.1f}",
    )
