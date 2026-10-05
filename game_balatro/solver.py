from __future__ import annotations

from collections import Counter
from itertools import combinations

from game_balatro.models import (
    BalatroDecision,
    BalatroSnapshot,
    HandValue,
    PlayingCard,
    Point,
    ScreenType,
)


_RANK_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
_CARD_CHIPS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 10, "Q": 10, "K": 10, "A": 11,
}
_HAND_BASE = {
    "High Card": (1, 5, 1),
    "Pair": (2, 10, 2),
    "Two Pair": (3, 20, 2),
    "Three of a Kind": (4, 30, 3),
    "Straight": (5, 30, 4),
    "Flush": (6, 35, 4),
    "Full House": (7, 40, 4),
    "Four of a Kind": (8, 60, 7),
    "Straight Flush": (9, 100, 8),
    "Five of a Kind": (10, 120, 12),
    "Flush House": (11, 140, 14),
    "Flush Five": (12, 160, 16),
}


def _straight_high(values: list[int]) -> int | None:
    unique = sorted(set(values), reverse=True)
    if 14 in unique:
        unique.append(1)
    pool = set(unique)
    for high in range(14, 4, -1):
        if {high - offset for offset in range(5)}.issubset(pool):
            return high
    return None


def evaluate_hand(cards: tuple[PlayingCard, ...]) -> HandValue:
    if not cards:
        raise ValueError("至少需要一张牌")
    ranks = [_RANK_VALUE.get(card.rank, 0) for card in cards]
    suits = [card.suit for card in cards]
    counts = Counter(ranks)
    flush = len(cards) == 5 and len(set(suits)) == 1
    straight_high = _straight_high(ranks) if len(cards) == 5 else None

    category = "High Card"
    tiebreak: tuple[int, ...] = tuple(sorted(ranks, reverse=True))
    count_values = sorted(counts.values(), reverse=True)

    if len(cards) == 5 and count_values == [5] and flush:
        category = "Flush Five"
        tiebreak = (max(ranks),)
    elif len(cards) == 5 and count_values == [3, 2] and flush:
        category = "Flush House"
        triple = max(rank for rank, count in counts.items() if count == 3)
        pair = max(rank for rank, count in counts.items() if count == 2)
        tiebreak = (triple, pair)
    elif len(cards) == 5 and count_values == [5]:
        category = "Five of a Kind"
        tiebreak = (max(ranks),)
    elif len(cards) == 5 and flush and straight_high is not None:
        category = "Straight Flush"
        tiebreak = (straight_high,)
    elif 4 in count_values:
        category = "Four of a Kind"
        four = max(rank for rank, count in counts.items() if count == 4)
        kickers = sorted((rank for rank, count in counts.items() if count != 4), reverse=True)
        tiebreak = (four, *kickers)
    elif len(cards) == 5 and count_values == [3, 2]:
        category = "Full House"
        triple = max(rank for rank, count in counts.items() if count == 3)
        pair = max(rank for rank, count in counts.items() if count == 2)
        tiebreak = (triple, pair)
    elif len(cards) == 5 and flush:
        category = "Flush"
    elif len(cards) == 5 and straight_high is not None:
        category = "Straight"
        tiebreak = (straight_high,)
    elif 3 in count_values:
        category = "Three of a Kind"
        triple = max(rank for rank, count in counts.items() if count == 3)
        kickers = sorted((rank for rank, count in counts.items() if count != 3), reverse=True)
        tiebreak = (triple, *kickers)
    elif count_values.count(2) >= 2:
        category = "Two Pair"
        pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)[:2]
        kickers = sorted((rank for rank, count in counts.items() if count == 1), reverse=True)
        tiebreak = (*pairs, *kickers)
    elif 2 in count_values:
        category = "Pair"
        pair = max(rank for rank, count in counts.items() if count == 2)
        kickers = sorted((rank for rank, count in counts.items() if count == 1), reverse=True)
        tiebreak = (pair, *kickers)

    category_rank, base_chips, base_mult = _HAND_BASE[category]
    if category == "High Card":
        scoring_ranks = {max(ranks)}
        score_limit = 1
    elif category == "Pair":
        scoring_ranks = {max(rank for rank, count in counts.items() if count == 2)}
        score_limit = 2
    elif category == "Two Pair":
        scoring_ranks = {rank for rank, count in counts.items() if count == 2}
        score_limit = 4
    elif category == "Three of a Kind":
        scoring_ranks = {max(rank for rank, count in counts.items() if count == 3)}
        score_limit = 3
    elif category == "Four of a Kind":
        scoring_ranks = {max(rank for rank, count in counts.items() if count == 4)}
        score_limit = 4
    else:
        scoring_ranks = set(ranks)
        score_limit = len(cards)

    scoring_cards = [
        card for card in cards
        if _RANK_VALUE.get(card.rank, 0) in scoring_ranks
    ][:score_limit]
    card_chips = sum(
        _CARD_CHIPS.get(card.rank, 0)
        for card in scoring_cards
        if not card.debuffed
    )
    # 这里只用于动作排序，不冒充游戏最终分数；Joker/牌型等级会在后续评分层叠加。
    strength = float((base_chips + card_chips) * base_mult)
    strength += sum(value / (100 ** (idx + 1)) for idx, value in enumerate(tiebreak))
    return HandValue(
        category=category,
        category_rank=category_rank,
        tiebreak=tiebreak,
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=card_chips,
        strength=strength,
    )


class BalatroSolver:
    """Balatro 第一版确定性求解器。

    牌桌动作完全由 Python 枚举；本地模型只负责把画面转录为结构化状态。
    """

    def __init__(self, *, reserve_money: int = 5) -> None:
        self.reserve_money = max(0, int(reserve_money))

    def best_play(
        self,
        hand: tuple[PlayingCard, ...],
    ) -> tuple[tuple[int, ...], HandValue]:
        if not hand:
            raise ValueError("没有可用于求解的手牌")
        best_indices: tuple[int, ...] | None = None
        best_value: HandValue | None = None
        limit = min(5, len(hand))
        for size in range(1, limit + 1):
            for indices in combinations(range(len(hand)), size):
                cards = tuple(hand[index] for index in indices)
                value = evaluate_hand(cards)
                key = (value.category_rank, value.strength, value.tiebreak)
                if best_value is None:
                    best_indices, best_value = indices, value
                    continue
                best_key = (
                    best_value.category_rank,
                    best_value.strength,
                    best_value.tiebreak,
                )
                if key > best_key:
                    best_indices, best_value = indices, value
        assert best_indices is not None and best_value is not None
        return best_indices, best_value

    def _discard_indices(
        self,
        hand: tuple[PlayingCard, ...],
        keep: tuple[int, ...],
    ) -> tuple[int, ...]:
        outside = [index for index in range(len(hand)) if index not in keep]
        if not outside:
            return ()
        outside.sort(key=lambda i: (_RANK_VALUE.get(hand[i].rank, 0), i))
        return tuple(outside[:5])

    def decide_play(self, snapshot: BalatroSnapshot) -> BalatroDecision:
        indices, value = self.best_play(snapshot.hand)
        target_remaining = max(0, snapshot.target_score - snapshot.current_score)
        hands_left = max(snapshot.hands_left, 1)
        need_per_hand = (
            target_remaining / hands_left
            if snapshot.target_score > 0
            else 0
        )

        discards = self._discard_indices(snapshot.hand, indices)
        weak = value.category_rank <= 1
        if (
            weak
            and snapshot.discards_left > 0
            and discards
            and snapshot.discard_button
        ):
            return BalatroDecision(
                action="discard",
                reason=(
                    f"当前最佳仅 {value.category}，仍有 {snapshot.discards_left} 次弃牌；"
                    f"保留核心牌并换掉 {len(discards)} 张低价值牌。"
                ),
                card_indices=discards,
                target=snapshot.discard_button,
                expected_strength=value.strength,
                metadata={
                    "need_per_hand": round(need_per_hand, 2),
                    "category": value.category,
                },
            )

        return BalatroDecision(
            action="play_hand",
            reason=(
                f"枚举全部 1~5 张合法组合后选择 {value.category}；"
                f"启发式强度={value.strength:.2f}。"
            ),
            card_indices=indices,
            target=snapshot.play_button,
            expected_strength=value.strength,
            metadata={
                "need_per_hand": round(need_per_hand, 2),
                "category": value.category,
            },
        )

    def decide_shop(self, snapshot: BalatroSnapshot) -> BalatroDecision:
        spendable = max(0, snapshot.money - self.reserve_money)
        affordable = [
            item
            for item in snapshot.shop_items
            if 0 <= item.price <= spendable
        ]

        if len(snapshot.jokers) < 5:
            jokers = [item for item in affordable if item.kind == "joker"]
            if jokers:
                item = max(jokers, key=lambda x: (x.price, x.confidence))
                return BalatroDecision(
                    "click",
                    f"Joker 槽未满，保留 ${self.reserve_money} 后购买 {item.name or 'Joker'}。",
                    target=Point(item.x, item.y),
                    metadata={
                        "candidate_type": "shop_buy",
                        "shop_item": item.name,
                        "kind": item.kind,
                        "price": item.price,
                        "api_group": item.api_group,
                        "api_index": item.api_index,
                    },
                )

        vouchers = [item for item in affordable if item.kind == "voucher"]
        if vouchers:
            item = max(vouchers, key=lambda x: (x.price, x.confidence))
            return BalatroDecision(
                "click",
                f"购买可负担 Voucher：{item.name or 'Voucher'}。",
                target=Point(item.x, item.y),
                metadata={
                    "candidate_type": "shop_buy",
                    "shop_item": item.name,
                    "kind": item.kind,
                    "price": item.price,
                    "api_group": item.api_group,
                    "api_index": item.api_index,
                },
            )

        packs = [
            item
            for item in affordable
            if (
                item.kind == "pack"
                and snapshot.money - item.price >= self.reserve_money + 3
            )
        ]
        if packs:
            item = max(packs, key=lambda x: (x.price, x.confidence))
            return BalatroDecision(
                "click",
                f"经济余量充足，开启 {item.name or 'Booster Pack'}。",
                target=Point(item.x, item.y),
                metadata={
                    "shop_item": item.name,
                    "kind": item.kind,
                    "price": item.price,
                },
            )

        if snapshot.next_round_button:
            return BalatroDecision(
                "click",
                "商店没有满足保守预算的购买项，进入下一回合。",
                target=snapshot.next_round_button,
                metadata={"candidate_type": "shop_leave"},
            )
        return BalatroDecision(
            "wait",
            "商店状态缺少下一回合按钮，等待重新识别。",
        )

    @staticmethod
    def _decision_key(decision: BalatroDecision) -> tuple[object, ...]:
        return (
            decision.action,
            decision.card_indices,
            (
                (decision.target.x, decision.target.y)
                if decision.target is not None
                else None
            ),
        )

    def rank_candidates(
        self,
        snapshot: BalatroSnapshot,
        *,
        top_n: int = 6,
    ) -> tuple[BalatroDecision, ...]:
        """生成给长期策略层使用的合法候选，首项始终是 Python 基线。

        本地模型只能在这里生成的候选中重排，绝不能创造新的牌索引、
        商店坐标或按钮坐标。这样即使模型判断失误，执行层仍只会触发
        Python 已验证过的动作。
        """
        limit = max(1, min(int(top_n), 10))
        base = self.decide(snapshot)
        candidates: list[BalatroDecision] = [base]
        seen = {self._decision_key(base)}

        def add(decision: BalatroDecision) -> None:
            key = self._decision_key(decision)
            if key in seen or len(candidates) >= limit:
                return
            seen.add(key)
            candidates.append(decision)

        if snapshot.screen_type == ScreenType.PLAY and snapshot.hand:
            ranked_plays: list[
                tuple[tuple[int, float, tuple[int, ...]], tuple[int, ...], HandValue]
            ] = []
            for size in range(1, min(5, len(snapshot.hand)) + 1):
                for indices in combinations(range(len(snapshot.hand)), size):
                    value = evaluate_hand(tuple(snapshot.hand[index] for index in indices))
                    ranked_plays.append(
                        (
                            (value.category_rank, value.strength, value.tiebreak),
                            indices,
                            value,
                        )
                    )
            ranked_plays.sort(key=lambda row: row[0], reverse=True)
            if snapshot.play_button is not None:
                # 优先给策略模型不同牌型，而不是四个几乎一样的组合。
                diverse: list[
                    tuple[tuple[int, float, tuple[int, ...]], tuple[int, ...], HandValue]
                ] = []
                categories: set[str] = set()
                for row in ranked_plays:
                    value = row[2]
                    if value.category in categories:
                        continue
                    categories.add(value.category)
                    diverse.append(row)
                    if len(diverse) >= 4:
                        break
                for row in ranked_plays:
                    if len(diverse) >= 4:
                        break
                    if row not in diverse:
                        diverse.append(row)

                for _key, indices, value in diverse:
                    add(
                        BalatroDecision(
                            action="play_hand",
                            reason=(
                                f"Python候选：{value.category}，"
                                f"启发式强度={value.strength:.2f}。"
                            ),
                            card_indices=indices,
                            target=snapshot.play_button,
                            expected_strength=value.strength,
                            metadata={
                                "category": value.category,
                                "candidate_type": "play",
                            },
                        )
                    )

            if snapshot.discards_left > 0 and snapshot.discard_button is not None:
                best_keep, _best_value = self.best_play(snapshot.hand)
                discard_variants: list[tuple[int, ...]] = []

                outside = tuple(
                    index
                    for index in range(len(snapshot.hand))
                    if index not in best_keep
                )
                if outside:
                    discard_variants.append(outside[:5])

                rank_counts = Counter(card.rank for card in snapshot.hand)
                keep_pairs = {
                    index
                    for index, card in enumerate(snapshot.hand)
                    if rank_counts[card.rank] >= 2
                }
                pair_discard = tuple(
                    sorted(
                        (
                            index
                            for index in range(len(snapshot.hand))
                            if index not in keep_pairs
                        ),
                        key=lambda i: _RANK_VALUE.get(snapshot.hand[i].rank, 0),
                    )[:5]
                )
                if pair_discard:
                    discard_variants.append(pair_discard)

                low_cards = tuple(
                    sorted(
                        range(len(snapshot.hand)),
                        key=lambda i: _RANK_VALUE.get(snapshot.hand[i].rank, 0),
                    )[: min(5, max(1, len(snapshot.hand) // 2))]
                )
                if low_cards:
                    discard_variants.append(low_cards)

                for indices in discard_variants:
                    add(
                        BalatroDecision(
                            action="discard",
                            reason=(
                                "Python候选：保留当前结构，"
                                f"弃掉 {len(indices)} 张牌寻找后续牌型。"
                            ),
                            card_indices=indices,
                            target=snapshot.discard_button,
                            expected_strength=0.0,
                            metadata={
                                "candidate_type": "discard",
                            },
                        )
                    )

        elif snapshot.screen_type == ScreenType.SHOP:
            spendable = max(0, snapshot.money - self.reserve_money)
            priority = {
                "joker": 6,
                "voucher": 5,
                "pack": 4,
                "planet": 3,
                "tarot": 2,
                "spectral": 2,
                "card": 1,
                "other": 0,
            }
            affordable = [
                item
                for item in snapshot.shop_items
                if 0 <= item.price <= spendable
            ]
            affordable.sort(
                key=lambda item: (
                    priority.get(item.kind, 0),
                    item.confidence,
                    item.price,
                ),
                reverse=True,
            )
            # “保留经济”和“重掷”属于长期策略关键选项，必须在商品很多时
            # 仍进入 Top-N，不能被前面的购买项挤掉。
            if snapshot.next_round_button is not None:
                add(
                    BalatroDecision(
                        action="click",
                        reason="Python候选：不购买，保留经济进入下一回合。",
                        target=snapshot.next_round_button,
                        metadata={"candidate_type": "shop_leave"},
                    )
                )
            if (
                snapshot.reroll_button is not None
                and snapshot.money >= self.reserve_money + 5
            ):
                add(
                    BalatroDecision(
                        action="click",
                        reason="Python候选：经济允许，重掷商店寻找更强协同。",
                        target=snapshot.reroll_button,
                        metadata={"candidate_type": "shop_reroll"},
                    )
                )
            for item in affordable:
                add(
                    BalatroDecision(
                        action="click",
                        reason=(
                            f"Python候选：购买 {item.name or item.kind} "
                            f"(${item.price})。"
                        ),
                        target=Point(item.x, item.y),
                        metadata={
                            "candidate_type": "shop_buy",
                            "shop_item": item.name,
                            "kind": item.kind,
                            "price": item.price,
                            "api_group": item.api_group,
                            "api_index": item.api_index,
                        },
                    )
                )

        elif snapshot.screen_type == ScreenType.PACK:
            if snapshot.skip_button is not None:
                add(
                    BalatroDecision(
                        action="click",
                        reason="Python候选：跳过当前卡包。",
                        target=snapshot.skip_button,
                        metadata={"candidate_type": "pack_skip"},
                    )
                )
            for item in sorted(
                snapshot.pack_items,
                key=lambda value: value.confidence,
                reverse=True,
            ):
                add(
                    BalatroDecision(
                        action="click",
                        reason=f"Python候选：选择卡包中的 {item.name or item.kind}。",
                        target=Point(item.x, item.y),
                        metadata={
                            "candidate_type": "pack_pick",
                            "shop_item": item.name,
                            "kind": item.kind,
                            "api_index": item.api_index,
                        },
                    )
                )

        elif snapshot.screen_type == ScreenType.BLIND_SELECT:
            if snapshot.primary_button is not None:
                add(
                    BalatroDecision(
                        action="click",
                        reason="Python候选：挑战当前 Blind。",
                        target=snapshot.primary_button,
                        metadata={"candidate_type": "blind_play"},
                    )
                )
            if snapshot.skip_button is not None:
                add(
                    BalatroDecision(
                        action="click",
                        reason="Python候选：跳过当前 Blind 获取 Tag。",
                        target=snapshot.skip_button,
                        metadata={"candidate_type": "blind_skip"},
                    )
                )

        return tuple(candidates[:limit])

    def decide(self, snapshot: BalatroSnapshot) -> BalatroDecision:
        if snapshot.screen_type == ScreenType.PLAY:
            if snapshot.hand and snapshot.play_button:
                return self.decide_play(snapshot)
            return BalatroDecision(
                "wait",
                "牌桌状态尚未识别完整，等待下一帧。",
            )

        if snapshot.screen_type == ScreenType.SHOP:
            return self.decide_shop(snapshot)

        if snapshot.screen_type == ScreenType.PACK:
            if snapshot.pack_items:
                item = max(
                    snapshot.pack_items,
                    key=lambda x: (x.confidence, x.price),
                )
                return BalatroDecision(
                    "click",
                    f"从卡包选择当前最高优先级的 {item.name or item.kind}。",
                    target=Point(item.x, item.y),
                    metadata={
                        "candidate_type": "pack_pick",
                        "shop_item": item.name,
                        "kind": item.kind,
                        "api_index": item.api_index,
                    },
                )
            if snapshot.skip_button:
                return BalatroDecision(
                    "click",
                    "卡包选项未可靠识别，安全跳过。",
                    target=snapshot.skip_button,
                    metadata={"candidate_type": "pack_skip"},
                )
            return BalatroDecision("wait", "等待卡包选项出现。")

        if snapshot.screen_type == ScreenType.ROUND_EVAL:
            return BalatroDecision(
                "cash_out",
                "本回合已结算，领取奖励并进入商店。",
                metadata={"candidate_type": "cash_out"},
            )

        if snapshot.screen_type in {
            ScreenType.START,
            ScreenType.BLIND_SELECT,
        }:
            if snapshot.primary_button:
                candidate_type = (
                    "start_run"
                    if snapshot.screen_type == ScreenType.START
                    else "blind_play"
                )
                return BalatroDecision(
                    "click",
                    "执行当前页面的主推进动作。",
                    target=snapshot.primary_button,
                    metadata={"candidate_type": candidate_type},
                )
            return BalatroDecision(
                "wait",
                "没有识别到可安全点击的主按钮。",
            )

        if snapshot.screen_type == ScreenType.GAME_OVER:
            return BalatroDecision("stop", "本局已结束。")

        return BalatroDecision(
            "wait",
            "当前页面未知，等待稳定后重新识别。",
        )
