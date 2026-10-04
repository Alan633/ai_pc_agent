from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from game_baque.models import Card, RANKS, SUITS

SEATS = ("left", "top", "right")
CardKey = str


def card_key(card: Card) -> CardKey:
    if card.wild:
        return "WILD"
    if card.rank is None or card.suit is None:
        raise ValueError("普通牌必须有点数和花色")
    return f"{card.rank}:{card.suit}"


def card_from_key(key: CardKey) -> Card:
    if key == "WILD":
        return Card(wild=True)
    rank, suit = key.split(":", 1)
    return Card(rank=rank, suit=suit)


def initial_deck_counts() -> Counter[CardKey]:
    """84 张点字成金牌池。

    两副 6-A 普通牌共 72 张；8 额外每花色 1 张，共新增 4 张；
    另有 4 张癞子、4 张机会牌。机会牌不进入普通胡牌计算。
    """
    counts: Counter[CardKey] = Counter()
    for rank in RANKS:
        copies_per_suit = 3 if rank == "8" else 2
        for suit in SUITS:
            counts[f"{rank}:{suit}"] = copies_per_suit
    counts["WILD"] = 4
    counts["OPPORTUNITY"] = 4
    assert sum(counts.values()) == 84
    return counts


@dataclass(slots=True)
class OpponentKnowledge:
    seat: str
    known_hand: Counter[CardKey] = field(default_factory=Counter)
    unknown_hand_count: int = 7
    discards: list[CardKey] = field(default_factory=list)
    known_takes: list[CardKey] = field(default_factory=list)
    deck_draws: int = 0
    finished: bool = False

    @property
    def hand_count(self) -> int:
        return sum(self.known_hand.values()) + self.unknown_hand_count


@dataclass(frozen=True, slots=True)
class TableEvent:
    sequence: int
    kind: str
    seat: str | None = None
    card: CardKey | None = None
    detail: str = ""


class TableTracker:
    """整局信息状态。

    只把“确实看见”的牌当作确定信息；对手从暗牌堆摸到的牌只增加 unknown_hand_count。
    因此不会因为一次视觉误判就虚构完整对手手牌。
    """

    def __init__(self) -> None:
        self.initial = initial_deck_counts()
        self.own_hand: Counter[CardKey] = Counter()
        self.public_discards: Counter[CardKey] = Counter()
        self.opponents = {seat: OpponentKnowledge(seat) for seat in SEATS}
        self.consumed_opportunity_cards = 0
        self.known_deck_top: list[CardKey] = []
        self.deck_count: int | None = 56
        self.events: list[TableEvent] = []
        self._seq = 0

    def reset_round(self, own_hand: Iterable[Card], deck_count: int | None = 56) -> None:
        self.own_hand = Counter(card_key(card) for card in own_hand)
        self.public_discards.clear()
        self.opponents = {seat: OpponentKnowledge(seat) for seat in SEATS}
        self.consumed_opportunity_cards = 0
        self.known_deck_top.clear()
        self.deck_count = deck_count
        self.events.clear()
        self._seq = 0
        self._validate_counts()
        self._event("round_start", detail=f"deck={deck_count}")

    def update_own_hand(self, cards: Iterable[Card]) -> None:
        self.own_hand = Counter(card_key(card) for card in cards)
        self._validate_counts()

    def set_deck_count(self, count: int | None) -> None:
        if count is not None and count < 0:
            raise ValueError("牌堆数量不能为负")
        self.deck_count = count

    def record_opponent_draw_deck(self, seat: str) -> None:
        opp = self._opponent(seat)
        opp.unknown_hand_count += 1
        opp.deck_draws += 1
        if self.known_deck_top:
            # 如果机会牌已经让我们知道牌堆顶部，则这次暗摸其实是确定牌。
            key = self.known_deck_top.pop(0)
            opp.unknown_hand_count -= 1
            opp.known_hand[key] += 1
            opp.known_takes.append(key)
            self._event("opponent_draw_known_top", seat, key)
        else:
            self._event("opponent_draw_deck", seat)
        if self.deck_count is not None:
            self.deck_count = max(0, self.deck_count - 1)

    def record_opponent_take_visible(self, seat: str, card: Card) -> None:
        key = card_key(card)
        opp = self._opponent(seat)
        opp.known_hand[key] += 1
        opp.known_takes.append(key)
        self._event("opponent_take_visible", seat, key)
        self._validate_counts()

    def record_opponent_discard(self, seat: str, card: Card) -> None:
        key = card_key(card)
        opp = self._opponent(seat)
        if opp.known_hand[key] > 0:
            opp.known_hand[key] -= 1
            if opp.known_hand[key] <= 0:
                del opp.known_hand[key]
        elif opp.unknown_hand_count > 0:
            opp.unknown_hand_count -= 1
        else:
            raise ValueError(f"{seat} 没有可供打出的手牌计数")
        opp.discards.append(key)
        self.public_discards[key] += 1
        self._event("opponent_discard", seat, key)
        self._validate_counts()

    def record_own_discard(self, card: Card) -> None:
        key = card_key(card)
        if self.own_hand[key] <= 0:
            raise ValueError(f"自己的当前手牌中没有 {key}")
        self.own_hand[key] -= 1
        if self.own_hand[key] <= 0:
            del self.own_hand[key]
        self.public_discards[key] += 1
        self._event("own_discard", "self", key)
        self._validate_counts()

    def record_opportunity_reveal(
        self,
        chooser: str,
        options: list[Card],
        chosen_index: int,
    ) -> None:
        if len(options) != 4:
            raise ValueError("机会牌必须翻出 4 张普通牌")
        if not 0 <= chosen_index < 4:
            raise IndexError(chosen_index)
        keys = [card_key(card) for card in options]
        chosen = keys[chosen_index]
        returned = [key for index, key in enumerate(keys) if index != chosen_index]

        self.consumed_opportunity_cards += 1
        if self.deck_count is not None:
            # 机会牌本身已从牌堆取出；四张候选中仅选中牌最终离开牌堆。
            self.deck_count = max(0, self.deck_count - 2)

        if chooser in self.opponents:
            opp = self.opponents[chooser]
            opp.known_hand[chosen] += 1
            opp.known_takes.append(chosen)
        elif chooser != "self":
            raise ValueError(f"未知座位：{chooser}")

        # 公开资料说明其余 3 张按原顺序回到牌堆顶部。
        self.known_deck_top = returned + self.known_deck_top
        self._event(
            "opportunity_pick",
            chooser,
            chosen,
            detail="returned_top=" + ",".join(returned),
        )
        self._validate_counts()

    def mark_finished(self, seat: str, finished: bool = True) -> None:
        self._opponent(seat).finished = finished
        self._event("finished" if finished else "resume", seat)

    def unseen_counts(self, *, include_opportunity: bool = False) -> Counter[CardKey]:
        counts = self.initial.copy()
        counts.subtract(self.own_hand)
        counts.subtract(self.public_discards)
        for opp in self.opponents.values():
            counts.subtract(opp.known_hand)
        counts["OPPORTUNITY"] -= self.consumed_opportunity_cards
        for key in list(counts):
            if counts[key] < 0:
                raise RuntimeError(f"记牌状态冲突：{key} 计数为 {counts[key]}")
            if counts[key] == 0 or (key == "OPPORTUNITY" and not include_opportunity):
                del counts[key]
        return counts

    def draw_distribution(self) -> dict[CardKey, float]:
        if self.known_deck_top:
            return {self.known_deck_top[0]: 1.0}
        unseen = self.unseen_counts(include_opportunity=False)
        total = sum(unseen.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in unseen.items()}

    def sample_opponent_hands(
        self,
        rng: random.Random | None = None,
    ) -> dict[str, list[Card]]:
        """从当前条件分布采样 3 家可能手牌。

        已知的亮牌/机会牌会固定在对应玩家手中；其余未知手牌从未见牌池无放回采样。
        """
        rng = rng or random.Random()
        pool = self.unseen_counts(include_opportunity=False)
        result: dict[str, list[Card]] = {}

        def draw_one() -> CardKey:
            total = sum(pool.values())
            if total <= 0:
                raise RuntimeError("未见牌池不足以采样对手手牌")
            pick = rng.randrange(total)
            cursor = 0
            for key, count in pool.items():
                cursor += count
                if pick < cursor:
                    pool[key] -= 1
                    if pool[key] == 0:
                        del pool[key]
                    return key
            raise AssertionError("加权采样失败")

        for seat, opp in self.opponents.items():
            cards: list[Card] = []
            for key, count in opp.known_hand.items():
                cards.extend(card_from_key(key) for _ in range(count))
            for _ in range(opp.unknown_hand_count):
                cards.append(card_from_key(draw_one()))
            result[seat] = cards
        return result

    def opponent_known_summary(self) -> dict[str, dict[str, object]]:
        return {
            seat: {
                "known_hand": dict(opp.known_hand),
                "unknown_hand_count": opp.unknown_hand_count,
                "discards": list(opp.discards),
                "known_takes": list(opp.known_takes),
                "deck_draws": opp.deck_draws,
                "finished": opp.finished,
            }
            for seat, opp in self.opponents.items()
        }

    def _opponent(self, seat: str) -> OpponentKnowledge:
        try:
            return self.opponents[seat]
        except KeyError as exc:
            raise ValueError(f"未知对手座位：{seat}") from exc

    def _event(self, kind: str, seat: str | None = None, card: str | None = None, detail: str = "") -> None:
        self._seq += 1
        self.events.append(TableEvent(self._seq, kind, seat, card, detail))

    def _validate_counts(self) -> None:
        # 通过 unseen_counts 统一检查是否出现重复记牌/视觉冲突。
        self.unseen_counts(include_opportunity=True)
