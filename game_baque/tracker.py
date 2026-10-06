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
        # 最近一次看到的桌面亮牌（含对手打出的牌）。本地视觉模式没有对手动作
        # 事件，这是得知对手弃牌的唯一来源；它们不可能再从牌堆摸到。
        self.table_visible: Counter[CardKey] = Counter()
        self.opponents = {seat: OpponentKnowledge(seat) for seat in SEATS}
        self.consumed_opportunity_cards = 0
        self.known_deck_top: list[CardKey] = []
        self.deck_count: int | None = 56
        self.events: list[TableEvent] = []
        self._seq = 0

    def _capped_own(self, cards: Iterable[Card]) -> Counter[CardKey]:
        """自己手牌计数不超过牌池张数；超出的必然是视觉误读（如读出 3 张 J♣）。"""
        own = Counter(card_key(card) for card in cards)
        return Counter({key: min(count, self.initial[key]) for key, count in own.items()})

    def reset_round(self, own_hand: Iterable[Card], deck_count: int | None = 56) -> None:
        self.own_hand = self._capped_own(own_hand)
        self.public_discards.clear()
        self.table_visible.clear()
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

    def record_own_draw_deck(self) -> None:
        if self.known_deck_top:
            key = self.known_deck_top.pop(0)
            self._event("own_draw_known_top", "self", key)
        else:
            self._event("own_draw_deck", "self")
        if self.deck_count is not None:
            self.deck_count = max(0, self.deck_count - 1)

    def sync_table(self, cards: Iterable[Card]) -> None:
        """以摸牌阶段看到的桌面亮牌为准，替换已知桌面牌。"""
        self.table_visible = Counter(card_key(card) for card in cards)

    def _leave_table(self, key: CardKey) -> None:
        """亮牌被拿走：它不再躺在桌面，也不再算“已出过的牌”，而是进了某家的手牌。

        漏扣 public_discards 会让这张牌同时算在“已出”和“手牌”里，计数变负，
        曾因此反复触发“记牌状态冲突，按当前手牌重建”，把三家记录全部清空。
        """
        for counter in (self.table_visible, self.public_discards):
            if counter[key] > 0:
                counter[key] -= 1
                if counter[key] <= 0:
                    del counter[key]

    def record_own_take_visible(self, card: Card) -> None:
        key = card_key(card)
        # 牌已从桌面进入自己手牌；不扣掉会在下次同步前被重复计数。
        self._leave_table(key)
        self._event("own_take_visible", "self", key)

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

    def record_opponent_draw_unknown(self, seat: str) -> None:
        """确认对手取得了一张牌，但无法可靠判断来自牌堆还是桌面亮牌。

        只增加未知手牌数量，不擅自扣牌堆数量；后续自己的稳定截图会用真实 deck_count 校正。
        """
        opp = self._opponent(seat)
        opp.unknown_hand_count += 1
        self._event("opponent_draw_unknown", seat)

    def record_opponent_turn(
        self,
        seat: str,
        *,
        draw_source: str,
        discard: Card,
        drawn_card: Card | None = None,
    ) -> None:
        """原子记录一名对手完成的一整轮“取 1 张 → 打 1 张”。

        这样即使视觉只识别到最终弃牌，也不会把对手手牌数错误地从 7 记成 6。
        """
        if draw_source == "visible" and drawn_card is not None:
            self.record_opponent_take_visible(seat, drawn_card)
        elif draw_source == "deck":
            self.record_opponent_draw_deck(seat)
        else:
            self.record_opponent_draw_unknown(seat)
            if self.known_deck_top:
                # 不知道这家是否摸走了已知牌堆顶：宁可丢掉这条信息，也不能让它过期。
                self.known_deck_top.clear()
                self._event("known_top_cleared", seat)
        self.record_opponent_discard(seat, discard)

    def record_opponent_take_visible(self, seat: str, card: Card) -> None:
        key = card_key(card)
        opp = self._opponent(seat)
        self._leave_table(key)
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
        for key, seen in self.table_visible.items():
            # 自己打出的牌已计入 public_discards 且仍躺在桌面，只扣超出部分；
            # 桌面信息只做“可能更少”的修正，绝不因视觉误差制造计数冲突。
            extra = seen - self.public_discards[key]
            if extra > 0 and counts[key] > 0:
                counts[key] -= min(extra, counts[key])
        for key in list(counts):
            if counts[key] < 0:
                raise RuntimeError(f"记牌状态冲突：{key} 计数为 {counts[key]}")
            if counts[key] == 0 or (key == "OPPORTUNITY" and not include_opportunity):
                del counts[key]
        return counts

    def draw_distribution(self, skip: int = 0) -> dict[CardKey, float]:
        """从牌堆再往下第 skip 张（0 = 下一张）的分布。

        已知牌堆顶只决定前几张：越过它们之后按未见牌池随机，且这几张已被摸走、
        不在池中。旧实现对任何深度都返回同一张已知牌，模拟“再摸三次”时会三次
        摸到同一张牌，严重扭曲摸牌/取亮牌的估值。
        """
        if skip < len(self.known_deck_top):
            return {self.known_deck_top[skip]: 1.0}
        unseen = self.unseen_counts(include_opportunity=False)
        unseen.subtract(Counter(self.known_deck_top))
        unseen = Counter({key: count for key, count in unseen.items() if count > 0})
        total = sum(unseen.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in unseen.items()}

    def sample_world(
        self,
        rng: random.Random | None = None,
    ) -> tuple[dict[str, list[Card]], Counter[CardKey]]:
        """采样三家隐藏手牌，并返回采样后仍可能位于牌堆中的剩余牌池。

        已知牌堆顶部会先从隐藏手牌抽样池中保留，避免把“已知未来牌”错误分配给对手。
        """
        rng = rng or random.Random()
        pool = self.unseen_counts(include_opportunity=False)

        reserved_top: Counter[CardKey] = Counter(self.known_deck_top)
        for key, count in reserved_top.items():
            if pool[key] < count:
                raise RuntimeError(f"已知牌堆顶部与牌池冲突：{key}")
            pool[key] -= count
            if pool[key] == 0:
                del pool[key]

        result: dict[str, list[Card]] = {}

        def draw_one() -> CardKey:
            total = sum(pool.values())
            if total <= 0:
                raise RuntimeError("未见牌池不足以采样对手手牌")
            pick = rng.randrange(total)
            cursor = 0
            for key, count in tuple(pool.items()):
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

        # 采样完对手隐藏手牌后，把已知顶部重新放回“牌堆剩余池”。
        pool.update(reserved_top)
        return result, pool

    def sample_opponent_hands(
        self,
        rng: random.Random | None = None,
    ) -> dict[str, list[Card]]:
        """兼容接口：仅返回从当前条件分布采样得到的三家可能手牌。"""
        hands, _ = self.sample_world(rng)
        return hands

    def state_fingerprint(self) -> tuple[object, ...]:
        """用于 EV/对手模型缓存；只有牌局已知信息变化时才改变。"""
        opponents = tuple(
            (
                seat,
                tuple(sorted(opp.known_hand.items())),
                opp.unknown_hand_count,
                tuple(opp.discards),
                tuple(opp.known_takes),
                opp.deck_draws,
                opp.finished,
            )
            for seat, opp in sorted(self.opponents.items())
        )
        return (
            tuple(sorted(self.own_hand.items())),
            tuple(sorted(self.public_discards.items())),
            tuple(sorted(self.table_visible.items())),
            opponents,
            self.consumed_opportunity_cards,
            tuple(self.known_deck_top),
            self.deck_count,
        )

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

    def repair_counts(self) -> list[CardKey]:
        """某张牌计数为负（多半是视觉认错牌）时，就地扣掉多记的来源，保留其余记牌。

        自己手牌来自当前截图，视为准确；先扣“已出过的牌”与桌面亮牌，再扣对手
        已知手牌（转为未知张数，保持对手手牌总数不变）。返回修正过的牌。
        """
        counts = self.initial.copy()
        counts.subtract(self.own_hand)
        counts.subtract(self.public_discards)
        for opp in self.opponents.values():
            counts.subtract(opp.known_hand)
        for key, seen in self.table_visible.items():
            extra = seen - self.public_discards[key]
            if extra > 0:
                counts[key] -= extra
        repaired: list[CardKey] = []
        for key, count in list(self.own_hand.items()):
            if count > self.initial[key]:
                # 自己手牌一般可信，但超出牌池张数的部分只能是误读。
                counts[key] += count - self.initial[key]
                self.own_hand[key] = self.initial[key]
                repaired.append(key)
        for key, value in list(counts.items()):
            if key == "OPPORTUNITY" or value >= 0:
                continue
            deficit = -value
            for counter in (self.public_discards, self.table_visible):
                take = min(deficit, counter[key])
                if take:
                    counter[key] -= take
                    deficit -= take
                    if counter[key] <= 0:
                        del counter[key]
            for opp in self.opponents.values():
                take = min(deficit, opp.known_hand[key])
                if take:
                    opp.known_hand[key] -= take
                    opp.unknown_hand_count += take
                    deficit -= take
                    if opp.known_hand[key] <= 0:
                        del opp.known_hand[key]
            repaired.append(key)
            self._event("repair", detail=key)
        self._validate_counts()
        return repaired

    def _validate_counts(self) -> None:
        # 通过 unseen_counts 统一检查是否出现重复记牌/视觉冲突。
        self.unseen_counts(include_opportunity=True)
