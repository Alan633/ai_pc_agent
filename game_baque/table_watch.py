"""对手回合记牌：比较亮牌行与牌堆数的前后变化，推断每家“取了什么、打了什么”。

不理解动画，只做状态差分：
- 牌堆数减 1：这家摸了牌堆（暗牌）；
- 亮牌少了一张且牌堆数不变：这家拿走了那张亮牌；
- 亮牌多了一张：这家打出了那张牌，轮到下一家。
自己出牌后依次行动的是下家、对家、上家，所以每看到一张新打出的牌就换下一家。
亮牌行满 4 张时新打出的牌会把最旧的一张挤出去；挤出不是“拿牌”，靠牌堆数区分。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from game_baque.models import Card
from game_baque.tracker import CardKey, card_from_key, card_key

SEAT_ORDER = ("right", "top", "left")
ROW_SLOTS = 4


@dataclass(frozen=True, slots=True)
class OpponentTurn:
    seat: str
    draw_source: str  # "deck" / "visible" / "unknown"
    drawn_card: Card | None
    discard: Card

    def describe(self) -> str:
        if self.draw_source == "visible" and self.drawn_card is not None:
            source = f"拿亮牌 {self.drawn_card.label}"
        elif self.draw_source == "deck":
            source = "摸牌堆"
        else:
            source = "来源不明"
        return f"{self.seat} {source} → 打 {self.discard.label}"


def _keys(cards: list[Card]) -> list[CardKey]:
    """亮牌行从左到右的牌（不含空位、问号、癞子以外的无效牌）。"""
    keys: list[CardKey] = []
    for card in cards:
        try:
            keys.append(card_key(card))
        except ValueError:
            continue
    return keys


class RowDiffTracker:
    """逐次比较亮牌行，产出对手完整的一轮动作。只做推断，不直接改记牌器。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._row: list[CardKey] | None = None
        self._deck: int | None = None
        self._actor = len(SEAT_ORDER)  # 未开始：自己出牌后才知道谁先动
        self._pending_take: CardKey | None = None
        self._pending_deck_draws = 0
        self._resync = False  # 下一次读数只作基准（结算后亮牌行有非出牌的变动）
        self.cycle_complete = False  # 自己出牌后三家的出牌都已看到（上家出完）
        # 开局阶段：谁先出牌是随机的，先暂存不分座位，轮到自己时再倒推。
        self._opening = False
        self._opening_turns: list[OpponentTurn] = []

    @property
    def pending_take(self) -> Card | None:
        """已看到拿走、还没看到打出的那张亮牌（拿牌后直接胡时就停在这里）。"""
        return card_from_key(self._pending_take) if self._pending_take is not None else None

    @property
    def active(self) -> bool:
        return self._row is not None and self._actor < len(SEAT_ORDER)

    def start_opening(self) -> None:
        """新一局开始：亮牌行为空，先出牌的人随机，暂存对手动作直到轮到自己。"""
        self.reset()
        self._row = []
        self._actor = 0
        self._opening = True

    def finish_opening(self) -> list[OpponentTurn]:
        """第一次轮到自己：自己前面出了 k 手，就是 下家/对家/上家 中的最后 k 家。"""
        if not self._opening:
            return []
        self._opening = False
        turns = self._opening_turns[-len(SEAT_ORDER):]
        self._opening_turns = []
        seats = SEAT_ORDER[len(SEAT_ORDER) - len(turns):]
        self._actor = len(SEAT_ORDER)
        return [
            OpponentTurn(seat, turn.draw_source, turn.drawn_card, turn.discard)
            for seat, turn in zip(seats, turns)
        ]

    def continue_after_hu(self, winner: str) -> Card | None:
        """有人胡牌：赢家这一轮到此结束（不打牌），接着轮到赢家的下家摸牌。

        返回赢家胡牌前拿走的亮牌（若有）。结算时亮牌行会变动（如翻出一张牌），
        所以下一次读数只作新基准、不做归属。
        """
        taken = self.pending_take if winner in SEAT_ORDER else None
        order = ("self", *SEAT_ORDER)
        if winner in order:
            # 座位顺序 自己→下家→对家→上家→自己；_actor 记的是 SEAT_ORDER 下标。
            next_index = (order.index(winner) + 1) % len(order)
            self._actor = next_index - 1 if next_index > 0 else len(SEAT_ORDER)
        else:
            self._actor = len(SEAT_ORDER)
        self._pending_take = None
        self._pending_deck_draws = 0
        self._resync = True
        self.cycle_complete = False
        return taken

    def start_after_own_turn(
        self,
        row_before: list[Card],
        deck_before: int | None,
        *,
        taken: Card | None,
        discarded: Card,
        drew_deck: bool,
    ) -> None:
        """自己出完牌：由摸牌阶段读到的亮牌行推算对手开始行动前的状态。

        row_before 为从左到右的亮牌。新打出的牌进最左边；拿走亮牌会留出空位，
        新牌补空位不挤人；没有空位且已满 4 张时，最右边（最旧）那张被挤出。
        """
        row = _keys(row_before)
        if taken is not None:
            key = card_key(taken)
            if key in row:
                row.remove(key)
        elif len(row) >= ROW_SLOTS:
            row = row[: ROW_SLOTS - 1]
        self._row = [card_key(discarded), *row]
        deck = deck_before
        if deck is not None and drew_deck:
            deck -= 1
        self._deck = deck
        self._actor = 0
        self._pending_take = None
        self._pending_deck_draws = 0
        self._resync = False
        self.cycle_complete = False
        self._opening = False
        self._opening_turns = []

    def observe(self, row_cards: list[Card], deck: int | None) -> list[OpponentTurn]:
        """读到一次稳定的亮牌行；返回这次新完成的对手回合（按行动顺序）。"""
        row = _keys(row_cards)
        if deck is not None and self._deck is not None and not 0 <= self._deck - deck <= 4:
            # 牌堆只会变少，两次读取之间最多四家各摸一张；动画中常把牌堆读错（如 37→52）。
            deck = None
        if self._row is None or self._resync:
            self._row, self._deck = row, deck if deck is not None else self._deck
            self._resync = False
            return []
        if self._actor >= len(SEAT_ORDER):
            # 三家都已打完（或尚未开始）：只更新基准，等自己出牌后再归属。
            self._row, self._deck = row, deck if deck is not None else self._deck
            return []

        before, after = Counter(self._row), Counter(row)
        added_counter = after - before
        removed = list((before - after).elements())
        deck_drop = 0
        if deck is not None and self._deck is not None:
            deck_drop = max(0, self._deck - deck)
        if not added_counter and deck_drop == 0 and len(removed) >= 2:
            # 一家一次最多拿走一张：少了两张以上又没有新牌，几乎都是读漏，丢弃这次读取。
            return []
        if sum(added_counter.values()) > len(SEAT_ORDER) - self._actor:
            # 新出现的牌比还没行动的家数还多：读数对不上，只重设基准，不做归属。
            self._row, self._deck = row, deck if deck is not None else self._deck
            self._pending_take = None
            return []
        self._pending_deck_draws += deck_drop

        # 新打出的牌按出现顺序处理：越靠左越新，所以从右往左归属。
        added: list[CardKey] = []
        remaining = Counter(added_counter)
        for key in reversed(row):
            if remaining[key] > 0:
                added.append(key)
                remaining[key] -= 1

        if not added:
            if removed and deck_drop == 0 and self._pending_take is None and len(removed) == 1:
                self._pending_take = removed[0]  # 拿走亮牌，出牌还没落下
        turns: list[OpponentTurn] = []
        for key in added:
            if self._actor >= len(SEAT_ORDER):
                break
            seat = SEAT_ORDER[self._actor]
            if self._pending_take is not None:
                source, drawn = "visible", card_from_key(self._pending_take)
            elif self._pending_deck_draws > 0:
                self._pending_deck_draws -= 1
                source, drawn = "deck", None
            elif len(added) == 1 and len(removed) == 1 and deck_drop == 0:
                # 拿牌和出牌在两次读取之间一起完成。
                source, drawn = "visible", card_from_key(removed[0])
            else:
                source, drawn = "unknown", None
            turns.append(OpponentTurn(seat, source, drawn, card_from_key(key)))
            self._actor += 1
            self._pending_take = None
            if self._actor >= len(SEAT_ORDER) and not self._opening:
                self.cycle_complete = True
        self._row = row
        if deck is not None:
            self._deck = deck
        if self._opening:
            # 开局还不知道是谁出的：先存起来（座位在 finish_opening 时倒推）。
            self._opening_turns.extend(turns)
            return []
        return turns
