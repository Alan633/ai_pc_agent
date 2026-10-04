from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

RANKS: tuple[str, ...] = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS: tuple[str, ...] = ("S", "H", "C", "D")
Rank = Literal["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
Suit = Literal["S", "H", "C", "D"]


@dataclass(frozen=True, slots=True)
class Card:
    rank: str | None = None
    suit: str | None = None
    wild: bool = False
    x: int | None = None
    y: int | None = None
    confidence: float = 1.0
    selected: bool = False

    def __post_init__(self) -> None:
        if self.wild:
            return
        if self.rank not in RANKS:
            raise ValueError(f"无效点数：{self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"无效花色：{self.suit}")

    @property
    def color(self) -> str | None:
        if self.wild or self.suit is None:
            return None
        return "R" if self.suit in {"H", "D"} else "B"

    @property
    def label(self) -> str:
        if self.wild:
            return "癞子"
        symbols = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}
        return f"{self.rank}{symbols.get(self.suit or '', '')}"


class ScreenType(str, Enum):
    START = "start"
    SWAP = "swap"
    DRAW = "draw"
    DISCARD = "discard"
    HU = "hu"
    WAITING = "waiting"
    SETTLEMENT = "settlement"
    RECHARGE = "recharge"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class OpponentEvent:
    seat: str
    action: str
    card: Card | None = None
    confidence: float = 1.0
    detail: str = ""


@dataclass(slots=True)
class GameSnapshot:
    screen_type: ScreenType
    gold_ranks: tuple[str, ...] = ()
    hand: list[Card] = field(default_factory=list)
    visible_cards: list[Card] = field(default_factory=list)
    opponent_events: list[OpponentEvent] = field(default_factory=list)
    deck_click: Point | None = None
    swap_button: Point | None = None
    discard_button: Point | None = None
    hu_button: Point | None = None
    start_button: Point | None = None
    continue_button: Point | None = None
    close_popup: Point | None = None
    deck_count: int | None = None
    message: str = ""

    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.screen_type.value,
            self.gold_ranks,
            tuple((c.rank, c.suit, c.wild, c.selected) for c in self.hand),
            tuple((c.rank, c.suit, c.wild) for c in self.visible_cards),
            self.deck_count,
        )


@dataclass(frozen=True, slots=True)
class HuResult:
    pattern: str
    pattern_multiplier: int
    base_points: int
    gold_count: int
    gold_multiplier: int
    hard: bool
    hard_multiplier: int
    total_multiplier: int
    theoretical_score: int
    wildcard_assignments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GameDecision:
    action: str
    card_indexes: tuple[int, ...] = ()
    visible_index: int | None = None
    reason: str = ""
    hu: HuResult | None = None
