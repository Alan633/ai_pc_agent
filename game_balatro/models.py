from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


RANK_ORDER = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("S", "H", "C", "D")


class ScreenType(str, Enum):
    START = "start"
    BLIND_SELECT = "blind_select"
    PLAY = "play"
    SHOP = "shop"
    PACK = "pack"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int
    visible: bool = True


@dataclass(frozen=True, slots=True)
class PlayingCard:
    rank: str
    suit: str
    x: int
    y: int
    selected: bool = False
    enhancement: str = ""
    edition: str = ""
    seal: str = ""
    debuffed: bool = False
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Joker:
    name: str
    x: int
    y: int
    edition: str = ""
    debuffed: bool = False
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ShopItem:
    name: str
    kind: str
    price: int
    x: int
    y: int
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class BalatroSnapshot:
    screen_type: ScreenType
    ante: int = -1
    round_no: int = -1
    blind_name: str = ""
    target_score: int = -1
    current_score: int = 0
    money: int = -1
    hands_left: int = -1
    discards_left: int = -1
    hand: tuple[PlayingCard, ...] = ()
    jokers: tuple[Joker, ...] = ()
    shop_items: tuple[ShopItem, ...] = ()
    pack_items: tuple[ShopItem, ...] = ()
    primary_button: Point | None = None
    play_button: Point | None = None
    discard_button: Point | None = None
    skip_button: Point | None = None
    reroll_button: Point | None = None
    next_round_button: Point | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class HandValue:
    category: str
    category_rank: int
    tiebreak: tuple[int, ...]
    base_chips: int
    base_mult: int
    card_chips: int
    strength: float


@dataclass(frozen=True, slots=True)
class BalatroDecision:
    action: str
    reason: str
    card_indices: tuple[int, ...] = ()
    target: Point | None = None
    expected_strength: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
