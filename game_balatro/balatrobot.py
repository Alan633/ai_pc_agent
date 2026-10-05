from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from game_balatro.models import (
    BalatroDecision,
    BalatroSnapshot,
    Joker,
    PlayingCard,
    Point,
    ScreenType,
    ShopItem,
)


class BalatroBotError(RuntimeError):
    """BalatroBot JSON-RPC 调用失败。"""


_STATE_MAP = {
    "MENU": ScreenType.START,
    "BLIND_SELECT": ScreenType.BLIND_SELECT,
    "SELECTING_HAND": ScreenType.PLAY,
    "ROUND_EVAL": ScreenType.ROUND_EVAL,
    "SHOP": ScreenType.SHOP,
    "SMODS_BOOSTER_OPENED": ScreenType.PACK,
    "GAME_OVER": ScreenType.GAME_OVER,
}


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_cards(area: object) -> list[dict[str, Any]]:
    cards = _as_dict(area).get("cards", [])
    if not isinstance(cards, list):
        return []
    return [card for card in cards if isinstance(card, dict)]


def _to_int(value: object, default: int = -1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _rank(value: object) -> str:
    text = str(value or "").upper()
    return "10" if text == "T" else text


def _card_label(card: dict[str, Any]) -> str:
    return str(
        card.get("label")
        or card.get("name")
        or card.get("key")
        or ""
    )


def _card_kind(card: dict[str, Any], fallback: str = "card") -> str:
    value = str(card.get("set") or fallback).strip().lower()
    aliases = {
        "booster": "pack",
        "default": "card",
        "enhanced": "card",
    }
    return aliases.get(value, value or fallback)


def _playing_card(card: dict[str, Any], index: int) -> PlayingCard:
    value = _as_dict(card.get("value"))
    modifier = _as_dict(card.get("modifier"))
    state = _as_dict(card.get("state"))
    return PlayingCard(
        rank=_rank(value.get("rank")),
        suit=str(value.get("suit") or "").upper(),
        # API 模式不依赖坐标；保留稳定占位值，禁止直接拿来做鼠标 fallback。
        x=100 + index * 100,
        y=720,
        selected=bool(state.get("highlight", False)),
        enhancement=str(modifier.get("enhancement") or ""),
        edition=str(modifier.get("edition") or ""),
        seal=str(modifier.get("seal") or ""),
        debuffed=bool(state.get("debuff", False)),
        confidence=1.0,
    )


def _joker(card: dict[str, Any], index: int) -> Joker:
    modifier = _as_dict(card.get("modifier"))
    state = _as_dict(card.get("state"))
    return Joker(
        name=_card_label(card),
        x=100 + index * 120,
        y=160,
        edition=str(modifier.get("edition") or ""),
        debuffed=bool(state.get("debuff", False)),
        confidence=1.0,
    )


def _shop_item(
    card: dict[str, Any],
    *,
    group: str,
    index: int,
) -> ShopItem:
    cost = _as_dict(card.get("cost"))
    return ShopItem(
        name=_card_label(card),
        kind=_card_kind(card, "pack" if group == "pack" else "card"),
        price=max(0, _to_int(cost.get("buy"), 0)),
        x=100 + index * 140,
        y=420,
        confidence=1.0,
        api_group=group,
        api_index=index,
    )


def _current_blind(
    state: dict[str, Any],
) -> tuple[str, int]:
    blinds = _as_dict(state.get("blinds"))
    rows: list[tuple[int, str, int]] = []
    status_priority = {
        "CURRENT": 3,
        "SELECT": 2,
        "UPCOMING": 1,
    }
    for value in blinds.values():
        blind = _as_dict(value)
        status = str(blind.get("status") or "").upper()
        priority = status_priority.get(status, 0)
        if priority <= 0:
            continue
        rows.append(
            (
                priority,
                str(blind.get("name") or blind.get("type") or ""),
                _to_int(blind.get("score"), -1),
            )
        )
    if not rows:
        return "", -1
    rows.sort(key=lambda row: row[0], reverse=True)
    _, name, score = rows[0]
    return name, score


def snapshot_from_gamestate(
    state: dict[str, Any],
) -> BalatroSnapshot:
    """把 BalatroBot GameState 转换成 AIAGENT 的统一 BalatroSnapshot。"""
    raw_state = str(state.get("state") or "").upper()
    screen_type = _STATE_MAP.get(raw_state, ScreenType.UNKNOWN)
    round_data = _as_dict(state.get("round"))
    blind_name, target_score = _current_blind(state)

    hand = tuple(
        _playing_card(card, index)
        for index, card in enumerate(_as_cards(state.get("hand")))
    )
    jokers = tuple(
        _joker(card, index)
        for index, card in enumerate(_as_cards(state.get("jokers")))
    )

    shop_items: list[ShopItem] = []
    for group, key in (
        ("card", "shop"),
        ("voucher", "vouchers"),
        ("pack", "packs"),
    ):
        shop_items.extend(
            _shop_item(card, group=group, index=index)
            for index, card in enumerate(_as_cards(state.get(key)))
        )

    pack_items = tuple(
        _shop_item(card, group="pack_pick", index=index)
        for index, card in enumerate(_as_cards(state.get("pack")))
    )

    # Solver 仍通过按钮是否存在判断动作是否合法。API 状态使用不可见占位点，
    # 执行层绝不会直接点击这些坐标。
    hidden = Point(0, 0, visible=False)
    primary_button = (
        hidden
        if screen_type in {ScreenType.START, ScreenType.BLIND_SELECT}
        else None
    )
    play_button = hidden if screen_type == ScreenType.PLAY else None
    discard_button = hidden if screen_type == ScreenType.PLAY else None
    skip_button = (
        hidden
        if screen_type in {ScreenType.BLIND_SELECT, ScreenType.PACK}
        else None
    )
    reroll_button = hidden if screen_type == ScreenType.SHOP else None
    next_round_button = hidden if screen_type == ScreenType.SHOP else None

    return BalatroSnapshot(
        screen_type=screen_type,
        ante=_to_int(state.get("ante_num"), -1),
        round_no=_to_int(state.get("round_num"), -1),
        blind_name=blind_name,
        target_score=target_score,
        current_score=max(0, _to_int(round_data.get("chips"), 0)),
        money=_to_int(state.get("money"), -1),
        hands_left=_to_int(round_data.get("hands_left"), -1),
        discards_left=_to_int(round_data.get("discards_left"), -1),
        hand=hand,
        jokers=jokers,
        shop_items=tuple(shop_items),
        pack_items=pack_items,
        primary_button=primary_button,
        play_button=play_button,
        discard_button=discard_button,
        skip_button=skip_button,
        reroll_button=reroll_button,
        next_round_button=next_round_button,
        message=f"BalatroBot API state={raw_state or 'UNKNOWN'}",
        metadata={
            "source": "balatrobot",
            "api_state": raw_state,
            "deck": str(state.get("deck") or ""),
            "stake": str(state.get("stake") or ""),
            "won": bool(state.get("won", False)),
            "reroll_cost": _to_int(round_data.get("reroll_cost"), -1),
        },
    )


class BalatroBotClient:
    """BalatroBot JSON-RPC 2.0 客户端。

    只负责确定性的状态读取和动作执行。策略仍由 BalatroSolver + 本地 Qwen
    负责，避免让语言模型直接生成 RPC 参数或鼠标坐标。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:12346",
        timeout_seconds: float = 20.0,
        deck: str = "RED",
        stake: str = "WHITE",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 60.0))
        self.deck = deck.strip().upper() or "RED"
        self.stake = stake.strip().upper() or "WHITE"
        self._request_id = 0

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._request_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": self._request_id,
            }
        ).encode("utf-8")
        request = Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else max(0.2, float(timeout_seconds))
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise BalatroBotError(
                f"BalatroBot 连接失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BalatroBotError("BalatroBot 返回了非法 JSON") from exc
        if not isinstance(message, dict):
            raise BalatroBotError("BalatroBot 返回值不是 JSON 对象")
        if "error" in message:
            error = _as_dict(message.get("error"))
            raise BalatroBotError(
                "BalatroBot RPC 错误 "
                f"{error.get('code', '')}: {error.get('message', error)}"
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise BalatroBotError("BalatroBot RPC 缺少 result 对象")
        return result

    def get_snapshot(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> BalatroSnapshot:
        return snapshot_from_gamestate(
            self.call(
                "gamestate",
                timeout_seconds=timeout_seconds,
            )
        )

    def command_for_decision(
        self,
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
    ) -> tuple[str, dict[str, Any]] | None:
        if decision.action == "play_hand":
            return "play", {"cards": list(decision.card_indices)}
        if decision.action == "discard":
            return "discard", {"cards": list(decision.card_indices)}
        if decision.action == "cash_out":
            return "cash_out", {}

        kind = str(decision.metadata.get("candidate_type") or "")
        if kind == "shop_leave":
            return "next_round", {}
        if kind == "shop_reroll":
            return "reroll", {}
        if kind == "shop_buy":
            group = str(decision.metadata.get("api_group") or "")
            index = _to_int(decision.metadata.get("api_index"), -1)
            if group in {"card", "voucher", "pack"} and index >= 0:
                return "buy", {group: index}
            return None
        if kind == "pack_skip":
            return "pack", {"skip": True}
        if kind == "pack_pick":
            index = _to_int(decision.metadata.get("api_index"), -1)
            if index >= 0:
                return "pack", {"card": index}
            return None
        if kind == "blind_play":
            return "select", {}
        if kind == "blind_skip":
            return "skip", {}
        if kind == "start_run":
            return "start", {"deck": self.deck, "stake": self.stake}

        # 兼容旧 Solver 基线：能由页面语义唯一确定的动作直接映射。
        if decision.action == "click":
            if snapshot.screen_type == ScreenType.BLIND_SELECT:
                return "select", {}
            if snapshot.screen_type == ScreenType.START:
                return "start", {"deck": self.deck, "stake": self.stake}
        return None

    def execute(
        self,
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
    ) -> dict[str, Any]:
        command = self.command_for_decision(snapshot, decision)
        if command is None:
            raise BalatroBotError(
                f"当前决策无法安全映射到 BalatroBot：{decision.action}"
            )
        method, params = command
        return self.call(method, params)
