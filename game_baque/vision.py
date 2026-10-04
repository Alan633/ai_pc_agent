from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from game_baque.models import Card, GameSnapshot, OpponentEvent, Point, RANKS, ScreenType, SUITS


VISION_PROMPT = """你只负责读取腾讯欢乐斗地主“百变八雀·点字成金”游戏截图，不负责制定策略。

请严格从截图中提取当前玩家自己的游戏状态。

识别规则：
1. 左上角的 3 张黄色小牌是本局“金牌点数”，只返回点数。
2. 当前玩家自己的手牌位于屏幕底部。必须从左到右逐张识别，牌可能互相遮挡。
3. 癞子单独返回 rank=WILD、suit=NONE、wild=true。
4. 普通牌点数只可能是 6、7、8、9、10、J、Q、K、A；花色 S/H/C/D 分别表示黑桃/红桃/梅花/方片。
5. 每张手牌的 x/y 要放在“该牌自己可点击的暴露区域”中心，优先取牌面上方的点数/花色附近，避免落在被右侧牌遮住的区域。若手牌明显被抬高/高亮表示已选中，selected=true。
6. visible_cards 只返回当前提示“请从牌堆或亮牌中选择一张牌”时，桌面上真正正面朝上且可选择的亮牌；问号背面牌、对手手牌、历史弃牌不要放进去。
7. deck_click 是当前牌堆（蓝色背面、通常带剩余张数）的可点击中心。
8. 识别按钮中心：换牌、出牌、胡牌、继续/再来一局。不可见时 visible=false。
9. 若出现“豆子不足啦”、充值、钻石购买等弹窗，screen_type 必须为 recharge。不要把任何购买按钮当作 continue_button。
10. screen_type：
   - swap：开局换2张；
   - draw：提示从牌堆或亮牌中选一张，自己手牌通常 7 张；
   - discard：自己已取得第8张，需要选择1张并点击出牌；
   - hu：已满足胡牌且界面提供胡牌相关操作；
   - waiting：等待其他玩家或动画；
   - settlement：正常结算/本局结束；
   - recharge：豆子不足、充值、购买弹窗；
   - unknown：无法可靠判断。
11. 坐标均使用“这张输入图片自身的像素坐标”，左上角为 (0,0)。
12. 看不清的牌不要猜。confidence 低于 0.65 时仍可返回，但如无法确定 rank/suit，应将整个 screen_type 设为 unknown，并在 message 中说明。
13. 不要输出策略建议，只做视觉转录。

对手事件识别：
14. 如果输入包含两张图片，第一张是上一帧、第二张是当前帧。只有能从两帧变化中明确确认时，才输出 opponent_events。
15. seat 只允许 left/top/right；action 只允许 draw_deck、take_visible、discard、finished。
16. draw_deck 表示该玩家从背面牌堆暗摸；take_visible 表示明确拿走桌面正面亮牌；discard 表示明确打出一张可见牌；finished 表示该玩家已经胡牌/结束本局。
17. discard/take_visible 必须可靠识别具体牌；看不清则不要猜，也不要输出该事件。draw_deck/finished 不需要 card。
18. 同一个动作不要重复输出。仅描述“上一帧到当前帧之间新发生”的动作。
19. 如果是首次截图、跨局、结算切换、新一局开始或无法确认动作，opponent_events 返回空数组。
"""


def _point_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "visible": {"type": "boolean"},
        },
        "required": ["x", "y", "visible"],
    }


CARD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rank": {"type": "string", "enum": [*RANKS, "WILD"]},
        "suit": {"type": "string", "enum": [*SUITS, "NONE"]},
        "wild": {"type": "boolean"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "confidence": {"type": "number"},
        "selected": {"type": "boolean"},
    },
    "required": ["rank", "suit", "wild", "x", "y", "confidence", "selected"],
}

OPPONENT_EVENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "seat": {"type": "string", "enum": ["left", "top", "right"]},
        "action": {"type": "string", "enum": ["draw_deck", "take_visible", "discard", "finished"]},
        "card_rank": {"type": "string", "enum": [*RANKS, "WILD", "NONE"]},
        "card_suit": {"type": "string", "enum": [*SUITS, "NONE"]},
        "card_wild": {"type": "boolean"},
        "confidence": {"type": "number"},
        "detail": {"type": "string"},
    },
    "required": ["seat", "action", "card_rank", "card_suit", "card_wild", "confidence", "detail"],
}


SNAPSHOT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "screen_type": {
            "type": "string",
            "enum": ["swap", "draw", "discard", "hu", "waiting", "settlement", "recharge", "unknown"],
        },
        "gold_ranks": {
            "type": "array",
            "items": {"type": "string", "enum": list(RANKS)},
            "maxItems": 3,
        },
        "hand": {"type": "array", "items": CARD_SCHEMA, "maxItems": 8},
        "visible_cards": {"type": "array", "items": CARD_SCHEMA, "maxItems": 8},
        "opponent_events": {"type": "array", "items": OPPONENT_EVENT_SCHEMA, "maxItems": 6},
        "deck_click": _point_schema(),
        "swap_button": _point_schema(),
        "discard_button": _point_schema(),
        "hu_button": _point_schema(),
        "continue_button": _point_schema(),
        "close_popup": _point_schema(),
        "deck_count": {"type": "integer"},
        "message": {"type": "string"},
    },
    "required": [
        "screen_type",
        "gold_ranks",
        "hand",
        "visible_cards",
        "opponent_events",
        "deck_click",
        "swap_button",
        "discard_button",
        "hu_button",
        "continue_button",
        "close_popup",
        "deck_count",
        "message",
    ],
}


class BaqueVision:
    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        client_key = api_key or ("local-model" if base_url else "")
        if not client_key:
            raise ValueError("点字成金视觉识别需要 OPENAI_API_KEY 或本地 GAME_VISION_BASE_URL")
        options: dict[str, Any] = {"api_key": client_key}
        if base_url:
            options["base_url"] = base_url.rstrip("/")
        self.client = AsyncOpenAI(**options)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._prefer_chat_completions = bool(self.base_url)

    @staticmethod
    def _data_url(image_path: Path) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{image_b64}"

    async def parse(
        self,
        image_path: Path,
        previous_image_path: Path | None = None,
    ) -> GameSnapshot:
        current_url = self._data_url(image_path)
        previous_url = (
            self._data_url(previous_image_path)
            if previous_image_path is not None and previous_image_path.exists()
            else None
        )

        if self._prefer_chat_completions:
            raw = await self._parse_chat_completions(current_url, previous_url)
        else:
            try:
                raw = await self._parse_responses(current_url, previous_url)
            except Exception:
                logging.getLogger(__name__).exception("Responses API 视觉解析失败，尝试 Chat Completions")
                raw = await self._parse_chat_completions(current_url, previous_url)

        payload = json.loads(raw)
        return self._to_snapshot(payload)

    async def _parse_responses(self, current_url: str, previous_url: str | None) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "若有两张图：第一张是上一帧，第二张是当前帧；识别当前状态以及两帧之间新发生的对手动作。"
                    if previous_url
                    else "这是首次截图，只识别当前状态，opponent_events 必须为空数组。"
                ),
            }
        ]
        if previous_url:
            content.append({"type": "input_image", "image_url": previous_url, "detail": "high"})
        content.append({"type": "input_image", "image_url": current_url, "detail": "high"})
        response = await self.client.responses.create(
            model=self.model,
            instructions=VISION_PROMPT,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "baque_snapshot",
                    "description": "百变八雀点字成金截图状态与对手事件",
                    "schema": SNAPSHOT_SCHEMA,
                    "strict": True,
                }
            },
        )
        return response.output_text

    async def _parse_chat_completions(self, current_url: str, previous_url: str | None) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    VISION_PROMPT
                    + "\n严格只输出 JSON 对象，不要 Markdown。JSON Schema：\n"
                    + json.dumps(SNAPSHOT_SCHEMA, ensure_ascii=False)
                    + (
                        "\n第一张图片是上一帧，第二张是当前帧。只输出两帧之间新发生的 opponent_events。"
                        if previous_url
                        else "\n这是首次截图，opponent_events 必须为空数组。"
                    )
                ),
            }
        ]
        if previous_url:
            content.append({"type": "image_url", "image_url": {"url": previous_url}})
        content.append({"type": "image_url", "image_url": {"url": current_url}})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        try:
            response = await self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "本地视觉接口不支持 response_format，回退到纯 JSON 提示词",
                exc_info=True,
            )
            response = await self.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            text = text.removesuffix("```").strip()
        # Qwen thinking/兼容接口偶尔会在 JSON 前后附加说明；只截取最外层对象。
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _to_point(value: dict[str, Any]) -> Point | None:
        if not value.get("visible", False):
            return None
        return Point(int(value["x"]), int(value["y"]))

    @staticmethod
    def _to_card(value: dict[str, Any]) -> Card:
        wild = bool(value["wild"]) or value["rank"] == "WILD"
        return Card(
            rank=None if wild else str(value["rank"]),
            suit=None if wild else str(value["suit"]),
            wild=wild,
            x=int(value["x"]),
            y=int(value["y"]),
            confidence=float(value["confidence"]),
            selected=bool(value.get("selected", False)),
        )

    @staticmethod
    def _to_opponent_event(value: dict[str, Any]) -> OpponentEvent:
        rank = str(value.get("card_rank", "NONE"))
        suit = str(value.get("card_suit", "NONE"))
        wild = bool(value.get("card_wild", False)) or rank == "WILD"
        card: Card | None = None
        if wild:
            card = Card(wild=True, confidence=float(value.get("confidence", 0.0)))
        elif rank in RANKS and suit in SUITS:
            card = Card(
                rank=rank,
                suit=suit,
                confidence=float(value.get("confidence", 0.0)),
            )
        return OpponentEvent(
            seat=str(value["seat"]),
            action=str(value["action"]),
            card=card,
            confidence=float(value.get("confidence", 0.0)),
            detail=str(value.get("detail", "")),
        )

    @classmethod
    def _to_snapshot(cls, payload: dict[str, Any]) -> GameSnapshot:
        screen_type = ScreenType(str(payload["screen_type"]))
        hand = [cls._to_card(item) for item in payload.get("hand", [])]
        visible = [cls._to_card(item) for item in payload.get("visible_cards", [])]
        events = [
            cls._to_opponent_event(item)
            for item in payload.get("opponent_events", [])
        ]
        return GameSnapshot(
            screen_type=screen_type,
            gold_ranks=tuple(str(rank) for rank in payload.get("gold_ranks", [])),
            hand=hand,
            visible_cards=visible,
            opponent_events=events,
            deck_click=cls._to_point(payload["deck_click"]),
            swap_button=cls._to_point(payload["swap_button"]),
            discard_button=cls._to_point(payload["discard_button"]),
            hu_button=cls._to_point(payload["hu_button"]),
            continue_button=cls._to_point(payload["continue_button"]),
            close_popup=cls._to_point(payload["close_popup"]),
            deck_count=None if int(payload.get("deck_count", -1)) < 0 else int(payload["deck_count"]),
            message=str(payload.get("message", "")),
        )
