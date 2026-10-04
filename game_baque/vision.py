from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from game_baque.models import Card, GameSnapshot, Point, RANKS, ScreenType, SUITS


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
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("点字成金视觉识别需要 OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def parse(self, image_path: Path) -> GameSnapshot:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:image/png;base64,{image_b64}"
        structured_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "读取这张游戏截图，严格按结构化格式返回。",
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                ],
            }
        ]
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=VISION_PROMPT,
                input=structured_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "baque_snapshot",
                        "description": "百变八雀点字成金截图状态",
                        "schema": SNAPSHOT_SCHEMA,
                        "strict": True,
                    }
                },
            )
            raw = response.output_text
        except TypeError:
            response = await self.client.responses.create(
                model=self.model,
                instructions=VISION_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "读取截图，仅输出一个 JSON 对象，不要 Markdown。字段必须符合："
                                    + json.dumps(SNAPSHOT_SCHEMA, ensure_ascii=False)
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
            )
            raw = response.output_text

        payload = json.loads(raw)
        return self._to_snapshot(payload)

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

    @classmethod
    def _to_snapshot(cls, payload: dict[str, Any]) -> GameSnapshot:
        screen_type = ScreenType(str(payload["screen_type"]))
        hand = [cls._to_card(item) for item in payload.get("hand", [])]
        visible = [cls._to_card(item) for item in payload.get("visible_cards", [])]
        return GameSnapshot(
            screen_type=screen_type,
            gold_ranks=tuple(str(rank) for rank in payload.get("gold_ranks", [])),
            hand=hand,
            visible_cards=visible,
            deck_click=cls._to_point(payload["deck_click"]),
            swap_button=cls._to_point(payload["swap_button"]),
            discard_button=cls._to_point(payload["discard_button"]),
            hu_button=cls._to_point(payload["hu_button"]),
            continue_button=cls._to_point(payload["continue_button"]),
            close_popup=cls._to_point(payload["close_popup"]),
            deck_count=None if int(payload.get("deck_count", -1)) < 0 else int(payload["deck_count"]),
            message=str(payload.get("message", "")),
        )
