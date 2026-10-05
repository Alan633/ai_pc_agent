from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from PIL import Image

from game_balatro.models import (
    BalatroSnapshot,
    Joker,
    PlayingCard,
    Point,
    ScreenType,
    ShopItem,
)


LOCAL_VISION_SIZE = (1280, 720)
LOCAL_COORDINATE_RANGE = 1000

VISION_PROMPT = """你只负责读取 Steam 游戏《Balatro / 小丑牌》的当前截图，不负责制定策略。
只输出一个 JSON 对象，不要 Markdown、不要解释。

坐标规则：
- 所有 x/y 使用 0-1000 归一化坐标，左上=(0,0)，右下=(1000,1000)。
- 只给真正可点击对象的中心坐标，看不清不要猜。

screen_type 只能是：
start：主菜单/继续游戏/新游戏等可推进页面；
blind_select：选择 Small/Big/Boss Blind 或看到“Select/选择”推进按钮；
play：正式牌局，底部能看到自己的手牌，并且有 Play Hand / Discard；
shop：商店，有 Joker/卡包/Voucher/重掷/下一回合；
pack：已经打开 Booster Pack，需要从若干选项中选牌/跳过；
game_over：本局失败/胜利/Run 结束；
unknown：无法可靠判断。

必须返回：
{
 "screen_type":"unknown",
 "ante":-1,
 "round_no":-1,
 "blind_name":"",
 "target_score":-1,
 "current_score":0,
 "money":-1,
 "hands_left":-1,
 "discards_left":-1,
 "hand":[],
 "jokers":[],
 "shop_items":[],
 "pack_items":[],
 "primary_button":{"x":0,"y":0,"visible":false},
 "play_button":{"x":0,"y":0,"visible":false},
 "discard_button":{"x":0,"y":0,"visible":false},
 "skip_button":{"x":0,"y":0,"visible":false},
 "reroll_button":{"x":0,"y":0,"visible":false},
 "next_round_button":{"x":0,"y":0,"visible":false},
 "message":""
}

识别细则：
1. hand 只返回玩家底部当前真正持有的扑克牌，从左到右。rank=2/3/4/5/6/7/8/9/10/J/Q/K/A，suit=S/H/C/D。
2. selected=true 只在该牌明显被抬高/选中时。debuffed=true 只在牌明确被 Boss Blind 禁用/灰化时。
3. enhancement/edition/seal 能可靠识别才填名称，看不清填空字符串。
4. jokers 从左到右，name 尽量照抄卡名；看不清名字也可返回空 name，但不要虚构。
5. shop_items 只返回当前商店可购买项。kind 只能是 joker/pack/voucher/card/tarot/planet/spectral/other；price 是屏幕显示价格，看不清填 -1。
6. pack_items 是已经打开卡包后可选择的项目；price 通常填 0。
7. primary_button：
   - start 页面：Continue/New Run/Play 等最合理的推进按钮；
   - blind_select 页面：当前 Blind 的 Select/选择按钮；
   其他页面通常 visible=false。
8. play_button 只对应 Play Hand/出牌；discard_button 只对应 Discard/弃牌。
9. next_round_button 是商店里的 Next Round/下一回合。
10. skip_button 是 Skip/跳过 Blind 或卡包跳过按钮。
11. 不要把卡牌/Joker 自己的位置当成按钮字段；它们的 x/y 放在各自对象中。
12. target_score/current_score/money/hands_left/discards_left 只照抄屏幕数字。带逗号也转换为整数；看不清填 -1（current_score 默认 0）。
13. Boss Blind 特殊效果可简短写进 message。
14. 视觉置信不足时宁可 screen_type=unknown，也不要提供可能导致误点击的坐标。

hand 每项：
{"rank":"A","suit":"S","x":500,"y":800,"selected":false,"enhancement":"","edition":"","seal":"","debuffed":false,"confidence":0.99}

jokers 每项：
{"name":"Joker","x":200,"y":180,"edition":"","debuffed":false,"confidence":0.99}

shop_items/pack_items 每项：
{"name":"Joker","kind":"joker","price":4,"x":500,"y":450,"confidence":0.99}
"""


class BalatroVision:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
    ) -> None:
        client_key = api_key or ("local-model" if base_url else "")
        if not client_key:
            raise ValueError(
                "Balatro 视觉识别需要 OPENAI_API_KEY "
                "或 BALATRO_VISION_BASE_URL"
            )
        options: dict[str, Any] = {"api_key": client_key}
        if base_url:
            options["base_url"] = base_url.rstrip("/")
        self.client = AsyncOpenAI(**options)
        self.model = model
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _extract_json(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(
                r"^```(?:json)?\s*",
                "",
                stripped,
                flags=re.I,
            )
            stripped = re.sub(r"\s*```$", "", stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Balatro 视觉模型没有返回 JSON 对象")
        return stripped[start : end + 1]

    @staticmethod
    def _image_data_url(
        image_path: Path,
    ) -> tuple[str, tuple[int, int]]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = image.size
            resized = image.resize(
                LOCAL_VISION_SIZE,
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            resized.save(
                buffer,
                "JPEG",
                quality=90,
                optimize=True,
            )
        data = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        return "data:image/jpeg;base64," + data, original_size

    async def parse(
        self,
        image_path: Path,
    ) -> BalatroSnapshot:
        image_url, original_size = self._image_data_url(image_path)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            reasoning_effort="none",
            response_format={
                "type": "json_object",
            },
        )
        raw = response.choices[0].message.content or ""
        payload = json.loads(self._extract_json(raw))
        if not isinstance(payload, dict):
            raise ValueError(
                "Balatro 视觉结果必须是 JSON 对象"
            )
        return self._to_snapshot(
            payload,
            original_size,
        )

    @staticmethod
    def _screen_type(
        value: Any,
    ) -> ScreenType:
        try:
            return ScreenType(
                str(
                    value or "unknown"
                ).strip().lower()
            )
        except ValueError:
            return ScreenType.UNKNOWN

    @staticmethod
    def _integer(
        value: Any,
        default: int = -1,
    ) -> int:
        if isinstance(value, bool):
            return default
        text = (
            str(
                value
                if value is not None
                else ""
            )
            .replace(",", "")
            .replace("$", "")
            .strip()
        )
        match = re.search(
            r"-?\d+",
            text,
        )
        if not match:
            return default
        try:
            return int(match.group(0))
        except ValueError:
            return default

    @staticmethod
    def _confidence(
        value: Any,
    ) -> float:
        try:
            return min(
                max(
                    float(value),
                    0.0,
                ),
                1.0,
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _scale_xy(
        x: Any,
        y: Any,
        size: tuple[int, int],
    ) -> tuple[int, int]:
        try:
            nx = int(float(x))
            ny = int(float(y))
        except (
            TypeError,
            ValueError,
        ):
            return 0, 0
        width, height = size
        return (
            min(
                max(
                    round(
                        nx
                        * width
                        / LOCAL_COORDINATE_RANGE
                    ),
                    0,
                ),
                max(
                    width - 1,
                    0,
                ),
            ),
            min(
                max(
                    round(
                        ny
                        * height
                        / LOCAL_COORDINATE_RANGE
                    ),
                    0,
                ),
                max(
                    height - 1,
                    0,
                ),
            ),
        )

    @classmethod
    def _point(
        cls,
        value: Any,
        size: tuple[int, int],
    ) -> Point | None:
        if (
            not isinstance(
                value,
                dict,
            )
            or not bool(
                value.get(
                    "visible",
                    False,
                )
            )
        ):
            return None
        x, y = cls._scale_xy(
            value.get("x"),
            value.get("y"),
            size,
        )
        return Point(
            x,
            y,
        )

    @classmethod
    def _cards(
        cls,
        values: Any,
        size: tuple[int, int],
    ) -> tuple[PlayingCard, ...]:
        result: list[PlayingCard] = []
        if not isinstance(
            values,
            list,
        ):
            return ()
        for item in values:
            if not isinstance(
                item,
                dict,
            ):
                continue
            rank = str(
                item.get(
                    "rank",
                    "",
                )
            ).upper().strip()
            suit = str(
                item.get(
                    "suit",
                    "",
                )
            ).upper().strip()
            confidence = cls._confidence(
                item.get(
                    "confidence",
                    0.0,
                )
            )
            if rank not in {
                "2","3","4","5","6","7",
                "8","9","10","J","Q","K","A",
            }:
                continue
            if (
                suit not in {
                    "S",
                    "H",
                    "C",
                    "D",
                }
                or confidence < 0.45
            ):
                continue
            x, y = cls._scale_xy(
                item.get("x"),
                item.get("y"),
                size,
            )
            result.append(
                PlayingCard(
                    rank=rank,
                    suit=suit,
                    x=x,
                    y=y,
                    selected=bool(
                        item.get(
                            "selected",
                            False,
                        )
                    ),
                    enhancement=str(
                        item.get(
                            "enhancement",
                            "",
                        )
                        or ""
                    ),
                    edition=str(
                        item.get(
                            "edition",
                            "",
                        )
                        or ""
                    ),
                    seal=str(
                        item.get(
                            "seal",
                            "",
                        )
                        or ""
                    ),
                    debuffed=bool(
                        item.get(
                            "debuffed",
                            False,
                        )
                    ),
                    confidence=confidence,
                )
            )
        result.sort(
            key=lambda card: card.x
        )
        return tuple(result)

    @classmethod
    def _jokers(
        cls,
        values: Any,
        size: tuple[int, int],
    ) -> tuple[Joker, ...]:
        result: list[Joker] = []
        if not isinstance(
            values,
            list,
        ):
            return ()
        for item in values:
            if not isinstance(
                item,
                dict,
            ):
                continue
            confidence = cls._confidence(
                item.get(
                    "confidence",
                    0.0,
                )
            )
            if confidence < 0.45:
                continue
            x, y = cls._scale_xy(
                item.get("x"),
                item.get("y"),
                size,
            )
            result.append(
                Joker(
                    name=str(
                        item.get(
                            "name",
                            "",
                        )
                        or ""
                    ).strip(),
                    x=x,
                    y=y,
                    edition=str(
                        item.get(
                            "edition",
                            "",
                        )
                        or ""
                    ),
                    debuffed=bool(
                        item.get(
                            "debuffed",
                            False,
                        )
                    ),
                    confidence=confidence,
                )
            )
        result.sort(
            key=lambda joker: joker.x
        )
        return tuple(result)

    @classmethod
    def _items(
        cls,
        values: Any,
        size: tuple[int, int],
    ) -> tuple[ShopItem, ...]:
        allowed = {
            "joker",
            "pack",
            "voucher",
            "card",
            "tarot",
            "planet",
            "spectral",
            "other",
        }
        result: list[ShopItem] = []
        if not isinstance(
            values,
            list,
        ):
            return ()
        for item in values:
            if not isinstance(
                item,
                dict,
            ):
                continue
            confidence = cls._confidence(
                item.get(
                    "confidence",
                    0.0,
                )
            )
            kind = str(
                item.get(
                    "kind",
                    "other",
                )
                or "other"
            ).strip().lower()
            if (
                kind not in allowed
                or confidence < 0.45
            ):
                continue
            x, y = cls._scale_xy(
                item.get("x"),
                item.get("y"),
                size,
            )
            result.append(
                ShopItem(
                    name=str(
                        item.get(
                            "name",
                            "",
                        )
                        or ""
                    ).strip(),
                    kind=kind,
                    price=cls._integer(
                        item.get("price"),
                        -1,
                    ),
                    x=x,
                    y=y,
                    confidence=confidence,
                )
            )
        return tuple(result)

    @classmethod
    def _to_snapshot(
        cls,
        payload: dict[str, Any],
        size: tuple[int, int],
    ) -> BalatroSnapshot:
        return BalatroSnapshot(
            screen_type=cls._screen_type(
                payload.get("screen_type")
            ),
            ante=cls._integer(
                payload.get("ante")
            ),
            round_no=cls._integer(
                payload.get("round_no")
            ),
            blind_name=str(
                payload.get(
                    "blind_name",
                    "",
                )
                or ""
            ).strip(),
            target_score=cls._integer(
                payload.get("target_score")
            ),
            current_score=cls._integer(
                payload.get("current_score"),
                0,
            ),
            money=cls._integer(
                payload.get("money")
            ),
            hands_left=cls._integer(
                payload.get("hands_left")
            ),
            discards_left=cls._integer(
                payload.get("discards_left")
            ),
            hand=cls._cards(
                payload.get("hand"),
                size,
            ),
            jokers=cls._jokers(
                payload.get("jokers"),
                size,
            ),
            shop_items=cls._items(
                payload.get("shop_items"),
                size,
            ),
            pack_items=cls._items(
                payload.get("pack_items"),
                size,
            ),
            primary_button=cls._point(
                payload.get("primary_button"),
                size,
            ),
            play_button=cls._point(
                payload.get("play_button"),
                size,
            ),
            discard_button=cls._point(
                payload.get("discard_button"),
                size,
            ),
            skip_button=cls._point(
                payload.get("skip_button"),
                size,
            ),
            reroll_button=cls._point(
                payload.get("reroll_button"),
                size,
            ),
            next_round_button=cls._point(
                payload.get("next_round_button"),
                size,
            ),
            message=str(
                payload.get(
                    "message",
                    "",
                )
                or ""
            ).strip(),
        )
