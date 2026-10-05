from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from openai import AsyncOpenAI
from PIL import Image

from game_baque.capture import find_action_buttons
from game_baque.models import Card, GameSnapshot, OpponentEvent, Point, RANKS, ScreenType, SUITS
from game_baque.rules import evaluate_hu
from game_baque.vision_coordinator import LocalVisionCoordinator


LOCAL_VISION_SIZE = (1280, 720)
LOCAL_COORDINATE_RANGE = 1000


VISION_PROMPT = """你只负责读取腾讯欢乐斗地主“百变八雀·点字成金”游戏截图，不负责制定策略。

请严格从截图中提取当前玩家自己的游戏状态。

识别规则：
1. 左上角的 3 张黄色小牌是本局“金牌点数”，只返回点数。
2. 当前玩家自己的手牌位于屏幕底部。必须从左到右逐张识别，牌可能互相遮挡。
3. 癞子单独返回 rank=WILD、suit=NONE、wild=true。
4. 普通牌点数只可能是 6、7、8、9、10、J、Q、K、A；花色 S/H/C/D 分别表示黑桃/红桃/梅花/方片。
5. 每张手牌的 x/y 要放在“该牌自己可点击的暴露区域”中心，优先取牌面上方的点数/花色附近，避免落在被右侧牌遮住的区域。若手牌明显被抬高/高亮表示已选中，selected=true。
6. visible_cards 只返回当前提示“请从牌堆或亮牌中选择一张牌”时，桌面上真正正面朝上且可选择的普通亮牌；对手手牌、历史弃牌不要放进去。
7. opportunity_click 是当前玩家底部手牌区中、正面显示橙色“?”和骰子图案的“选牌”卡中心。它不是癞子、不是普通手牌，也不是中央牌堆旁较小的“?”预览。可见时必须返回；deck_click 是中央蓝色背面牌堆中心。选牌卡在出牌阶段选中并点击“出牌”后会弹出四选一。
8. 识别按钮中心：开始游戏、换牌、出牌、胡牌、继续/再来一局。不可见时 visible=false。
9. 若出现“豆子不足啦”、充值、钻石购买等弹窗，screen_type 必须为 recharge。不要把任何购买按钮当作 start_button 或 continue_button。
10. screen_type：
   - start：已经进入“百变八雀·点字成金”桌面，但牌局尚未开始，界面中央有明显橙色“开始游戏”按钮；
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


LOCAL_VISION_PROMPT = """识别腾讯欢乐斗地主“百变八雀·点字成金”截图，只输出一个 JSON 对象，不要 Markdown/解释。

规则：
- 所有牌和按钮坐标使用 Qwen 视觉的 0-1000 归一化坐标：左上=(0,0)，右下=(1000,1000)。
- screen_type: start/swap/draw/discard/hu/waiting/settlement/recharge/unknown。
- start=中央有橙色“开始游戏”；recharge=豆子不足/充值/购买，绝不能误点购买。
- gold_ranks 必须是三个点数字符串，例如 ["6","8","9"]，绝不能返回对象。
- 金牌是左上 3 张黄色小牌；自己的手牌在底部，按左到右。
- 普通牌 rank=6/7/8/9/10/J/Q/K/A，suit=S/H/C/D；癞子 rank=WILD,suit=NONE,wild=true。
- swap 阶段 hand 只返回底部 6 张普通明牌。最左侧带“?”和骰子图案的是“选牌/机会牌入口”，不是手牌、不是癞子，绝不能放进 hand；visible_cards=[]。
- draw 阶段通常有 7 张普通手牌；若底部最左侧有橙色“?”选牌卡，则 opportunity_click 填它的中心，hand 只返回其余 6 张普通牌。discard/hu 阶段 hand 必须有 8 张普通牌。
- 中央蓝色牌背上的 56 等数字是 deck_count，不是 visible_cards，也不是牌的 rank。
- 底部手牌区最左侧、正面为橙色“?”加骰子的牌是自己的“选牌”卡：填写 opportunity_click。它不是癞子、不是普通手牌；中央牌堆右侧较小的“?”只是预览，绝不能当作 opportunity_click。
- 看不清不要猜，低置信时 screen_type=unknown。
- visible_cards 只放当前真正可选的正面亮牌。
- 两张图时第一张=上一帧、第二张=当前帧；只记录新发生的对手动作。首次截图 opponent_events=[]。
- 对手 seat=left/top/right；action=draw_deck/take_visible/discard/finished；看不清具体牌时不要虚构。

必须返回这些字段，缺失按钮用 visible=false，未知 deck_count=-1：
{"screen_type":"unknown","gold_ranks":[],"hand":[],"visible_cards":[],"opponent_events":[],
"opportunity_click":{"x":0,"y":0,"visible":false},"deck_click":{"x":0,"y":0,"visible":false},"swap_button":{"x":0,"y":0,"visible":false},
"discard_button":{"x":0,"y":0,"visible":false},"hu_button":{"x":0,"y":0,"visible":false},
"start_button":{"x":0,"y":0,"visible":false},"continue_button":{"x":0,"y":0,"visible":false},
"close_popup":{"x":0,"y":0,"visible":false},"deck_count":-1,"message":""}

hand/visible_cards 每项：
{"rank":"6","suit":"S","wild":false,"x":0,"y":0,"confidence":0.99,"selected":false}

opponent_events 每项：
{"seat":"left","action":"discard","card_rank":"A","card_suit":"D","card_wild":false,"confidence":0.99,"detail":""}
"""


LOCAL_SWAP_PROMPT = """只识别腾讯欢乐斗地主“百变八雀”换牌页，只输出 JSON，不要解释。
格式：{"gold_ranks":["6","7","J"],"hand":[{"rank":"?","suit":"NONE","x":0,"y":0},{"rank":"A","suit":"H","x":0,"y":0}],"swap_button":{"x":0,"y":0}}。
坐标使用 0-1000：左上=(0,0)，右下=(1000,1000)。
gold_ranks 读取左上三张金牌。hand 按底部所有可见项从左到右返回；橙色“?”加骰子是选牌入口，必须用 rank="?"、suit="NONE" 作为位置占位，后续不会把它当手牌。
普通牌 rank 仅限 6/7/8/9/10/J/Q/K/A，suit 仅限 S/H/C/D；写着“癞子”的牌是真实手牌，用 rank="WILD"、suit="NONE"，不能漏掉。x/y 取每张牌上方可点击的点数花色处。
swap_button 返回灰色“换牌”按钮中心。"""


LOCAL_TURN_PROMPT = """只识别腾讯欢乐斗地主“百变八雀”自己回合截图，只输出 JSON，不要解释。
格式：{"gold_ranks":["6","7","J"],"hand":[{"rank":"?","suit":"NONE"},{"rank":"A","suit":"H","up":false}],"table":[{"rank":"9","suit":"D","x":490}],"deck_count":52,"hu_button":null}
gold_ranks：左上三张黄色金牌的点数。
hand：底部自己的牌从左到右逐张返回，通常共 7 或 8 张，牌互相遮挡也必须数全，每张只返回一次。橙色“?”加骰子的选牌卡用 rank="?"、suit="NONE" 占位。牌被明显抬高时 up=true。
table：中央长框内蓝色牌堆右侧正面朝上的亮牌，x 为其中心横坐标（0-1000），没有则 []。中央牌堆右侧的小“?”不是亮牌。
deck_count：蓝色牌堆背面上的数字。
hu_button：画面有“胡”按钮时返回 {"x":0,"y":0}（0-1000 坐标），否则 null。
普通牌 rank 仅限 6/7/8/9/10/J/Q/K/A，suit 仅限 S/H/C/D；癞子 rank="WILD"、suit="NONE"。"""


# 回合内的快速识别只看局部裁剪图，并用“点数+花色字母”的紧凑写法，输出 token
# 远少于完整 JSON。实测（qwen3.5-9b）：手牌裁剪 0.5~1.2 秒、亮牌裁剪约 0.4 秒，
# 完整回合提示词 5~6 秒；且手牌裁剪对花色的识别反而更准（10♠/10♣ 不再混淆）。
# 一副牌每种两张，手里可能同时有两张一模一样的牌（实测 10♦ 两张）。旧写法
# “每张只写一次”会让模型把重复牌去掉，出牌阶段永远只读到 7 张而卡住；
# 现用“分别写两次”为主提示，张数不符时再用旧提示重读一次（各约 0.7 秒）。
HAND_COMPACT_PROMPT = """只识别图中自己的手牌，只输出 JSON，不要解释。
格式：{"hand":["WILD","AH","10S"]}
从左到右逐张返回，牌互相遮挡也必须数全，两张一模一样的牌要分别写两次。每张写成 点数+花色字母：点数 6/7/8/9/10/J/Q/K/A，花色 S♠ H♥ C♣ D♦。
写着“癞子”的牌写 "WILD"；橙色“?”加骰子的选牌卡写 "?"。牌被明显抬高时在后面加 "^"，如 "AH^"。"""

HAND_COMPACT_FALLBACK_PROMPT = """只识别图中自己的手牌，只输出 JSON，不要解释。
格式：{"hand":["WILD","AH","10S"]}
从左到右逐张返回，牌互相遮挡也必须数全，每张只写一次。每张写成 点数+花色字母：点数 6/7/8/9/10/J/Q/K/A，花色 S♠ H♥ C♣ D♦。
写着“癞子”的牌写 "WILD"；橙色“?”加骰子的选牌卡写 "?"。牌被明显抬高时在后面加 "^"，如 "AH^"。"""

# 出牌/摸牌阶段手牌区应有的张数（含“?”选牌卡）。
TURN_HAND_COUNTS = {"draw": 7, "discard": 8}
# 牌角“30分/20分/10分”有时被抄进牌面，如 "A30H"、"J20S"。
_CARD_TOKEN_RE = re.compile(r"^(10|[6-9JQKA])(?:10|20|30)?([SHCD])$")

TABLE_COMPACT_PROMPT = """图中是牌桌中央：最左边是蓝色背面牌堆（上面有数字），右边依次是 4 个亮牌位，最右侧小“?”图标不是牌。只输出 JSON，不要解释。
格式：{"deck":52,"slots":["","9D","",""]}
deck：牌堆背面上的数字。slots：牌堆右边 4 个位置从左到右，空位写 ""，有正面牌写 点数+花色字母（点数 6/7/8/9/10/J/Q/K/A，花色 S♠ H♥ C♣ D♦）。牌上的“吃”只是提示。"""

# 裁剪区域（相对窗口宽高）与放大倍数。
HAND_CROP_BOX = (0.24, 0.74, 0.76, 1.0)
TABLE_CROP_BOX = (0.355, 0.32, 0.67, 0.45)
TABLE_CROP_SCALE = 2
_SUIT_SYMBOLS = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}

# 各场次胡一把每家输赢封顶（欢乐豆），三家分别计算。
ROOM_WIN_CAPS: dict[str, float] = {
    "新手": 20_000.0,
    "普通": 300_000.0,
    "高级": 3_000_000.0,
    "顶级": 8_000_000.0,
}

# 本局信息：让模型只照抄裁剪出的文字，字段由代码解析。整图让模型直接填
# 字段时会照抄提示里的示例值（把普通场读成“新手场、底分 10”）；裁剪照抄
# 实测 10/10 正确（新手/普通/高级场、两种窗口尺寸），约 1.1 秒。
TITLE_COPY_PROMPT = """照抄图中这一行文字，只输出 JSON：{"text":"..."}"""
BEANS_COPY_PROMPT = """图中从上到下有三行，每行是一个欢乐豆数量（黄色豆子图标右边的数字，可能带“万”）。从上到下照抄，只输出 JSON：{"beans":["","",""]}"""
TITLE_CROP_BOX = (0.33, 0.18, 0.67, 0.24)
# 左、上、右三家欢乐豆所在区域。
BEAN_CROP_BOXES = (
    (0.015, 0.40, 0.11, 0.48),
    (0.49, 0.09, 0.57, 0.145),
    (0.905, 0.40, 0.995, 0.48),
)

LOCAL_CHOICE_PROMPT = """只识别腾讯欢乐斗地主“百变八雀”截图中央“请选择想要的牌”弹窗，只输出 JSON，不要解释。
格式：{"options":[{"rank":"6","suit":"D","x":327}],"hand":[{"rank":"J","suit":"S"}]}
options：弹窗里可选的牌，从左到右，x 为牌中心横坐标（0-1000）。牌下方的“吃”只是提示，不是牌。
hand：屏幕底部自己的手牌，从左到右逐张返回。
普通牌 rank 仅限 6/7/8/9/10/J/Q/K/A，suit 仅限 S/H/C/D；癞子 rank="WILD"、suit="NONE"。"""


# “请选择想要的牌”弹窗：4 张候选时的固定牌位（0-1000，按 1936x1088 实图标定）。
CHOICE_SLOTS: dict[int, tuple[int, ...]] = {4: (327, 441, 556, 671)}
CHOICE_Y = 416


# 回合专用识别的固定布局（0-1000 归一化，按 1936x1088 实图标定）。扇形手牌
# 互相遮挡，模型坐标越靠右误差越大；牌位只由总张数决定，模型只负责读牌面。
TURN_HAND_Y = 850
TURN_HAND_SLOTS: dict[int, tuple[int, ...]] = {
    7: tuple(283 + index * 62 for index in range(7)),
    8: tuple(round(282 + index * 53.4) for index in range(8)),
}
TURN_TABLE_SLOTS: tuple[int, ...] = (438, 492, 546, 600)
TURN_TABLE_Y = 386
TURN_DECK_POINT = Point(384, 386)
TURN_DISCARD_BUTTON = Point(536, 677)


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
            "enum": ["start", "swap", "draw", "discard", "hu", "waiting", "settlement", "recharge", "unknown"],
        },
        "gold_ranks": {
            "type": "array",
            "items": {"type": "string", "enum": list(RANKS)},
            "maxItems": 3,
        },
        "hand": {"type": "array", "items": CARD_SCHEMA, "maxItems": 8},
        "visible_cards": {"type": "array", "items": CARD_SCHEMA, "maxItems": 8},
        "opponent_events": {"type": "array", "items": OPPONENT_EVENT_SCHEMA, "maxItems": 6},
        "opportunity_click": _point_schema(),
        "deck_click": _point_schema(),
        "swap_button": _point_schema(),
        "discard_button": _point_schema(),
        "hu_button": _point_schema(),
        "start_button": _point_schema(),
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
        "opportunity_click",
        "deck_click",
        "swap_button",
        "discard_button",
        "hu_button",
        "start_button",
        "continue_button",
        "close_popup",
        "deck_count",
        "message",
    ],
}


T = TypeVar("T")


class BaqueVision:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        *,
        coordinator: LocalVisionCoordinator | None = None,
    ) -> None:
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
        self.coordinator = coordinator

    async def _run_local_request(self, request: Callable[[], Awaitable[T]]) -> T:
        if self.coordinator is None:
            return await request()
        return await self.coordinator.run_own(request)

    @staticmethod
    def _data_url(image_path: Path) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{image_b64}"

    @staticmethod
    def _local_data_url(image_path: Path) -> tuple[str, tuple[int, int]]:
        """把本地视觉输入固定为 1280x720，并保留原图尺寸供坐标反算。"""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = image.size
            resized = image.resize(LOCAL_VISION_SIZE, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, "JPEG", quality=90, optimize=True)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}", original_size

    async def parse(
        self,
        image_path: Path,
        previous_image_path: Path | None = None,
    ) -> GameSnapshot:
        if self._prefer_chat_completions:
            current_url, original_size = self._local_data_url(image_path)
            previous_url = (
                self._local_data_url(previous_image_path)[0]
                if previous_image_path is not None and previous_image_path.exists()
                else None
            )
            raw = await self._parse_chat_completions(current_url, previous_url)
            payload = json.loads(raw)
            return self._scale_local_snapshot(self._to_snapshot(payload), original_size)
        else:
            current_url = self._data_url(image_path)
            previous_url = (
                self._data_url(previous_image_path)
                if previous_image_path is not None and previous_image_path.exists()
                else None
            )
            try:
                raw = await self._parse_responses(current_url, previous_url)
            except Exception:
                logging.getLogger(__name__).exception("Responses API 视觉解析失败，尝试 Chat Completions")
                raw = await self._parse_chat_completions(current_url, previous_url)

        payload = json.loads(raw)
        return self._to_snapshot(payload)

    async def parse_swap(self, image_path: Path) -> GameSnapshot:
        """使用最小输出专门识别短倒计时的开局换牌页。"""
        if not self._prefer_chat_completions:
            return await self.parse(image_path)

        current_url, original_size = self._local_data_url(image_path)
        payload = await self._quick_json(LOCAL_SWAP_PROMPT, current_url, "换牌")
        hand_values = payload.get("hand", [])
        if isinstance(hand_values, list):
            placeholder = next(
                (
                    item
                    for item in hand_values
                    if isinstance(item, dict) and str(item.get("rank", "")) == "?"
                ),
                None,
            )
            # 癞子是真实手牌；只按普通点数过滤会把它丢掉，6 张手牌变 5 张，
            # 换牌校验失败并错过倒计时。
            hand_values = [
                {**item, "rank": rank, "wild": rank == "WILD"}
                for item in hand_values
                if isinstance(item, dict)
                for rank in [self._normalize_rank(item.get("rank"))]
                if rank in RANKS or rank == "WILD"
            ]
            # 扇形手牌互相遮挡，Qwen 给出的 x 越靠右误差越大。游戏在换牌页
            # 使用固定牌位：问号+6张普通牌时普通牌从 350 开始；选牌已经翻成
            # 普通牌、共7张时从 285 开始。牌面仍由模型识别，点击点使用实图
            # 标定后的稳定暴露区域中心。
            if len(hand_values) == 7 and placeholder is None:
                for index, item in enumerate(hand_values):
                    item["x"] = 285 + index * 58
                    item["y"] = 850
            elif len(hand_values) == 6:
                for index, item in enumerate(hand_values):
                    item["x"] = 350 + index * 60
                    item["y"] = 850
            for item in hand_values:
                # 专用协议刻意省略这些恒定字段以减少输出 token；合法普通牌已
                # 经过 rank/suit 白名单约束，赋予可执行识别的默认置信度。
                item.setdefault("wild", False)
                item.setdefault("confidence", 0.95)
                item.setdefault("selected", False)
        snapshot = GameSnapshot(
            screen_type=ScreenType.SWAP,
            gold_ranks=self._to_gold_ranks(payload.get("gold_ranks", [])),
            hand=self._to_cards(hand_values, field_name="swap_hand"),
            swap_button=self._to_point(
                {
                    **(payload.get("swap_button") or {}),
                    "visible": True,
                }
            ),
            deck_count=56,
            message="开局换牌专用快速识别",
        )
        return self._scale_local_snapshot(snapshot, original_size)

    async def parse_turn(
        self,
        image_path: Path,
        action_state: str,
        *,
        gold_ranks: tuple[str, ...] = (),
    ) -> GameSnapshot:
        """自己回合专用快速识别；阶段由本地像素门给出，模型只读牌面。

        已知本局金牌时走局部裁剪快速路径（摸牌约 1.5 秒、出牌约 1 秒）；
        否则走整图回合提示词（约 5 秒），顺带读出金牌。
        """
        if action_state == "swap":
            return await self.parse_swap(image_path)
        if not self._prefer_chat_completions:
            return await self.parse(image_path)
        if len(gold_ranks) == 3:
            snapshot = await self._parse_turn_crops(image_path, action_state, gold_ranks)
            if not (
                snapshot.screen_type == ScreenType.DISCARD
                and len(snapshot.hand) == 8
                and evaluate_hu(snapshot.hand, gold_ranks) is not None
            ):
                return snapshot
            # 能胡时按钮布局变为 [出牌][胡]，按颜色形状直接定位，不必整图识别。
            with Image.open(image_path) as image:
                buttons = find_action_buttons(image)
            if "hu" in buttons:
                return replace(
                    snapshot,
                    screen_type=ScreenType.HU,
                    hu_button=buttons["hu"],
                    discard_button=buttons.get("discard", snapshot.discard_button),
                )
        return await self._parse_turn_full(image_path, action_state)

    async def _parse_turn_crops(
        self,
        image_path: Path,
        action_state: str,
        gold_ranks: tuple[str, ...],
    ) -> GameSnapshot:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = image.size
            hand_url = self._crop_data_url(image, HAND_CROP_BOX)
            table_url = (
                self._crop_data_url(image, TABLE_CROP_BOX, TABLE_CROP_SCALE)
                if action_state == "draw"
                else None
            )
        hand_items = self._compact_hand_items(
            (await self._quick_json(HAND_COMPACT_PROMPT, hand_url, "手牌")).get("hand", [])
        )
        expected = TURN_HAND_COUNTS.get(action_state)
        if expected is not None and len(hand_items) != expected:
            retry_items = self._compact_hand_items(
                (await self._quick_json(HAND_COMPACT_FALLBACK_PROMPT, hand_url, "手牌重读")).get("hand", [])
            )
            if len(retry_items) == expected:
                hand_items = retry_items
        payload: dict[str, Any] = {
            "gold_ranks": list(gold_ranks),
            "hand": hand_items,
            "table": [],
            "deck_count": -1,
            "hu_button": None,
        }
        if table_url is not None:
            table_payload = await self._quick_json(TABLE_COMPACT_PROMPT, table_url, "亮牌")
            payload["deck_count"] = table_payload.get("deck", -1)
            slots = table_payload.get("slots", [])
            for slot_x, token in zip(TURN_TABLE_SLOTS, slots if isinstance(slots, list) else []):
                card = self._parse_card_token(token)
                if card is not None and card["rank"] != "?":
                    payload["table"].append({**card, "x": slot_x})
        return self._scale_local_snapshot(
            self._turn_snapshot(payload, action_state),
            original_size,
        )

    @staticmethod
    def _crop_data_url(
        image: Image.Image,
        box: tuple[float, float, float, float],
        scale: int = 1,
    ) -> str:
        width, height = image.size
        crop = image.crop(
            (
                round(width * box[0]),
                round(height * box[1]),
                round(width * box[2]),
                round(height * box[3]),
            )
        )
        if scale != 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @classmethod
    def _parse_card_token(cls, token: Any) -> dict[str, Any] | None:
        """解析 "10S"、"AH^"、"J D"、"WILD"、"?" 这类紧凑牌面写法。"""
        text = "".join(str(token or "").split()).upper()
        for symbol, letter in _SUIT_SYMBOLS.items():
            text = text.replace(symbol, letter)
        selected = text.endswith("^")
        text = text.rstrip("^")
        if not text:
            return None
        if text == "?":
            return {"rank": "?", "suit": "NONE", "up": selected}
        if cls._normalize_rank(text) == "WILD" or text in {"癞子", "赖子"}:
            return {"rank": "WILD", "suit": "NONE", "up": selected}
        match = _CARD_TOKEN_RE.match(text)
        if match is None:
            return None
        return {"rank": match.group(1), "suit": match.group(2), "up": selected}

    @classmethod
    def _compact_hand_items(cls, tokens: Any) -> list[dict[str, Any]]:
        items = []
        for token in tokens if isinstance(tokens, list) else []:
            card = cls._parse_card_token(token)
            if card is not None:
                items.append(card)
        return items

    async def _parse_turn_full(self, image_path: Path, action_state: str) -> GameSnapshot:
        current_url, original_size = self._local_data_url(image_path)
        payload = await self._quick_json(LOCAL_TURN_PROMPT, current_url, "回合")
        return self._scale_local_snapshot(
            self._turn_snapshot(payload, action_state),
            original_size,
        )

    async def parse_round_info(
        self,
        image_path: Path,
    ) -> tuple[float | None, list[float], float | None]:
        """读取本局底分、三家对手欢乐豆与场次封顶，返回（底分, 对手豆数列表, 场次封顶）。

        实测座位偶尔左右/上下对调，但三个数和底分本身都正确；封顶只用合计。
        """
        if not self._prefer_chat_completions:
            raise RuntimeError("本局信息识别仅支持本地视觉模型")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            title_url = self._crop_data_url(image, TITLE_CROP_BOX)
            beans_url = self._encode_jpeg(self._stack_crops(image, BEAN_CROP_BOXES))
        title = str((await self._quick_json(TITLE_COPY_PROMPT, title_url, "场次底分")).get("text", ""))
        raw_beans = (await self._quick_json(BEANS_COPY_PROMPT, beans_url, "对手欢乐豆")).get("beans", [])
        balances = [
            value
            for value in (self._parse_beans(item) for item in (raw_beans if isinstance(raw_beans, list) else []))
            if value is not None
        ]
        room_match = re.search(r"(新手|普通|高级|顶级)场", title)
        room_cap = ROOM_WIN_CAPS.get(room_match.group(1)) if room_match else None
        base_match = re.search(r"底分\D*(\d+)", title)
        base = float(base_match.group(1)) if base_match else None
        return (base if base and base > 0 else None), balances, room_cap

    @classmethod
    def _stack_crops(
        cls,
        image: Image.Image,
        boxes: tuple[tuple[float, float, float, float], ...],
    ) -> Image.Image:
        """把多个小区域放大后自上而下拼成一张图，一次请求读完。"""
        width, height = image.size
        crops = []
        for left, top, right, bottom in boxes:
            crop = image.crop(
                (round(width * left), round(height * top), round(width * right), round(height * bottom))
            )
            crops.append(crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS))
        gap = 20
        canvas = Image.new(
            "RGB",
            (max(crop.width for crop in crops), sum(crop.height for crop in crops) + gap * (len(crops) - 1)),
            (255, 255, 255),
        )
        y = 0
        for crop in crops:
            canvas.paste(crop, (0, y))
            y += crop.height + gap
        return canvas

    @staticmethod
    def _encode_jpeg(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _parse_beans(value: Any) -> float | None:
        """解析 "2700"、"1.258万"、"10.04万"、10 这类豆数。"""
        text = "".join(str(value if value is not None else "").split()).replace(",", "")
        multiplier = 1.0
        if text.endswith("万"):
            multiplier, text = 10000.0, text[:-1]
        elif text.endswith("亿"):
            multiplier, text = 100000000.0, text[:-1]
        try:
            number = float(text) * multiplier
        except ValueError:
            return None
        return number if number >= 0 else None

    async def parse_choice(self, image_path: Path) -> tuple[list[Card], list[Card]]:
        """识别“请选择想要的牌”弹窗，返回（候选牌, 手牌），坐标已换算为截图像素。"""
        if not self._prefer_chat_completions:
            raise RuntimeError("选牌弹窗快速识别仅支持本地视觉模型")
        current_url, original_size = self._local_data_url(image_path)
        payload = await self._quick_json(LOCAL_CHOICE_PROMPT, current_url, "选牌弹窗")
        options, hand = self._choice_cards(payload)
        scaled = self._scale_local_snapshot(
            GameSnapshot(screen_type=ScreenType.DRAW, hand=hand, visible_cards=options),
            original_size,
        )
        return scaled.visible_cards, scaled.hand

    @classmethod
    def _choice_cards(cls, payload: dict[str, Any]) -> tuple[list[Card], list[Card]]:
        raw_options = payload.get("options", [])
        items = (
            [item for item in raw_options if isinstance(item, dict)]
            if isinstance(raw_options, list)
            else []
        )
        slots = CHOICE_SLOTS.get(len(items))
        option_values = []
        for index, item in enumerate(items):
            rank = cls._normalize_rank(item.get("rank"))
            try:
                model_x = int(item.get("x", 0))
            except (TypeError, ValueError):
                model_x = 0
            option_values.append(
                {
                    "rank": rank,
                    "suit": item.get("suit"),
                    "wild": rank == "WILD",
                    "x": slots[index] if slots is not None else model_x,
                    "y": CHOICE_Y,
                    "confidence": 0.95,
                }
            )
        raw_hand = payload.get("hand", [])
        hand_values = []
        for item in raw_hand if isinstance(raw_hand, list) else []:
            if not isinstance(item, dict):
                continue
            rank = cls._normalize_rank(item.get("rank"))
            if rank == "?":
                continue
            # 弹窗阶段只用手牌估值，不点击手牌，坐标无意义。
            hand_values.append(
                {"rank": rank, "suit": item.get("suit"), "wild": rank == "WILD", "confidence": 0.95}
            )
        return (
            cls._to_cards(option_values, field_name="choice_options"),
            cls._to_cards(hand_values, field_name="choice_hand"),
        )

    async def _quick_json(self, prompt: str, image_url: str, label: str) -> dict[str, Any]:
        async def send() -> Any:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )

        response = await self._run_local_request(send)
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise RuntimeError(f"本地视觉模型未返回{label}识别结果")
        payload = json.loads(self._extract_json(raw))
        if not isinstance(payload, dict):
            raise RuntimeError(f"本地视觉模型{label}识别结果不是 JSON 对象")
        return payload

    @staticmethod
    def _normalize_rank(value: Any) -> str:
        rank = str(value or "").strip()
        return "WILD" if rank.upper() == "WILD" or rank in {"癞子", "赖子"} else rank

    @classmethod
    def _turn_snapshot(cls, payload: dict[str, Any], action_state: str) -> GameSnapshot:
        """把回合专用协议转换为 0-1000 坐标下的 GameSnapshot。"""
        raw_hand = payload.get("hand", [])
        items = (
            [item for item in raw_hand if isinstance(item, dict)]
            if isinstance(raw_hand, list)
            else []
        )
        slots = TURN_HAND_SLOTS.get(len(items))
        opportunity_click: Point | None = None
        hand_values: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            # 张数不符合任何已知布局时不给坐标，控制器会按张数校验拒绝执行。
            x = slots[index] if slots is not None else 0
            rank = cls._normalize_rank(item.get("rank"))
            if rank == "?":
                if slots is not None:
                    opportunity_click = Point(x, TURN_HAND_Y)
                continue
            hand_values.append(
                {
                    "rank": rank,
                    "suit": item.get("suit"),
                    "wild": rank == "WILD",
                    "x": x,
                    "y": TURN_HAND_Y,
                    "confidence": 0.95,
                    "selected": bool(item.get("up", False)),
                }
            )
        hand = cls._to_cards(hand_values, field_name="turn_hand")

        visible: list[Card] = []
        if action_state == "draw":
            used_slots: set[int] = set()
            raw_table = payload.get("table", [])
            for item in raw_table if isinstance(raw_table, list) else []:
                if not isinstance(item, dict):
                    continue
                try:
                    raw_x = int(item.get("x", -1))
                except (TypeError, ValueError):
                    continue
                slot_x = min(TURN_TABLE_SLOTS, key=lambda value: abs(value - raw_x))
                if abs(slot_x - raw_x) > 40 or slot_x in used_slots:
                    continue
                rank = cls._normalize_rank(item.get("rank"))
                cards = cls._to_cards(
                    [
                        {
                            "rank": rank,
                            "suit": item.get("suit"),
                            "wild": rank == "WILD",
                            "x": slot_x,
                            "y": TURN_TABLE_Y,
                            "confidence": 0.95,
                        }
                    ],
                    field_name="turn_table",
                )
                if cards:
                    used_slots.add(slot_x)
                    visible.extend(cards)
            visible.sort(key=lambda card: card.x or 0)

        hu_value = payload.get("hu_button")
        hu_button = (
            cls._to_point({**hu_value, "visible": True})
            if isinstance(hu_value, dict) and action_state == "discard"
            else None
        )
        if action_state == "draw":
            screen_type = ScreenType.DRAW
        elif hu_button is not None:
            screen_type = ScreenType.HU
        else:
            screen_type = ScreenType.DISCARD

        return GameSnapshot(
            screen_type=screen_type,
            gold_ranks=cls._to_gold_ranks(payload.get("gold_ranks", [])),
            hand=hand,
            visible_cards=visible,
            opportunity_click=opportunity_click,
            deck_click=TURN_DECK_POINT if action_state == "draw" else None,
            discard_button=TURN_DISCARD_BUTTON if action_state == "discard" else None,
            hu_button=hu_button,
            deck_count=cls._to_deck_count(payload.get("deck_count", -1)),
            message="回合专用快速识别",
        )

    @staticmethod
    def _scale_local_snapshot(
        snapshot: GameSnapshot,
        original_size: tuple[int, int],
    ) -> GameSnapshot:
        target_width, target_height = original_size
        scale_x = target_width / LOCAL_COORDINATE_RANGE
        scale_y = target_height / LOCAL_COORDINATE_RANGE

        def point(value: Point | None) -> Point | None:
            if value is None:
                return None
            return Point(round(value.x * scale_x), round(value.y * scale_y))

        def card(value: Card) -> Card:
            if value.x is None or value.y is None:
                return value
            return replace(
                value,
                x=round(value.x * scale_x),
                y=round(value.y * scale_y),
            )

        return GameSnapshot(
            screen_type=snapshot.screen_type,
            gold_ranks=snapshot.gold_ranks,
            hand=[card(value) for value in snapshot.hand],
            visible_cards=[card(value) for value in snapshot.visible_cards],
            opponent_events=snapshot.opponent_events,
            opportunity_click=point(snapshot.opportunity_click),
            deck_click=point(snapshot.deck_click),
            swap_button=point(snapshot.swap_button),
            discard_button=point(snapshot.discard_button),
            hu_button=point(snapshot.hu_button),
            start_button=point(snapshot.start_button),
            continue_button=point(snapshot.continue_button),
            close_popup=point(snapshot.close_popup),
            deck_count=snapshot.deck_count,
            message=snapshot.message,
        )

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
        prompt = LOCAL_VISION_PROMPT + (
            "\n当前有两张图片：只比较上一帧→当前帧的新事件。"
            if previous_url
            else "\n当前只有一张首次截图：opponent_events 必须为空数组。"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if previous_url:
            content.append({"type": "image_url", "image_url": {"url": previous_url}})
        content.append({"type": "image_url", "image_url": {"url": current_url}})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            # Ollama/Qwen 默认会输出很长的隐藏推理；视觉状态只需要结构化转录。
            # 关闭思考可显著降低延迟，并避免推理耗尽输出预算后 content 为空。
            "reasoning_effort": "none",
        }
        async def send(*, structured: bool) -> Any:
            request_kwargs = dict(kwargs)
            if structured:
                request_kwargs["response_format"] = {"type": "json_object"}
            return await self.client.chat.completions.create(**request_kwargs)

        try:
            response = await self._run_local_request(
                lambda: send(structured=True)
            )
        except Exception as exc:
            if self._is_context_length_error(exc):
                raise RuntimeError(
                    "本地视觉模型上下文仍不足。当前已使用紧凑提示词；"
                    "请运行 scripts/setup_ollama_qwen16k.bat 创建 16K 上下文模型，并重启 Agent。"
                ) from exc
            logging.getLogger(__name__).warning(
                "本地视觉接口不支持 response_format，已回退到普通 JSON 输出：%s",
                type(exc).__name__,
            )
            response = await self._run_local_request(
                lambda: send(structured=False)
            )
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            logging.getLogger(__name__).warning(
                "本地视觉模型返回空 content，取消 response_format 后重试一次"
            )
            response = await self._run_local_request(
                lambda: send(structured=False)
            )
            raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise RuntimeError("本地视觉模型连续两次返回空内容")
        return self._extract_json(raw)

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "context size" in text
            or "context_length" in text
            or "exceed_context" in text
            or ("n_ctx" in text and "exceed" in text)
        )

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
    def _to_point(value: dict[str, Any] | None) -> Point | None:
        if not isinstance(value, dict) or not value.get("visible", False):
            return None
        try:
            return Point(int(value.get("x", 0)), int(value.get("y", 0)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_card(value: dict[str, Any]) -> Card:
        rank = str(value.get("rank", ""))
        suit = str(value.get("suit", ""))
        wild = bool(value.get("wild", False)) or rank == "WILD"
        if not wild and (rank not in RANKS or suit not in SUITS):
            raise ValueError(f"非法牌：rank={rank!r} suit={suit!r}")
        return Card(
            rank=None if wild else rank,
            suit=None if wild else suit,
            wild=wild,
            x=int(value.get("x", 0)),
            y=int(value.get("y", 0)),
            confidence=float(value.get("confidence", 0.0)),
            selected=bool(value.get("selected", False)),
        )

    @classmethod
    def _to_cards(cls, values: Any, *, field_name: str) -> list[Card]:
        cards: list[Card] = []
        if not isinstance(values, list):
            return cards
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                logging.getLogger(__name__).warning(
                    "忽略非对象牌字段 field=%s index=%s value=%r",
                    field_name,
                    index,
                    value,
                )
                continue
            try:
                cards.append(cls._to_card(value))
            except (KeyError, TypeError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "忽略非法牌字段 field=%s index=%s value=%r error=%s",
                    field_name,
                    index,
                    value,
                    exc,
                )
        return cards

    @staticmethod
    def _to_gold_ranks(values: Any) -> tuple[str, ...]:
        if not isinstance(values, list):
            return ()
        ranks: list[str] = []
        for value in values:
            raw = value.get("rank") if isinstance(value, dict) else value
            rank = str(raw)
            if rank in RANKS:
                ranks.append(rank)
        return tuple(ranks)

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

    @staticmethod
    def _to_deck_count(value: Any) -> int | None:
        try:
            count = int(value)
        except (TypeError, ValueError):
            return None
        return None if count < 0 else count

    @classmethod
    def _to_snapshot(cls, payload: dict[str, Any]) -> GameSnapshot:
        try:
            screen_type = ScreenType(str(payload.get("screen_type", "unknown")))
        except ValueError:
            screen_type = ScreenType.UNKNOWN
        hand = cls._to_cards(payload.get("hand", []), field_name="hand")

        # 本地 Qwen 偶尔会在已经发牌后仍把整局标成 start。牌局阶段的手牌数是
        # 更强、也更稳定的结构信号：换牌固定 6 张，摸牌固定 7 张，取牌后固定
        # 8 张。必须先纠正阶段，再决定是否保留 visible_cards，否则错误的 start
        # 标签会把真正可选的亮牌清空，并让启动流程持续寻找“开始游戏”。
        raw_swap_button = cls._to_point(payload.get("swap_button"))
        if raw_swap_button is not None:
            # “换牌”按钮是最直接的阶段证据。换牌时可能是 6 张普通牌加一个
            # 尚未完成的“选牌”入口，也可能已经选牌完成而显示 7 张普通牌；
            # 不能再用“7 张牌=摸牌”覆盖这个明确按钮。
            screen_type = ScreenType.SWAP
        elif len(hand) == 6:
            screen_type = ScreenType.SWAP
        elif len(hand) == 7:
            screen_type = ScreenType.DRAW
        elif len(hand) == 8:
            screen_type = (
                ScreenType.HU
                if cls._to_point(payload.get("hu_button")) is not None
                else ScreenType.DISCARD
            )

        if screen_type == ScreenType.SWAP and len(hand) in {6, 7}:
            # 游戏通过“牌向上抬起”表示已选中。模型有时能给出正确坐标却漏掉
            # selected=true，因此用同一排牌的纵坐标差补充判断。这里的坐标仍是
            # 0-1000 归一化值，30 对应画面高度约 3%，足以排除轻微扇形差异。
            ys = sorted(card.y for card in hand if card.y is not None)
            if len(ys) == len(hand) and ys[-1] - ys[0] >= 40:
                baseline_y = ys[len(ys) // 2]
                hand = [
                    replace(
                        card,
                        selected=(
                            card.selected
                            or (card.y is not None and card.y <= baseline_y - 30)
                        ),
                    )
                    for card in hand
                ]

        visible = cls._to_cards(
            payload.get("visible_cards", []),
            field_name="visible_cards",
        )
        raw_opportunity_click = cls._to_point(payload.get("opportunity_click"))
        if raw_opportunity_click is not None:
            # 本地模型可能同时把同一个“?”入口错误抄成 WILD 亮牌。按坐标去重，
            # 保留独立的选牌入口，不能让它参与普通牌 EV 或被当成癞子。
            visible = [
                card
                for card in visible
                if not (
                    card.wild
                    and card.x is not None
                    and card.y is not None
                    and abs(card.x - raw_opportunity_click.x) <= 80
                    and abs(card.y - raw_opportunity_click.y) <= 80
                )
            ]
            # “?”选牌卡位于自己的手牌区。本地模型有时为了满足 draw=7 张的
            # 提示，又把它重复抄成 WILD 手牌；去掉这份重复表示。选牌卡不是
            # 癞子，控制器通过 opportunity_click 单独执行它。
            hand = [
                card
                for card in hand
                if not (
                    card.wild
                    and card.x is not None
                    and card.y is not None
                    and abs(card.x - raw_opportunity_click.x) <= 100
                    and abs(card.y - raw_opportunity_click.y) <= 100
                )
            ]
        if screen_type != ScreenType.DRAW:
            visible = []
        events = [
            cls._to_opponent_event(item)
            for item in payload.get("opponent_events", [])
        ]
        opportunity_click = raw_opportunity_click
        deck_click = cls._to_point(payload.get("deck_click"))
        swap_button = raw_swap_button
        discard_button = cls._to_point(payload.get("discard_button"))
        hu_button = cls._to_point(payload.get("hu_button"))
        continue_button = cls._to_point(payload.get("continue_button"))
        start_value = payload.get("start_button")
        start_button = (
            cls._to_point(start_value)
            if isinstance(start_value, dict)
            else None
        )
        # 兼容本地模型偶尔仍把“开始游戏”填入旧的 continue_button 字段。
        if screen_type == ScreenType.START and start_button is None:
            start_button = continue_button
        message = str(payload.get("message", ""))

        # 正式牌局里不存在“开始游戏”按钮。清除模型在头像、计时器等区域上
        # 臆造的 start_button，避免控制器误点并让日志准确反映当前阶段。
        if screen_type in {
            ScreenType.SWAP,
            ScreenType.DRAW,
            ScreenType.DISCARD,
            ScreenType.HU,
        }:
            start_button = None

        if screen_type == ScreenType.START and start_button is None:
            # 匹配/发牌动画有时会被本地模型误标为 start；真正开始页必须存在
            # 可点击按钮。没有按钮时保守等待，不能中断自动启动或盲点坐标。
            screen_type = ScreenType.WAITING
            message = message or "未检测到开始按钮，按匹配/发牌等待状态处理。"

        return GameSnapshot(
            screen_type=screen_type,
            gold_ranks=cls._to_gold_ranks(payload.get("gold_ranks", [])),
            hand=hand,
            visible_cards=visible,
            opponent_events=events,
            opportunity_click=opportunity_click,
            deck_click=deck_click,
            swap_button=swap_button,
            discard_button=discard_button,
            hu_button=hu_button,
            start_button=start_button,
            continue_button=continue_button,
            close_popup=cls._to_point(payload.get("close_popup")),
            deck_count=cls._to_deck_count(payload.get("deck_count", -1)),
            message=message,
        )
