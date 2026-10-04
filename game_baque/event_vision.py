from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from game_baque.models import Card, RANKS, SUITS

_CARD_RANKS = [*RANKS, "WILD", "NONE"]
_CARD_SUITS = [*SUITS, "NONE"]


@dataclass(frozen=True, slots=True)
class ObservedCard:
    rank: str = "NONE"
    suit: str = "NONE"
    confidence: float = 0.0

    @property
    def known(self) -> bool:
        return self.rank != "NONE"

    def to_card(self) -> Card | None:
        if not self.known:
            return None
        if self.rank == "WILD":
            return Card(wild=True, confidence=self.confidence)
        if self.rank not in RANKS or self.suit not in SUITS:
            return None
        return Card(
            rank=self.rank,
            suit=self.suit,
            confidence=self.confidence,
        )

    @property
    def label(self) -> str:
        if self.rank == "NONE":
            return "未知"
        if self.rank == "WILD":
            return "癞子"
        symbols = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}
        return f"{self.rank}{symbols.get(self.suit, '')}"


@dataclass(frozen=True, slots=True)
class TableObservation:
    actor: str
    event_type: str
    draw_source: str
    drawn_card: ObservedCard
    discard_card: ObservedCard
    opportunity_cards: tuple[ObservedCard, ...]
    opportunity_chosen_index: int
    deck_before: int
    deck_after: int
    confidence: float
    evidence: str

    @property
    def actionable_opponent_turn(self) -> bool:
        return (
            self.actor in {"left", "top", "right"}
            and self.event_type == "turn"
            and self.discard_card.known
        )

    def summary(self) -> str:
        if self.event_type == "turn":
            draw = self.draw_source
            if self.drawn_card.known:
                draw += f" {self.drawn_card.label}"
            return (
                f"{self.actor} turn：{draw} → 打 {self.discard_card.label} "
                f"(conf={self.confidence:.2f})"
            )
        if self.event_type == "opportunity":
            cards = " ".join(card.label for card in self.opportunity_cards)
            return (
                f"{self.actor} 机会牌 [{cards}] chosen={self.opportunity_chosen_index} "
                f"(conf={self.confidence:.2f})"
            )
        return f"{self.actor} {self.event_type} (conf={self.confidence:.2f})"


EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "actor": {
            "type": "string",
            "enum": ["left", "top", "right", "self", "none"],
        },
        "event_type": {
            "type": "string",
            "enum": [
                "turn",
                "opportunity",
                "hu",
                "settlement",
                "round_start",
                "animation",
                "unknown",
            ],
        },
        "draw_source": {
            "type": "string",
            "enum": ["deck", "visible", "opportunity", "none", "unknown"],
        },
        "drawn_card": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rank": {"type": "string", "enum": _CARD_RANKS},
                "suit": {"type": "string", "enum": _CARD_SUITS},
                "confidence": {"type": "number"},
            },
            "required": ["rank", "suit", "confidence"],
        },
        "discard_card": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rank": {"type": "string", "enum": _CARD_RANKS},
                "suit": {"type": "string", "enum": _CARD_SUITS},
                "confidence": {"type": "number"},
            },
            "required": ["rank", "suit", "confidence"],
        },
        "opportunity_cards": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rank": {"type": "string", "enum": _CARD_RANKS},
                    "suit": {"type": "string", "enum": _CARD_SUITS},
                    "confidence": {"type": "number"},
                },
                "required": ["rank", "suit", "confidence"],
            },
        },
        "opportunity_chosen_index": {"type": "integer"},
        "deck_before": {"type": "integer"},
        "deck_after": {"type": "integer"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": [
        "actor",
        "event_type",
        "draw_source",
        "drawn_card",
        "discard_card",
        "opportunity_cards",
        "opportunity_chosen_index",
        "deck_before",
        "deck_after",
        "confidence",
        "evidence",
    ],
}


EVENT_PROMPT = """你负责分析腾讯欢乐斗地主“百变八雀·点字成金”的一个动作片段。

输入依次是：
1. BEFORE：动作发生前相对稳定画面；
2. PEAK：动作/动画变化最大时画面；
3. AFTER：动作结束后的相对稳定画面。

座位固定：
- left = 左侧玩家
- top = 上方玩家
- right = 右侧玩家
- self = 屏幕底部自己的玩家

目标是记录三名对手真实发生的拿牌/摸牌/出牌事件，用于记牌和概率计算。

严格规则：
1. 不要因为“硬气”“比翼为邻”“金×2/×3”等全屏特效就虚构玩家动作；纯特效 event_type=animation。
2. 普通一轮“玩家取得一张牌后再打出一张牌”尽量合并为 event_type=turn。
3. draw_source：
   - deck：牌堆剩余张数下降，或明确从蓝色牌背牌堆摸牌；
   - visible：明确拿走中央正面亮牌；
   - opportunity：通过机会牌四选一得到牌；
   - unknown：能确认该玩家完成了一轮，但看不清从哪里取牌；
   - none：不是 turn。
4. discard_card 是该玩家本轮最终打到桌面上的那张正面牌。必须同时识别点数和花色；看不清则 rank=NONE。
5. drawn_card 只有在明确看到拿走哪张正面牌时填写。暗摸牌不能猜，rank=NONE。
6. 癞子 rank=WILD、suit=NONE。
7. 普通点数只有 6、7、8、9、10、J、Q、K、A；花色 S/H/C/D=黑桃/红桃/梅花/方片。
8. 机会牌 event_type=opportunity 时，若能看到四张候选，按画面从左到右写 opportunity_cards；chosen_index 0~3。无法确定则 -1。
9. deck_before/deck_after 只在确实能读到牌堆数字时填写，否则 -1。
10. actor_hint 只是本地图像差分提示，不是事实；若截图证据与 hint 冲突，以截图为准。
11. 如果 before/peak/after 跨越了多个玩家动作，或者无法可靠归因，event_type=unknown，confidence 降低。
12. 只输出结构化结果，不给策略建议。
"""


class OpponentEventVision:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "",
    ) -> None:
        if not api_key and not base_url:
            raise ValueError("对手动作视觉识别需要 OPENAI_API_KEY 或兼容视觉接口")
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        else:
            kwargs["api_key"] = "local"
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._prefer_chat_completions = bool(self.base_url)

    async def classify(
        self,
        before: Path,
        peak: Path,
        after: Path,
        *,
        actor_hint: str = "unknown",
        peak_scores: dict[str, float] | None = None,
    ) -> TableObservation:
        if self._prefer_chat_completions:
            raw = await self._classify_chat(
                before,
                peak,
                after,
                actor_hint=actor_hint,
                peak_scores=peak_scores or {},
            )
        else:
            try:
                raw = await self._classify_responses(
                    before,
                    peak,
                    after,
                    actor_hint=actor_hint,
                    peak_scores=peak_scores or {},
                )
            except Exception:
                raw = await self._classify_chat(
                    before,
                    peak,
                    after,
                    actor_hint=actor_hint,
                    peak_scores=peak_scores or {},
                )
        return self._parse(json.loads(self._clean_json(raw)))

    async def _classify_responses(
        self,
        before: Path,
        peak: Path,
        after: Path,
        *,
        actor_hint: str,
        peak_scores: dict[str, float],
    ) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"actor_hint={actor_hint}; "
                    f"local_peak_scores={json.dumps(peak_scores, ensure_ascii=False)}。" 
                    "下面三张图按 BEFORE、PEAK、AFTER 顺序。"
                ),
            }
        ]
        for label, path in (("BEFORE", before), ("PEAK", peak), ("AFTER", after)):
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "input_text", "text": label})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{data}",
                    "detail": "high",
                }
            )

        response = await self.client.responses.create(
            model=self.model,
            instructions=EVENT_PROMPT,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "baque_table_event",
                    "description": "百变八雀一个动作片段的牌桌事件",
                    "schema": EVENT_SCHEMA,
                    "strict": True,
                }
            },
        )
        return response.output_text

    async def _classify_chat(
        self,
        before: Path,
        peak: Path,
        after: Path,
        *,
        actor_hint: str,
        peak_scores: dict[str, float],
    ) -> str:
        prompt = (
            EVENT_PROMPT
            + "\nactor_hint="
            + actor_hint
            + "; local_peak_scores="
            + json.dumps(peak_scores, ensure_ascii=False)
            + "\n严格只输出 JSON 对象，不要 Markdown。JSON Schema：\n"
            + json.dumps(EVENT_SCHEMA, ensure_ascii=False)
            + "\n下面三张图片依次是 BEFORE、PEAK、AFTER。"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in (before, peak, after):
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                }
            )

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
            response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _clean_json(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            text = text.removesuffix("```").strip()
        # Qwen3.5 的 thinking/兼容层偶尔会在 JSON 前后附加说明。
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text
    @classmethod
    def _parse(cls, payload: dict[str, Any]) -> TableObservation:
        def card(value: dict[str, Any]) -> ObservedCard:
            return ObservedCard(
                rank=str(value.get("rank", "NONE")),
                suit=str(value.get("suit", "NONE")),
                confidence=float(value.get("confidence", 0.0)),
            )

        return TableObservation(
            actor=str(payload["actor"]),
            event_type=str(payload["event_type"]),
            draw_source=str(payload["draw_source"]),
            drawn_card=card(payload["drawn_card"]),
            discard_card=card(payload["discard_card"]),
            opportunity_cards=tuple(card(item) for item in payload.get("opportunity_cards", [])),
            opportunity_chosen_index=int(payload.get("opportunity_chosen_index", -1)),
            deck_before=int(payload.get("deck_before", -1)),
            deck_after=int(payload.get("deck_after", -1)),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=str(payload.get("evidence", "")),
        )
