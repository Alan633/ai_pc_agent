from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from openai import AsyncOpenAI

from game_balatro.models import BalatroDecision, BalatroSnapshot, ScreenType


LOGGER = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class BalatroStrategyResult:
    decision: BalatroDecision
    used_model: bool
    confidence: float
    choice_id: str | None
    reason: str
    build_plan: str


class BalatroLocalStrategist:
    """让本地 Qwen3.5 在 Python 合法候选中做长期策略选择。

    模型不能返回鼠标坐标，也不能创造牌索引。最终可执行动作必须来自
    BalatroSolver.rank_candidates()。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 6.0,
        top_n: int = 6,
        min_confidence: float = 0.55,
    ) -> None:
        if not base_url:
            raise ValueError("Balatro 本地策略模型必须配置 base_url")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 20.0))
        self.top_n = max(2, min(int(top_n), 10))
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.client = AsyncOpenAI(
            api_key=api_key or "local-model",
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    async def rerank(
        self,
        *,
        snapshot: BalatroSnapshot,
        candidates: tuple[BalatroDecision, ...],
        strategy_memory: tuple[str, ...] = (),
    ) -> BalatroStrategyResult:
        selected = candidates[: self.top_n]
        if len(selected) < 2:
            return BalatroStrategyResult(
                decision=selected[0],
                used_model=False,
                confidence=0.0,
                choice_id=None,
                reason="合法候选不足，直接使用 Python 基线。",
                build_plan="",
            )

        rows = [
            self._candidate_payload(index, decision, snapshot)
            for index, decision in enumerate(selected)
        ]
        prompt = (
            "你是单机游戏 Balatro 的长期策略重排器。"
            "Python 已经生成并校验了所有候选动作；你只能从 candidates 的 "
            "choice_id 中选一个，禁止创造新动作、坐标或牌索引。"
            "你的目标不是只看眼前牌型大小，而是综合过关概率、剩余 Hands/Discards、"
            "Joker 协同、经济利息、商店成长、Blind 风险和当前 Build 的长期收益。"
            "如果信息不足，优先选择 Python 基线 A。"
            "只返回 JSON："
            '{"choice_id":"A","confidence":0.82,"reason":"不超过100字",'
            '"build_plan":"不超过120字，概括当前长期构筑方向"}\n'
            + json.dumps(
                {
                    "state": self._snapshot_payload(snapshot),
                    "recent_strategy_memory": list(strategy_memory[-8:]),
                    "candidates": rows,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        try:
            raw = await asyncio.wait_for(
                self._request(prompt),
                timeout=self.timeout_seconds + 0.5,
            )
            payload = self._parse_json(raw)
            return self._validate(payload, selected)
        except Exception as exc:
            LOGGER.warning(
                "Balatro HYBRID 本地策略失败，回退 Python：%s: %s",
                type(exc).__name__,
                exc,
            )
            return BalatroStrategyResult(
                decision=selected[0],
                used_model=False,
                confidence=0.0,
                choice_id=None,
                reason=f"本地策略模型不可用，回退 Python：{type(exc).__name__}",
                build_plan="",
            )

    async def _request(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "reasoning_effort": "none",
        }
        try:
            response = await self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            text = str(exc).lower()
            if (
                "response_format" not in text
                and "json_object" not in text
                and "unsupported" not in text
            ):
                raise
            response = await self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("Balatro 本地策略模型没有返回结果")
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_RE.search(text)
            if match is None:
                raise ValueError("Balatro 本地策略模型没有返回 JSON")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Balatro 本地策略结果不是 JSON 对象")
        return value

    def _validate(
        self,
        payload: dict[str, Any],
        candidates: tuple[BalatroDecision, ...],
    ) -> BalatroStrategyResult:
        choice_id = str(payload.get("choice_id", "")).strip().upper()
        index = ord(choice_id) - ord("A") if len(choice_id) == 1 else -1
        if not 0 <= index < len(candidates):
            raise ValueError(f"模型选择了非法 choice_id：{choice_id or '<empty>'}")

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        reason = str(payload.get("reason", "")).strip()[:240]
        build_plan = str(payload.get("build_plan", "")).strip()[:240]

        if confidence < self.min_confidence:
            return BalatroStrategyResult(
                decision=candidates[0],
                used_model=False,
                confidence=confidence,
                choice_id=choice_id,
                reason=(
                    f"模型置信度 {confidence:.2f} 低于 "
                    f"{self.min_confidence:.2f}，回退 Python。"
                ),
                build_plan=build_plan,
            )

        chosen = candidates[index]
        metadata = dict(chosen.metadata)
        metadata.update(
            {
                "decision_source": "qwen_hybrid",
                "qwen_choice_id": choice_id,
                "qwen_confidence": round(confidence, 4),
                "qwen_build_plan": build_plan,
            }
        )
        chosen = replace(
            chosen,
            reason=(
                f"HYBRID/Qwen {choice_id}（{confidence:.2f}）："
                f"{reason or chosen.reason}"
            ),
            metadata=metadata,
        )
        return BalatroStrategyResult(
            decision=chosen,
            used_model=True,
            confidence=confidence,
            choice_id=choice_id,
            reason=reason,
            build_plan=build_plan,
        )

    @staticmethod
    def _candidate_payload(
        index: int,
        decision: BalatroDecision,
        snapshot: BalatroSnapshot,
    ) -> dict[str, Any]:
        cards = [
            f"{snapshot.hand[i].rank}{snapshot.hand[i].suit}"
            for i in decision.card_indices
            if 0 <= i < len(snapshot.hand)
        ]
        return {
            "choice_id": chr(ord("A") + index),
            "action": decision.action,
            "cards": cards,
            "card_indices": list(decision.card_indices),
            "expected_strength": round(decision.expected_strength, 3),
            "metadata": decision.metadata,
            "python_reason": decision.reason,
            "python_baseline": index == 0,
        }

    @staticmethod
    def _snapshot_payload(snapshot: BalatroSnapshot) -> dict[str, Any]:
        return {
            "screen_type": snapshot.screen_type.value,
            "ante": snapshot.ante,
            "round_no": snapshot.round_no,
            "blind_name": snapshot.blind_name,
            "target_score": snapshot.target_score,
            "current_score": snapshot.current_score,
            "money": snapshot.money,
            "hands_left": snapshot.hands_left,
            "discards_left": snapshot.discards_left,
            "hand": [
                {
                    "index": index,
                    "card": f"{card.rank}{card.suit}",
                    "selected": card.selected,
                    "enhancement": card.enhancement,
                    "edition": card.edition,
                    "seal": card.seal,
                    "debuffed": card.debuffed,
                }
                for index, card in enumerate(snapshot.hand)
            ],
            "jokers": [
                {
                    "name": joker.name,
                    "edition": joker.edition,
                    "debuffed": joker.debuffed,
                }
                for joker in snapshot.jokers
            ],
            "shop_items": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "price": item.price,
                }
                for item in snapshot.shop_items
            ],
            "pack_items": [
                {
                    "name": item.name,
                    "kind": item.kind,
                }
                for item in snapshot.pack_items
            ],
            "screen_note": snapshot.message,
        }
