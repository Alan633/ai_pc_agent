from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from game_baque.ev import ActionValue
from game_baque.models import Card, GameDecision
from game_baque.rules import evaluate_hu
from game_baque.tracker import TableTracker

LOGGER = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class HybridDecisionResult:
    decision: GameDecision
    used_model: bool
    confidence: float
    choice_id: str | None
    reason: str


class LocalDecisionReranker:
    """让本地 Qwen 在 Python EV 候选中做最终重排。

    关键约束：
    - Python 先生成全部合法动作与 EV；
    - 模型只能从 Top-N 的 choice_id 中选择，不能创造动作；
    - 低置信、超时、非法输出或 EV 跌幅过大时自动回退 Python EV；
    - 这里只处理 8 张牌的“胡/弃牌”决策，摸牌仍由 Python EV 负责。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 4.0,
        top_n: int = 4,
        min_confidence: float = 0.55,
        max_ev_drop: float = 18.0,
    ) -> None:
        if not base_url:
            raise ValueError("本地决策模型必须配置 base_url")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.top_n = max(2, min(int(top_n), 8))
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.max_ev_drop = max(0.0, float(max_ev_drop))
        self.client = OpenAI(
            api_key=api_key or "local-model",
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            # 实时回合里宁可立刻回退 EV，也不要 SDK 自动重试拖过倒计时。
            max_retries=0,
        )

    def rerank_8_card_action(
        self,
        *,
        hand: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        tracker: TableTracker,
        ev_decision: GameDecision,
        values: list[ActionValue],
    ) -> HybridDecisionResult:
        candidates = self._candidate_rows(hand, values)
        if len(candidates) < 2:
            return HybridDecisionResult(
                decision=ev_decision,
                used_model=False,
                confidence=0.0,
                choice_id=None,
                reason="合法候选不足，沿用 Python EV。",
            )

        payload = {
            "hand": [self._card_payload(card, index) for index, card in enumerate(hand)],
            "gold_ranks": list(gold_ranks),
            "deck_count": tracker.deck_count,
            "opponents": tracker.opponent_known_summary(),
            "candidates": candidates,
            "instruction": (
                "只能从 candidates 的 choice_id 中选一个。Python EV 是强基线；"
                "只有在牌型结构、金牌保留、对手已知弃牌/拿牌等信息明确支持时才改选。"
            ),
        }
        prompt = (
            "你是百变八雀出牌重排器。不要创造动作，不要改变牌索引。"
            "根据 Python 已计算的 EV、未来胡牌价值、放牌风险和已知对手信息，"
            "从候选中选最终动作。只返回 JSON："
            '{"choice_id":"A","confidence":0.82,"reason":"不超过60字"}\n'
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        )

        try:
            raw = self._request(prompt)
            parsed = self._parse_json(raw)
            return self._validate_choice(
                parsed,
                hand=hand,
                gold_ranks=gold_ranks,
                ev_decision=ev_decision,
                values=values,
                candidates=candidates,
            )
        except Exception as exc:
            LOGGER.warning(
                "八雀HYBRID本地模型失败，回退Python EV：%s: %s",
                type(exc).__name__,
                exc,
            )
            return HybridDecisionResult(
                decision=ev_decision,
                used_model=False,
                confidence=0.0,
                choice_id=None,
                reason=f"本地模型不可用，回退 Python EV：{type(exc).__name__}",
            )

    def _request(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # Qwen 默认先输出很长的隐藏推理（实测同类出牌问题约 7000 token、
            # 110+ 秒），必然超时；重排只需要短 JSON。
            "reasoning_effort": "none",
        }
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Ollama 版本若不支持 response_format，再降级一次普通 JSON 输出；
            # 超时/连接错误不重复请求，避免错过 20 秒回合。
            text = str(exc).lower()
            if (
                "response_format" not in text
                and "json_object" not in text
                and "unsupported" not in text
            ):
                raise
            response = self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("本地决策模型没有返回结果")
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_RE.search(text)
            if match is None:
                raise ValueError("本地模型没有返回 JSON")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("本地模型结果不是 JSON 对象")
        return value

    def _candidate_rows(
        self,
        hand: list[Card],
        values: list[ActionValue],
    ) -> list[dict[str, Any]]:
        ordered = sorted(values, key=lambda item: item.expected_value, reverse=True)
        selected = ordered[: self.top_n]

        # 只要当前确实可以胡，就始终把 hu 候选交给模型，避免 Top-N 截断。
        hu_value = next((item for item in ordered if item.action == "hu"), None)
        if hu_value is not None and hu_value not in selected:
            if len(selected) >= self.top_n:
                selected[-1] = hu_value
            else:
                selected.append(hu_value)

        rows: list[dict[str, Any]] = []
        for index, item in enumerate(selected):
            row: dict[str, Any] = {
                "choice_id": chr(ord("A") + index),
                "action": item.action,
                "card_index": item.card_index,
                "ev": round(item.expected_value, 3),
                "future_hu": round(item.future_hu_value, 3),
                "opponent_risk": round(item.opponent_risk, 3),
            }
            if item.card_index is not None and 0 <= item.card_index < len(hand):
                row["card"] = hand[item.card_index].label
            rows.append(row)
        return rows

    def _validate_choice(
        self,
        payload: dict[str, Any],
        *,
        hand: list[Card],
        gold_ranks: tuple[str, ...] | list[str],
        ev_decision: GameDecision,
        values: list[ActionValue],
        candidates: list[dict[str, Any]],
    ) -> HybridDecisionResult:
        choice_id = str(payload.get("choice_id", "")).strip().upper()
        candidate = next(
            (item for item in candidates if item["choice_id"] == choice_id),
            None,
        )
        if candidate is None:
            raise ValueError(f"模型选择了非法 choice_id：{choice_id or '<empty>'}")

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        model_reason = str(payload.get("reason", "")).strip()[:180]

        if confidence < self.min_confidence:
            return HybridDecisionResult(
                decision=ev_decision,
                used_model=False,
                confidence=confidence,
                choice_id=choice_id,
                reason=f"模型置信度 {confidence:.2f} 过低，沿用 Python EV。",
            )

        action = str(candidate["action"])
        card_index = candidate.get("card_index")
        chosen_value = self._match_action_value(values, action, card_index)
        best_value = max(values, key=lambda item: item.expected_value)
        ev_drop = best_value.expected_value - chosen_value.expected_value
        if ev_drop > self.max_ev_drop:
            return HybridDecisionResult(
                decision=ev_decision,
                used_model=False,
                confidence=confidence,
                choice_id=choice_id,
                reason=(
                    f"模型候选比最高EV低 {ev_drop:.1f}，超过安全阈值 "
                    f"{self.max_ev_drop:.1f}，沿用 Python EV。"
                ),
            )

        if action == "hu":
            hu = evaluate_hu(hand, gold_ranks)
            if hu is None:
                raise ValueError("模型选择 hu，但规则引擎判定当前不能胡")
            decision = GameDecision(
                "hu",
                reason=(
                    f"HYBRID/Qwen 选择 {choice_id}（置信度 {confidence:.2f}）："
                    f"{model_reason or '选择立即胡牌'}；候选EV {chosen_value.expected_value:.1f}"
                ),
                hu=hu,
            )
        elif action == "discard":
            if not isinstance(card_index, int) or not 0 <= card_index < len(hand):
                raise ValueError("模型选择的弃牌索引非法")
            decision = GameDecision(
                "discard",
                card_indexes=(card_index,),
                reason=(
                    f"HYBRID/Qwen 选择 {choice_id} 打出 {hand[card_index].label}"
                    f"（置信度 {confidence:.2f}）：{model_reason or '综合重排'}；"
                    f"候选EV {chosen_value.expected_value:.1f}"
                ),
            )
        else:
            raise ValueError(f"HYBRID 不支持动作：{action}")

        return HybridDecisionResult(
            decision=decision,
            used_model=True,
            confidence=confidence,
            choice_id=choice_id,
            reason=model_reason,
        )

    @staticmethod
    def _match_action_value(
        values: list[ActionValue],
        action: str,
        card_index: int | None,
    ) -> ActionValue:
        for item in values:
            if item.action == action and item.card_index == card_index:
                return item
        raise ValueError("模型选择的候选不在 Python 合法动作集合中")

    @staticmethod
    def _card_payload(card: Card, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "card": card.label,
            "rank": card.rank,
            "suit": card.suit,
            "wild": card.wild,
        }
