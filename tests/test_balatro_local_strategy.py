from __future__ import annotations

import pytest

from game_balatro.local_strategy import BalatroLocalStrategist
from game_balatro.models import BalatroDecision, Point


def strategist() -> BalatroLocalStrategist:
    return BalatroLocalStrategist(
        base_url="http://127.0.0.1:11434/v1",
        model="local-test",
        min_confidence=0.55,
    )


def candidates() -> tuple[BalatroDecision, ...]:
    return (
        BalatroDecision(
            action="play_hand",
            reason="Python基线",
            card_indices=(0, 1),
            target=Point(800, 650),
            expected_strength=120.0,
        ),
        BalatroDecision(
            action="discard",
            reason="保留对子继续找牌",
            card_indices=(2, 3, 4),
            target=Point(650, 650),
            expected_strength=0.0,
        ),
    )


def test_qwen_can_only_select_existing_candidate() -> None:
    result = strategist()._validate(
        {
            "choice_id": "B",
            "confidence": 0.88,
            "reason": "当前分数压力不大，继续强化对子构筑",
            "build_plan": "对子/两对方向，优先找相关Joker并保留经济",
        },
        candidates(),
    )

    assert result.used_model is True
    assert result.decision.action == "discard"
    assert result.decision.card_indices == (2, 3, 4)
    assert result.decision.metadata["decision_source"] == "qwen_hybrid"
    assert result.build_plan


def test_low_confidence_falls_back_to_python_baseline() -> None:
    values = candidates()
    result = strategist()._validate(
        {
            "choice_id": "B",
            "confidence": 0.31,
            "reason": "不确定",
            "build_plan": "",
        },
        values,
    )

    assert result.used_model is False
    assert result.decision == values[0]


def test_illegal_choice_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        strategist()._validate(
            {
                "choice_id": "Z",
                "confidence": 0.99,
                "reason": "非法候选",
                "build_plan": "",
            },
            candidates(),
        )
