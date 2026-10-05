from __future__ import annotations

from game_baque.ev import ActionValue
from game_baque.local_decision import LocalDecisionReranker
from game_baque.models import Card, GameDecision
from game_baque.tracker import TableTracker


def _hand() -> list[Card]:
    return [
        Card(rank="6", suit="S"),
        Card(rank="6", suit="H"),
        Card(rank="8", suit="C"),
        Card(rank="9", suit="D"),
        Card(rank="10", suit="S"),
        Card(rank="J", suit="H"),
        Card(rank="K", suit="C"),
        Card(rank="A", suit="D"),
    ]


def _values() -> list[ActionValue]:
    return [
        ActionValue("discard", 2, 100.0, 0.0, 115.0, 10.0, "打8"),
        ActionValue("discard", 3, 95.0, 0.0, 112.0, 8.0, "打9"),
        ActionValue("discard", 4, 70.0, 0.0, 90.0, 15.0, "打10"),
        ActionValue("discard", 5, 60.0, 0.0, 82.0, 16.0, "打J"),
    ]


def _reranker(**kwargs: object) -> LocalDecisionReranker:
    return LocalDecisionReranker(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5-9b-16k",
        timeout_seconds=2,
        **kwargs,
    )


def test_hybrid_qwen_can_rerank_within_ev_guard(monkeypatch) -> None:
    reranker = _reranker(max_ev_drop=18.0, min_confidence=0.55)
    monkeypatch.setattr(
        reranker,
        "_request",
        lambda prompt: '{"choice_id":"B","confidence":0.88,"reason":"放牌风险更低"}',
    )
    hand = _hand()
    ev = GameDecision("discard", card_indexes=(2,), reason="EV最高")

    result = reranker.rerank_8_card_action(
        hand=hand,
        gold_ranks=("7", "Q", "A"),
        tracker=TableTracker(),
        ev_decision=ev,
        values=_values(),
    )

    assert result.used_model is True
    assert result.decision.action == "discard"
    assert result.decision.card_indexes == (3,)
    assert "HYBRID/Qwen" in result.decision.reason


def test_hybrid_rejects_large_ev_drop(monkeypatch) -> None:
    reranker = _reranker(max_ev_drop=18.0, min_confidence=0.55, top_n=4)
    monkeypatch.setattr(
        reranker,
        "_request",
        lambda prompt: '{"choice_id":"D","confidence":0.99,"reason":"想打J"}',
    )
    ev = GameDecision("discard", card_indexes=(2,), reason="EV最高")

    result = reranker.rerank_8_card_action(
        hand=_hand(),
        gold_ranks=("7", "Q", "A"),
        tracker=TableTracker(),
        ev_decision=ev,
        values=_values(),
    )

    assert result.used_model is False
    assert result.decision == ev
    assert "超过安全阈值" in result.reason


def test_hybrid_low_confidence_falls_back_to_ev(monkeypatch) -> None:
    reranker = _reranker(min_confidence=0.70)
    monkeypatch.setattr(
        reranker,
        "_request",
        lambda prompt: '{"choice_id":"B","confidence":0.40,"reason":"不确定"}',
    )
    ev = GameDecision("discard", card_indexes=(2,), reason="EV最高")

    result = reranker.rerank_8_card_action(
        hand=_hand(),
        gold_ranks=("7", "Q", "A"),
        tracker=TableTracker(),
        ev_decision=ev,
        values=_values(),
    )

    assert result.used_model is False
    assert result.decision == ev
    assert result.confidence == 0.40


def test_hybrid_illegal_choice_falls_back_to_ev(monkeypatch) -> None:
    reranker = _reranker()
    monkeypatch.setattr(
        reranker,
        "_request",
        lambda prompt: '{"choice_id":"Z","confidence":0.99,"reason":"非法"}',
    )
    ev = GameDecision("discard", card_indexes=(2,), reason="EV最高")

    result = reranker.rerank_8_card_action(
        hand=_hand(),
        gold_ranks=("7", "Q", "A"),
        tracker=TableTracker(),
        ev_decision=ev,
        values=_values(),
    )

    assert result.used_model is False
    assert result.decision == ev


def test_hybrid_prompt_contains_only_legal_candidates(monkeypatch) -> None:
    reranker = _reranker(top_n=2)
    seen: list[str] = []

    def fake_request(prompt: str) -> str:
        seen.append(prompt)
        return '{"choice_id":"A","confidence":0.9,"reason":"保持最高EV"}'

    monkeypatch.setattr(reranker, "_request", fake_request)
    ev = GameDecision("discard", card_indexes=(2,), reason="EV最高")
    result = reranker.rerank_8_card_action(
        hand=_hand(),
        gold_ranks=("7", "Q", "A"),
        tracker=TableTracker(),
        ev_decision=ev,
        values=_values(),
    )

    assert result.used_model is True
    assert len(seen) == 1
    assert '"choice_id":"A"' in seen[0]
    assert '"choice_id":"B"' in seen[0]
    assert '"choice_id":"C"' not in seen[0]
