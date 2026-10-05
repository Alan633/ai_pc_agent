from __future__ import annotations

from game_balatro.balatrobot import BalatroBotClient, snapshot_from_gamestate
from game_balatro.models import BalatroDecision, ScreenType
from game_balatro.solver import BalatroSolver


def test_balatrobot_state_maps_to_solver_snapshot() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SELECTING_HAND",
            "round_num": 2,
            "ante_num": 1,
            "money": 9,
            "round": {
                "hands_left": 3,
                "discards_left": 2,
                "chips": 125,
                "reroll_cost": 5,
            },
            "blinds": {
                "small": {
                    "status": "CURRENT",
                    "name": "Small Blind",
                    "score": 300,
                }
            },
            "hand": {
                "cards": [
                    {
                        "label": "Ten of Hearts",
                        "set": "DEFAULT",
                        "value": {"rank": "T", "suit": "H"},
                        "modifier": {},
                        "state": {"highlight": False, "debuff": False},
                        "cost": {"buy": 0},
                    },
                    {
                        "label": "Ace of Spades",
                        "set": "DEFAULT",
                        "value": {"rank": "A", "suit": "S"},
                        "modifier": {"enhancement": "BONUS"},
                        "state": {"highlight": True, "debuff": False},
                        "cost": {"buy": 0},
                    },
                ]
            },
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )

    assert snapshot.screen_type == ScreenType.PLAY
    assert snapshot.ante == 1
    assert snapshot.round_no == 2
    assert snapshot.current_score == 125
    assert snapshot.target_score == 300
    assert snapshot.hands_left == 3
    assert snapshot.discards_left == 2
    assert snapshot.hand[0].rank == "10"
    assert snapshot.hand[0].suit == "H"
    assert snapshot.hand[1].selected is True
    assert snapshot.hand[1].enhancement == "BONUS"
    assert snapshot.play_button is not None
    assert snapshot.play_button.visible is False
    assert snapshot.metadata["source"] == "balatrobot"


def test_balatrobot_shop_keeps_real_api_indices() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {
                "cards": [
                    {
                        "label": "Joker A",
                        "set": "JOKER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "vouchers": {
                "cards": [
                    {
                        "label": "Voucher A",
                        "set": "VOUCHER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 10},
                    }
                ]
            },
            "packs": {
                "cards": [
                    {
                        "label": "Arcana Pack",
                        "set": "BOOSTER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "pack": {"cards": []},
        }
    )

    assert [(item.api_group, item.api_index) for item in snapshot.shop_items] == [
        ("card", 0),
        ("voucher", 0),
        ("pack", 0),
    ]
    assert [item.kind for item in snapshot.shop_items] == [
        "joker",
        "voucher",
        "pack",
    ]


def test_balatrobot_maps_decisions_without_mouse_coordinates() -> None:
    client = BalatroBotClient(
        base_url="http://127.0.0.1:12346",
        deck="RED",
        stake="WHITE",
    )
    play_snapshot = snapshot_from_gamestate(
        {
            "state": "SELECTING_HAND",
            "round": {"hands_left": 4, "discards_left": 3, "chips": 0},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert client.command_for_decision(
        play_snapshot,
        BalatroDecision(
            action="play_hand",
            reason="test",
            card_indices=(0, 2, 4),
        ),
    ) == ("play", {"cards": [0, 2, 4]})

    shop_snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "round": {},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert client.command_for_decision(
        shop_snapshot,
        BalatroDecision(
            action="click",
            reason="buy",
            metadata={
                "candidate_type": "shop_buy",
                "api_group": "voucher",
                "api_index": 1,
            },
        ),
    ) == ("buy", {"voucher": 1})


def test_round_eval_maps_to_cash_out_screen() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "ROUND_EVAL",
            "round": {"chips": 500},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert snapshot.screen_type == ScreenType.ROUND_EVAL


def test_api_shop_candidates_keep_leave_and_reroll_separate() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {
                "cards": [
                    {
                        "label": "Joker A",
                        "set": "JOKER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    candidates = BalatroSolver(reserve_money=5).rank_candidates(
        snapshot,
        top_n=6,
    )
    kinds = {
        str(decision.metadata.get("candidate_type") or "")
        for decision in candidates
    }

    assert "shop_buy" in kinds
    assert "shop_leave" in kinds
    assert "shop_reroll" in kinds


def test_api_shop_booster_baseline_keeps_buy_group_and_index() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {
                "cards": [
                    {
                        "label": "Arcana Pack",
                        "set": "BOOSTER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "pack": {"cards": []},
        }
    )
    decision = BalatroSolver(reserve_money=5).decide(snapshot)

    assert decision.metadata["candidate_type"] == "shop_buy"
    assert decision.metadata["api_group"] == "pack"
    assert decision.metadata["api_index"] == 0
