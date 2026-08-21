from __future__ import annotations

import pytest

from DanRL_retrieval.retrieval.context import build_context
from DanRL_retrieval.retrieval.models import ActionCandidate
from DanRL_retrieval.training.type_suppression import (
    BASE_RESPONSE_RISK,
    candidate_response_profile,
)


def _ctx(history: list[dict[str, object]]):
    played = [card for event in history for card in event.get("cards", [])]
    return build_context(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "curRank": "A",
            "public_counts": [10, 10, 10, 10],
            "current_kind": "Lead",
            "known_hand_cards": {"0": []},
            "played_cards": played,
            "history": history,
        }
    )


def _action(kind: str, rank: str, cards: tuple[str, ...]) -> ActionCandidate:
    return ActionCandidate(index=0, kind=kind, rank=rank, cards=cards)


def test_pass_on_pair_suppresses_same_or_harder_pair_more_than_easier_pair() -> None:
    ctx = _ctx(
        [
            {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
            {"pos": 1, "kind": "PASS", "cards": []},
        ]
    )
    harder = candidate_response_profile(ctx, _action("Pair", "8", ("S8", "H8")))
    easier = candidate_response_profile(ctx, _action("Pair", "4", ("S4", "H4")))

    assert harder.left_evidence > easier.left_evidence
    assert harder.left_risk < easier.left_risk
    assert harder.next_suppression > easier.next_suppression


def test_teammate_pass_is_much_weaker_than_opponent_pass() -> None:
    opponent_ctx = _ctx(
        [
            {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
            {"pos": 1, "kind": "PASS", "cards": []},
        ]
    )
    teammate_ctx = _ctx(
        [
            {"pos": 3, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
            {"pos": 1, "kind": "PASS", "cards": []},
        ]
    )
    action = _action("Pair", "8", ("S8", "H8"))
    opponent = candidate_response_profile(opponent_ctx, action)
    teammate = candidate_response_profile(teammate_ctx, action)

    assert opponent.left_evidence > teammate.left_evidence * 3.0
    assert opponent.left_risk < teammate.left_risk


def test_successful_response_is_positive_counter_evidence() -> None:
    pass_ctx = _ctx(
        [
            {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
            {"pos": 1, "kind": "PASS", "cards": []},
        ]
    )
    response_ctx = _ctx(
        [
            {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
            {"pos": 1, "kind": "Pair", "rank": "8", "cards": ["S8", "H8"]},
        ]
    )
    action = _action("Pair", "7", ("S7", "H7"))

    assert candidate_response_profile(response_ctx, action).left_risk > candidate_response_profile(pass_ctx, action).left_risk


def test_spent_straight_is_mild_depletion_evidence_not_hard_zero() -> None:
    empty = _ctx([])
    spent = _ctx(
        [
            {
                "pos": 1,
                "kind": "Straight",
                "rank": "7",
                "cards": ["S3", "H4", "C5", "D6", "S7"],
            }
        ]
    )
    action = _action("Straight", "9", ("S5", "H6", "C7", "D8", "S9"))
    empty_profile = candidate_response_profile(empty, action)
    spent_profile = candidate_response_profile(spent, action)

    assert 0.05 < spent_profile.left_risk < empty_profile.left_risk
    assert 0.0 < spent_profile.left_evidence < 0.5


def test_no_evidence_has_zero_combined_network_signal() -> None:
    profile = candidate_response_profile(
        _ctx([]), _action("Pair", "8", ("S8", "H8"))
    )
    assert profile.evidence_strength == pytest.approx(0.0)
    assert profile.combined_suppression == pytest.approx(0.0)


def test_pass_profiles_the_current_public_target() -> None:
    ctx = build_context(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "curRank": "A",
            "public_counts": [10, 10, 10, 10],
            "current_kind": "Pair",
            "current_rank": "6",
            "current_size": 2,
            "last_player": 0,
            "known_hand_cards": {"0": []},
            "played_cards": ["S6", "H6"],
            "history": [
                {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
                {"pos": 1, "kind": "PASS", "cards": []},
            ],
        }
    )

    profile = candidate_response_profile(
        ctx, _action("PASS", "PASS", ()),
    )

    assert profile.left_evidence > 0.0
    assert profile.left_risk < BASE_RESPONSE_RISK
