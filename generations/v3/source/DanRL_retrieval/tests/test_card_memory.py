from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from DanRL_retrieval.retrieval import card_memory as card_memory_module
from DanRL_retrieval.retrieval.card_memory import (
    PLAY_STAT_FIELDS,
    CardMemoryIntegrityError,
    build_card_memory,
)
from DanRL_retrieval.retrieval.cards import ALL_CARDS, CARD_INDEX
from DanRL_retrieval.retrieval.context import build_context
from DanRL_retrieval.retrieval.models import ActionCandidate
from DanRL_retrieval.training.featurizer import (
    _card_memory_features,
    _pressure_state_features,
    state_features,
)
from DanRL_retrieval.training.schema import (
    CARD_MEMORY_DIM,
    CARD_MEMORY_STAT_DIM,
    FEATURE_VERSION,
    PRESSURE_STATE_DIM,
    STATE_DIM,
)
from DanRL_retrieval.training.type_suppression import candidate_response_profile


def complete_history_state() -> dict[str, object]:
    history = [
        {
            "pos": 2,
            "kind": "Straight",
            "rank": "5",
            "cards": ["SA", "S2", "S3", "S4", "S5"],
        },
        {
            "pos": 3,
            "kind": "TriplePlus",
            "rank": "7",
            "cards": ["H7", "C7", "D7", "H9", "C9"],
        },
        {
            "pos": 0,
            "kind": "Bomb",
            "rank": "8",
            "cards": ["S8", "H8", "C8", "D8"],
        },
        {
            "pos": 1,
            "kind": "StraightFlush",
            "rank": "A",
            "cards": ["HT", "HJ", "HQ", "HK", "HA"],
        },
        {
            "pos": 0,
            "kind": "FourKings",
            "rank": "RJ",
            "cards": ["BJ", "BJ", "RJ", "RJ"],
        },
        {"pos": 3, "kind": "PASS", "rank": None, "cards": []},
    ]
    played = [card for event in history for card in event["cards"]]
    return {
        "curRank": "6",
        "my_seat": 0,
        "history_my_seat": 2,
        "history_is_complete": True,
        "known_hand_cards": {"0": ["D2"]},
        "played_cards": played,
        "history": history,
    }


def test_card_memory_rotates_seats_and_records_requested_structures() -> None:
    memory = build_card_memory(complete_history_state())

    own, left, teammate, right = memory.seats
    assert own.straight_actions == 1
    assert own.played_cards == ("SA", "S2", "S3", "S4", "S5")

    assert left.triple_plus_actions == 1
    assert left.pass_actions == 1
    assert left.play_actions == 1

    assert teammate.bomb_actions == 2
    assert teammate.normal_bomb_actions == 1
    assert teammate.four_kings_actions == 1
    assert teammate.black_jokers == 2
    assert teammate.red_jokers == 2

    assert right.bomb_actions == 1
    assert right.straight_flush_actions == 1
    assert right.played_cards == ("HT", "HJ", "HQ", "HK", "HA")

    assert [(action.seat, action.kind) for action in memory.actions] == [
        (0, "Straight"),
        (1, "TriplePlus"),
        (2, "Bomb"),
        (3, "StraightFlush"),
        (2, "FourKings"),
        (1, "PASS"),
    ]
    assert memory.valid
    assert memory.complete


def test_pressure_features_encode_enemy_run_failed_team_responses_and_bomb_retake() -> None:
    history = [
        {"pos": 1, "kind": "TriplePlus", "rank": "5", "cards": ["S5", "H5", "C5", "S3", "H3"]},
        {"pos": 2, "kind": "PASS", "cards": []},
        {"pos": 3, "kind": "PASS", "cards": []},
        {"pos": 0, "kind": "PASS", "cards": []},
        {"pos": 1, "kind": "TriplePlus", "rank": "6", "cards": ["S6", "H6", "C6", "S4", "H4"]},
        {"pos": 2, "kind": "PASS", "cards": []},
    ]
    played = [card for event in history for card in event.get("cards", [])]
    ctx = build_context(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "known_hand_cards": {"0": ["D2"]},
            "played_cards": played,
            "history": history,
            "current_kind": "TriplePlus",
            "current_rank": "6",
            "current_size": 5,
            "last_player": 1,
        }
    )

    pressure = _pressure_state_features(ctx)

    np.testing.assert_allclose(
        pressure,
        np.array([2 / 8, 1 / 3, 2 / 4, 10 / 27, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(state_features(["D2"], ctx)[-5:], pressure)

    bomb_ctx = build_context(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "known_hand_cards": {"0": ["D2"]},
            "played_cards": ["S7", "S8", "H8", "C8", "D8"],
            "history": [
                {"pos": 1, "kind": "Single", "rank": "7", "cards": ["S7"]},
                {"pos": 2, "kind": "PASS", "cards": []},
                {"pos": 0, "kind": "Bomb", "rank": "8", "cards": ["S8", "H8", "C8", "D8"]},
            ],
            "current_kind": "Bomb",
            "current_rank": "8",
            "current_size": 4,
            "last_player": 0,
        }
    )
    np.testing.assert_allclose(
        _pressure_state_features(bomb_ctx),
        np.array([0, 0, 0, 0, 1], dtype=np.float32),
    )


def test_pressure_features_are_relative_to_an_odd_actor_seat() -> None:
    history = [
        {"pos": 0, "kind": "TriplePlus", "rank": "5", "cards": ["S5", "H5", "C5", "S3", "H3"]},
        {"pos": 1, "kind": "PASS", "cards": []},
        {"pos": 2, "kind": "PASS", "cards": []},
        {"pos": 3, "kind": "PASS", "cards": []},
        {"pos": 0, "kind": "TriplePlus", "rank": "6", "cards": ["S6", "H6", "C6", "S4", "H4"]},
        {"pos": 1, "kind": "PASS", "cards": []},
    ]
    ctx = build_context(
        {
            "my_seat": 1,
            "history_my_seat": 1,
            "history_is_complete": True,
            "known_hand_cards": {"1": ["D2"]},
            "played_cards": [
                card for event in history for card in event.get("cards", [])
            ],
            "history": history,
            "current_kind": "TriplePlus",
            "current_rank": "6",
            "current_size": 5,
            "last_player": 0,
        }
    )

    np.testing.assert_allclose(
        _pressure_state_features(ctx),
        np.array([2 / 8, 1 / 3, 2 / 4, 10 / 27, 0], dtype=np.float32),
    )


def test_card_memory_exact_counts_and_remaining_pool_are_consistent() -> None:
    state = complete_history_state()
    memory = build_card_memory(state)
    all_used = Counter(state["played_cards"] + state["known_hand_cards"]["0"])

    for card in ALL_CARDS:
        expected = 2 - all_used[card]
        assert memory.remaining_exact[CARD_INDEX[card]] == expected
        assert sum(seat.played_cards.count(card) for seat in memory.seats) == Counter(state["played_cards"])[card]


def test_card_memory_summary_lists_what_each_seat_played() -> None:
    summary = build_card_memory(complete_history_state()).to_dict()
    assert summary["seats"][0]["relative_name"] == "self"
    assert summary["seats"][1]["relative_name"] == "left_opponent"
    assert summary["seats"][2]["relative_name"] == "teammate"
    assert summary["seats"][3]["relative_name"] == "right_opponent"
    assert summary["seats"][2]["kind_counts"] == {"Bomb": 1, "FourKings": 1}
    assert summary["seats"][2]["exact_card_counts"]["BJ"] == 2
    assert summary["seats"][2]["actions"][-1] == {
        "kind": "FourKings",
        "rank": "RJ",
        "cards": ["BJ", "BJ", "RJ", "RJ"],
    }


def test_card_memory_binds_pass_and_response_to_current_winning_action() -> None:
    history = [
        {"pos": 0, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
        {"pos": 1, "kind": "PASS", "cards": []},
        {"pos": 2, "kind": "PASS", "cards": []},
        {"pos": 3, "kind": "Pair", "rank": "8", "cards": ["S8", "H8"]},
    ]
    memory = build_card_memory(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "known_hand_cards": {"0": []},
            "played_cards": ["S6", "H6", "S8", "H8"],
            "history": history,
        }
    )

    left_event = memory.seats[1].response_events[0]
    teammate_event = memory.seats[2].response_events[0]
    right_event = memory.seats[3].response_events[0]
    assert (left_event.passed, left_event.target_seat, left_event.target_kind) == (True, 0, "Pair")
    assert left_event.target_rank == "6"
    assert left_event.target_cards == ("S6", "H6")
    assert not left_event.target_was_teammate
    assert teammate_event.passed
    assert teammate_event.target_was_teammate
    assert not right_event.passed
    assert right_event.response_kind == "Pair"
    assert right_event.response_rank == "8"

    summary = memory.to_dict()
    assert summary["seats"][1]["response_events"][0]["target_rank"] == "6"


def test_candidate_response_profile_exposes_teammate_relative_evidence() -> None:
    history = [
        {"pos": 1, "kind": "Pair", "rank": "6", "cards": ["S6", "H6"]},
        {"pos": 2, "kind": "PASS", "cards": []},
        {"pos": 3, "kind": "Pair", "rank": "8", "cards": ["S8", "H8"]},
    ]
    ctx = build_context(
        {
            "my_seat": 0,
            "history_my_seat": 0,
            "history_is_complete": True,
            "known_hand_cards": {"0": ["D2"]},
            "played_cards": ["S6", "H6", "S8", "H8"],
            "history": history,
        }
    )

    details = candidate_response_profile(
        ctx,
        ActionCandidate(index=0, kind="Pair", rank="6", cards=("C6", "D6")),
    ).to_details()

    assert details["teammate_pass_evidence"] > 0.0
    assert details["teammate_response_risk"] < details["left_response_risk"]
    assert details["right_positive_response_evidence"] > 0.0
    assert details["left_pass_evidence"] == 0.0


def test_card_memory_rejects_an_impossible_third_copy() -> None:
    with pytest.raises(CardMemoryIntegrityError, match="S3.*3.*2"):
        build_card_memory(
            {
                "history_is_complete": True,
                "known_hand_cards": {"0": ["S3", "S3"]},
                "played_cards": ["S3"],
                "history": [{"pos": 1, "kind": "Single", "cards": ["S3"]}],
            }
        )


@pytest.mark.parametrize("size", range(4, 11))
def test_card_memory_tracks_every_normal_bomb_size(size: int) -> None:
    physical_threes = [
        "S3", "S3", "H3", "H3", "C3", "C3", "D3", "D3",
        "H2", "H2",
    ]
    cards = physical_threes[:size]
    memory = build_card_memory(
        {
            "history_is_complete": True,
            "known_hand_cards": {"0": []},
            "played_cards": cards,
            "history": [
                {"pos": 1, "kind": "Bomb", "rank": "3", "cards": cards}
            ],
        }
    )
    seat = memory.seats[1]
    assert seat.normal_bomb_actions == 1
    assert getattr(seat, f"bomb_{size}_actions") == 1


def test_complete_history_must_match_explicit_played_cards() -> None:
    with pytest.raises(CardMemoryIntegrityError, match="history.*played_cards"):
        build_card_memory(
            {
                "history_is_complete": True,
                "known_hand_cards": {"0": []},
                "played_cards": ["S3"],
                "history": [{"pos": 1, "kind": "Single", "cards": ["H3"]}],
            }
        )


def test_partial_history_preserves_explicit_played_card_precedence() -> None:
    memory = build_card_memory(
        {
            "known_hand_cards": {"0": ["D3"]},
            "played_cards": ["S3"],
            "history": [{"pos": 1, "kind": "Single", "cards": ["H3"]}],
        }
    )
    assert not memory.complete
    assert memory.remaining_exact[CARD_INDEX["S3"]] == 1
    assert memory.remaining_exact[CARD_INDEX["H3"]] == 2
    assert memory.seats[1].played_cards == ("H3",)


def test_context_and_selector_features_expose_full_card_memory() -> None:
    state = complete_history_state()
    ctx = build_context(state)
    features = _card_memory_features(ctx)

    assert features.shape == (CARD_MEMORY_DIM,)
    network_fields = getattr(card_memory_module, "NETWORK_PLAY_STAT_FIELDS", ())
    network_scales = getattr(card_memory_module, "NETWORK_PLAY_STAT_SCALES", ())
    assert CARD_MEMORY_STAT_DIM == 4 * len(network_fields)
    assert len(network_fields) == len(network_scales) == 13
    assert set(network_fields).isdisjoint(
        {
            "played_cards_count",
            "bomb_actions",
            "normal_bomb_actions",
            "black_jokers",
            "red_jokers",
        }
    )
    assert features[CARD_INDEX["D2"]] == pytest.approx(0.5)
    assert features[CARD_INDEX["BJ"]] == pytest.approx(0.0)

    exact_offset = len(ALL_CARDS)
    teammate_offset = exact_offset + 2 * len(ALL_CARDS)
    assert features[teammate_offset + CARD_INDEX["BJ"]] == pytest.approx(1.0)
    assert features[teammate_offset + CARD_INDEX["S8"]] == pytest.approx(0.5)

    full_state = state_features(["D2"], ctx)
    assert full_state.shape == (STATE_DIM,)
    np.testing.assert_array_equal(
        full_state[-(CARD_MEMORY_DIM + PRESSURE_STATE_DIM):-PRESSURE_STATE_DIM],
        features,
    )
    assert FEATURE_VERSION == "top10_selector_v12_calibrated_tactics1"
