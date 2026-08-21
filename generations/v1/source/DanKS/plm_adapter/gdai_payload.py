"""Local PLM/GDAI payload adapter for the retrieval service.

This module intentionally contains only the runtime pieces used by
``scripts/interactive_server.py``.  The retrieval project can run without
depending on another project directory.
"""

from __future__ import annotations

import os
from typing import Any

# The release bundles the minimal Dan_platform Python rules under
# KSplatform/Dan_platform/python. The launcher adds that directory to
# PYTHONPATH before importing this module.
os.environ.setdefault("GUANDAN_MOVE_ENGINE", "python")

from engine import Moves
from engine.python_rules import _plm_sequence_value
from engine.python_rules import _plm_triple_plus_value


PLM_VALUE_TO_RANK = {
    1: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "B",
    15: "R",
}

PLM_PATTERN_TO_OGD_TYPE = {
    1: "Single",
    2: "Pair",
    3: "ThreePair",
    4: "Trips",
    5: "TwoTrips",
    6: "Straight",
    7: "ThreeWithTwo",
}

PLM_PATTERN_NAME_TO_OGD_TYPE = {
    "单张": "Single",
    "对子": "Pair",
    "木板": "ThreePair",
    "三连对": "ThreePair",
    "三张": "Trips",
    "钢板": "TwoTrips",
    "二连三": "TwoTrips",
    "顺子": "Straight",
    "三带二": "ThreeWithTwo",
    "炸弹": "Bomb",
    "Single": "Single",
    "Pair": "Pair",
    "ThreePair": "ThreePair",
    "Trips": "Trips",
    "Triple": "Trips",
    "TwoTrips": "TwoTrips",
    "StraightTriple": "TwoTrips",
    "Straight": "Straight",
    "ThreeWithTwo": "ThreeWithTwo",
    "TriplePlus": "ThreeWithTwo",
    "Bomb": "Bomb",
    "StraightFlush": "StraightFlush",
    "FourKings": "FourKings",
}

CARD_TO_NUM = {
    "H2": 0,
    "H3": 1,
    "H4": 2,
    "H5": 3,
    "H6": 4,
    "H7": 5,
    "H8": 6,
    "H9": 7,
    "HT": 8,
    "HJ": 9,
    "HQ": 10,
    "HK": 11,
    "HA": 12,
    "S2": 13,
    "S3": 14,
    "S4": 15,
    "S5": 16,
    "S6": 17,
    "S7": 18,
    "S8": 19,
    "S9": 20,
    "ST": 21,
    "SJ": 22,
    "SQ": 23,
    "SK": 24,
    "SA": 25,
    "C2": 26,
    "C3": 27,
    "C4": 28,
    "C5": 29,
    "C6": 30,
    "C7": 31,
    "C8": 32,
    "C9": 33,
    "CT": 34,
    "CJ": 35,
    "CQ": 36,
    "CK": 37,
    "CA": 38,
    "D2": 39,
    "D3": 40,
    "D4": 41,
    "D5": 42,
    "D6": 43,
    "D7": 44,
    "D8": 45,
    "D9": 46,
    "DT": 47,
    "DJ": 48,
    "DQ": 49,
    "DK": 50,
    "DA": 51,
    "SB": 52,
    "HR": 53,
}


def plm_tile_to_ogd_label(tile: Any) -> str | None:
    tile = int(tile)
    suit = tile >> 4
    value = tile & 0x0F
    if suit == 5 and value == 14:
        return "SB"
    if suit == 5 and value == 15:
        return "HR"
    suit_char = {1: "S", 2: "H", 3: "C", 4: "D"}.get(suit)
    rank_char = PLM_VALUE_TO_RANK.get(value)
    if suit_char is None or rank_char is None or rank_char in ("B", "R"):
        return None
    return suit_char + rank_char


def danrl_num_to_plm_tile(num: Any) -> int:
    num = int(num)
    if num == 52:
        return 0x5E
    if num == 53:
        return 0x5F
    suit_block, rank_idx = divmod(num, 13)
    suit = [2, 1, 3, 4][suit_block]
    value = 1 if rank_idx == 12 else rank_idx + 2
    return (suit << 4) | value


def ogd_label_to_plm_tile(label: str) -> int:
    return danrl_num_to_plm_tile(CARD_TO_NUM[label])


def tiles_to_ogd_labels(tiles: list[Any] | tuple[Any, ...] | None) -> list[str]:
    out: list[str] = []
    for tile in tiles or []:
        label = plm_tile_to_ogd_label(tile)
        if label is not None:
            out.append(label)
    return out


def infer_seat_maps(payload: dict[str, Any]) -> tuple[int, dict[int, int]]:
    players = payload.get("players") or []
    by_uid = {int(p.get("uid")): int(p.get("chair_id", 0)) for p in players if p.get("uid") is not None}
    self_uid = int(payload.get("uid", 0) or 0)
    self_chair = by_uid.get(self_uid, int(payload.get("chair_id", 1) or 1))
    chair_to_pos = {chair: (chair - self_chair) % 4 for chair in range(1, 5)}
    uid_to_pos = {uid: chair_to_pos.get(chair, 0) for uid, chair in by_uid.items()}
    uid_to_pos[self_uid] = 0
    return self_uid, uid_to_pos


def _rank_char_from_plm_value(value: Any) -> str:
    return PLM_VALUE_TO_RANK.get(int(value or 0), "")


def _rank_value_for_ogd(rank: str, current_rank: str) -> int:
    if rank == current_rank:
        return 15
    return {
        "A": 14,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "B": 16,
        "R": 17,
    }.get(rank, 0)


def _sequence_tail_rank(ranks: list[str], current_rank: str) -> str:
    del current_rank  # Sequence comparisons never promote the level rank.
    unique = set(ranks)
    if not unique:
        return ""
    sequence = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
    width = len(unique)
    for start in range(len(sequence) - width + 1):
        window = sequence[start : start + width]
        if len(set(window)) == width and set(window) == unique:
            return window[-1]
    # Keep degraded history deterministic; legal sequences take the branch above.
    natural = {rank: value for value, rank in enumerate(sequence[:-1], start=1)}
    return max(unique, key=lambda rank: natural.get(rank, -1))


def _infer_action_rank(action_type: str, cards: list[str], current_rank: str) -> str:
    ranks = [card[1] for card in cards if card not in ("SB", "HR")]
    if not cards:
        return ""
    if action_type == "FourKings":
        return "R"
    sequence_group_size = {"Straight": 1, "StraightFlush": 1, "ThreePair": 2, "TwoTrips": 3}.get(action_type)
    if sequence_group_size is not None:
        return _plm_sequence_value(cards, current_rank, sequence_group_size) or _sequence_tail_rank(ranks, current_rank)
    counts: dict[str, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    if action_type == "ThreeWithTwo":
        return _plm_triple_plus_value(cards, current_rank) or ""
    if action_type == "Bomb":
        if sorted(cards) == ["HR", "HR", "SB", "SB"]:
            return "R"
    if action_type in ("Pair", "Trips", "Bomb"):
        if cards and all(card == "SB" for card in cards):
            return "B"
        if cards and all(card == "HR" for card in cards):
            return "R"
        wild = f"H{current_rank}"
        natural_ranks = [card[1] for card in cards if card != wild and card not in ("SB", "HR")]
        if not natural_ranks:
            return current_rank
        natural_counts = {rank: natural_ranks.count(rank) for rank in set(natural_ranks)}
        return max(natural_counts, key=natural_counts.get)
    if cards[0] == "SB":
        return "B"
    if cards[0] == "HR":
        return "R"
    return cards[0][1]


def _infer_ogd_action_type(item: dict[str, Any], current_rank: str) -> str:
    pattern = int(item.get("pattern", 0) or 0)
    sub = int(item.get("sub", 0) or 0)
    cards = tiles_to_ogd_labels(item.get("cards") or [])
    if pattern == 8:
        if sub == 3 or sorted(cards) == ["HR", "HR", "SB", "SB"]:
            return "FourKings"
        if sub == 2:
            return "StraightFlush"
        return "Bomb"
    if pattern:
        return PLM_PATTERN_TO_OGD_TYPE.get(pattern, "")
    pattern_name = str(item.get("pattern_name", "") or "")
    action_type = PLM_PATTERN_NAME_TO_OGD_TYPE.get(pattern_name, "")
    if action_type == "Bomb":
        if sorted(cards) == ["HR", "HR", "SB", "SB"]:
            return "FourKings"
        if len(cards) == 5:
            wild = f"H{current_rank}"
            natural = [card for card in cards if card != wild]
            if (
                natural
                and all(card not in ("SB", "HR") for card in natural)
                and len({card[0] for card in natural}) == 1
                and _plm_sequence_value(cards, current_rank, 1) is not None
            ):
                return "StraightFlush"
    return action_type


def latest_greater_action(payload: dict[str, Any]) -> list[Any] | None:
    current_rank = PLM_VALUE_TO_RANK.get(int(payload.get("level_value", 2)), "2")
    for item in reversed(payload.get("play_history") or []):
        if str(item.get("action", "")).lower() != "play":
            continue
        cards = tiles_to_ogd_labels(item.get("cards") or [])
        action_type = _infer_ogd_action_type(item, current_rank)
        rank = _rank_char_from_plm_value(item.get("value")) or str(item.get("rank") or "")
        rank = rank or _infer_action_rank(action_type, cards, current_rank)
        if action_type and cards:
            return [action_type, rank, cards]
    return None


def build_dan_platform_action_list(payload: dict[str, Any]) -> list[Any]:
    current_rank = PLM_VALUE_TO_RANK.get(int(payload.get("level_value", 2)), "2")
    hand_labels = tiles_to_ogd_labels(payload.get("self_hand") or [])
    hearts_num = sum(1 for card in hand_labels if card == f"H{current_rank}")
    moves = Moves()
    if bool(payload.get("must_discard", False)):
        moves.parse_first_action(hand_labels, hearts_num, current_rank)
    else:
        greater = latest_greater_action(payload)
        moves.parse_second_action(hand_labels, hearts_num, current_rank, greater, None, None)
    return moves.action_list
