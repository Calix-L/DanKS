"""Minimal action semantics used by structural scoring.

Legal-action generation and final legality checks belong to Dan_platform.
This module only normalizes public action labels and compares
already-formed actions for control-value estimation.
"""

from __future__ import annotations

from .models import ActionCandidate, CardGroup


RANK_POINTS = {
    "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
    "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12,
    "A": 13, "BJ": 14, "RJ": 15,
}
KIND_ALIASES = {
    "Boom": "Bomb",
    "BOOM": "Bomb",
    "ThreeWithTwo": "TriplePlus",
    "ThreePair": "StraightPair",
    "ThreePairs": "StraightPair",
    "TwoTrips": "StraightTriple",
    "FullHouse": "TriplePlus",
    "Trips": "Triple",
}


def normalize_rank(rank: str | None) -> str | None:
    if rank is None:
        return None
    text = str(rank).strip().upper().replace("10", "T")
    text = {"B": "BJ", "SB": "BJ", "R": "RJ", "HR": "RJ"}.get(text, text)
    return text if text in RANK_POINTS else None


def normalize_kind(kind: str | None) -> str:
    text = str(kind or "PASS")
    return KIND_ALIASES.get(text, text)


def plain_points(rank: str | None) -> int:
    return RANK_POINTS.get(normalize_rank(rank) or "", 0)


def _level_rank_beats(level: str | None, lower: str | None, higher: str | None) -> bool:
    lower = normalize_rank(lower)
    higher = normalize_rank(higher)
    level = normalize_rank(level)
    if lower is None or higher is None or lower == higher:
        return False
    if level and higher == level:
        return lower not in {"BJ", "RJ"}
    if level and lower == level:
        return higher in {"BJ", "RJ"}
    return plain_points(higher) > plain_points(lower)


def _bomb_power(action: ActionCandidate | CardGroup, level: str | None) -> tuple[int, int, int]:
    kind = normalize_kind(action.kind)
    size = len(action.cards)
    if kind == "FourKings":
        return 100, 0, 0
    if kind == "StraightFlush":
        return 70, 0, plain_points(action.rank)
    tier = 80 + min(size, 8) if size >= 6 else (60 if size == 5 else 50)
    rank = normalize_rank(action.rank)
    rank_value = 98 if rank and rank == normalize_rank(level) else plain_points(rank)
    return tier, size, rank_value


def can_beat(
    candidate: ActionCandidate | CardGroup,
    target: ActionCandidate | CardGroup,
    level: str | None,
) -> bool:
    candidate_kind = normalize_kind(candidate.kind)
    target_kind = normalize_kind(target.kind)
    if candidate_kind == "PASS" or target_kind in {"PASS", "Lead"}:
        return False
    bomb_kinds = {"Bomb", "StraightFlush", "FourKings"}
    candidate_is_bomb = candidate_kind in bomb_kinds
    target_is_bomb = target_kind in bomb_kinds
    if candidate_is_bomb and target_is_bomb:
        return _bomb_power(candidate, level) > _bomb_power(target, level)
    if candidate_kind != target_kind:
        return candidate_is_bomb and not target_is_bomb
    if len(candidate.cards) != len(target.cards):
        return False
    if candidate_kind in {"Single", "Pair", "Triple", "TriplePlus"}:
        return _level_rank_beats(level, target.rank, candidate.rank)
    if candidate_kind in {"Straight", "StraightPair", "StraightTriple"}:
        return plain_points(candidate.rank) > plain_points(target.rank)
    return False
