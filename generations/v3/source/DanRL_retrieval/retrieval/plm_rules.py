from __future__ import annotations

from .models import ActionCandidate, CardGroup


RANK_TO_VALUE = {
    "A": 1,
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
    "BJ": 14,
    "RJ": 15,
}
VALUE_TO_RANK = {value: rank for rank, value in RANK_TO_VALUE.items()}
POINTS = {
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "T": 9,
    "J": 10,
    "Q": 11,
    "K": 12,
    "A": 13,
    "BJ": 14,
    "RJ": 15,
}
PLM_KIND = {
    "Single",
    "Pair",
    "StraightPair",
    "Triple",
    "StraightTriple",
    "Straight",
    "TriplePlus",
    "Bomb",
    "StraightFlush",
    "FourKings",
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
BLACK_JOKER_ALIASES = {"B", "SB"}
RED_JOKER_ALIASES = {"R", "HR"}


def normalize_rank(rank: str | None) -> str | None:
    if rank is None:
        return None
    text = str(rank).strip().upper().replace("10", "T")
    if text in BLACK_JOKER_ALIASES:
        return "BJ"
    if text in RED_JOKER_ALIASES:
        return "RJ"
    return text if text in RANK_TO_VALUE else None


def normalize_kind(kind: str | None) -> str:
    text = str(kind or "PASS")
    return KIND_ALIASES.get(text, text)


def is_bomb_kind(kind: str) -> bool:
    return normalize_kind(kind) in {"Bomb", "StraightFlush", "FourKings"}


def follow_rank(level: str | None, lower: str | None, higher: str | None) -> bool:
    """PLM numeric comparison for singles/pairs/triples/triple-plus.

    Mirrors common.Follow: level rank is above A and below jokers; jokers are
    still top. Sequence comparison should use `plain_points` instead.
    """

    lower = normalize_rank(lower)
    higher = normalize_rank(higher)
    level = normalize_rank(level)
    if lower is None or higher is None or lower == higher:
        return False
    if level and higher == level:
        return lower not in {"BJ", "RJ"}
    if level and lower == level:
        return higher in {"BJ", "RJ"}
    return POINTS[higher] > POINTS[lower]


def plain_points(rank: str | None) -> int:
    rank = normalize_rank(rank)
    return POINTS.get(rank or "", 0)


def action_minor(kind: str, size: int = 0, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    kind = normalize_kind(kind)
    if kind == "FourKings":
        return "joker"
    if kind == "StraightFlush":
        return "flush"
    if kind == "Bomb":
        return f"normal{size or 4}"
    return ""


def bomb_power(kind: str, rank: str | None, size: int, level: str | None, minor: str | None = None) -> tuple[int, int, int]:
    kind = normalize_kind(kind)
    minor = action_minor(kind, size, minor)
    if kind == "FourKings" or minor == "joker":
        return (100, 0, 0)
    if kind == "StraightFlush" or minor == "flush":
        return (70, 0, plain_points(rank))
    normal_size = size or 4
    if minor.startswith("normal"):
        try:
            normal_size = int(minor.removeprefix("normal"))
        except ValueError:
            normal_size = size or 4
    if normal_size >= 6:
        tier = 80 + min(normal_size, 8)
    elif normal_size == 5:
        tier = 60
    else:
        tier = 50
    return (tier, normal_size, rank_order_for_same_bomb(level, rank))


def rank_order_for_same_bomb(level: str | None, rank: str | None) -> int:
    rank = normalize_rank(rank)
    if rank is None:
        return 0
    if rank == "RJ":
        return 100
    if rank == "BJ":
        return 99
    if level and rank == normalize_rank(level):
        return 98
    return plain_points(rank)


def can_follow_action(candidate: ActionCandidate | CardGroup, target: ActionCandidate | CardGroup, level: str | None) -> bool:
    ck = normalize_kind(candidate.kind)
    tk = normalize_kind(target.kind)
    if ck == "PASS" or tk in {"PASS", "Lead"}:
        return False
    csize = len(candidate.cards)
    tsize = len(target.cards)
    if is_bomb_kind(ck) and is_bomb_kind(tk):
        cp = bomb_power(ck, candidate.rank, csize, level, getattr(candidate, "meta", {}).get("minor") if isinstance(candidate, CardGroup) else None)
        tp = bomb_power(tk, target.rank, tsize, level, getattr(target, "meta", {}).get("minor") if isinstance(target, CardGroup) else None)
        return cp > tp
    if ck != tk:
        return is_bomb_kind(ck) and not is_bomb_kind(tk)
    if ck in {"Single", "Pair", "Triple", "TriplePlus"}:
        return csize == tsize and follow_rank(level, target.rank, candidate.rank)
    if ck in {"Straight", "StraightPair", "StraightTriple"}:
        return csize == tsize and plain_points(candidate.rank) > plain_points(target.rank)
    return False


def group_to_action(group: CardGroup, index: int = 0) -> ActionCandidate:
    return ActionCandidate(index=index, kind=normalize_kind(group.kind), cards=group.cards, rank=group.rank)
