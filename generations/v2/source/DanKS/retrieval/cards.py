from __future__ import annotations

from collections import Counter
from typing import Iterable


NORMAL_RANKS = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2")
STRAIGHT_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANKS = NORMAL_RANKS + ("BJ", "RJ")
SUITS = ("S", "H", "C", "D")
JOKERS = ("BJ", "RJ")
RANK_VALUE = {rank: i for i, rank in enumerate(RANKS)}
CONTROL_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "BJ", "RJ")
CONTROL_RANK_VALUE = {rank: i for i, rank in enumerate(CONTROL_RANKS)}
CONTROL_RANK_STRENGTH_DENOM = max(1, len(CONTROL_RANKS) - 1)
LEVEL_RANK_STRENGTH = (CONTROL_RANK_VALUE["A"] + 0.5) / CONTROL_RANK_STRENGTH_DENOM
ALL_CARDS = tuple(f"{suit}{rank}" for rank in NORMAL_RANKS for suit in SUITS) + JOKERS
CARD_INDEX = {card: i for i, card in enumerate(ALL_CARDS)}
CARD_SORT_KEYS = {card: (RANK_VALUE[card if card in JOKERS else card[1]], card) for card in ALL_CARDS}


def normalize_card(card: object) -> str:
    if isinstance(card, str):
        if card in CARD_INDEX:
            return card
    text = str(card).strip().upper()
    text = text.replace("10", "T")
    if text in JOKERS:
        return text
    if len(text) == 2 and text[0] in SUITS and text[1] in NORMAL_RANKS:
        return text
    raise ValueError(f"invalid card label: {card!r}")


def normalize_cards(cards: Iterable[object]) -> list[str]:
    return [normalize_card(card) for card in cards]


def card_rank(card: str) -> str:
    if card in JOKERS:
        return card
    if isinstance(card, str) and len(card) == 2 and card[0] in SUITS and card[1] in NORMAL_RANKS:
        return card[1]
    card = normalize_card(card)
    return card if card in JOKERS else card[1]


def card_suit(card: str) -> str | None:
    if card in JOKERS:
        return None
    if isinstance(card, str) and len(card) == 2 and card[0] in SUITS and card[1] in NORMAL_RANKS:
        return card[0]
    card = normalize_card(card)
    return None if card in JOKERS else card[0]


def heart_level_card(cur_rank: str | None) -> str | None:
    if cur_rank in NORMAL_RANKS:
        return f"H{cur_rank}"
    return None


def rank_counter(cards: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for card in cards:
        counts[card_rank(card)] += 1
    return counts


def sorted_cards(cards: Iterable[str]) -> tuple[str, ...]:
    items = tuple(cards)
    if len(items) < 2:
        return items
    prev_key = card_sort_key(items[0])
    for card in items[1:]:
        key = card_sort_key(card)
        if prev_key > key:
            return tuple(sorted(items, key=card_sort_key))
        prev_key = key
    return items


def card_sort_key(card: str) -> tuple[int, str]:
    cached = CARD_SORT_KEYS.get(card)
    if cached is not None:
        return cached
    if card in JOKERS:
        return (RANK_VALUE[card], card)
    if isinstance(card, str) and len(card) == 2 and card[0] in SUITS and card[1] in NORMAL_RANKS:
        return (RANK_VALUE[card[1]], card)
    return (RANK_VALUE[card_rank(card)], card)


def remove_cards(hand: Iterable[str], action: Iterable[str]) -> list[str]:
    counts = Counter(normalize_cards(hand))
    for card in normalize_cards(action):
        if counts[card] <= 0:
            raise ValueError(f"action card {card} not in hand")
        counts[card] -= 1
    out: list[str] = []
    for card, count in counts.items():
        out.extend([card] * count)
    return sorted(out, key=card_sort_key)


def rank_strength(rank: str, cur_rank: str | None = None, remaining_detail: dict[str, int] | None = None) -> float:
    """Return a coarse control strength in [0, 1].

    Already-played key cards are represented through `remaining_detail`: if
    higher jokers are gone, BJ/RJ strengths naturally move toward the top.
    """
    base = CONTROL_RANK_VALUE.get(rank, 0) / CONTROL_RANK_STRENGTH_DENOM
    if cur_rank and rank == cur_rank:
        base = max(base, LEVEL_RANK_STRENGTH)
    if remaining_detail:
        if rank == "BJ" and remaining_detail.get("RJ", 0) == 0:
            base = 0.98
        if rank == "RJ":
            base = 1.0
    return base


def action_type_size(kind: str) -> int:
    return {
        "PASS": 0,
        "Single": 1,
        "Pair": 2,
        "Triple": 3,
        "Straight": 5,
        "StraightPair": 6,
        "StraightTriple": 6,
        "FullHouse": 5,
        "TriplePlus": 5,
        "Bomb": 4,
        "StraightFlush": 5,
        "FourKings": 4,
    }.get(kind, 1)
