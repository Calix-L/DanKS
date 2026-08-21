from __future__ import annotations

from collections import Counter
from typing import Any


RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "BJ", "RJ"]
NORMAL_RANKS = RANK_ORDER[:13]
SUITS = ["S", "H", "C", "D"]
JOKERS = ["BJ", "RJ"]


def _full_deck() -> Counter[str]:
    deck: Counter[str] = Counter()
    for rank in NORMAL_RANKS:
        for suit in SUITS:
            deck[f"{suit}{rank}"] = 2
    for joker in JOKERS:
        deck[joker] = 2
    return deck


FULL_DECK = _full_deck()


def _rank(card: str) -> str | None:
    if card in JOKERS:
        return card
    if len(card) == 2 and card[0] in SUITS and card[1] in NORMAL_RANKS:
        return card[1]
    return None


def count_remaining_cards(state: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    """Compute the frozen unseen-card features without external imports."""

    known: list[str] = []
    for cards in (state.get("known_hand_cards") or {}).values():
        if isinstance(cards, (list, tuple)):
            known.extend(str(card) for card in cards)
    played = [str(card) for card in (state.get("played_cards") or [])]
    if not played:
        for event in state.get("history") or []:
            if isinstance(event, dict):
                played.extend(str(card) for card in (event.get("cards") or []))

    remaining = FULL_DECK.copy()
    for card, count in Counter(known + played).items():
        if card in remaining:
            remaining[card] -= count

    detail = {
        card: max(0, int(remaining.get(card, 0)))
        for rank in NORMAL_RANKS
        for card in (f"S{rank}", f"H{rank}", f"C{rank}", f"D{rank}")
    }
    detail.update({joker: max(0, int(remaining.get(joker, 0))) for joker in JOKERS})
    by_rank = {rank: 0 for rank in RANK_ORDER}
    for card, count in remaining.items():
        rank = _rank(card)
        if rank is not None:
            by_rank[rank] += max(0, int(count))
    return detail, by_rank
