from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cards import normalize_cards
from .card_counter import count_remaining_cards


@dataclass(frozen=True)
class RetrievalContext:
    my_seat: int = 0
    cur_rank: str | None = None
    public_counts: tuple[int, int, int, int] = (27, 27, 27, 27)
    current_kind: str = "Lead"
    current_rank: str | None = None
    current_size: int = 0
    current_minor: str | None = None
    last_player: int | None = None
    played_cards: tuple[str, ...] = ()
    known_hand_cards: dict[str, tuple[str, ...]] = field(default_factory=dict)
    remaining_detail: dict[str, int] = field(default_factory=dict)
    remaining_by_rank: dict[str, int] = field(default_factory=dict)


def _remaining_from_state(state: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    detail = {
        str(card): int(count)
        for card, count in dict(
            state.get("remaining_detail") or state.get("remaining_cards_detail") or {}
        ).items()
    }
    raw_by_rank = state.get("remaining_by_rank") or {}
    if isinstance(raw_by_rank, dict):
        by_rank = {str(rank): int(count) for rank, count in raw_by_rank.items()}
    else:
        by_rank = {
            str(row["rank"]): int(row["count"])
            for row in raw_by_rank
            if isinstance(row, dict) and "rank" in row and "count" in row
        }
    if detail or by_rank:
        return detail, by_rank
    return count_remaining_cards(state)


def build_context(state: dict[str, Any]) -> RetrievalContext:
    """Build retrieval context from a loose game-state dict."""

    public_counts = state.get("public_counts") or state.get("remaining_counts") or [27, 27, 27, 27]
    public_counts = tuple(int(x) for x in list(public_counts)[:4])
    if len(public_counts) != 4:
        public_counts = (27, 27, 27, 27)
    known = {}
    for seat, cards in (state.get("known_hand_cards") or {}).items():
        known[str(seat)] = tuple(normalize_cards(cards))
    played = tuple(normalize_cards(state.get("played_cards") or []))
    detail, by_rank = _remaining_from_state(state)
    return RetrievalContext(
        my_seat=int(state.get("my_seat", 0)),
        cur_rank=str(state.get("curRank")) if state.get("curRank") else None,
        public_counts=public_counts,
        current_kind=str(state.get("current_kind") or state.get("greater_kind") or "Lead"),
        current_rank=str(state.get("current_rank") or state.get("greater_rank")) if (state.get("current_rank") or state.get("greater_rank")) else None,
        current_size=int(state.get("current_size") or state.get("greater_size") or 0),
        current_minor=str(state.get("current_minor") or state.get("greater_minor")) if (state.get("current_minor") or state.get("greater_minor")) else None,
        last_player=state.get("last_player"),
        played_cards=played,
        known_hand_cards=known,
        remaining_detail=detail,
        remaining_by_rank=by_rank,
    )
