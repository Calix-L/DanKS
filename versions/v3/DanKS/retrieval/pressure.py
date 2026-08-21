from __future__ import annotations

import math

from .cards import action_type_size
from .context import RetrievalContext


def _position_factor(ctx: RetrievalContext, seat: int) -> float:
    distance = (seat - ctx.my_seat) % 4
    return {1: 1.0, 2: 0.45, 3: 0.7}.get(distance, 0.0)


def _is_opponent(ctx: RetrievalContext, seat: int) -> bool:
    return (seat % 2) != (ctx.my_seat % 2)


def pressure(ctx: RetrievalContext, kind: str, tau: float = 1.5) -> float:
    """Continuous pressure for a card-type dimension.

    High when an opponent's remaining card count is close to this kind's size,
    the opponent acts soon, and the game is in a race-like state.
    """

    target_size = action_type_size(kind)
    if target_size <= 0:
        return 0.0
    best = 0.0
    for seat, left in enumerate(ctx.public_counts):
        if seat == ctx.my_seat or not _is_opponent(ctx, seat) or left <= 0:
            continue
        count_match = math.exp(-abs(left - target_size) / tau)
        threat = min(1.0, 10.0 / max(1.0, float(left)))
        best = max(best, count_match * _position_factor(ctx, seat) * threat)
    return max(0.0, min(1.0, best))
