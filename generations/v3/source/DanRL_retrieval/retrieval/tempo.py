from __future__ import annotations

import math
from dataclasses import dataclass

from .cards import action_type_size
from .context import RetrievalContext
from .models import ActionCandidate, Partition
from .plm_rules import normalize_kind
from .scoring import control_by_kind, current_control_value


@dataclass(frozen=True)
class TempoFeatures:
    my_min_steps: float
    partner_min_steps: float
    opponent_min_steps_min: float
    opponent_short_pressure: float
    my_retake_count: float
    partner_follow_help: float
    must_block: float
    can_race: float


@dataclass(frozen=True)
class TempoContext:
    partner_min_steps: float
    opponent_min_steps_min: float
    opponent_short_pressure: float
    partner_follow_help: float
    must_block: float
    partner_seat: int
    last_player_is_opponent: bool
    last_player_is_partner: bool
    current_kind: str


def _partner_seat(ctx: RetrievalContext) -> int:
    return (ctx.my_seat + 2) % 4


def _is_opponent(ctx: RetrievalContext, seat: int) -> bool:
    return (seat % 2) != (ctx.my_seat % 2)


def _position_factor(ctx: RetrievalContext, seat: int) -> float:
    distance = (seat - ctx.my_seat) % 4
    return {1: 1.0, 2: 0.45, 3: 0.7}.get(distance, 0.0)


def _estimated_steps(card_count: int) -> float:
    if card_count <= 0:
        return 0.0
    # Unknown hands average between singles and 5-card structures. This is only
    # a pressure estimate; exact search is available only for our own hand.
    return max(1.0, math.ceil(card_count / 3.5))


def opponent_short_pressure(ctx: RetrievalContext) -> float:
    best = 0.0
    for seat, left in enumerate(ctx.public_counts):
        if seat == ctx.my_seat or not _is_opponent(ctx, seat) or left <= 0:
            continue
        if left == 1:
            base = 1.0
        elif left == 2:
            base = 0.82
        elif left <= 5:
            base = 0.56
        elif left <= 8:
            base = 0.28
        else:
            base = 0.0
        best = max(best, base * _position_factor(ctx, seat))
    return max(0.0, min(1.0, best))


def _opponent_min_steps(ctx: RetrievalContext) -> float:
    values = [
        _estimated_steps(left)
        for seat, left in enumerate(ctx.public_counts)
        if seat != ctx.my_seat and _is_opponent(ctx, seat) and left > 0
    ]
    return min(values) if values else 0.0


def _partner_steps(ctx: RetrievalContext) -> float:
    return _estimated_steps(ctx.public_counts[_partner_seat(ctx)])


def _retake_count(partition: Partition, ctx: RetrievalContext) -> float:
    kinds = ("Single", "Pair", "Triple", "TriplePlus", "Straight", "StraightPair", "StraightTriple", "Bomb")
    total = 0.0
    for kind in kinds:
        best = max((control_by_kind(group, ctx, kind) for group in partition.groups), default=0.0)
        if best >= 0.82:
            total += 1.0
        elif best >= 0.62:
            total += 0.55
        elif best >= 0.45:
            total += 0.25
    return total


def _partner_help(ctx: RetrievalContext) -> float:
    partner = _partner_seat(ctx)
    left = ctx.public_counts[partner]
    help_score = 0.0
    if left <= 2:
        help_score += 0.75
    elif left <= 5:
        help_score += 0.45
    elif left <= 8:
        help_score += 0.2
    if ctx.last_player == partner:
        help_score += 0.35
    return max(0.0, min(1.0, help_score))


def _must_block(ctx: RetrievalContext) -> float:
    kind = normalize_kind(ctx.current_kind)
    if kind == "Lead":
        return 0.0
    target_size = max(1, ctx.current_size or action_type_size(kind))
    best = 0.0
    for seat, left in enumerate(ctx.public_counts):
        if seat == ctx.my_seat or not _is_opponent(ctx, seat) or left <= 0:
            continue
        if ctx.last_player == seat and left <= target_size:
            best = max(best, 1.0)
        elif left <= target_size:
            best = max(best, 0.82 * _position_factor(ctx, seat))
        elif left <= target_size + 2:
            best = max(best, 0.52 * _position_factor(ctx, seat))
    return max(0.0, min(1.0, best))


def build_tempo_context(ctx: RetrievalContext) -> TempoContext:
    partner = _partner_seat(ctx)
    last_player = int(ctx.last_player) if ctx.last_player is not None else None
    return TempoContext(
        partner_min_steps=_partner_steps(ctx),
        opponent_min_steps_min=_opponent_min_steps(ctx),
        opponent_short_pressure=opponent_short_pressure(ctx),
        partner_follow_help=_partner_help(ctx),
        must_block=_must_block(ctx),
        partner_seat=partner,
        last_player_is_opponent=last_player is not None and _is_opponent(ctx, last_player),
        last_player_is_partner=last_player == partner,
        current_kind=normalize_kind(ctx.current_kind),
    )


def tempo_features(partition: Partition, ctx: RetrievalContext, retake_count: float | None = None) -> TempoFeatures:
    return tempo_features_from_context(partition, build_tempo_context(ctx), retake_count=retake_count)


def tempo_features_from_context(
    partition: Partition,
    tempo_ctx: TempoContext,
    retake_count: float | None = None,
    ctx: RetrievalContext | None = None,
) -> TempoFeatures:
    my_steps = float(partition.hand_count)
    if retake_count is None:
        if ctx is None:
            raise ValueError("ctx is required when retake_count is not provided")
        retake_count = _retake_count(partition, ctx)
    can_race = 1.0 if (my_steps <= 2 and retake_count >= 0.5) or (my_steps <= tempo_ctx.opponent_min_steps_min and retake_count >= 1.2) else 0.0
    return TempoFeatures(
        my_min_steps=my_steps,
        partner_min_steps=tempo_ctx.partner_min_steps,
        opponent_min_steps_min=tempo_ctx.opponent_min_steps_min,
        opponent_short_pressure=tempo_ctx.opponent_short_pressure,
        my_retake_count=retake_count,
        partner_follow_help=tempo_ctx.partner_follow_help,
        must_block=tempo_ctx.must_block,
        can_race=can_race,
    )


def tempo_action_value(
    action: ActionCandidate,
    partition: Partition,
    ctx: RetrievalContext,
    *,
    before_min_steps: float,
) -> tuple[float, TempoFeatures]:
    features = tempo_features(partition, ctx)
    return tempo_action_value_from_features(
        action,
        features,
        ctx,
        before_min_steps=before_min_steps,
        current_control=current_control_value(action, ctx) / 100.0,
    )


def tempo_action_value_from_features(
    action: ActionCandidate,
    features: TempoFeatures,
    ctx: RetrievalContext,
    *,
    before_min_steps: float,
    current_control: float,
) -> tuple[float, TempoFeatures]:
    return tempo_action_value_from_context(
        action,
        features,
        build_tempo_context(ctx),
        before_min_steps=before_min_steps,
        current_control=current_control,
    )


def tempo_action_value_from_context(
    action: ActionCandidate,
    features: TempoFeatures,
    tempo_ctx: TempoContext,
    *,
    before_min_steps: float,
    current_control: float,
) -> tuple[float, TempoFeatures]:
    kind = normalize_kind(action.kind)
    is_pass = kind == "PASS"
    value = 0.0

    progress = max(0.0, before_min_steps - features.my_min_steps)
    value += 70.0 * progress

    pressure = features.opponent_short_pressure
    if features.must_block > 0.0:
        if is_pass:
            value -= 360.0 * features.must_block
        else:
            value += 300.0 * features.must_block * max(0.25, current_control)

    if is_pass and pressure > 0.0 and tempo_ctx.last_player_is_opponent:
        value -= 210.0 * pressure
    elif not is_pass and pressure > 0.0 and tempo_ctx.current_kind != "Lead":
        value += 110.0 * pressure * current_control

    if features.can_race > 0.0 and not is_pass:
        value += 28.0 * min(6, action.size)
        if tempo_ctx.current_kind == "Lead":
            value += 55.0

    if is_pass and features.can_race > 0.0 and features.partner_follow_help < 0.35:
        value -= 80.0

    # If partner is currently winning this trick and can help, passing is often
    # a useful cooperation move rather than passivity.
    if is_pass and tempo_ctx.last_player_is_partner:
        value += 90.0 * features.partner_follow_help

    return value, features
