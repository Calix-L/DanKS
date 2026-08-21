from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from typing import Mapping

from .cards import NORMAL_RANKS, card_rank, rank_strength
from .context import RetrievalContext
from .models import ActionCandidate, CardGroup, Partition, effective_group_cost
from .rules import POINTS, can_follow_action, normalize_kind, plain_points
from .pressure import pressure


@dataclass(frozen=True)
class ScoreWeights:
    hand_count: float = 18.0
    card_value: float = 1.0
    retake: float = 6.0
    current_control: float = 16.0
    pass_pressure: float = 120.0
    bomb_spend: float = 360.0
    control_spend: float = 620.0
    teammate_overcall: float = 520.0
    break_group: float = 30.0
    low_break_preference: float = 18.0
    escape_risk: float = 260.0
    lead_action: float = 1.0
    residue: float = 20.0
    tempo: float = 1.0


@dataclass(frozen=True)
class BreakPenaltyProfile:
    name: str
    base_by_kind: Mapping[str, float]
    straight_flush_to_bomb: float
    recommended_group_weight: float
    group_value_base: Mapping[str, float]
    bomb_size_bonus: float
    bomb_break_size_bonus: float


RETAKE_KINDS = ("Single", "Pair", "Triple", "TriplePlus", "Straight", "StraightPair", "StraightTriple", "Bomb")
RETAKE_KIND_INDEX = {kind: idx for idx, kind in enumerate(RETAKE_KINDS)}

BREAK_BASE_BY_KIND = {
    "FourKings": 30.0,
    "StraightFlush": 28.0,
    "Bomb": 22.0,
    "StraightTriple": 17.0,
    "StraightPair": 15.0,
    "TriplePlus": 12.0,
    "Straight": 12.0,
    "Triple": 4.0,
    "Pair": 2.0,
    "Single": 0.0,
}
GROUP_VALUE_BASE = {
    "FourKings": 180.0,
    "StraightFlush": 130.0,
    "Bomb": 90.0,
    "StraightTriple": 40.0,
    "StraightPair": 34.0,
    "TriplePlus": 28.0,
    "Straight": 24.0,
    "Triple": 18.0,
    "Pair": 10.0,
    "Single": 4.0,
}
ASSET_TIER_GROUP_VALUE_BASE = {
    "FourKings": 175.0,
    "StraightFlush": 190.0,
    "Bomb": 130.0,
    "StraightTriple": 40.0,
    "StraightPair": 34.0,
    "TriplePlus": 28.0,
    "Straight": 24.0,
    "Triple": 18.0,
    "Pair": 10.0,
    "Single": 4.0,
}
DEFAULT_BREAK_PENALTY_PROFILE = BreakPenaltyProfile(
    name="default",
    base_by_kind=BREAK_BASE_BY_KIND,
    straight_flush_to_bomb=12.0,
    recommended_group_weight=30.0,
    group_value_base=GROUP_VALUE_BASE,
    bomb_size_bonus=14.0,
    bomb_break_size_bonus=2.0,
)
ASSET_TIER_BREAK_PENALTY_PROFILE = BreakPenaltyProfile(
    name="asset-tier-v1",
    base_by_kind={
        "FourKings": 36.0,
        "StraightFlush": 38.0,
        "Bomb": 30.0,
        "StraightTriple": 14.0,
        "StraightPair": 10.0,
        "TriplePlus": 10.0,
        "Straight": 8.0,
        "Triple": 4.0,
        "Pair": 2.0,
        "Single": 0.0,
    },
    straight_flush_to_bomb=28.0,
    recommended_group_weight=36.0,
    group_value_base=ASSET_TIER_GROUP_VALUE_BASE,
    bomb_size_bonus=32.0,
    bomb_break_size_bonus=5.0,
)
ASSET_TIER_V2_BREAK_PENALTY_PROFILE = replace(
    ASSET_TIER_BREAK_PENALTY_PROFILE,
    name="asset-tier-v2",
    recommended_group_weight=46.0,
)
BREAK_PENALTY_PROFILES = {
    DEFAULT_BREAK_PENALTY_PROFILE.name: DEFAULT_BREAK_PENALTY_PROFILE,
    ASSET_TIER_BREAK_PENALTY_PROFILE.name: ASSET_TIER_BREAK_PENALTY_PROFILE,
    ASSET_TIER_V2_BREAK_PENALTY_PROFILE.name: ASSET_TIER_V2_BREAK_PENALTY_PROFILE,
}
BREAK_SEQUENCE_KINDS = {"Straight", "StraightPair", "StraightTriple", "StraightFlush"}
BOMB_KINDS = {"Bomb", "StraightFlush", "FourKings"}


def group_value(
    group: CardGroup,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    base = profile.group_value_base.get(group.kind, 0.0)
    size_bonus = max(0, group.size - 4) * profile.bomb_size_bonus if group.kind == "Bomb" else 0.0
    return base + size_bonus + group.strength * 12.0


def card_value(
    partition: Partition,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    return sum(group_value(group, profile) for group in partition.groups)


def residue_penalty(partition: Partition, ctx: RetrievalContext) -> float:
    """Debt carried by fragmented after-hand structures.

    Hand count alone treats every group equally. In practice, several isolated
    low/mid singles are much worse than one cohesive group because they need
    future leads or control cards to clear. High-control singles are not treated
    as real residue; they are still useful for retaking.
    """

    debt = 0.0
    single_debt = 0.0
    single_count = 0
    low_single_count = 0
    for group in partition.groups:
        kind = normalize_kind(group.kind)
        if kind != "Single":
            continue
        single_count += 1
        rank = group.rank or card_rank(group.cards[0])
        strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail)
        if strength >= 0.92:
            item = 0.08
        elif strength >= 0.78:
            item = 0.28
        elif strength >= 0.62:
            item = 0.65
        else:
            item = 1.05 + (0.62 - strength) * 0.75
            low_single_count += 1
        single_debt += item

    if single_count >= 3:
        single_debt += (single_count - 2) * 0.35
    if low_single_count >= 2:
        single_debt += (low_single_count - 1) * 0.45

    # Too many fragments means the hand needs too many future entries even if
    # some fragments are valuable. Keep this weaker than the explicit singles
    # debt so bombs/straight flushes are not punished just for being separate.
    debt += max(0, partition.hand_count - 7) * 0.22
    debt += single_debt
    return debt


def _unknown_total(ctx: RetrievalContext) -> int:
    if ctx.remaining_by_rank:
        return max(1, sum(max(0, int(count)) for count in ctx.remaining_by_rank.values()))
    return 54


def _higher_unknown_count(ctx: RetrievalContext, rank: str | None, *, use_level: bool) -> int:
    if not rank or not ctx.remaining_by_rank:
        return 0
    out = 0
    for other, count in ctx.remaining_by_rank.items():
        if count <= 0:
            continue
        if use_level:
            if rank_strength(other, ctx.cur_rank, ctx.remaining_detail) > rank_strength(rank, ctx.cur_rank, ctx.remaining_detail):
                out += int(count)
        elif plain_points(other) > plain_points(rank):
            out += int(count)
    return out


def _same_kind_control(group: CardGroup, ctx: RetrievalContext, target_kind: str) -> float:
    kind = normalize_kind(group.kind)
    if kind != normalize_kind(target_kind):
        return 0.0
    if kind in {"Single", "Pair", "Triple", "TriplePlus"}:
        base = rank_strength(group.rank, ctx.cur_rank, ctx.remaining_detail)
        higher = _higher_unknown_count(ctx, group.rank, use_level=True)
    elif kind in {"Straight", "StraightPair", "StraightTriple"}:
        base = plain_points(group.rank) / max(1, POINTS["A"])
        higher = _higher_unknown_count(ctx, group.rank, use_level=False)
    elif kind in BOMB_KINDS:
        base = 0.62 + min(0.34, max(0, group.size - 4) * 0.08)
        if kind == "StraightFlush":
            base = 0.74 + plain_points(group.rank) / 100.0
        if kind == "FourKings":
            base = 1.0
        higher = 0
    else:
        return 0.0
    scarcity = 1.0 - min(0.75, higher / _unknown_total(ctx))
    return 0.35 * base + 0.65 * scarcity


def _bomb_cover_control(group: CardGroup, target_kind: str) -> float:
    kind = normalize_kind(group.kind)
    target_kind = normalize_kind(target_kind)
    if target_kind in BOMB_KINDS:
        return 0.0
    if kind == "FourKings":
        return 0.96
    if kind == "StraightFlush":
        return 0.84
    if kind == "Bomb":
        return 0.70 + min(0.18, max(0, group.size - 4) * 0.06)
    return 0.0


def _current_trick_control(group: CardGroup, ctx: RetrievalContext) -> float:
    kind = normalize_kind(ctx.current_kind)
    if kind == "Lead" or not ctx.current_rank:
        return 0.0
    target = ActionCandidate(index=-1, kind=kind, cards=tuple("__" for _ in range(ctx.current_size or group.size)), rank=ctx.current_rank)
    if not can_follow_action(group, target, ctx.cur_rank):
        return 0.0
    control = _same_kind_control(group, ctx, kind)
    return max(control, _bomb_cover_control(group, kind))


def control_by_kind(group: CardGroup, ctx: RetrievalContext, target_kind: str) -> float:
    target_kind = normalize_kind(target_kind)
    if ctx.current_rank and target_kind == normalize_kind(ctx.current_kind):
        return _current_trick_control(group, ctx)
    return max(_same_kind_control(group, ctx, target_kind), _bomb_cover_control(group, target_kind))


def retake_value(partition: Partition, ctx: RetrievalContext, primary_kind: str | None = None) -> float:
    kinds = ["Single", "Pair", "Triple", "TriplePlus", "Straight", "StraightPair", "StraightTriple", "Bomb"]
    total = 0.0
    for kind in kinds:
        p = pressure(ctx, kind)
        if primary_kind and kind == primary_kind:
            p = max(p, 0.55)
        best_control = max((control_by_kind(group, ctx, kind) for group in partition.groups), default=0.0)
        total += p * best_control
    return total * 100.0


def current_control_value(action: ActionCandidate, ctx: RetrievalContext) -> float:
    """Control value of the current action in the current trick dimension."""

    if action.kind == "PASS" or not action.cards:
        return 0.0
    if normalize_kind(ctx.current_kind) == "Lead":
        return 0.0
    kind = ctx.current_kind if action.kind in BOMB_KINDS else action.kind
    p = max(pressure(ctx, kind), 0.35 if kind == ctx.current_kind else 0.0)
    if action.rank:
        strength = rank_strength(action.rank, ctx.cur_rank, ctx.remaining_detail)
    else:
        strength = max(
            (rank_strength(card_rank(card), ctx.cur_rank, ctx.remaining_detail) for card in action.cards),
            default=0.0,
        )
    if action.kind in BOMB_KINDS:
        strength = max(strength, 0.75)
    return p * strength * 100.0


def lead_action_value(action: ActionCandidate, ctx: RetrievalContext) -> float:
    """Lead-only value for proactively starting a complete hand structure."""

    if normalize_kind(ctx.current_kind) != "Lead" or action.kind == "PASS" or not action.cards:
        return 0.0
    kind = normalize_kind(action.kind)
    if kind == "Single":
        strength = rank_strength(action.rank, ctx.cur_rank, ctx.remaining_detail) if action.rank else 0.0
        return 8.0 + (1.0 - strength) * 35.0

    cards_released = max(0, action.size - 1)
    tempo = 95.0 * cards_released
    group = CardGroup(kind=kind, cards=action.cards, rank=action.rank)
    intrinsic = group_value(group)
    structure_multiplier = {
        "Pair": 32.0,
        "Triple": 14.0,
        "TriplePlus": 10.0,
        "Straight": 12.0,
        "StraightPair": 13.0,
        "StraightTriple": 14.0,
        "StraightFlush": 1.2,
        "Bomb": 0.8,
        "FourKings": 0.4,
    }.get(kind, 0.0)
    return tempo + intrinsic * structure_multiplier


def action_spend_penalty(action: ActionCandidate, ctx: RetrievalContext) -> float:
    """Cost for spending scarce assets in the current action."""

    if action.kind == "PASS" or not action.cards:
        return 0.0
    if action.kind == "FourKings":
        return 2.5
    if action.kind == "StraightFlush":
        return 1.8
    if action.kind == "Bomb":
        return 1.0 + max(0, action.size - 4) * 0.25
    if action.kind == "Single":
        strength = rank_strength(action.rank, ctx.cur_rank, ctx.remaining_detail) if action.rank else 0.0
        return max(0.0, strength - 0.72)
    if action.kind == "Pair":
        strength = rank_strength(action.rank, ctx.cur_rank, ctx.remaining_detail) if action.rank else 0.0
        return max(0.0, strength - 0.78) * 1.2
    return 0.0


def _counter_subset(left: Counter[str], right: Counter[str]) -> bool:
    return all(count <= right[card] for card, count in left.items())


def _counter_overlap(left: Counter[str], right: Counter[str]) -> int:
    return sum(min(count, right.get(card, 0)) for card, count in left.items())


def _break_severity(
    group: CardGroup,
    action: ActionCandidate,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    if action.kind == "PASS" or not action.cards:
        return 0.0
    group_counter = Counter(group.cards)
    action_counter = Counter(action.cards)
    return _break_severity_from_counters(group, action, group_counter, action_counter, profile)


def _break_severity_from_counters(
    group: CardGroup,
    action: ActionCandidate,
    group_counter: Counter[str],
    action_counter: Counter[str],
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    overlap = _counter_overlap(action_counter, group_counter)
    if overlap <= 0:
        return 0.0
    group_kind = normalize_kind(group.kind)
    action_kind = normalize_kind(action.kind)
    same_cards = overlap == group.size == action.size
    if same_cards and action_kind == group_kind:
        return 0.0
    if group_kind == "StraightFlush" and action_kind == "Bomb":
        return profile.straight_flush_to_bomb
    if same_cards:
        return 0.15
    base = profile.base_by_kind.get(group_kind, 0.0)
    if group_kind == "Bomb":
        base += (group.size - 4) * profile.bomb_break_size_bonus
    if group_kind in BREAK_SEQUENCE_KINDS:
        base += (group.size - 5) * 1.6
    fraction = overlap / group.size
    return base * (0.55 + 0.45 * fraction)


def _break_severity_from_precomputed(
    group_kind: str,
    group_size: int,
    group_counter: dict[str, int] | Counter[str],
    action_kind: str,
    action_size: int,
    action_items: tuple[tuple[str, int], ...],
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    overlap = 0
    for card, count in action_items:
        other = group_counter.get(card, 0)
        overlap += count if count <= other else other
    if overlap <= 0:
        return 0.0
    same_cards = overlap == group_size == action_size
    if same_cards and action_kind == group_kind:
        return 0.0
    if group_kind == "StraightFlush" and action_kind == "Bomb":
        return profile.straight_flush_to_bomb
    if same_cards:
        return 0.15
    base = profile.base_by_kind.get(group_kind, 0.0)
    if group_kind == "Bomb":
        base += (group_size - 4) * profile.bomb_break_size_bonus
    if group_kind in BREAK_SEQUENCE_KINDS:
        base += (group_size - 5) * 1.6
    fraction = overlap / group_size
    return base * (0.55 + 0.45 * fraction)


def break_group_penalty(
    action: ActionCandidate,
    before_partitions,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> float:
    """Penalty for playing a strict subset of a valuable current partition group.

    We evaluate several plausible current partitions and keep the strongest
    break signal. If one strong interpretation says this action breaks a
    valuable group, the action should pay for that structural damage.
    """

    if action.kind == "PASS" or not action.cards or not before_partitions:
        return 0.0
    candidates: list[float] = []
    action_counter = Counter(action.cards)
    for partition in before_partitions:
        touched = [
            (group, group_counter)
            for group in partition.groups
            for group_counter in (Counter(group.cards),)
            if _counter_overlap(action_counter, group_counter) > 0
        ]
        if not touched:
            candidates.append(0.0)
            continue
        candidates.append(
            max(
                _break_severity_from_counters(group, action, group_counter, action_counter, profile)
                for group, group_counter in touched
            )
        )
    return max(candidates) if candidates else 0.0


def break_group_penalty_table(
    actions: list[ActionCandidate],
    before_partitions,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> dict[int, float]:
    """Exact break penalties for many actions from one pass over partitions."""

    active, penalties = prepare_break_group_penalty_table(actions)
    if not active:
        return penalties

    for partition in before_partitions:
        update_break_group_penalty_table(active, penalties, partition, profile=profile)
    return penalties


def break_group_penalty_indexed(
    actions: list[ActionCandidate],
    before_partitions,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> dict[int, float]:
    """Exact break penalties using a card-to-action reverse index."""

    active_by_card, penalties = prepare_break_group_penalty_index(actions)
    if not active_by_card:
        return penalties
    group_counter_cache: dict[int, dict[str, int]] = {}
    for partition in before_partitions:
        update_break_group_penalty_index(
            active_by_card,
            penalties,
            partition,
            group_counter_cache,
            profile,
        )
    return penalties


def prepare_break_group_penalty_table(actions: list[ActionCandidate]) -> tuple[list[tuple[int, ActionCandidate, Counter[str]]], dict[int, float]]:
    active: list[tuple[int, ActionCandidate, Counter[str]]] = []
    penalties: dict[int, float] = {}
    for action in actions:
        penalties[action.index] = 0.0
        if action.kind == "PASS" or not action.cards:
            continue
        active.append((action.index, action, Counter(action.cards)))
    return active, penalties


def prepare_break_group_penalty_index(
    actions: list[ActionCandidate],
) -> tuple[dict[str, list[tuple[int, str, int, tuple[tuple[str, int], ...]]]], dict[int, float]]:
    active_by_card: dict[str, list[tuple[int, str, int, tuple[tuple[str, int], ...]]]] = {}
    penalties: dict[int, float] = {}
    for action in actions:
        penalties[action.index] = 0.0
        if action.kind == "PASS" or not action.cards:
            continue
        action_counter = Counter(action.cards)
        entry = (
            action.index,
            normalize_kind(action.kind),
            action.size,
            tuple(action_counter.items()),
        )
        for card in action_counter:
            active_by_card.setdefault(card, []).append(entry)
    return active_by_card, penalties


def update_break_group_penalty_table(
    active: list[tuple[int, ActionCandidate, Counter[str]]],
    penalties: dict[int, float],
    partition: Partition,
    group_counter_cache: dict[int, Counter[str]] | None = None,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> None:
    if not active:
        return
    group_counters: list[tuple[CardGroup, Counter[str]]] = []
    for group in partition.groups:
        key = id(group)
        if group_counter_cache is None:
            group_counter = Counter(group.cards)
        else:
            group_counter = group_counter_cache.get(key)
            if group_counter is None:
                group_counter = Counter(group.cards)
                group_counter_cache[key] = group_counter
        group_counters.append((group, group_counter))
    for action_index, action, action_counter in active:
        best = 0.0
        for group, group_counter in group_counters:
            if _counter_overlap(action_counter, group_counter) <= 0:
                continue
            best = max(best, _break_severity_from_counters(group, action, group_counter, action_counter, profile))
        if best > penalties[action_index]:
            penalties[action_index] = best


def update_break_group_penalty_index(
    active_by_card: dict[str, list[tuple[int, str, int, tuple[tuple[str, int], ...]]]],
    penalties: dict[int, float],
    partition: Partition,
    group_counter_cache: dict[int, dict[str, int]] | None = None,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> None:
    if not active_by_card:
        return
    for group in partition.groups:
        update_break_group_penalty_for_group(active_by_card, penalties, group, group_counter_cache, profile=profile)


def break_group_penalty_from_groups(
    actions: list[ActionCandidate],
    groups: list[CardGroup],
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> dict[int, float]:
    """Exact break penalties from the generated current-hand group universe.

    In full-search mode every generated group can be extended to an exact cover
    because remaining physical cards can always be covered by Single groups.
    Therefore the max break signal over all exact-cover partitions is identical
    to the max break signal over all generated groups, without iterating every
    partition just to compute this penalty.
    """

    active_by_card, penalties = prepare_break_group_penalty_index(actions)
    if not active_by_card:
        return penalties
    seen_action_marks: dict[int, int] = {}
    stamp = 0
    for group in groups:
        if normalize_kind(group.kind) == "Single":
            continue
        stamp += 1
        update_break_group_penalty_for_group(
            active_by_card,
            penalties,
            group,
            seen_action_marks=seen_action_marks,
            stamp=stamp,
            profile=profile,
        )
    return penalties


def update_break_group_penalty_for_group(
    active_by_card: dict[str, list[tuple[int, str, int, tuple[tuple[str, int], ...]]]],
    penalties: dict[int, float],
    group: CardGroup,
    group_counter_cache: dict[int, dict[str, int]] | None = None,
    seen_action_marks: dict[int, int] | None = None,
    stamp: int = 0,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> None:
    if normalize_kind(group.kind) == "Single":
        return
    if group_counter_cache is None:
        group_counter: dict[str, int] = {}
        for card in group.cards:
            group_counter[card] = group_counter.get(card, 0) + 1
    else:
        key = id(group)
        group_counter = group_counter_cache.get(key)
        if group_counter is None:
            group_counter = {}
            for card in group.cards:
                group_counter[card] = group_counter.get(card, 0) + 1
            group_counter_cache[key] = group_counter

    group_kind = normalize_kind(group.kind)
    group_size = group.size
    local_seen: set[int] | None = None if seen_action_marks is not None else set()
    for card in group_counter:
        for action_index, action_kind, action_size, action_items in active_by_card.get(card, ()):
            if seen_action_marks is not None:
                if seen_action_marks.get(action_index) == stamp:
                    continue
                seen_action_marks[action_index] = stamp
            else:
                assert local_seen is not None
                if action_index in local_seen:
                    continue
                local_seen.add(action_index)
            severity = _break_severity_from_precomputed(
                group_kind,
                group_size,
                group_counter,
                action_kind,
                action_size,
                action_items,
                profile,
            )
            if severity > penalties[action_index]:
                penalties[action_index] = severity


def pass_pressure_penalty(ctx: RetrievalContext) -> float:
    """Penalty for passing when the current trick type creates short-hand pressure."""

    if not ctx.current_kind or ctx.current_kind == "Lead":
        return 0.0
    return pressure(ctx, ctx.current_kind)


def short_opponent_pressure(ctx: RetrievalContext) -> float:
    """Pressure to avoid letting an opponent with 1-2 cards escape."""

    best = 0.0
    for seat, left in enumerate(ctx.public_counts):
        if seat == ctx.my_seat or left <= 0:
            continue
        if (seat % 2) == (ctx.my_seat % 2):
            continue
        if left == 1:
            best = max(best, 1.0)
        elif left == 2:
            best = max(best, 0.75)
        elif left <= 4:
            best = max(best, 0.35)
    return best


def escape_risk_penalty(action: ActionCandidate, ctx: RetrievalContext) -> float:
    """Penalty for passing or making a low-control play while opponents are short."""

    return escape_risk_penalty_from_current_control(action, ctx, current_control_value(action, ctx))


def escape_risk_penalty_from_current_control(
    action: ActionCandidate,
    ctx: RetrievalContext,
    current_control_score: float,
) -> float:
    """Same as escape_risk_penalty when current_control_value was already computed."""

    p = short_opponent_pressure(ctx)
    if p <= 0.0:
        return 0.0
    if action.kind == "PASS" or not action.cards:
        return p
    control = current_control_score / 100.0
    if normalize_kind(ctx.current_kind) == "Lead":
        needed = 0.55
    else:
        needed = 0.45
    return p * max(0.0, needed - control)


def teammate_overcall_penalty(action: ActionCandidate, ctx: RetrievalContext) -> float:
    """Cost for beating a trick currently held by partner."""

    if action.kind == "PASS" or not ctx.current_kind or ctx.current_kind == "Lead":
        return 0.0
    if ctx.last_player is None:
        return 0.0
    teammate = (ctx.my_seat + 2) % 4
    if int(ctx.last_player) != teammate:
        return 0.0
    if action.kind in BOMB_KINDS:
        return 1.25
    return 1.0


def score_partition(
    partition: Partition,
    ctx: RetrievalContext,
    weights: ScoreWeights = ScoreWeights(),
    primary_kind: str | None = None,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> tuple[float, dict[str, float]]:
    return PartitionScorer(ctx, weights, profile).score(partition, primary_kind)


class PartitionScorer:
    """Cache per-group scoring terms while evaluating many partitions."""

    def __init__(
        self,
        ctx: RetrievalContext,
        weights: ScoreWeights = ScoreWeights(),
        profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
    ) -> None:
        self.ctx = ctx
        self.weights = weights
        self.profile = profile
        self.current_kind = normalize_kind(ctx.current_kind)
        self.has_current_rank = bool(ctx.current_rank)
        self.current_target = (
            ActionCandidate(
                index=-1,
                kind=self.current_kind,
                cards=tuple("__" for _ in range(ctx.current_size)),
                rank=ctx.current_rank,
            )
            if self.current_kind != "Lead" and ctx.current_rank and ctx.current_size
            else None
        )
        self.unknown_total = _unknown_total(ctx)
        self.higher_unknown_cache: dict[tuple[str | None, bool], int] = {}
        self.pressure_values = tuple(pressure(ctx, kind) for kind in RETAKE_KINDS)
        self.primary_pressure_cache: dict[str | None, tuple[float, ...]] = {None: self.pressure_values}
        self.pressure_by_kind = dict(zip(RETAKE_KINDS, self.pressure_values))
        self._native_weight_values = [weights.hand_count, weights.card_value, weights.retake, weights.residue]
        self.group_score_cache: dict[tuple, tuple[float, int, int, float, tuple[tuple[int, float], ...]]] = {}
        self.group_value_cache: dict[tuple, float] = {}
        self.group_residue_cache: dict[tuple, tuple[int, int, float]] = {}
        self.group_control_cache: dict[tuple, tuple[float, ...]] = {}
        self.flat_cover_score_entry_cache: dict[tuple, list[float | int]] = {}
        self.rank_strength_cache: dict[str | None, float] = {}

    @staticmethod
    def _group_cache_key(group: CardGroup) -> tuple:
        meta = group.meta or {}
        cached_key = meta.get("__score_key")
        if cached_key is not None:
            return cached_key
        return (
            group.kind,
            group.rank,
            group.size,
            group.strength,
        )

    def _higher_unknown_count_cached(self, rank: str | None, *, use_level: bool) -> int:
        key = (rank, use_level)
        cached = self.higher_unknown_cache.get(key)
        if cached is None:
            cached = _higher_unknown_count(self.ctx, rank, use_level=use_level)
            self.higher_unknown_cache[key] = cached
        return cached

    def _rank_strength_cached(self, rank: str | None) -> float:
        cached = self.rank_strength_cache.get(rank)
        if cached is None:
            cached = rank_strength(rank, self.ctx.cur_rank, self.ctx.remaining_detail) if rank else 0.0
            self.rank_strength_cache[rank] = cached
        return cached

    def _pressure_values_for_primary(self, primary_kind: str | None) -> tuple[float, ...]:
        cached = self.primary_pressure_cache.get(primary_kind)
        if cached is not None:
            return cached
        values = list(self.pressure_values)
        idx = RETAKE_KIND_INDEX.get(primary_kind)
        if idx is not None and values[idx] < 0.55:
            values[idx] = 0.55
        cached = tuple(values)
        self.primary_pressure_cache[primary_kind] = cached
        return cached

    def _same_kind_control_normalized(self, group: CardGroup, group_kind: str, target_kind: str) -> float:
        if group_kind != target_kind:
            return 0.0
        ctx = self.ctx
        if group_kind in {"Single", "Pair", "Triple", "TriplePlus"}:
            base = self._rank_strength_cached(group.rank)
            higher = self._higher_unknown_count_cached(group.rank, use_level=True)
        elif group_kind in {"Straight", "StraightPair", "StraightTriple"}:
            base = plain_points(group.rank) / max(1, POINTS["A"])
            higher = self._higher_unknown_count_cached(group.rank, use_level=False)
        elif group_kind in BOMB_KINDS:
            base = 0.62 + min(0.34, max(0, group.size - 4) * 0.08)
            if group_kind == "StraightFlush":
                base = 0.74 + plain_points(group.rank) / 100.0
            if group_kind == "FourKings":
                base = 1.0
            higher = 0
        else:
            return 0.0
        scarcity = 1.0 - min(0.75, higher / self.unknown_total)
        return 0.35 * base + 0.65 * scarcity

    @staticmethod
    def _bomb_cover_control_normalized(group: CardGroup, group_kind: str, target_kind: str) -> float:
        if target_kind in BOMB_KINDS:
            return 0.0
        if group_kind == "FourKings":
            return 0.96
        if group_kind == "StraightFlush":
            return 0.84
        if group_kind == "Bomb":
            return 0.70 + min(0.18, max(0, group.size - 4) * 0.06)
        return 0.0

    def _current_trick_control_cached(self, group: CardGroup, group_kind: str) -> float:
        if self.current_kind == "Lead" or not self.ctx.current_rank:
            return 0.0
        current_kind = self.current_kind
        group_is_bomb = group_kind in BOMB_KINDS
        current_is_bomb = current_kind in BOMB_KINDS
        if not group_is_bomb and group_kind != current_kind:
            return 0.0
        if current_is_bomb and not group_is_bomb:
            return 0.0
        if group_is_bomb and not current_is_bomb:
            return self._bomb_cover_control_normalized(group, group_kind, current_kind)
        target = self.current_target
        if target is None:
            target = ActionCandidate(
                index=-1,
                kind=current_kind,
                cards=tuple("__" for _ in range(self.ctx.current_size or group.size)),
                rank=self.ctx.current_rank,
            )
        if not can_follow_action(group, target, self.ctx.cur_rank):
            return 0.0
        control = self._same_kind_control_normalized(group, group_kind, current_kind)
        bomb_control = self._bomb_cover_control_normalized(group, group_kind, current_kind)
        return control if control >= bomb_control else bomb_control

    def _controls_for_group(self, group: CardGroup) -> tuple[float, ...]:
        group_kind = normalize_kind(group.kind)
        current_idx = RETAKE_KIND_INDEX.get(self.current_kind, -1) if self.has_current_rank else -1
        single = pair = triple = triple_plus = straight = straight_pair = straight_triple = bomb = 0.0

        if group_kind == "Single" and current_idx != 0:
            single = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "Pair" and current_idx != 1:
            pair = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "Triple" and current_idx != 2:
            triple = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "TriplePlus" and current_idx != 3:
            triple_plus = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "Straight" and current_idx != 4:
            straight = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "StraightPair" and current_idx != 5:
            straight_pair = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "StraightTriple" and current_idx != 6:
            straight_triple = self._same_kind_control_normalized(group, group_kind, group_kind)
        elif group_kind == "Bomb" and current_idx != 7:
            bomb = self._same_kind_control_normalized(group, group_kind, group_kind)

        if group_kind in BOMB_KINDS:
            cover = self._bomb_cover_control_normalized(group, group_kind, "Single")
            if current_idx != 0:
                single = max(single, cover)
            if current_idx != 1:
                pair = max(pair, cover)
            if current_idx != 2:
                triple = max(triple, cover)
            if current_idx != 3:
                triple_plus = max(triple_plus, cover)
            if current_idx != 4:
                straight = max(straight, cover)
            if current_idx != 5:
                straight_pair = max(straight_pair, cover)
            if current_idx != 6:
                straight_triple = max(straight_triple, cover)

        if current_idx >= 0:
            current_control = self._current_trick_control_cached(group, group_kind)
            if current_idx == 0:
                single = current_control
            elif current_idx == 1:
                pair = current_control
            elif current_idx == 2:
                triple = current_control
            elif current_idx == 3:
                triple_plus = current_control
            elif current_idx == 4:
                straight = current_control
            elif current_idx == 5:
                straight_pair = current_control
            elif current_idx == 6:
                straight_triple = current_control
            elif current_idx == 7:
                bomb = current_control

        return (single, pair, triple, triple_plus, straight, straight_pair, straight_triple, bomb)

    def _group_value(self, group: CardGroup) -> float:
        key = self._group_cache_key(group)
        cached = self.group_value_cache.get(key)
        if cached is None:
            cached = group_value(group, self.profile)
            self.group_value_cache[key] = cached
        return cached

    def _group_residue(self, group: CardGroup) -> tuple[int, int, float]:
        key = self._group_cache_key(group)
        cached = self.group_residue_cache.get(key)
        if cached is not None:
            return cached
        if group.kind != "Single":
            cached = (0, 0, 0.0)
        else:
            rank = group.rank or card_rank(group.cards[0])
            strength = self._rank_strength_cached(rank)
            if strength >= 0.92:
                item = 0.08
                low = 0
            elif strength >= 0.78:
                item = 0.28
                low = 0
            elif strength >= 0.62:
                item = 0.65
                low = 0
            else:
                item = 1.05 + (0.62 - strength) * 0.75
                low = 1
            cached = (1, low, item)
        self.group_residue_cache[key] = cached
        return cached

    def _group_controls(self, group: CardGroup) -> tuple[float, ...]:
        key = self._group_cache_key(group)
        cached = self.group_control_cache.get(key)
        if cached is None:
            cached = self._controls_for_group(group)
            self.group_control_cache[key] = cached
        return cached

    def group_score_entry(self, group: CardGroup, key: tuple | None = None) -> tuple[float, int, int, float, tuple[tuple[int, float], ...]]:
        if key is None:
            key = self._group_cache_key(group)
        cached = self.group_score_cache.get(key)
        if cached is not None:
            return cached
        ctx = self.ctx
        cached_value = group_value(group, self.profile)
        if group.kind != "Single":
            group_single_count = 0
            group_low_count = 0
            group_debt = 0.0
        else:
            rank = group.rank or card_rank(group.cards[0])
            strength = self._rank_strength_cached(rank)
            if strength >= 0.92:
                item = 0.08
                low = 0
            elif strength >= 0.78:
                item = 0.28
                low = 0
            elif strength >= 0.62:
                item = 0.65
                low = 0
            else:
                item = 1.05 + (0.62 - strength) * 0.75
                low = 1
            group_single_count = 1
            group_low_count = low
            group_debt = item
        controls = self._controls_for_group(group)
        nonzero: list[tuple[int, float]] = []
        for idx, control in enumerate(controls):
            if control > 0.0:
                nonzero.append((idx, control))
        nonzero_controls = tuple(nonzero)
        cached = (cached_value, group_single_count, group_low_count, group_debt, nonzero_controls)
        self.group_score_cache[key] = cached
        return cached

    def cover_score_entry(self, group: CardGroup, key: tuple | None = None) -> tuple[float, int, int, float, int, int, float, int, tuple[tuple[int, float], ...]]:
        cached_value, group_single_count, group_low_count, group_debt, nonzero_controls = self.group_score_entry(group, key)
        nonzero_count = len(nonzero_controls)
        if nonzero_count == 1:
            idx, control = nonzero_controls[0]
        else:
            idx = -1
            control = 0.0
        return (
            cached_value,
            group_single_count,
            group_low_count,
            group_debt,
            nonzero_count,
            idx,
            control,
            effective_group_cost(group),
            nonzero_controls,
        )

    def flat_cover_score_entry(self, group: CardGroup) -> list[float | int]:
        key = self._group_cache_key(group)
        cached = self.flat_cover_score_entry_cache.get(key)
        if cached is not None:
            return cached
        (
            cached_value,
            group_single_count,
            group_low_count,
            group_debt,
            nonzero_count,
            idx,
            control,
            group_effective_cost,
            nonzero_controls,
        ) = self.cover_score_entry(group, key)
        entry: list[float | int] = [
            cached_value,
            group_single_count,
            group_low_count,
            group_debt,
            nonzero_count,
            idx,
            control,
            group_effective_cost,
        ]
        if nonzero_count > 1:
            for control_idx, item_control in nonzero_controls:
                entry.extend((control_idx, item_control))
        self.flat_cover_score_entry_cache[key] = entry
        return entry

    def flat_cover_score_entries(self, groups: list[CardGroup] | tuple[CardGroup, ...]) -> list[list[float | int]]:
        return [self.flat_cover_score_entry(group) for group in groups]

    def native_weight_values(self) -> list[float]:
        return self._native_weight_values

    def retake_count_value(self, partition: Partition) -> float:
        best_single = 0.0
        best_pair = 0.0
        best_triple = 0.0
        best_triple_plus = 0.0
        best_straight = 0.0
        best_straight_pair = 0.0
        best_straight_triple = 0.0
        best_bomb = 0.0
        group_score_cache = self.group_score_cache
        for group in partition.groups:
            cached = group_score_cache.get(self._group_cache_key(group))
            nonzero_controls = cached[4] if cached is not None else self.group_score_entry(group)[4]
            nonzero_count = len(nonzero_controls)
            if nonzero_count == 1:
                idx, control = nonzero_controls[0]
                if idx == 0:
                    if control > best_single:
                        best_single = control
                elif idx == 1:
                    if control > best_pair:
                        best_pair = control
                elif idx == 2:
                    if control > best_triple:
                        best_triple = control
                elif idx == 3:
                    if control > best_triple_plus:
                        best_triple_plus = control
                elif idx == 4:
                    if control > best_straight:
                        best_straight = control
                elif idx == 5:
                    if control > best_straight_pair:
                        best_straight_pair = control
                elif idx == 6:
                    if control > best_straight_triple:
                        best_straight_triple = control
                elif control > best_bomb:
                    best_bomb = control
            elif nonzero_count:
                for idx, control in nonzero_controls:
                    if idx == 0:
                        if control > best_single:
                            best_single = control
                    elif idx == 1:
                        if control > best_pair:
                            best_pair = control
                    elif idx == 2:
                        if control > best_triple:
                            best_triple = control
                    elif idx == 3:
                        if control > best_triple_plus:
                            best_triple_plus = control
                    elif idx == 4:
                        if control > best_straight:
                            best_straight = control
                    elif idx == 5:
                        if control > best_straight_pair:
                            best_straight_pair = control
                    elif idx == 6:
                        if control > best_straight_triple:
                            best_straight_triple = control
                    elif control > best_bomb:
                        best_bomb = control

        total = 0.0
        for best in (
            best_single,
            best_pair,
            best_triple,
            best_triple_plus,
            best_straight,
            best_straight_pair,
            best_straight_triple,
            best_bomb,
        ):
            if best >= 0.82:
                total += 1.0
            elif best >= 0.62:
                total += 0.55
            elif best >= 0.45:
                total += 0.25
        return total

    def score(self, partition: Partition, primary_kind: str | None = None) -> tuple[float, dict[str, float]]:
        total, hand_count_score, card_value_score, retake_score, residue_score = self.score_values(partition, primary_kind)
        return total, self.parts_from_values(hand_count_score, card_value_score, retake_score, residue_score)

    @staticmethod
    def parts_from_values(
        hand_count_score: float,
        card_value_score: float,
        retake_score: float,
        residue_score: float,
    ) -> dict[str, float]:
        return {
            "hand_count_score": hand_count_score,
            "card_value_score": card_value_score,
            "retake_score": retake_score,
            "residue_penalty_score": residue_score,
        }

    def score_values(
        self,
        partition: Partition,
        primary_kind: str | None = None,
        pressure_values: tuple[float, ...] | None = None,
    ) -> tuple[float, float, float, float, float]:
        return self.score_group_values(partition.groups, primary_kind, pressure_values)

    def score_group_values(
        self,
        groups: tuple[CardGroup, ...],
        primary_kind: str | None = None,
        pressure_values: tuple[float, ...] | None = None,
    ) -> tuple[float, float, float, float, float]:
        weights = self.weights
        hand_count = len(groups)
        effective_hand_count = sum(effective_group_cost(group) for group in groups)
        hand_count_score = -float(effective_hand_count)
        card_value_score = 0.0
        single_debt = 0.0
        single_count = 0
        low_single_count = 0
        best_single = 0.0
        best_pair = 0.0
        best_triple = 0.0
        best_triple_plus = 0.0
        best_straight = 0.0
        best_straight_pair = 0.0
        best_straight_triple = 0.0
        best_bomb = 0.0
        group_score_cache = self.group_score_cache
        ctx = self.ctx

        for group in groups:
            cached = self.group_score_entry(group)
            cached_value, group_single_count, group_low_count, group_debt, nonzero_controls = cached
            card_value_score += cached_value
            single_count += group_single_count
            low_single_count += group_low_count
            single_debt += group_debt

            nonzero_count = len(nonzero_controls)
            if nonzero_count == 1:
                idx, control = nonzero_controls[0]
                if idx == 0:
                    if control > best_single:
                        best_single = control
                elif idx == 1:
                    if control > best_pair:
                        best_pair = control
                elif idx == 2:
                    if control > best_triple:
                        best_triple = control
                elif idx == 3:
                    if control > best_triple_plus:
                        best_triple_plus = control
                elif idx == 4:
                    if control > best_straight:
                        best_straight = control
                elif idx == 5:
                    if control > best_straight_pair:
                        best_straight_pair = control
                elif idx == 6:
                    if control > best_straight_triple:
                        best_straight_triple = control
                elif control > best_bomb:
                    best_bomb = control
            elif nonzero_count:
                for idx, control in nonzero_controls:
                    if idx == 0:
                        if control > best_single:
                            best_single = control
                    elif idx == 1:
                        if control > best_pair:
                            best_pair = control
                    elif idx == 2:
                        if control > best_triple:
                            best_triple = control
                    elif idx == 3:
                        if control > best_triple_plus:
                            best_triple_plus = control
                    elif idx == 4:
                        if control > best_straight:
                            best_straight = control
                    elif idx == 5:
                        if control > best_straight_pair:
                            best_straight_pair = control
                    elif idx == 6:
                        if control > best_straight_triple:
                            best_straight_triple = control
                    elif control > best_bomb:
                        best_bomb = control

        residue_score = max(0, hand_count - 7) * 0.22 + single_debt
        if single_count >= 3:
            residue_score += (single_count - 2) * 0.35
        if low_single_count >= 2:
            residue_score += (low_single_count - 1) * 0.45

        (
            p_single,
            p_pair,
            p_triple,
            p_triple_plus,
            p_straight,
            p_straight_pair,
            p_straight_triple,
            p_bomb,
        ) = pressure_values if pressure_values is not None else self._pressure_values_for_primary(primary_kind)
        retake_total = (
            p_single * best_single
            + p_pair * best_pair
            + p_triple * best_triple
            + p_triple_plus * best_triple_plus
            + p_straight * best_straight
            + p_straight_pair * best_straight_pair
            + p_straight_triple * best_straight_triple
            + p_bomb * best_bomb
        )
        retake_score = retake_total * 100.0

        total = (
            weights.hand_count * hand_count_score
            + weights.card_value * card_value_score
            + weights.retake * retake_score
            - weights.residue * residue_score
        )
        return total, hand_count_score, card_value_score, retake_score, residue_score

    def score_cover_values(
        self,
        cover: list[int] | tuple[int, ...],
        group_entries: list[tuple[float, int, int, float, int, int, float, int, tuple[tuple[int, float], ...]]],
        primary_kind: str | None = None,
        pressure_values: tuple[float, ...] | None = None,
    ) -> tuple[float, float, float, float, float]:
        weights = self.weights
        w_hand_count = weights.hand_count
        w_card_value = weights.card_value
        w_retake = weights.retake
        w_residue = weights.residue
        hand_count = len(cover)
        effective_hand_count = 0
        card_value_score = 0.0
        single_debt = 0.0
        single_count = 0
        low_single_count = 0
        best_single = 0.0
        best_pair = 0.0
        best_triple = 0.0
        best_triple_plus = 0.0
        best_straight = 0.0
        best_straight_pair = 0.0
        best_straight_triple = 0.0
        best_bomb = 0.0

        for group_id in cover:
            (
                cached_value,
                group_single_count,
                group_low_count,
                group_debt,
                nonzero_count,
                idx,
                control,
                group_effective_cost,
                nonzero_controls,
            ) = group_entries[group_id]
            effective_hand_count += group_effective_cost
            card_value_score += cached_value
            single_count += group_single_count
            low_single_count += group_low_count
            single_debt += group_debt

            if nonzero_count == 1:
                if idx == 0:
                    if control > best_single:
                        best_single = control
                elif idx == 1:
                    if control > best_pair:
                        best_pair = control
                elif idx == 2:
                    if control > best_triple:
                        best_triple = control
                elif idx == 3:
                    if control > best_triple_plus:
                        best_triple_plus = control
                elif idx == 4:
                    if control > best_straight:
                        best_straight = control
                elif idx == 5:
                    if control > best_straight_pair:
                        best_straight_pair = control
                elif idx == 6:
                    if control > best_straight_triple:
                        best_straight_triple = control
                elif control > best_bomb:
                    best_bomb = control
            elif nonzero_count:
                for idx, control in nonzero_controls:
                    if idx == 0:
                        if control > best_single:
                            best_single = control
                    elif idx == 1:
                        if control > best_pair:
                            best_pair = control
                    elif idx == 2:
                        if control > best_triple:
                            best_triple = control
                    elif idx == 3:
                        if control > best_triple_plus:
                            best_triple_plus = control
                    elif idx == 4:
                        if control > best_straight:
                            best_straight = control
                    elif idx == 5:
                        if control > best_straight_pair:
                            best_straight_pair = control
                    elif idx == 6:
                        if control > best_straight_triple:
                            best_straight_triple = control
                    elif control > best_bomb:
                        best_bomb = control

        hand_count_score = -float(effective_hand_count)
        residue_score = ((hand_count - 7) * 0.22 if hand_count > 7 else 0.0) + single_debt
        if single_count >= 3:
            residue_score += (single_count - 2) * 0.35
        if low_single_count >= 2:
            residue_score += (low_single_count - 1) * 0.45

        (
            p_single,
            p_pair,
            p_triple,
            p_triple_plus,
            p_straight,
            p_straight_pair,
            p_straight_triple,
            p_bomb,
        ) = pressure_values if pressure_values is not None else self._pressure_values_for_primary(primary_kind)
        retake_total = (
            p_single * best_single
            + p_pair * best_pair
            + p_triple * best_triple
            + p_triple_plus * best_triple_plus
            + p_straight * best_straight
            + p_straight_pair * best_straight_pair
            + p_straight_triple * best_straight_triple
            + p_bomb * best_bomb
        )
        retake_score = retake_total * 100.0

        total = (
            w_hand_count * hand_count_score
            + w_card_value * card_value_score
            + w_retake * retake_score
            - w_residue * residue_score
        )
        return total, hand_count_score, card_value_score, retake_score, residue_score

    def best_cover_values(
        self,
        covers: list[list[int]],
        group_entries: list[tuple[float, int, int, float, int, int, float, int, tuple[tuple[int, float], ...]]],
        primary_kind: str | None = None,
        pressure_values: tuple[float, ...] | None = None,
    ) -> tuple[list[int], float, tuple[float, float, float, float]] | None:
        weights = self.weights
        w_hand_count = weights.hand_count
        w_card_value = weights.card_value
        w_retake = weights.retake
        w_residue = weights.residue
        (
            p_single,
            p_pair,
            p_triple,
            p_triple_plus,
            p_straight,
            p_straight_pair,
            p_straight_triple,
            p_bomb,
        ) = pressure_values if pressure_values is not None else self._pressure_values_for_primary(primary_kind)
        best_cover: list[int] | None = None
        best_total = 0.0
        best_parts: tuple[float, float, float, float] | None = None

        for cover in covers:
            hand_count = len(cover)
            effective_hand_count = 0
            card_value_score = 0.0
            single_debt = 0.0
            single_count = 0
            low_single_count = 0
            best_single = 0.0
            best_pair = 0.0
            best_triple = 0.0
            best_triple_plus = 0.0
            best_straight = 0.0
            best_straight_pair = 0.0
            best_straight_triple = 0.0
            best_bomb = 0.0

            for group_id in cover:
                (
                    cached_value,
                    group_single_count,
                    group_low_count,
                    group_debt,
                    nonzero_count,
                    idx,
                    control,
                    group_effective_cost,
                    nonzero_controls,
                ) = group_entries[group_id]
                effective_hand_count += group_effective_cost
                card_value_score += cached_value
                single_count += group_single_count
                low_single_count += group_low_count
                single_debt += group_debt

                if nonzero_count == 1:
                    if idx == 0:
                        if control > best_single:
                            best_single = control
                    elif idx == 1:
                        if control > best_pair:
                            best_pair = control
                    elif idx == 2:
                        if control > best_triple:
                            best_triple = control
                    elif idx == 3:
                        if control > best_triple_plus:
                            best_triple_plus = control
                    elif idx == 4:
                        if control > best_straight:
                            best_straight = control
                    elif idx == 5:
                        if control > best_straight_pair:
                            best_straight_pair = control
                    elif idx == 6:
                        if control > best_straight_triple:
                            best_straight_triple = control
                    elif control > best_bomb:
                        best_bomb = control
                elif nonzero_count:
                    for idx, control in nonzero_controls:
                        if idx == 0:
                            if control > best_single:
                                best_single = control
                        elif idx == 1:
                            if control > best_pair:
                                best_pair = control
                        elif idx == 2:
                            if control > best_triple:
                                best_triple = control
                        elif idx == 3:
                            if control > best_triple_plus:
                                best_triple_plus = control
                        elif idx == 4:
                            if control > best_straight:
                                best_straight = control
                        elif idx == 5:
                            if control > best_straight_pair:
                                best_straight_pair = control
                        elif idx == 6:
                            if control > best_straight_triple:
                                best_straight_triple = control
                        elif control > best_bomb:
                            best_bomb = control

            hand_count_score = -float(effective_hand_count)
            residue_score = ((hand_count - 7) * 0.22 if hand_count > 7 else 0.0) + single_debt
            if single_count >= 3:
                residue_score += (single_count - 2) * 0.35
            if low_single_count >= 2:
                residue_score += (low_single_count - 1) * 0.45

            retake_total = (
                p_single * best_single
                + p_pair * best_pair
                + p_triple * best_triple
                + p_triple_plus * best_triple_plus
                + p_straight * best_straight
                + p_straight_pair * best_straight_pair
                + p_straight_triple * best_straight_triple
                + p_bomb * best_bomb
            )
            retake_score = retake_total * 100.0
            total = (
                w_hand_count * hand_count_score
                + w_card_value * card_value_score
                + w_retake * retake_score
                - w_residue * residue_score
            )
            if best_cover is None or total > best_total:
                best_cover = cover
                best_total = total
                best_parts = (hand_count_score, card_value_score, retake_score, residue_score)

        if best_cover is None or best_parts is None:
            return None
        return best_cover, best_total, best_parts


def _score_partition_uncached(
    partition: Partition,
    ctx: RetrievalContext,
    weights: ScoreWeights = ScoreWeights(),
    primary_kind: str | None = None,
    profile: BreakPenaltyProfile = DEFAULT_BREAK_PENALTY_PROFILE,
) -> tuple[float, dict[str, float]]:
    hand_count_score = -float(partition.effective_hand_count)
    card_value_score = 0.0

    single_debt = 0.0
    single_count = 0
    low_single_count = 0

    best_control = {kind: 0.0 for kind in RETAKE_KINDS}

    for group in partition.groups:
        card_value_score += group_value(group, profile)

        kind = normalize_kind(group.kind)
        if kind == "Single":
            single_count += 1
            rank = group.rank or card_rank(group.cards[0])
            strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail)
            if strength >= 0.92:
                item = 0.08
            elif strength >= 0.78:
                item = 0.28
            elif strength >= 0.62:
                item = 0.65
            else:
                item = 1.05 + (0.62 - strength) * 0.75
                low_single_count += 1
            single_debt += item

        for retake_kind in RETAKE_KINDS:
            control = control_by_kind(group, ctx, retake_kind)
            if control > best_control[retake_kind]:
                best_control[retake_kind] = control

    residue_score = max(0, partition.hand_count - 7) * 0.22 + single_debt
    if single_count >= 3:
        residue_score += (single_count - 2) * 0.35
    if low_single_count >= 2:
        residue_score += (low_single_count - 1) * 0.45

    retake_total = 0.0
    for kind in RETAKE_KINDS:
        p = pressure(ctx, kind)
        if primary_kind and kind == primary_kind:
            p = max(p, 0.55)
        retake_total += p * best_control[kind]
    retake_score = retake_total * 100.0

    total = (
        weights.hand_count * hand_count_score
        + weights.card_value * card_value_score
        + weights.retake * retake_score
        - weights.residue * residue_score
    )
    return total, {
        "hand_count_score": hand_count_score,
        "card_value_score": card_value_score,
        "retake_score": retake_score,
        "residue_penalty_score": residue_score,
    }
