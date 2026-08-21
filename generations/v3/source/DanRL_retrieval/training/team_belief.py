from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from DanRL_retrieval.retrieval.action_generator import ActionGenerator
from DanRL_retrieval.retrieval.context import RetrievalContext
from DanRL_retrieval.retrieval.cards import CARD_INDEX, card_rank
from DanRL_retrieval.retrieval.models import (
    ActionCandidate,
    CardGroup,
    ScoredAction,
)
from DanRL_retrieval.retrieval.partitioner import FullSearchPartitioner
from DanRL_retrieval.retrieval.rules import (
    can_follow_action,
    group_to_action,
    is_bomb_kind,
    normalize_kind,
    normalize_rank,
)
from DanRL_retrieval.training.schema import (
    TEAM_BELIEF_SEAT_COUNT,
    TEAM_BELIEF_TARGET_DIM,
)


ENGINE_CARD_ALIASES = {"SB": "BJ", "B": "BJ", "HR": "RJ", "R": "RJ"}
BOMB_RESOURCE_KINDS = {"Bomb", "StraightFlush", "FourKings"}
NON_BOMB_RESPONSE = 1 << 0
BOMB_RESPONSE = 1 << 1
FINISHING_RESPONSE = 1 << 2
LOW_COST_RESPONSE = 1 << 3
BOMB_PRESERVING_RESPONSE = 1 << 4
CONTROL_PRESERVING_RESPONSE = 1 << 5
FIVE_CARD_RESPONSE = 1 << 6
CompactCardMultiset = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CompactBombResourceIndex:
    allowances: tuple[tuple[CompactCardMultiset, ...], ...]


@dataclass(slots=True)
class _ResponseClass:
    action: ActionCandidate
    signature: tuple[str, str | None, int]
    groups: tuple[CardGroup, ...]
    property_mask: int | None = None


class _SeatResponses(NamedTuple):
    hand_size: int
    hand_counter: Counter[str]
    bomb_resources: dict[tuple, list[Counter[str]]] | _CompactBombResourceIndex
    minimum_effective_groups: set[tuple]
    classes: tuple[_ResponseClass, ...]
    control_card_mask: int


def _candidate_action(candidate: ScoredAction | ActionCandidate) -> ActionCandidate:
    return candidate.action if isinstance(candidate, ScoredAction) else candidate


def _follow_signature(action: ActionCandidate) -> tuple[str, str | None, int]:
    """Return the fields consumed by canonical action-follow rules."""

    return normalize_kind(action.kind), normalize_rank(action.rank), len(action.cards)


def _group_key(
    kind: str,
    rank: str | None,
    cards: Sequence[str],
    *,
    compact: CompactCardMultiset | None = None,
) -> tuple:
    return (
        normalize_kind(kind),
        rank,
        compact if compact is not None else _compact_card_multiset(cards),
    )


def _bomb_resource_categories(groups) -> dict[tuple, list[Counter[str]]]:
    categories: dict[tuple, list[Counter[str]]] = {}
    for group in groups:
        kind = normalize_kind(group.kind)
        if kind not in BOMB_RESOURCE_KINDS:
            continue
        category = (kind, group.rank, len(group.cards))
        categories.setdefault(category, []).append(Counter(group.cards))
    return categories


def _compact_card_multiset(cards: Sequence[str]) -> CompactCardMultiset:
    """Encode a valid double-deck multiset as >=1 and >=2 bit planes."""

    one_or_more = 0
    two_or_more = 0
    for card in cards:
        try:
            bit = 1 << CARD_INDEX[card]
        except KeyError as exc:
            raise ValueError(f"unknown card in compact multiset: {card!r}") from exc
        if two_or_more & bit:
            raise ValueError(f"more than two copies of card {card!r}")
        if one_or_more & bit:
            two_or_more |= bit
        else:
            one_or_more |= bit
    return one_or_more, two_or_more


def _control_card_mask(cur_rank: str | None) -> int:
    """Encode jokers and current-level cards in the compact card bit space."""

    mask = (1 << CARD_INDEX["BJ"]) | (1 << CARD_INDEX["RJ"])
    if cur_rank is None:
        return mask
    for card, index in CARD_INDEX.items():
        if card not in {"BJ", "RJ"} and card_rank(card) == cur_rank:
            mask |= 1 << index
    return mask


def _remaining_allowance(
    hand: CompactCardMultiset,
    resource: CompactCardMultiset,
) -> CompactCardMultiset:
    """Return the largest response multiset that leaves `resource` intact."""

    hand_one, hand_two = hand
    resource_one, resource_two = resource
    if resource_one & ~hand_one or resource_two & ~hand_two:
        raise ValueError("bomb resource is not contained in its source hand")
    allowance_two = hand_two & ~resource_one
    allowance_one = (
        (hand_one & ~resource_one)
        | (hand_two & resource_one & ~resource_two)
    )
    return allowance_one, allowance_two


def _compile_bomb_resource_index(
    hand: Counter[str],
    resources: dict[tuple, list[Counter[str]]],
) -> _CompactBombResourceIndex:
    hand_compact = _compact_card_multiset(tuple(hand.elements()))
    return _CompactBombResourceIndex(
        tuple(
            tuple(
                _remaining_allowance(
                    hand_compact,
                    _compact_card_multiset(tuple(resource.elements())),
                )
                for resource in alternatives
            )
            for alternatives in resources.values()
        )
    )


def _compact_preserves_all_resources(
    response_one: int,
    response_two: int,
    resources: tuple[tuple[CompactCardMultiset, ...], ...],
) -> bool:
    """Return whether every bomb category retains at least one alternative."""

    for alternatives in resources:
        preserves_category = False
        for allowance_one, allowance_two in alternatives:
            if not (response_one & ~allowance_one) and not (
                response_two & ~allowance_two
            ):
                preserves_category = True
                break
        if not preserves_category:
            return False
    return True


def _preserves_bomb_resources(
    hand: Counter[str],
    response_cards: Sequence[str],
    resources: dict[tuple, list[Counter[str]]] | _CompactBombResourceIndex,
    *,
    response_compact: CompactCardMultiset | None = None,
) -> bool:
    if isinstance(resources, _CompactBombResourceIndex):
        if not resources.allowances:
            return True
        response_one, response_two = (
            response_compact
            if response_compact is not None
            else _compact_card_multiset(response_cards)
        )
        return _compact_preserves_all_resources(
            response_one,
            response_two,
            resources.allowances,
        )
    if not resources:
        return True
    response = Counter(response_cards)
    return all(
        any(
            all(
                hand[card] - response[card] >= count
                for card, count in resource.items()
            )
            for resource in alternatives
        )
        for alternatives in resources.values()
    )


def _minimum_effective_response_keys(
    partitioner: FullSearchPartitioner,
    hand: Sequence[str],
    ctx: RetrievalContext,
    groups: Sequence[CardGroup],
) -> set[tuple]:
    return {
        _group_key(group.kind, group.rank, group.cards)
        for group in partitioner.minimum_effective_groups(hand, ctx, groups)
    }


def _compile_response_classes(
    groups: Sequence[CardGroup],
) -> tuple[_ResponseClass, ...]:
    """Compile physical groups into follow-equivalent existential responses.

    `can_follow_action` only observes normalized kind, normalized rank, and
    card count for ActionCandidate inputs.  Physical groups with that same
    signature can share one follow check; their label traits are OR-ed lazily
    only when that response class matches a candidate target.
    """

    response_groups: dict[
        tuple[str, str | None, int], tuple[ActionCandidate, list[CardGroup]],
    ] = {}

    for index, response_group in enumerate(groups):
        signature = (
            normalize_kind(response_group.kind),
            normalize_rank(response_group.rank),
            len(response_group.cards),
        )
        previous = response_groups.get(signature)
        if previous is None:
            response = group_to_action(response_group, index)
            response_groups[signature] = (response, [response_group])
        else:
            previous[1].append(response_group)

    return tuple(
        _ResponseClass(action, signature, tuple(class_groups))
        for signature, (action, class_groups) in response_groups.items()
    )


def _response_property_mask(
    response_class: _ResponseClass,
    seat: _SeatResponses,
    ctx: RetrievalContext,
) -> int:
    cached = response_class.property_mask
    if cached is not None:
        return cached

    non_bomb = not is_bomb_kind(response_class.action.kind)
    property_mask = NON_BOMB_RESPONSE if non_bomb else BOMB_RESPONSE
    response_size = len(response_class.action.cards)
    if response_size == seat.hand_size:
        property_mask |= FINISHING_RESPONSE
    if response_size == 5:
        property_mask |= FIVE_CARD_RESPONSE
    required_quality_mask = (
        BOMB_PRESERVING_RESPONSE | CONTROL_PRESERVING_RESPONSE
    )
    if non_bomb:
        required_quality_mask |= LOW_COST_RESPONSE

    for response_group in response_class.groups:
        response_cards = response_group.cards
        response_compact = _compact_card_multiset(response_cards)
        bomb_preserving = _preserves_bomb_resources(
            seat.hand_counter,
            response_cards,
            seat.bomb_resources,
            response_compact=response_compact,
        )
        control_preserving = not bool(response_compact[0] & seat.control_card_mask)
        if (
            non_bomb
            and bomb_preserving
            and control_preserving
            and _group_key(
                response_group.kind,
                response_group.rank,
                response_cards,
                compact=response_compact,
            ) in seat.minimum_effective_groups
        ):
            property_mask |= LOW_COST_RESPONSE
        if bomb_preserving:
            property_mask |= BOMB_PRESERVING_RESPONSE
        if control_preserving:
            property_mask |= CONTROL_PRESERVING_RESPONSE
        if property_mask & required_quality_mask == required_quality_mask:
            break

    response_class.property_mask = property_mask
    return property_mask


def _hidden_seat_responses(
    partitioner: FullSearchPartitioner,
    hand: list[str],
    ctx: RetrievalContext,
    absolute_seat: int,
) -> _SeatResponses:
    """Build or reuse the latest immutable analysis for one absolute seat."""

    cache = None
    cache_key = None
    if partitioner.__class__ is FullSearchPartitioner:
        # This cache holds at most one immutable analysis per absolute seat.
        # Keep it independent from the much larger general partitioner LRUs:
        # platform actors intentionally disable those LRUs to bound memory,
        # but adjacent decisions still reuse two or three unchanged hidden
        # hands while building privileged team-belief labels.
        cache = getattr(partitioner, "_team_belief_latest_seat_cache", None)
        if cache is None:
            cache = {}
            setattr(partitioner, "_team_belief_latest_seat_cache", cache)
        cache_key = (ctx.cur_rank, _compact_card_multiset(hand))
        cached = cache.get(absolute_seat)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

    groups = partitioner._generate_groups(hand, ctx)
    hand_counter = Counter(hand)
    bomb_resources = _bomb_resource_categories(groups)
    result = _SeatResponses(
        len(hand),
        hand_counter,
        _compile_bomb_resource_index(hand_counter, bomb_resources),
        _minimum_effective_response_keys(partitioner, hand, ctx, groups),
        _compile_response_classes(groups),
        _control_card_mask(ctx.cur_rank),
    )
    if cache is not None:
        cache[absolute_seat] = (cache_key, result)
    return result


def build_team_belief_labels(
    all_hands: Sequence[Sequence[str]],
    *,
    actor_seat: int,
    ctx: RetrievalContext,
    candidates: Sequence[ScoredAction | ActionCandidate],
    capacity: int,
    partitioner: FullSearchPartitioner | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build privileged self-play response labels without exposing hands to policy.

    The returned tensor is candidate x relative-seat x target.  Targets cover
    non-bomb, bomb, finishing, low-cost, bomb-preserving, control-preserving,
    and five-card responses.  Only this label builder receives true hands.
    """

    if len(all_hands) != 4:
        raise ValueError("team-belief labels require hidden hands for all four seats")
    if actor_seat not in range(4):
        raise ValueError(f"actor_seat must be in [0,3], got {actor_seat}")
    if capacity <= 0 or len(candidates) > capacity:
        raise ValueError("capacity must cover every candidate")

    labels = np.zeros(
        (capacity, TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM),
        dtype=np.float32,
    )
    label_mask = np.zeros(
        (capacity, TEAM_BELIEF_SEAT_COUNT), dtype=np.float32,
    )
    if not candidates:
        return labels, label_mask

    partitioner = partitioner or FullSearchPartitioner(use_native_all=True)
    public_target = ActionGenerator(partitioner=partitioner)._target_action(ctx)
    targets = [
        public_target if _candidate_action(candidate).kind == "PASS"
        else _candidate_action(candidate)
        for candidate in candidates
    ]
    seat_responses: list[_SeatResponses] = []
    for relative_seat in (1, 2, 3):
        absolute_seat = (actor_seat + relative_seat) % 4
        hand = [
            ENGINE_CARD_ALIASES.get(str(card).strip().upper(), str(card))
            for card in all_hands[absolute_seat]
        ]
        seat_responses.append(
            _hidden_seat_responses(partitioner, hand, ctx, absolute_seat)
        )

    target_cache: dict[tuple[str, str | None, int], np.ndarray] = {}
    for candidate_index, target in enumerate(targets):
        if target is None:
            # PASS while leading is not a meaningful candidate-response query.
            continue
        label_mask[candidate_index, :] = 1.0
        signature = _follow_signature(target)
        cached_labels = target_cache.get(signature)
        if cached_labels is None:
            cached_labels = np.zeros(
                (TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM),
                dtype=np.float32,
            )
            for seat_index, seat in enumerate(seat_responses):
                property_mask = 0
                for response_class in seat.classes:
                    if can_follow_action(
                        response_class.action, target, ctx.cur_rank,
                    ):
                        property_mask |= _response_property_mask(
                            response_class, seat, ctx,
                        )
                for target_index in range(TEAM_BELIEF_TARGET_DIM):
                    if property_mask & (1 << target_index):
                        cached_labels[seat_index, target_index] = 1.0
            target_cache[signature] = cached_labels
        labels[candidate_index] = cached_labels
    return labels, label_mask
