"""Inference-only rules over a frozen V3 choice.

The TriplePlus rule accepts only a strict whole-hand partition equivalence:
the chosen and replacement actions use the same physical triple, and their
best residual partitions differ only by swapping two standalone Pair groups.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from DanKS.retrieval.cards import card_rank, heart_level_card, normalize_card, rank_strength
from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.models import ActionCandidate, CardGroup, Partition, ScoredAction
from DanKS.retrieval.rules import can_follow_action, is_bomb_kind, normalize_kind
from . import _guarded as legacy

PROTOCOL = "v3pro_guarded_compare_equivalent_pair_swap_v1"


@dataclass(frozen=True)
class RuleDecision:
    original_action_index: int
    final_action_index: int
    trace: dict[str, Any]


def _cards(action: ActionCandidate) -> tuple[str, ...]:
    return tuple(normalize_card(card) for card in action.cards)


def _view(row: ScoredAction) -> dict[str, Any]:
    return {
        "action_index": int(row.action.index),
        "kind": normalize_kind(row.action.kind),
        "rank": row.action.rank,
        "cards": _cards(row.action),
        "cards_zh": [legacy.card_zh(card) for card in row.action.cards],
    }


def _group_view(group: CardGroup) -> dict[str, Any]:
    return {
        "kind": normalize_kind(group.kind),
        "rank": group.rank,
        "cards": tuple(sorted(normalize_card(card) for card in group.cards)),
        "cards_zh": [legacy.card_zh(card) for card in group.cards],
    }


def _roles(action: ActionCandidate, ctx: RetrievalContext):
    if normalize_kind(action.kind) != "TriplePlus" or action.size != 5 or not action.rank:
        return None
    cards = _cards(action)
    wild = heart_level_card(ctx.cur_rank)
    if wild and wild in cards:
        return None
    triple = tuple(sorted(card for card in cards if card_rank(card) == action.rank))
    pair = tuple(sorted(card for card in cards if card_rank(card) != action.rank))
    if len(triple) != 3 or len(pair) != 2 or card_rank(pair[0]) != card_rank(pair[1]):
        return None
    return triple, pair, card_rank(pair[0])


def lower_pair_legal_actions(
    selected: ActionCandidate,
    legal_actions: Sequence[ActionCandidate],
    ctx: RetrievalContext,
) -> list[ActionCandidate]:
    """Return engine-generated lower-pair actions with the same physical triple."""
    selected_roles = _roles(selected, ctx)
    if selected_roles is None:
        return []
    triple, _pair, selected_rank = selected_roles
    selected_strength = rank_strength(selected_rank, ctx.cur_rank, ctx.remaining_detail)
    out = []
    for action in legal_actions:
        roles = _roles(action, ctx)
        if roles is None or roles[0] != triple or int(action.index) == int(selected.index):
            continue
        if rank_strength(roles[2], ctx.cur_rank, ctx.remaining_detail) < selected_strength:
            out.append(action)
    return out


def _pair_group_index(partition: Partition, pair: tuple[str, ...]) -> int | None:
    wanted = Counter(pair)
    matches = [
        index for index, group in enumerate(partition.groups)
        if normalize_kind(group.kind) == "Pair"
        and Counter(_cards(ActionCandidate(-1, "Pair", tuple(group.cards), group.rank))) == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def _remaining_signature(partition: Partition, omitted_index: int) -> tuple:
    signatures = []
    for index, group in enumerate(partition.groups):
        if index == omitted_index:
            continue
        signatures.append((
            normalize_kind(group.kind),
            str(group.rank),
            tuple(sorted(normalize_card(card) for card in group.cards)),
        ))
    return tuple(sorted(signatures))


def _equivalent_pair_swap(
    selected: ScoredAction,
    alternative: ScoredAction,
    ctx: RetrievalContext,
) -> dict[str, Any] | None:
    original_roles = _roles(selected.action, ctx)
    target_roles = _roles(alternative.action, ctx)
    if original_roles is None or target_roles is None or original_roles[0] != target_roles[0]:
        return None
    _triple, original_pair, original_rank = original_roles
    _triple, target_pair, target_rank = target_roles
    if rank_strength(target_rank, ctx.cur_rank, ctx.remaining_detail) >= rank_strength(
        original_rank, ctx.cur_rank, ctx.remaining_detail
    ):
        return None
    # After playing the original action, the target pair must remain as one
    # standalone Pair. After playing the alternative, the original pair must
    # remain as one standalone Pair.
    original_pair_slot = _pair_group_index(selected.partition, target_pair)
    target_pair_slot = _pair_group_index(alternative.partition, original_pair)
    if original_pair_slot is None or target_pair_slot is None:
        return None
    if selected.partition.hand_count != alternative.partition.hand_count:
        return None
    if selected.partition.effective_hand_count != alternative.partition.effective_hand_count:
        return None
    original_rest = _remaining_signature(selected.partition, original_pair_slot)
    target_rest = _remaining_signature(alternative.partition, target_pair_slot)
    if original_rest != target_rest:
        return None
    return {
        "original_pair": original_pair,
        "original_pair_rank": original_rank,
        "target_pair": target_pair,
        "target_pair_rank": target_rank,
        "shared_rest_signature": original_rest,
    }


def _validate_support(hand, ctx, support, selected_slot, alternatives):
    if ctx.cur_rank not in tuple("23456789TJQKA"):
        raise ValueError("a known current level rank is required")
    if not 1 <= len(support) <= 10:
        raise ValueError("expected the actual 1..10 policy candidates")
    if not 0 <= selected_slot < len(support):
        raise IndexError("selected slot outside policy candidates")
    support_indices = [int(row.action.index) for row in support]
    alternative_indices = [int(row.action.index) for row in alternatives]
    if len(set(support_indices)) != len(support_indices):
        raise ValueError("duplicate action indices in policy support")
    if len(set(alternative_indices)) != len(alternative_indices):
        raise ValueError("duplicate action indices in scored alternatives")
    all_rows = list(support) + list(alternatives)
    available = Counter(normalize_card(card) for card in hand)
    target = legacy._target_action(ctx)
    for row in all_rows:
        if Counter(_cards(row.action)) - available:
            raise ValueError("candidate uses cards absent from this hand")
        if target is not None and normalize_kind(row.action.kind) != "PASS" and not can_follow_action(
            row.action, target, ctx.cur_rank
        ):
            raise ValueError("candidate cannot follow the current target")


def _check_bomb_metrics_finite(rows, ctx):
    for row in rows:
        if not is_bomb_kind(row.action.kind):
            continue
        values = list(asdict(legacy._metrics(row, ctx)).values())
        values += [row.details.get(key, 0.0) for key in (
            "must_block", "opponent_short_pressure", "partner_follow_help"
        )]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("non-finite bomb comparison metrics")


def decide(
    hand: Sequence[str],
    ctx: RetrievalContext,
    support: Sequence[ScoredAction],
    selected_slot: int,
    *,
    equivalent_rows: Sequence[ScoredAction] = (),
    mode: str = "apply",
) -> RuleDecision:
    if mode not in {"apply", "shadow", "off"}:
        raise ValueError("mode must be apply, shadow, or off")
    _validate_support(hand, ctx, support, selected_slot, equivalent_rows)
    selected = support[selected_slot]
    original = int(selected.action.index)
    trace = {
        "protocol": PROTOCOL,
        "mode": mode,
        "rule": None,
        "selected": _view(selected),
        "support_action_indices": [int(row.action.index) for row in support],
        "scored_alternative_indices": [int(row.action.index) for row in equivalent_rows],
        "would_replace": False,
        "applied": False,
        "reason": None,
        "training_eligible": False,
    }
    recommendation = None
    if mode == "off":
        trace["reason"] = "disabled"
    elif is_bomb_kind(selected.action.kind):
        _check_bomb_metrics_finite(support, ctx)
        result = legacy.analyze_guarded_compare(hand, ctx, support, selected_slot)
        trace.update(rule="minimum_sufficient_bomb", bomb_comparison=result.to_dict())
        if result.recommended_slot is not None:
            recommendation = support[result.recommended_slot]
            trace["reason"] = "minimum_sufficient_bomb_found"
        else:
            trace["reason"] = "minimum_sufficient_bomb_not_found"
    elif normalize_kind(selected.action.kind) == "TriplePlus":
        trace["rule"] = "equivalent_whole_hand_pair_swap"
        selected_roles = _roles(selected.action, ctx)
        if selected_roles is None:
            trace["reason"] = "triple_plus_roles_ambiguous"
        elif not equivalent_rows:
            trace["reason"] = "no_lower_pair_legal_action"
        else:
            eligible = []
            for row in equivalent_rows:
                proof = _equivalent_pair_swap(selected, row, ctx)
                if proof is not None:
                    roles = _roles(row.action, ctx)
                    assert roles is not None
                    eligible.append((
                        rank_strength(roles[2], ctx.cur_rank, ctx.remaining_detail),
                        roles[1],
                        int(row.action.index),
                        row,
                        proof,
                    ))
            if not eligible:
                trace["reason"] = "no_strictly_equivalent_residual_partition"
            else:
                _strength, _pair, _index, recommendation, proof = min(eligible)
                trace.update(
                    reason="strict_equivalent_pair_swap",
                    target=_view(recommendation),
                    original_partition=[_group_view(group) for group in selected.partition.groups],
                    target_partition=[_group_view(group) for group in recommendation.partition.groups],
                    equivalence_proof=proof,
                )
    else:
        trace["reason"] = "no_enabled_rule_for_selected_action"

    would_replace = recommendation is not None
    applied = would_replace and mode == "apply"
    final = int(recommendation.action.index) if applied else original
    trace.update(
        recommended_action_index=(None if recommendation is None else int(recommendation.action.index)),
        would_replace=would_replace,
        applied=applied,
        final_action_index=final,
    )
    return RuleDecision(original, final, trace)
