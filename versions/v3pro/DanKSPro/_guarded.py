from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from DanKS.retrieval.cards import (
    card_rank,
    heart_level_card,
    normalize_card,
    rank_strength,
)
from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.models import ActionCandidate, ScoredAction
from DanKS.retrieval.rules import (
    bomb_power,
    can_follow_action,
    is_bomb_kind,
    normalize_kind,
)
from DanKS.training.type_suppression import candidate_response_profile


_EPSILON = 1.0e-9
_SUIT_ZH = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方片"}
_RANK_ZH = {"T": "10", "BJ": "小王", "RJ": "大王"}


@dataclass(frozen=True, slots=True)
class GuardedMetrics:
    opponent_response_risk: float
    my_min_steps: int
    retake_score: float
    my_retake_count: float
    card_value: float
    break_group: float
    spend_penalty: float


@dataclass(frozen=True, slots=True)
class CandidateView:
    slot: int
    action_index: int
    kind: str
    rank: str | None
    cards: tuple[str, ...]
    cards_zh: tuple[str, ...]
    bomb_power: tuple[int, int, int] | None
    kicker_rank: str | None
    metrics: GuardedMetrics


@dataclass(frozen=True, slots=True)
class GuardedComparison:
    reference: CandidateView
    dominates_on_observed_metrics: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuardedCompareTrace:
    protocol: str
    mode: str
    opportunity: str | None
    selected: CandidateView
    comparisons: tuple[GuardedComparison, ...]
    recommended_slot: int | None
    would_replace: bool
    applied: bool
    fail_closed_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def card_zh(card: str) -> str:
    normalized = normalize_card(card)
    if normalized in {"BJ", "RJ"}:
        return _RANK_ZH[normalized]
    return f"{_SUIT_ZH[normalized[0]]}{_RANK_ZH.get(normalized[1], normalized[1])}"


def _target_action(ctx: RetrievalContext) -> ActionCandidate | None:
    kind = normalize_kind(ctx.current_kind)
    if kind in {"Lead", "PASS"} or not ctx.current_rank:
        return None
    size = max(1, int(ctx.current_size or 1))
    return ActionCandidate(
        index=-1,
        kind=kind,
        cards=tuple("__target__" for _ in range(size)),
        rank=ctx.current_rank,
    )


def _is_legal_follow(row: ScoredAction, ctx: RetrievalContext) -> bool:
    target = _target_action(ctx)
    return target is not None and can_follow_action(row.action, target, ctx.cur_rank)


def _uses_wildcard(action: ActionCandidate, ctx: RetrievalContext) -> bool:
    wildcard = heart_level_card(ctx.cur_rank)
    return wildcard is not None and wildcard in action.cards


def _promoted_four_bomb(action: ActionCandidate, ctx: RetrievalContext) -> bool:
    if normalize_kind(action.kind) != "Bomb" or action.size != 4:
        return False
    wildcard = heart_level_card(ctx.cur_rank)
    if wildcard is None or action.cards.count(wildcard) != 1:
        return False
    if not action.rank or action.rank == ctx.cur_rank:
        return False
    natural = [card for card in action.cards if card != wildcard]
    return len(natural) == 3 and all(card_rank(card) == action.rank for card in natural)


def _natural_triple_plus_kicker(
    action: ActionCandidate,
    ctx: RetrievalContext,
) -> tuple[str, str] | None:
    if normalize_kind(action.kind) != "TriplePlus" or action.size != 5 or not action.rank:
        return None
    wildcard = heart_level_card(ctx.cur_rank)
    if wildcard is not None and wildcard in action.cards:
        # CardGroup semantic roles are not present in ActionCandidate.  A
        # wildcard TriplePlus can therefore be ambiguous; shadow mode must not
        # invent whether the wildcard belongs to the triple or the pair.
        return None
    counts = Counter(card_rank(card) for card in action.cards)
    if counts.get(action.rank) != 3:
        return None
    kickers = [rank for rank, count in counts.items() if rank != action.rank and count == 2]
    if len(kickers) != 1 or sum(counts.values()) != 5:
        return None
    return action.rank, kickers[0]


def _metrics(row: ScoredAction, ctx: RetrievalContext) -> GuardedMetrics:
    response = candidate_response_profile(ctx, row.action)
    details = row.details
    return GuardedMetrics(
        opponent_response_risk=max(float(response.left_risk), float(response.right_risk)),
        my_min_steps=int(row.partition.hand_count),
        retake_score=float(row.retake_score),
        my_retake_count=float(details.get("my_retake_count", 0.0)),
        card_value=float(row.card_value_score),
        break_group=float(details.get("break_group_penalty", 0.0)),
        spend_penalty=float(details.get("spend_penalty", 0.0)),
    )


def _view(slot: int, row: ScoredAction, ctx: RetrievalContext) -> CandidateView:
    action = row.action
    kicker = _natural_triple_plus_kicker(action, ctx)
    return CandidateView(
        slot=slot,
        action_index=int(action.index),
        kind=normalize_kind(action.kind),
        rank=action.rank,
        cards=tuple(action.cards),
        cards_zh=tuple(card_zh(card) for card in action.cards),
        bomb_power=(
            bomb_power(action.kind, action.rank, action.size, ctx.cur_rank)
            if is_bomb_kind(action.kind)
            else None
        ),
        kicker_rank=None if kicker is None else kicker[1],
        metrics=_metrics(row, ctx),
    )


def _dominance_blockers(
    selected: ScoredAction,
    reference: ScoredAction,
    ctx: RetrievalContext,
    *,
    hand_size: int,
    opportunity: str | None,
) -> tuple[str, ...]:
    selected_metrics = _metrics(selected, ctx)
    reference_metrics = _metrics(reference, ctx)
    blockers: list[str] = []
    if reference_metrics.opponent_response_risk > selected_metrics.opponent_response_risk + _EPSILON:
        blockers.append("opponent_response_risk_higher")
    if reference_metrics.my_min_steps > selected_metrics.my_min_steps:
        blockers.append("remaining_steps_more")
    if reference_metrics.retake_score + _EPSILON < selected_metrics.retake_score:
        blockers.append("retake_score_lower")
    if reference_metrics.my_retake_count + _EPSILON < selected_metrics.my_retake_count:
        blockers.append("retake_count_lower")
    if reference_metrics.card_value + _EPSILON < selected_metrics.card_value:
        blockers.append("remaining_card_value_lower")
    if reference_metrics.break_group > selected_metrics.break_group + _EPSILON:
        blockers.append("break_group_worse")
    if reference_metrics.spend_penalty > selected_metrics.spend_penalty + _EPSILON:
        blockers.append("spend_penalty_worse")
    if selected.action.size >= hand_size or selected_metrics.my_min_steps == 0:
        blockers.append("selected_finishes_hand")
    details = selected.details
    if float(details.get("must_block", 0.0)) > _EPSILON:
        blockers.append("must_block_active")
    if float(details.get("opponent_short_pressure", 0.0)) > _EPSILON:
        blockers.append("opponent_short_pressure_active")
    if float(details.get("partner_follow_help", 0.0)) > _EPSILON:
        blockers.append("partner_help_active")
    if opportunity == "ordinary_wildcard_vs_promoted_four" and _target_action(ctx) is None:
        # On lead, an ordinary action and a bomb do not serve the same control
        # objective. Keep the opportunity in the audit, but never claim that
        # the bomb dominates merely because observed scalar metrics align.
        blockers.append("lead_control_goal_not_comparable")
    return tuple(blockers)


def _bomb_references(
    candidates: Sequence[ScoredAction],
    selected_slot: int,
    ctx: RetrievalContext,
) -> tuple[str | None, list[int]]:
    selected = candidates[selected_slot]
    action = selected.action
    target = _target_action(ctx)
    selected_is_bomb = is_bomb_kind(action.kind)
    ordinary_wildcard = not selected_is_bomb and _uses_wildcard(action, ctx)
    if not selected_is_bomb and not ordinary_wildcard:
        return None, []
    if target is not None and not _is_legal_follow(selected, ctx):
        return None, []
    if target is None and selected_is_bomb:
        # "Minimum sufficient bomb" only has a defined meaning when following
        # an existing target. Lead-bomb comparisons remain fail-closed.
        return None, []
    selected_power = (
        bomb_power(action.kind, action.rank, action.size, ctx.cur_rank)
        if selected_is_bomb
        else None
    )
    refs: list[int] = []
    for slot, row in enumerate(candidates):
        if slot == selected_slot or not is_bomb_kind(row.action.kind):
            continue
        if target is not None and not _is_legal_follow(row, ctx):
            continue
        power = bomb_power(
            row.action.kind, row.action.rank, row.action.size, ctx.cur_rank,
        )
        if selected_power is not None:
            if power < selected_power:
                refs.append(slot)
        elif ordinary_wildcard and _promoted_four_bomb(row.action, ctx):
            refs.append(slot)
    refs.sort(
        key=lambda slot: bomb_power(
            candidates[slot].action.kind,
            candidates[slot].action.rank,
            candidates[slot].action.size,
            ctx.cur_rank,
        )
    )
    opportunity = (
        "minimum_sufficient_bomb"
        if selected_is_bomb and refs
        else "ordinary_wildcard_vs_promoted_four"
        if ordinary_wildcard and refs
        else None
    )
    return opportunity, refs


def _triple_plus_references(
    candidates: Sequence[ScoredAction],
    selected_slot: int,
    ctx: RetrievalContext,
) -> list[int]:
    selected_roles = _natural_triple_plus_kicker(candidates[selected_slot].action, ctx)
    if selected_roles is None:
        return []
    triple_rank, selected_kicker = selected_roles
    selected_strength = rank_strength(
        selected_kicker, ctx.cur_rank, ctx.remaining_detail,
    )
    refs: list[tuple[float, int]] = []
    for slot, row in enumerate(candidates):
        if slot == selected_slot:
            continue
        roles = _natural_triple_plus_kicker(row.action, ctx)
        if roles is None or roles[0] != triple_rank:
            continue
        kicker_strength = rank_strength(
            roles[1], ctx.cur_rank, ctx.remaining_detail,
        )
        if kicker_strength + _EPSILON < selected_strength:
            refs.append((kicker_strength, slot))
    return [slot for _, slot in sorted(refs)]


def analyze_guarded_compare(
    hand: Sequence[str],
    ctx: RetrievalContext,
    candidates: Sequence[ScoredAction],
    selected_slot: int,
) -> GuardedCompareTrace:
    """Audit conservative candidate substitutions without changing the action.

    The function is deliberately shadow-only.  It reports the slot that would
    be preferred under observed-metric dominance, but `applied` is always
    false and the caller must continue executing `selected_slot`.
    """

    if not candidates:
        raise ValueError("guarded compare requires at least one candidate")
    if selected_slot < 0 or selected_slot >= len(candidates):
        raise IndexError("selected slot is outside candidate support")
    selected = candidates[selected_slot]
    opportunity: str | None = None
    reference_slots = _triple_plus_references(candidates, selected_slot, ctx)
    fail_closed_reason: str | None = None
    if reference_slots:
        opportunity = "triple_plus_lower_kicker"
    elif normalize_kind(selected.action.kind) == "TriplePlus":
        if _natural_triple_plus_kicker(selected.action, ctx) is None:
            fail_closed_reason = "triple_plus_roles_ambiguous"
    else:
        opportunity, reference_slots = _bomb_references(
            candidates, selected_slot, ctx,
        )

    comparisons = []
    for slot in reference_slots:
        reference = candidates[slot]
        blockers = _dominance_blockers(
            selected,
            reference,
            ctx,
            hand_size=len(hand),
            opportunity=opportunity,
        )
        comparisons.append(
            GuardedComparison(
                reference=_view(slot, reference, ctx),
                dominates_on_observed_metrics=not blockers,
                blockers=blockers,
            )
        )

    recommended_slot = next(
        (
            comparison.reference.slot
            for comparison in comparisons
            if comparison.dominates_on_observed_metrics
        ),
        None,
    )
    return GuardedCompareTrace(
        protocol="v3pro_guarded_compare_shadow_v1",
        mode="shadow",
        opportunity=opportunity,
        selected=_view(selected_slot, selected, ctx),
        comparisons=tuple(comparisons),
        recommended_slot=recommended_slot,
        would_replace=recommended_slot is not None,
        applied=False,
        fail_closed_reason=fail_closed_reason,
    )
