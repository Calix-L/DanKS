from __future__ import annotations

from dataclasses import dataclass
import math

from DanKS.retrieval.card_memory import ResponseEvent, SeatPlayStats
from DanKS.retrieval.cards import action_type_size, rank_strength
from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.models import ActionCandidate
from DanKS.retrieval.rules import normalize_kind, plain_points


BASE_RESPONSE_RISK = 0.65
MIN_RESPONSE_RISK = 0.05
MAX_RESPONSE_RISK = 0.95
SEQUENCE_KINDS = {"Straight", "StraightPair", "StraightTriple", "StraightFlush"}
BOMB_KINDS = {"Bomb", "StraightFlush", "FourKings"}


@dataclass(frozen=True)
class SeatResponseEstimate:
    risk: float
    evidence: float
    pass_evidence: float
    response_evidence: float
    spent_evidence: float


@dataclass(frozen=True)
class CandidateResponseProfile:
    left_risk: float
    teammate_risk: float
    right_risk: float
    left_evidence: float
    teammate_evidence: float
    right_evidence: float
    left_pass_evidence: float
    teammate_pass_evidence: float
    right_pass_evidence: float
    left_positive_response_evidence: float
    teammate_positive_response_evidence: float
    right_positive_response_evidence: float
    left_spent_evidence: float
    teammate_spent_evidence: float
    right_spent_evidence: float
    next_risk: float
    opponent_risk: float
    evidence_strength: float
    type_suppression: float
    next_suppression: float
    combined_suppression: float

    def to_details(self) -> dict[str, float]:
        return {
            "left_response_risk": self.left_risk,
            "teammate_response_risk": self.teammate_risk,
            "right_response_risk": self.right_risk,
            "left_response_evidence": self.left_evidence,
            "teammate_response_evidence": self.teammate_evidence,
            "right_response_evidence": self.right_evidence,
            "left_pass_evidence": self.left_pass_evidence,
            "teammate_pass_evidence": self.teammate_pass_evidence,
            "right_pass_evidence": self.right_pass_evidence,
            "left_positive_response_evidence": self.left_positive_response_evidence,
            "teammate_positive_response_evidence": self.teammate_positive_response_evidence,
            "right_positive_response_evidence": self.right_positive_response_evidence,
            "left_spent_evidence": self.left_spent_evidence,
            "teammate_spent_evidence": self.teammate_spent_evidence,
            "right_spent_evidence": self.right_spent_evidence,
            "next_response_risk": self.next_risk,
            "opponent_response_risk": self.opponent_risk,
            "type_suppression_evidence": self.evidence_strength,
            "type_suppression_score": self.type_suppression,
            "next_suppression_score": self.next_suppression,
            "type_suppression_combined": self.combined_suppression,
        }


def _difficulty(kind: str, rank: str | None, cur_rank: str | None) -> float:
    if not rank:
        return 0.0
    if kind in SEQUENCE_KINDS:
        return min(1.0, max(0.0, plain_points(rank) / 15.0))
    return min(1.0, max(0.0, rank_strength(rank, cur_rank)))


def _kind_matches(candidate_kind: str, target_kind: str) -> bool:
    candidate_kind = normalize_kind(candidate_kind)
    target_kind = normalize_kind(target_kind)
    if candidate_kind == target_kind:
        return True
    return candidate_kind in BOMB_KINDS and target_kind in BOMB_KINDS


def _target_relevance(
    action: ActionCandidate,
    event: ResponseEvent,
    cur_rank: str | None,
) -> float:
    kind = normalize_kind(action.kind)
    target_kind = normalize_kind(event.target_kind)
    if not _kind_matches(kind, target_kind):
        return 0.0
    if kind in SEQUENCE_KINDS and action.size != event.target_size:
        return 0.0
    if kind == "Bomb" and target_kind == "Bomb":
        if action.size > event.target_size:
            return 1.0
        if action.size < event.target_size:
            return 0.25
    candidate_difficulty = _difficulty(kind, action.rank, cur_rank)
    target_difficulty = _difficulty(target_kind, event.target_rank, cur_rank)
    if candidate_difficulty >= target_difficulty:
        return 1.0
    if target_difficulty <= 0.0:
        return 0.5
    return min(0.5, 0.2 + 0.3 * candidate_difficulty / target_difficulty)


def _spent_count(seat: SeatPlayStats, kind: str) -> int:
    kind = normalize_kind(kind)
    if kind == "Straight":
        return seat.straight_actions
    if kind == "TriplePlus":
        return seat.triple_plus_actions
    if kind in BOMB_KINDS:
        return seat.bomb_actions
    return sum(1 for action in seat.actions if normalize_kind(action.kind) == kind)


def estimate_seat_response(
    ctx: RetrievalContext,
    action: ActionCandidate,
    seat_index: int,
) -> SeatResponseEstimate:
    memory = ctx.card_memory
    if memory is None or action.kind == "PASS" or not action.cards:
        return SeatResponseEstimate(BASE_RESPONSE_RISK, 0.0, 0.0, 0.0, 0.0)
    seat = memory.seats[seat_index]
    event_count = max(1, len(memory.actions))
    pass_evidence = 0.0
    response_evidence = 0.0
    for event in seat.response_events:
        relevance = _target_relevance(action, event, ctx.cur_rank)
        if relevance <= 0.0:
            continue
        recency = 0.6 + 0.4 * min(1.0, (event.action_index + 1) / event_count)
        if event.passed:
            strategic_weight = 0.15 if event.target_was_teammate else 1.0
            pass_evidence += relevance * recency * strategic_weight
        else:
            # Actually beating a matching target is positive capability evidence.
            response_evidence += relevance * recency * 0.85

    spent_evidence = min(3.0, float(_spent_count(seat, action.kind))) * 0.35
    prior_logit = math.log(BASE_RESPONSE_RISK / (1.0 - BASE_RESPONSE_RISK))
    logit = (
        prior_logit
        - 1.6 * pass_evidence
        + 1.1 * response_evidence
        - 0.28 * spent_evidence
    )
    risk = 1.0 / (1.0 + math.exp(-logit))
    risk = min(MAX_RESPONSE_RISK, max(MIN_RESPONSE_RISK, risk))
    evidence = 1.0 - math.exp(
        -(pass_evidence + response_evidence + spent_evidence)
    )
    return SeatResponseEstimate(
        risk=risk,
        evidence=min(1.0, max(0.0, evidence)),
        pass_evidence=pass_evidence,
        response_evidence=response_evidence,
        spent_evidence=spent_evidence,
    )


def candidate_response_profile(
    ctx: RetrievalContext,
    action: ActionCandidate,
) -> CandidateResponseProfile:
    if normalize_kind(action.kind) == "PASS":
        target_kind = normalize_kind(ctx.current_kind)
        if target_kind != "Lead":
            target_size = max(
                1,
                int(ctx.current_size or action_type_size(target_kind)),
            )
            action = ActionCandidate(
                index=-1,
                kind=target_kind,
                cards=tuple("__target__" for _ in range(target_size)),
                rank=ctx.current_rank,
            )
    left = estimate_seat_response(ctx, action, 1)
    teammate = estimate_seat_response(ctx, action, 2)
    right = estimate_seat_response(ctx, action, 3)
    evidence_strength = max(left.evidence, right.evidence)
    opponent_risk = max(left.risk, right.risk)
    type_suppression = evidence_strength * (1.0 - opponent_risk)
    next_suppression = left.evidence * (1.0 - left.risk)
    combined_suppression = 0.6 * next_suppression + 0.4 * type_suppression
    return CandidateResponseProfile(
        left_risk=left.risk,
        teammate_risk=teammate.risk,
        right_risk=right.risk,
        left_evidence=left.evidence,
        teammate_evidence=teammate.evidence,
        right_evidence=right.evidence,
        left_pass_evidence=left.pass_evidence,
        teammate_pass_evidence=teammate.pass_evidence,
        right_pass_evidence=right.pass_evidence,
        left_positive_response_evidence=left.response_evidence,
        teammate_positive_response_evidence=teammate.response_evidence,
        right_positive_response_evidence=right.response_evidence,
        left_spent_evidence=left.spent_evidence,
        teammate_spent_evidence=teammate.spent_evidence,
        right_spent_evidence=right.spent_evidence,
        next_risk=left.risk,
        opponent_risk=opponent_risk,
        evidence_strength=evidence_strength,
        type_suppression=type_suppression,
        next_suppression=next_suppression,
        combined_suppression=combined_suppression,
    )
