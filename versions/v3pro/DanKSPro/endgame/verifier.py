"""Progressively verify endgame proposals against the frozen V3Pro baseline."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Callable


@dataclass(frozen=True)
class VerificationAttempt:
    proposal_index: int
    proposal_rank: int
    status: str
    selected_index: int
    reason: str
    search_result: Any


@dataclass(frozen=True)
class ProgressiveVerificationResult:
    selected_index: int
    applied: bool
    accepted_rank: int | None
    reason: str
    attempts: tuple[VerificationAttempt, ...]
    short_circuited: bool


@dataclass(frozen=True)
class CascadedVerificationAttempt:
    proposal_index: int
    proposal_rank: int
    stage: str
    status: str
    selected_index: int
    reason: str
    search_result: Any


@dataclass(frozen=True)
class CascadedVerificationResult:
    selected_index: int
    applied: bool
    accepted_rank: int | None
    reason: str
    attempts: tuple[CascadedVerificationAttempt, ...]
    short_circuited: bool


@dataclass(frozen=True)
class BlueprintJointVerificationResult:
    selected_index: int
    applied: bool
    accepted_rank: int | None
    reason: str
    approved_indices: tuple[int, ...]
    blueprint_attempts: tuple[VerificationAttempt, ...]
    joint_result: Any | None


def verify_candidate_pool_sequentially(
    *,
    baseline_index: int,
    proposed_indices: tuple[int, ...],
    verify_one: Callable[[int, int], Any],
) -> ProgressiveVerificationResult:
    """Try proposals in priority order and stop at the first safe conclusion.

    A complete baseline selection is the only outcome that advances to the next
    proposal.  Every incomplete or structurally invalid result fails closed.
    """

    baseline = int(baseline_index)
    proposals = tuple(int(value) for value in proposed_indices)
    if not proposals:
        raise ValueError("proposal pool must not be empty")
    if baseline in proposals:
        raise ValueError("proposal pool must not contain the baseline")
    if len(set(proposals)) != len(proposals):
        raise ValueError("proposal pool must not contain duplicates")

    attempts: list[VerificationAttempt] = []
    for rank, proposal in enumerate(proposals, 1):
        search_result = verify_one(proposal, rank)
        status = str(search_result.status)
        selected = int(search_result.selected_index)
        if status != "complete":
            reason = "verification_incomplete"
        elif selected == baseline:
            reason = "verification_rejected"
        elif selected == proposal:
            reason = "verified"
        else:
            reason = "verification_unexpected_selection"
        attempts.append(
            VerificationAttempt(
                proposal_index=proposal,
                proposal_rank=rank,
                status=status,
                selected_index=selected,
                reason=reason,
                search_result=search_result,
            )
        )

        has_remaining = rank < len(proposals)
        if reason == "verification_rejected":
            continue
        if reason == "verified":
            return ProgressiveVerificationResult(
                selected_index=proposal,
                applied=True,
                accepted_rank=rank,
                reason=reason,
                attempts=tuple(attempts),
                short_circuited=has_remaining,
            )
        return ProgressiveVerificationResult(
            selected_index=baseline,
            applied=False,
            accepted_rank=None,
            reason=reason,
            attempts=tuple(attempts),
            short_circuited=has_remaining,
        )

    return ProgressiveVerificationResult(
        selected_index=baseline,
        applied=False,
        accepted_rank=None,
        reason="verification_rejected",
        attempts=tuple(attempts),
        short_circuited=False,
    )


def verify_candidate_pool_blueprint_first(
    *,
    baseline_index: int,
    proposed_indices: tuple[int, ...],
    verify_blueprint: Callable[[int, int], Any],
    verify_minimax: Callable[[int, int], Any],
) -> CascadedVerificationResult:
    """Require a cheap blueprint proof before exact minimax for each proposal."""

    baseline = int(baseline_index)
    proposals = tuple(int(value) for value in proposed_indices)
    if not proposals:
        raise ValueError("proposal pool must not be empty")
    if baseline in proposals:
        raise ValueError("proposal pool must not contain the baseline")
    if len(set(proposals)) != len(proposals):
        raise ValueError("proposal pool must not contain duplicates")

    attempts: list[CascadedVerificationAttempt] = []
    for rank, proposal in enumerate(proposals, 1):
        blueprint_result = verify_blueprint(proposal, rank)
        blueprint_status = str(blueprint_result.status)
        blueprint_selected = int(blueprint_result.selected_index)
        if blueprint_status != "complete":
            blueprint_reason = "blueprint_guard_incomplete"
        elif blueprint_selected == baseline:
            blueprint_reason = "blueprint_guard_rejected"
        elif blueprint_selected == proposal:
            blueprint_reason = "blueprint_verified"
        else:
            blueprint_reason = "blueprint_guard_unexpected_selection"
        attempts.append(
            CascadedVerificationAttempt(
                proposal_index=proposal,
                proposal_rank=rank,
                stage="blueprint",
                status=blueprint_status,
                selected_index=blueprint_selected,
                reason=blueprint_reason,
                search_result=blueprint_result,
            )
        )
        if blueprint_reason == "blueprint_guard_rejected":
            continue
        if blueprint_reason != "blueprint_verified":
            return CascadedVerificationResult(
                selected_index=baseline,
                applied=False,
                accepted_rank=None,
                reason=blueprint_reason,
                attempts=tuple(attempts),
                short_circuited=rank < len(proposals),
            )

        minimax_result = verify_minimax(proposal, rank)
        minimax_status = str(minimax_result.status)
        minimax_selected = int(minimax_result.selected_index)
        if minimax_status != "complete":
            minimax_reason = "verification_incomplete"
        elif minimax_selected == baseline:
            minimax_reason = "verification_rejected"
        elif minimax_selected == proposal:
            minimax_reason = "verified"
        else:
            minimax_reason = "verification_unexpected_selection"
        attempts.append(
            CascadedVerificationAttempt(
                proposal_index=proposal,
                proposal_rank=rank,
                stage="minimax",
                status=minimax_status,
                selected_index=minimax_selected,
                reason=minimax_reason,
                search_result=minimax_result,
            )
        )
        if minimax_reason == "verification_rejected":
            continue
        if minimax_reason == "verified":
            return CascadedVerificationResult(
                selected_index=proposal,
                applied=True,
                accepted_rank=rank,
                reason="verified_minimax_and_blueprint",
                attempts=tuple(attempts),
                short_circuited=rank < len(proposals),
            )
        return CascadedVerificationResult(
            selected_index=baseline,
            applied=False,
            accepted_rank=None,
            reason=minimax_reason,
            attempts=tuple(attempts),
            short_circuited=rank < len(proposals),
        )

    return CascadedVerificationResult(
        selected_index=baseline,
        applied=False,
        accepted_rank=None,
        reason="blueprint_guard_rejected",
        attempts=tuple(attempts),
        short_circuited=False,
    )


def verify_candidate_pool_blueprint_then_joint(
    *,
    baseline_index: int,
    proposed_indices: tuple[int, ...],
    verify_blueprint: Callable[[int, int], Any],
    verify_joint: Callable[[tuple[int, ...]], Any],
) -> BlueprintJointVerificationResult:
    """Filter every proposal with Blueprint, then compare survivors jointly.

    Every Blueprint result must be complete and structurally valid.  An
    incomplete or unexpected result fails the entire root decision closed;
    rejected candidates are simply excluded.  The joint verifier may select
    only the baseline or one of the Blueprint-approved alternatives.
    """

    baseline = int(baseline_index)
    proposals = tuple(int(value) for value in proposed_indices)
    if not proposals:
        raise ValueError("proposal pool must not be empty")
    if baseline in proposals:
        raise ValueError("proposal pool must not contain the baseline")
    if len(set(proposals)) != len(proposals):
        raise ValueError("proposal pool must not contain duplicates")

    approved: list[int] = []
    attempts: list[VerificationAttempt] = []
    for rank, proposal in enumerate(proposals, 1):
        result = verify_blueprint(proposal, rank)
        status = str(result.status)
        selected = int(result.selected_index)
        if status != "complete":
            reason = "blueprint_guard_incomplete"
        elif selected == baseline:
            reason = "blueprint_guard_rejected"
        elif selected == proposal:
            reason = "blueprint_verified"
        else:
            reason = "blueprint_guard_unexpected_selection"
        attempts.append(
            VerificationAttempt(
                proposal_index=proposal,
                proposal_rank=rank,
                status=status,
                selected_index=selected,
                reason=reason,
                search_result=result,
            )
        )
        if reason == "blueprint_guard_rejected":
            continue
        if reason == "blueprint_verified":
            approved.append(proposal)
            continue
        return BlueprintJointVerificationResult(
            selected_index=baseline,
            applied=False,
            accepted_rank=None,
            reason=reason,
            approved_indices=tuple(approved),
            blueprint_attempts=tuple(attempts),
            joint_result=None,
        )

    if not approved:
        return BlueprintJointVerificationResult(
            selected_index=baseline,
            applied=False,
            accepted_rank=None,
            reason="blueprint_guard_rejected",
            approved_indices=(),
            blueprint_attempts=tuple(attempts),
            joint_result=None,
        )

    joint_result = verify_joint(tuple(approved))
    joint_status = str(joint_result.status)
    joint_selected = int(joint_result.selected_index)
    if joint_status != "complete":
        reason = "verification_incomplete"
        applied = False
    elif joint_selected == baseline:
        reason = "verification_rejected"
        applied = False
    elif joint_selected in approved:
        reason = "verified_minimax_and_blueprint"
        applied = True
    else:
        reason = "verification_unexpected_selection"
        applied = False

    return BlueprintJointVerificationResult(
        selected_index=joint_selected if applied else baseline,
        applied=applied,
        accepted_rank=(proposals.index(joint_selected) + 1) if applied else None,
        reason=reason,
        approved_indices=tuple(approved),
        blueprint_attempts=tuple(attempts),
        joint_result=joint_result,
    )


__all__ = [
    "BlueprintJointVerificationResult",
    "CascadedVerificationAttempt",
    "CascadedVerificationResult",
    "ProgressiveVerificationResult",
    "VerificationAttempt",
    "verify_candidate_pool_blueprint_first",
    "verify_candidate_pool_blueprint_then_joint",
    "verify_candidate_pool_sequentially",
]
