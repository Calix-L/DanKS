"""Bounded, isolated two-root recovery; never relabels the original pool."""

from __future__ import annotations

import copy

from dataclasses import asdict, is_dataclass

import math

CONTRACT = "r13_isolated_complete_singleton_v1"

INCOMPLETE = {"node_budget_exceeded", "total_node_budget_exceeded"}

INITIAL_NODES = 500000

RETRY_NODES = 2000000

PER_WORLD_NODES = 250000


def _strict_candidate(row, helper):
    return (
        row["feasible"] is True
        and row["rejection_reason"] is None
        and row["mean_value_margin"] > 1e-12
        and all(row[k] >= 0 for k in helper.MARGINS)
    )


def _eligible(
    decision, baseline, proposals, config, route, calls, helper, retry_budget
):
    helper.validate_blueprint_tie_contract(config)
    if (
        route not in ("extended", "extended_outer")
        or config.get("verify_blueprint_tie_fallback") is not True
        or config.get("verify_max_total_nodes") != INITIAL_NODES
        or config.get("verify_max_nodes_per_allocation") != PER_WORLD_NODES
        or config.get("verify_retry_max_total_nodes") != RETRY_NODES
        or not helper._indices(proposals)
        or baseline in proposals
        or type(baseline) is not int
        or baseline < 0
        or not is_dataclass(decision)
        or isinstance(decision, type)
        or decision.applied is not False
        or type(decision.selected_index) is not int
        or decision.selected_index != baseline
        or decision.accepted_rank is not None
        or decision.reason != "verification_incomplete"
        or not helper._indices(decision.approved_indices)
        or len(decision.approved_indices) < 2
    ):
        raise ValueError("ineligible original decision or contract")
    joint = decision.joint_result
    if not is_dataclass(joint) or joint.status not in INCOMPLETE:
        raise ValueError("no budget-incomplete original pool")
    expected_stages = (
        ("initial", "retry")
        if retry_budget(route=route, status=joint.status, config=config) is not None
        else ("initial",)
    )
    if (
        type(calls) not in (tuple, list)
        or tuple(c[0] for c in calls) != expected_stages
        or calls[-1][1] is not joint
    ):
        raise ValueError("original retry sequence not exhausted")
    if type(decision.blueprint_attempts) is not tuple or len(
        decision.blueprint_attempts
    ) != len(proposals):
        raise ValueError("missing full blueprint pool")
    approved, candidates, world_counts = [], [], None
    alpha = config.get("verify_risk_cvar_alpha", 1 / 3)
    if not helper._finite(alpha) or not 0 < alpha <= 1:
        raise ValueError("invalid alpha")
    for rank, (index, attempt) in enumerate(
        zip(proposals, decision.blueprint_attempts, strict=True), 1
    ):
        if (
            not is_dataclass(attempt)
            or type(attempt.proposal_rank) is not int
            or attempt.proposal_rank != rank
            or type(attempt.proposal_index) is not int
            or attempt.proposal_index != index
            or attempt.status != "complete"
            or type(attempt.selected_index) is not int
        ):
            raise ValueError("invalid blueprint attempt metadata")
        rows, counts = helper._proof_rows(
            attempt.search_result,
            baseline=baseline,
            indices=(baseline, index),
            alpha=alpha,
        )
        if world_counts is not None and counts != world_counts:
            raise ValueError("blueprint world mismatch")
        world_counts = counts
        if attempt.selected_index != attempt.search_result.selected_index:
            raise ValueError("blueprint selection mismatch")
        if (
            attempt.reason == "blueprint_guard_rejected"
            and attempt.selected_index == baseline
        ):
            continue
        if (
            attempt.reason != "blueprint_verified"
            or attempt.selected_index != index
            or not _strict_candidate(rows[index], helper)
        ):
            raise ValueError("blueprint approval lacks strict proof")
        approved.append(index)
        candidates.append(
            (rows[index]["mean_value_margin"], rank, index, attempt.search_result)
        )
    if tuple(approved) != decision.approved_indices:
        raise ValueError("approved pool disagrees with full blueprint attempts")
    for _, old in calls:
        if (
            not is_dataclass(old)
            or old.status not in INCOMPLETE
            or type(old.incomplete_reason) is not str
            or not old.incomplete_reason
            or type(old.selected_index) is not int
            or old.selected_index != baseline
            or type(old.proposed_index) is not int
            or old.proposed_index != baseline
            or old.selection_reason != "incomplete_preserves_v3pro"
            or old.risk_contract != "v3pro_all_candidate_risk_constrained_v2"
            or old.belief_contract != "uniform_physical_prior_v1"
            or old.risk_metrics != {}
            or old.action_rows != ()
            or old.candidate_risk_rows != ()
            or type(old.root_action_count) is not int
            or old.root_action_count != 1 + len(approved)
            or type(old.total_allocations) is not int
            or type(old.total_physical_allocations) is not int
            or (old.total_allocations, old.total_physical_allocations) != world_counts
            or any(
                type(v) is not int or v < 0
                for v in (
                    old.nodes,
                    old.completed_allocations,
                    old.completed_physical_weight,
                )
            )
            or old.completed_allocations >= world_counts[0]
            or old.completed_physical_weight >= world_counts[1]
            or old.completed_physical_weight < old.completed_allocations
            or (
                (old.completed_allocations == 0) != (old.completed_physical_weight == 0)
            )
        ):
            raise ValueError("invalid original incomplete proof metadata")
    return sorted(candidates, key=lambda c: (-c[0], c[1]))[:2], world_counts


def recover(
    *,
    decision,
    baseline_index,
    proposed_indices,
    route,
    config,
    table,
    helper,
    verifier,
    enabled,
    original_calls,
    retry_budget,
    verify_exact,
):
    """Try independent singleton subproblems, preserving every original object.

    ``verify_exact`` must apply the supplied TT override and restore engine state.
    Initial and retry within one candidate share a clone, never the episode TT;
    each next candidate starts from a fresh copy of the original post-r11 TT.
    """
    out = dict(
        contract=CONTRACT,
        status="not_eligible",
        applied=False,
        selected_index=baseline_index,
        original_proposal_rank=None,
        local_proposal_rank=None,
        calls=[],
        certificate=None,
        certificate_kind=None,
        isolation_nodes=0,
        isolation_retry_nodes=0,
        isolation_elapsed_sec=0.0,
        nominal_node_budget=0,
    )
    if enabled is not True:
        out["status"] = "disabled"
        return out
    try:
        candidates, counts = _eligible(
            decision,
            baseline_index,
            proposed_indices,
            config,
            route,
            original_calls,
            helper,
            retry_budget,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        out["not_eligible_reason"] = str(exc)
        return out
    out.update(
        status="no_complete_certificate",
        full_pool=asdict(decision),
        full_pool_actual_root_indices=[baseline_index, *decision.approved_indices],
        full_pool_calls=[
            dict(
                phase="original_full_pool",
                stage=s,
                actual_root_indices=[baseline_index, *decision.approved_indices],
                search_result=asdict(r),
            )
            for s, r in original_calls
        ],
        priority=[
            dict(proposal_index=i, original_proposal_rank=r, blueprint_gain=g)
            for g, r, i, _ in candidates
        ],
        maximum_additional_nominal_nodes=2 * (INITIAL_NODES + RETRY_NODES),
    )
    for _, original_rank, index, bp in candidates:
        branch_table = copy.deepcopy(table)
        roots = (baseline_index, index)
        for stage, budget in (("initial", INITIAL_NODES), ("retry", RETRY_NODES)):
            if (
                stage == "retry"
                and retry_budget(route=route, status=exact.status, config=config)
                is None
            ):
                break
            out["nominal_node_budget"] += budget
            exact = verify_exact(
                (index,), max_total_nodes=budget, table_override=branch_table
            )
            if (
                type(exact.nodes) is not int
                or exact.nodes < 0
                or type(exact.elapsed_sec) not in (float, int)
                or not math.isfinite(exact.elapsed_sec)
                or exact.elapsed_sec < 0
            ):
                raise ValueError("invalid isolated work accounting")
            out["isolation_nodes"] += exact.nodes
            out["isolation_retry_nodes"] += exact.nodes if stage == "retry" else 0
            out["isolation_elapsed_sec"] += exact.elapsed_sec
            out["calls"].append(
                dict(
                    call_id=len(out["calls"]) + 1,
                    phase="isolated_singleton",
                    stage=stage,
                    actual_root_indices=list(roots),
                    original_proposal_rank=original_rank,
                    local_proposal_rank=1,
                    max_total_nodes=budget,
                    max_nodes_per_allocation=PER_WORLD_NODES,
                    max_allocations=config["verify_max_allocations"],
                    search_result=asdict(exact),
                )
            )
            if exact.status == "complete":
                break
            if exact.status not in INCOMPLETE:
                break
        try:
            rows, exact_counts = helper._proof_rows(
                exact,
                baseline=baseline_index,
                indices=roots,
                alpha=config.get("verify_risk_cvar_alpha", 1 / 3),
            )
            if exact_counts != counts:
                raise ValueError("singleton worlds differ from original blueprint")
            singleton = verifier.verify_candidate_pool_blueprint_then_joint(
                baseline_index=baseline_index,
                proposed_indices=(index,),
                verify_blueprint=lambda _i, _rank: bp,
                verify_joint=lambda _indices: exact,
            )
            tie_proof = None
            if singleton.applied:
                if singleton.selected_index != index or not _strict_candidate(
                    rows[index], helper
                ):
                    raise ValueError("strict exact approval lacks strict row proof")
                kind = "strict_exact_and_blueprint"
            else:
                singleton, tie_proof = helper.apply_blueprint_tie_fallback(
                    singleton,
                    baseline_index=baseline_index,
                    proposed_indices=(index,),
                    route="extended_outer",
                    config=config,
                )
                kind = "complete_exact_tie_and_blueprint_gain"
            if not singleton.applied:
                continue
            if singleton.accepted_rank != 1 or singleton.selected_index != index:
                raise ValueError("invalid singleton selection")
            out.update(
                status="recovered",
                applied=True,
                selected_index=index,
                original_proposal_rank=original_rank,
                local_proposal_rank=1,
                certificate=asdict(singleton),
                certificate_kind=kind,
                tie_proof=tie_proof,
                certificate_actual_root_indices=list(roots),
            )
            return out
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            out["calls"][-1]["certificate_rejection"] = str(exc)
    return out
