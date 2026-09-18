"""Complete-proof secondary Blueprint tie-break; never changes search results."""

from __future__ import annotations

import math

from dataclasses import is_dataclass, replace

from typing import Any

MARGINS = (
    "mean_value_margin",
    "worst_value_margin",
    "win_probability_margin",
    "paired_regret_min",
    "paired_regret_cvar",
)

THRESHOLDS = (
    "minimum_mean_value_margin",
    "minimum_worst_value_margin",
    "minimum_win_probability_margin",
    "minimum_paired_regret_min",
    "minimum_paired_cvar_margin",
)

VALUES = ("mean_value", "min_value", "win_probability")

REASON = "verified_minimax_tie_and_blueprint_gain"

CONSISTENCY_ABS_TOL = 1e-12


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_blueprint_tie_contract(config: dict[str, Any]) -> None:
    """Keep r4 defaults; require the exact declared opt-in experiment contract."""
    enabled = config.get("verify_blueprint_tie_fallback", False)
    if enabled is False or enabled is None:
        return
    if (
        enabled is not True
        or any(
            config.get(key) is not True
            for key in (
                "verify_complete_small_proposals",
                "extended_full_legal_proposals",
                "exact_verify_specialist",
                "verify_blueprint_guard",
                "verify_root_transaction",
            )
        )
        or type(config.get("extended_max_total_remaining")) is not int
        or config.get("extended_max_total_remaining") != 16
        or type(config.get("extended_max_allocations")) is not int
        or config.get("extended_max_allocations") != 128
        or config.get("verify_risk_mode") != "strict"
        or config.get("verify_candidate_mode") != "joint"
        or config.get("verify_blueprint_order") != "all_before_joint"
        or any(
            not _finite(config.get("verify_" + key)) or config["verify_" + key] != 0.0
            for key in THRESHOLDS
        )
    ):
        raise ValueError(
            "verify-blueprint-tie-fallback requires literal complete-small/full-legal/"
            "exact/Blueprint/transactional flags, integer max16/cap128, strict "
            "joint/all-before-joint and all five finite zero risk margins"
        )


def _indices(values: Any) -> bool:
    return (
        type(values) is tuple
        and all(type(value) is int and value >= 0 for value in values)
        and len(set(values)) == len(values)
    )


def _rows_by_index(rows: Any, indices: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    if type(rows) not in (tuple, list) or len(rows) != len(indices):
        raise ValueError("missing or duplicate proof rows")
    result = {}
    for row in rows:
        if type(row) is not dict or type(row.get("action_index")) is not int:
            raise ValueError("malformed proof row")
        index = row["action_index"]
        if index in result or index not in indices:
            raise ValueError("ambiguous proof row")
        result[index] = row
    if set(result) != set(indices):
        raise ValueError("proof rows do not cover actual candidates")
    return result


def _proof_rows(
    result: Any, *, baseline: int, indices: tuple[int, ...], alpha: float
) -> tuple[dict[int, dict[str, Any]], tuple[int, int]]:
    """Validate frozen result shape and row consistency, without inventing proof."""
    if not is_dataclass(result) or isinstance(result, type):
        raise ValueError("not a structured search result")
    if (
        result.status != "complete"
        or result.incomplete_reason is not None
        or type(result.selected_index) is not int
        or result.selected_index not in indices
        or type(result.proposed_index) is not int
        or result.proposed_index not in indices
        or result.risk_contract != "v3pro_all_candidate_risk_constrained_v2"
        or result.belief_contract != "uniform_physical_prior_v1"
        or type(result.root_action_count) is not int
        or result.root_action_count != len(indices)
    ):
        raise ValueError("incomplete or unexpected search contract")
    counts = (
        result.total_allocations,
        result.total_physical_allocations,
        result.completed_allocations,
        result.completed_physical_weight,
    )
    if (
        any(type(value) is not int or value <= 0 for value in counts)
        or counts[0] != counts[2]
        or counts[1] != counts[3]
        or counts[1] < counts[0]
    ):
        raise ValueError("allocation proof is not complete")
    rows = _rows_by_index(result.candidate_risk_rows, indices)
    actions = _rows_by_index(result.action_rows, indices)
    baseline_row = rows[baseline]
    for row in rows.values():
        if (
            any(not _finite(row.get(key)) for key in (*VALUES, *MARGINS))
            or not 0 <= row["win_probability"] <= 1
            or row["min_value"] > row["mean_value"] + CONSISTENCY_ABS_TOL
            or type(row.get("feasible")) is not bool
            or (row["feasible"] and row.get("rejection_reason") is not None)
            or (not row["feasible"] and type(row.get("rejection_reason")) is not str)
        ):
            raise ValueError("invalid candidate risk row")
        for value_key, margin_key in (
            ("mean_value", "mean_value_margin"),
            ("min_value", "worst_value_margin"),
            ("win_probability", "win_probability_margin"),
        ):
            if row[value_key] - baseline_row[value_key] != row[margin_key]:
                raise ValueError("risk margins disagree with baseline values")
        action = actions[row["action_index"]]
        for key in ("mean_value", "min_value"):
            if not _finite(action.get(key)) or action[key] != row[key]:
                raise ValueError("action and risk values disagree")
        if (
            not _finite(action.get("max_value"))
            or action["max_value"] < row["mean_value"] - CONSISTENCY_ABS_TOL
            or not _finite(action.get("lower_tail_cvar"))
            or not row["min_value"] - CONSISTENCY_ABS_TOL
            <= action["lower_tail_cvar"]
            <= action["max_value"] + CONSISTENCY_ABS_TOL
        ):
            raise ValueError("invalid action value bounds")
        for suffix, total in (
            ("physical_weight", counts[1]),
            ("unique_allocations", counts[0]),
        ):
            weights = [
                action.get(sign + "_" + suffix)
                for sign in ("positive", "negative", "zero")
            ]
            if (
                any(type(value) is not int or value < 0 for value in weights)
                or sum(weights) != total
            ):
                raise ValueError("action counts do not match completed proof")
        if not math.isclose(
            action["positive_physical_weight"] / counts[1],
            row["win_probability"],
            rel_tol=0.0,
            abs_tol=CONSISTENCY_ABS_TOL,
        ):
            raise ValueError("win probability disagrees with physical weights")
    if (
        baseline_row["feasible"] is not True
        or baseline_row["rejection_reason"] is not None
        or any(baseline_row[key] != 0.0 for key in MARGINS)
    ):
        raise ValueError("baseline is not the zero-margin reference")
    if result.selected_index == baseline and result.proposed_index == baseline:
        reason = "baseline_is_mean_optimal"
    elif result.selected_index == baseline:
        reason = "rejected_" + str(rows[result.proposed_index]["rejection_reason"])
    elif result.selected_index == result.proposed_index:
        reason = "risk_approved_override"
    else:
        reason = "selected_best_feasible_override"
    if result.selection_reason != reason:
        raise ValueError("selection reason disagrees with search decision")
    metrics = result.risk_metrics
    if (
        type(metrics) is not dict
        or metrics.get("risk_mode") != "strict"
        or not _finite(metrics.get("cvar_alpha"))
        or metrics["cvar_alpha"] != alpha
        or any(
            not _finite(metrics.get(key)) or metrics[key] != 0.0 for key in THRESHOLDS
        )
    ):
        raise ValueError("unexpected risk thresholds")
    for key in VALUES:
        value = metrics.get("baseline_" + key)
        if not _finite(value) or value != baseline_row[key]:
            raise ValueError("invalid baseline metrics")
    for key in (*VALUES, *MARGINS):
        if (
            not _finite(metrics.get(key))
            or metrics[key] != rows[result.selected_index][key]
        ):
            raise ValueError("selected metrics disagree with selected row")
    return rows, counts[:2]


def apply_blueprint_tie_fallback(
    decision: Any,
    *,
    baseline_index: int,
    proposed_indices: tuple[int, ...],
    route: str,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Return the original decision by identity unless every tie proof is valid."""
    if (
        config.get("verify_blueprint_tie_fallback") is not True
        or route != "extended_outer"
    ):
        return decision, None
    if getattr(decision, "applied", None) is True:
        return decision, None
    try:
        validate_blueprint_tie_contract(config)
        if (
            not is_dataclass(decision)
            or isinstance(decision, type)
            or type(baseline_index) is not int
            or baseline_index < 0
            or not _indices(proposed_indices)
            or not proposed_indices
            or baseline_index in proposed_indices
            or type(decision.selected_index) is not int
            or decision.selected_index != baseline_index
            or decision.applied is not False
            or decision.accepted_rank is not None
            or decision.reason != "verification_rejected"
            or not _indices(decision.approved_indices)
            or not decision.approved_indices
            or any(index not in proposed_indices for index in decision.approved_indices)
        ):
            return decision, None
        alpha = config.get("verify_risk_cvar_alpha", 1 / 3)
        if not _finite(alpha) or not 0 < alpha <= 1:
            return decision, None
        exact = decision.joint_result
        rows, counts = _proof_rows(
            exact,
            baseline=baseline_index,
            indices=(baseline_index, *decision.approved_indices),
            alpha=alpha,
        )
        if exact.selected_index != baseline_index:
            return decision, None
        attempts = decision.blueprint_attempts
        if type(attempts) is not tuple or len(attempts) != len(proposed_indices):
            return decision, None
        approved = []
        eligible = []
        blueprint_rows = {}
        for rank, (proposal, attempt) in enumerate(
            zip(proposed_indices, attempts, strict=True), 1
        ):
            if (
                not is_dataclass(attempt)
                or isinstance(attempt, type)
                or type(attempt.proposal_index) is not int
                or attempt.proposal_index != proposal
                or type(attempt.proposal_rank) is not int
                or attempt.proposal_rank != rank
                or attempt.status != "complete"
                or type(attempt.selected_index) is not int
            ):
                return decision, None
            bp = attempt.search_result
            bp_rows, bp_counts = _proof_rows(
                bp,
                baseline=baseline_index,
                indices=(baseline_index, proposal),
                alpha=alpha,
            )
            if bp_counts != counts or bp.selected_index != attempt.selected_index:
                return decision, None
            if (
                attempt.reason == "blueprint_guard_rejected"
                and attempt.selected_index == baseline_index
            ):
                continue
            if (
                attempt.reason != "blueprint_verified"
                or attempt.selected_index != proposal
            ):
                return decision, None
            approved.append(proposal)
            if proposal not in decision.approved_indices:
                return decision, None
            candidate, bp_candidate = rows[proposal], bp_rows[proposal]
            blueprint_rows[proposal] = bp_candidate
            if (
                candidate["feasible"] is False
                and candidate["rejection_reason"] == "insufficient_mean_improvement"
                and all(candidate[key] == 0.0 for key in MARGINS)
                and bp_candidate["feasible"] is True
                and bp_candidate["rejection_reason"] is None
                and bp_candidate["mean_value_margin"] > 1e-12
                and all(
                    bp_candidate[key] >= 0.0
                    for key in MARGINS
                    if key != "mean_value_margin"
                )
            ):
                eligible.append((bp_candidate["mean_value_margin"], -rank, proposal))
        if tuple(approved) != decision.approved_indices or not eligible:
            return decision, None
        gain, negative_rank, selected = max(eligible)
        rank = -negative_rank
        proof = {
            "contract": "complete_exact_tie_strict_blueprint_gain_v1",
            "reason": REASON,
            "selected_index": selected,
            "proposal_rank": rank,
            "exact_selected_index": exact.selected_index,
            "exact_selection_reason": exact.selection_reason,
            "exact_margins": {key: rows[selected][key] for key in MARGINS},
            "blueprint_mean_gain": gain,
            "blueprint_margins": {
                key: blueprint_rows[selected][key] for key in MARGINS
            },
            "total_allocations": counts[0],
            "total_physical_allocations": counts[1],
            "risk_contract": exact.risk_contract,
            "belief_contract": exact.belief_contract,
        }
        return (
            replace(
                decision,
                selected_index=selected,
                applied=True,
                accepted_rank=rank,
                reason=REASON,
            ),
            proof,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return decision, None
