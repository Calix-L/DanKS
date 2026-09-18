"""Exhaustive hidden-allocation search for very small V3Pro endgames.

This enumerates every publicly legal determinization and solves every root action
with perfect-information partnership minimax.  It is exhaustive over
determinizations, but is not a strategy-fusion-free imperfect-information solver.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Callable
from .belief import (
    LEGAL_GRADES,
    PUBLIC_BELIEF_CONTRACT,
    UNIFORM_BELIEF_CONTRACT,
    posterior_from_log_likelihoods,
    weighted_grade_distribution,
)

RISK_SELECTION_CONTRACT = "v3pro_all_candidate_risk_constrained_v2"
RISK_CONFIG_KEYS = (
    "risk_mode",
    "risk_cvar_alpha",
    "minimum_mean_value_margin",
    "minimum_worst_value_margin",
    "minimum_paired_cvar_margin",
    "minimum_paired_regret_min",
    "minimum_win_probability_margin",
)


def _action_kind_and_size(action: Any) -> tuple[str, int]:
    convert = getattr(action, "to_json", None)
    if callable(convert):
        action = convert()
    if isinstance(action, dict):
        kind = action.get("kind", action.get("type", ""))
        cards = action.get("cards") or ()
    elif isinstance(action, (list, tuple)) and len(action) >= 3:
        kind = action[0]
        cards = action[2] if isinstance(action[2], (list, tuple)) else ()
    else:
        kind = getattr(action, "kind", getattr(action, "type", ""))
        cards = getattr(action, "cards", ()) or ()
    normalized = {"single": "Single", "pair": "Pair", "pass": "PASS"}.get(
        str(kind).strip().lower(), str(kind).strip()
    )
    return (normalized, len(cards))


def downstream_feed_action_indices(
    actions: list[Any],
    *,
    actor_seat: int,
    actor_hand_size: int,
    public_counts: list[int],
) -> tuple[int, ...]:
    """Identify nonterminal single/pair leads matching next opponent's hand size."""
    if len(public_counts) != 4 or actor_seat not in range(4):
        return ()
    downstream = int(public_counts[(int(actor_seat) + 1) % 4])
    risky_kind = "Single" if downstream == 1 else "Pair" if downstream == 2 else None
    if risky_kind is None:
        return ()
    result = []
    for index, action in enumerate(actions):
        kind, size = _action_kind_and_size(action)
        if kind == risky_kind and size < int(actor_hand_size):
            result.append(index)
    return tuple(result)


@dataclass(frozen=True)
class ExhaustiveDeterminizationResult:
    status: str
    selected_index: int
    proposed_index: int
    selection_reason: str
    risk_contract: str
    risk_metrics: dict[str, float]
    belief_contract: str
    belief_diagnostics: dict[str, float | int]
    total_allocations: int
    total_physical_allocations: int
    completed_allocations: int
    completed_physical_weight: int
    root_action_count: int
    nodes: int
    policy_decisions: int
    expanded_actions: int
    cutoffs: int
    cache_lookups: int
    cache_hits: int
    cache_entries: int
    elapsed_sec: float
    action_rows: tuple[dict[str, Any], ...]
    candidate_risk_rows: tuple[dict[str, Any], ...] = ()
    incomplete_reason: str | None = None
    grouped_nodes: int = 0
    grouped_worlds: int = 0
    changed_groups: int = 0
    feed_shape_dominance_groups: int = 0
    grouped_nodes_by_depth: tuple[tuple[int, int], ...] = ()
    grouped_worlds_by_depth: tuple[tuple[int, int], ...] = ()
    frozen_nodes_by_depth: tuple[tuple[int, int], ...] = ()
    grouped_nodes_by_role: tuple[tuple[str, int], ...] = ()
    changed_groups_by_role: tuple[tuple[str, int], ...] = ()
    zero_weight_groups: int = 0
    bound_pruned_actions: int = 0
    bound_pruned_world_branches: int = 0


def _validate_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def risk_mode_contract(risk_mode: str) -> str:
    contracts = {
        "strict": "strict_world_nonregression_v1",
        "cvar": "paired_lower_tail_cvar_v1",
    }
    try:
        return contracts[str(risk_mode)]
    except KeyError as exc:
        raise ValueError("risk_mode must be 'strict' or 'cvar'") from exc


def _weighted_lower_tail_mean(
    values: list[float], weights: list[int | float], alpha: float
) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and have equal length")
    alpha = _validate_finite("cvar_alpha", alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0,1]")
    normalized_weights = [
        _validate_nonnegative_weight("world weight", weight) for weight in weights
    ]
    total_weight = math.fsum(normalized_weights)
    if total_weight <= 0.0:
        raise ValueError("at least one world weight must be positive")
    normalized_weights = [weight / total_weight for weight in normalized_weights]
    tail_mass = alpha
    remaining = tail_mass
    weighted_sum = 0.0
    for value, weight in sorted(
        zip(values, normalized_weights, strict=True), key=lambda item: item[0]
    ):
        take = min(float(weight), remaining)
        weighted_sum += float(value) * take
        remaining -= take
        if remaining <= 1e-12 * tail_mass:
            break
    return weighted_sum / tail_mass


def select_risk_guarded_action(
    *,
    values_by_allocation: list[dict[int, float]],
    physical_weights: list[int | float],
    baseline_index: int,
    proposed_index: int,
    cvar_alpha: float,
    minimum_mean_value_margin: float,
    minimum_worst_value_margin: float,
    minimum_paired_cvar_margin: float,
) -> dict[str, Any]:
    """Approve a mean-optimal override only when it clears V3Pro risk guards."""
    if len(values_by_allocation) != len(physical_weights) or not values_by_allocation:
        raise ValueError(
            "values_by_allocation and physical_weights must be non-empty and aligned"
        )
    cvar_alpha = _validate_finite("cvar_alpha", cvar_alpha)
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0,1]")
    minimum_mean_value_margin = _validate_finite(
        "minimum_mean_value_margin", minimum_mean_value_margin
    )
    minimum_worst_value_margin = _validate_finite(
        "minimum_worst_value_margin", minimum_worst_value_margin
    )
    minimum_paired_cvar_margin = _validate_finite(
        "minimum_paired_cvar_margin", minimum_paired_cvar_margin
    )
    weights = [
        _validate_nonnegative_weight("world weight", weight)
        for weight in physical_weights
    ]
    if math.fsum(weights) <= 0.0:
        raise ValueError("at least one world weight must be positive")
    baseline_index = int(baseline_index)
    proposed_index = int(proposed_index)
    try:
        baseline_values = [float(row[baseline_index]) for row in values_by_allocation]
        proposed_values = [float(row[proposed_index]) for row in values_by_allocation]
    except KeyError as exc:
        raise ValueError(
            "baseline/proposed action is absent from an allocation"
        ) from exc
    if not all((math.isfinite(value) for value in baseline_values + proposed_values)):
        raise ValueError("all solved action values must be finite")
    total_weight = math.fsum(weights)
    baseline_mean = (
        math.fsum(
            (
                value * weight
                for value, weight in zip(baseline_values, weights, strict=True)
            )
        )
        / total_weight
    )
    proposed_mean = (
        math.fsum(
            (
                value * weight
                for value, weight in zip(proposed_values, weights, strict=True)
            )
        )
        / total_weight
    )
    paired_regrets = [
        proposed - baseline
        for proposed, baseline in zip(proposed_values, baseline_values, strict=True)
    ]
    metrics = {
        "cvar_alpha": cvar_alpha,
        "baseline_mean_value": baseline_mean,
        "proposed_mean_value": proposed_mean,
        "mean_value_margin": proposed_mean - baseline_mean,
        "baseline_min_value": min(baseline_values),
        "proposed_min_value": min(proposed_values),
        "worst_value_margin": min(proposed_values) - min(baseline_values),
        "paired_regret_min": min(paired_regrets),
        "paired_regret_cvar": _weighted_lower_tail_mean(
            paired_regrets, weights, cvar_alpha
        ),
        "minimum_mean_value_margin": minimum_mean_value_margin,
        "minimum_worst_value_margin": minimum_worst_value_margin,
        "minimum_paired_cvar_margin": minimum_paired_cvar_margin,
    }
    if proposed_index == baseline_index:
        selected = baseline_index
        reason = "baseline_is_mean_optimal"
    elif metrics["mean_value_margin"] <= minimum_mean_value_margin + 1e-12:
        selected = baseline_index
        reason = "rejected_insufficient_mean_improvement"
    elif metrics["worst_value_margin"] < minimum_worst_value_margin - 1e-12:
        selected = baseline_index
        reason = "rejected_worst_value_regression"
    elif metrics["paired_regret_cvar"] < minimum_paired_cvar_margin - 1e-12:
        selected = baseline_index
        reason = "rejected_paired_tail_regret"
    else:
        selected = proposed_index
        reason = "risk_approved_override"
    return {
        "selected_index": selected,
        "selection_reason": reason,
        "metrics": metrics,
        "contract": RISK_SELECTION_CONTRACT,
    }


def select_risk_constrained_action(
    *,
    values_by_allocation: list[dict[int, float]],
    physical_weights: list[int | float],
    baseline_index: int,
    candidate_order: list[int],
    risk_mode: str,
    cvar_alpha: float,
    minimum_mean_value_margin: float,
    minimum_worst_value_margin: float,
    minimum_paired_cvar_margin: float,
    minimum_paired_regret_min: float,
    minimum_win_probability_margin: float,
) -> dict[str, Any]:
    """Select the highest-value candidate that clears the configured risk contract."""
    if len(values_by_allocation) != len(physical_weights) or not values_by_allocation:
        raise ValueError(
            "values_by_allocation and physical_weights must be non-empty and aligned"
        )
    mode_contract = risk_mode_contract(risk_mode)
    cvar_alpha = _validate_finite("cvar_alpha", cvar_alpha)
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0,1]")
    thresholds = {
        "minimum_mean_value_margin": _validate_finite(
            "minimum_mean_value_margin", minimum_mean_value_margin
        ),
        "minimum_worst_value_margin": _validate_finite(
            "minimum_worst_value_margin", minimum_worst_value_margin
        ),
        "minimum_paired_cvar_margin": _validate_finite(
            "minimum_paired_cvar_margin", minimum_paired_cvar_margin
        ),
        "minimum_paired_regret_min": _validate_finite(
            "minimum_paired_regret_min", minimum_paired_regret_min
        ),
        "minimum_win_probability_margin": _validate_finite(
            "minimum_win_probability_margin", minimum_win_probability_margin
        ),
    }
    weights = [
        _validate_nonnegative_weight("world weight", weight)
        for weight in physical_weights
    ]
    if math.fsum(weights) <= 0.0:
        raise ValueError("at least one world weight must be positive")
    total_weight = math.fsum(weights)
    baseline_index = int(baseline_index)
    order: list[int] = []
    for raw_index in candidate_order:
        index = int(raw_index)
        if index not in order:
            order.append(index)
    if baseline_index not in order:
        order.append(baseline_index)
    available = set(values_by_allocation[0])
    if any((set(row) != available for row in values_by_allocation)):
        raise ValueError("all allocations must contain the same candidate actions")
    if baseline_index not in available or any(
        (index not in available for index in order)
    ):
        raise ValueError("candidate_order/baseline contains an absent action")
    for index in sorted(available):
        if index not in order:
            order.append(index)

    def weighted_mean(values: list[float]) -> float:
        return (
            math.fsum(
                (value * weight for value, weight in zip(values, weights, strict=True))
            )
            / total_weight
        )

    def win_probability(values: list[float]) -> float:
        return (
            math.fsum(
                (
                    weight
                    for value, weight in zip(values, weights, strict=True)
                    if value > 0
                )
            )
            / total_weight
        )

    baseline_values = [float(row[baseline_index]) for row in values_by_allocation]
    if not all((math.isfinite(value) for value in baseline_values)):
        raise ValueError("all solved action values must be finite")
    baseline_mean = weighted_mean(baseline_values)
    baseline_min = min(baseline_values)
    baseline_win_probability = win_probability(baseline_values)
    rows: list[dict[str, Any]] = []
    for index in order:
        values = [float(row[index]) for row in values_by_allocation]
        if not all((math.isfinite(value) for value in values)):
            raise ValueError("all solved action values must be finite")
        mean_value = weighted_mean(values)
        regrets = [
            value - baseline
            for value, baseline in zip(values, baseline_values, strict=True)
        ]
        row = {
            "action_index": index,
            "mean_value": mean_value,
            "mean_value_margin": mean_value - baseline_mean,
            "min_value": min(values),
            "worst_value_margin": min(values) - baseline_min,
            "paired_regret_min": min(regrets),
            "paired_regret_cvar": _weighted_lower_tail_mean(
                regrets, weights, cvar_alpha
            ),
            "win_probability": win_probability(values),
            "win_probability_margin": win_probability(values)
            - baseline_win_probability,
            "feasible": True,
            "rejection_reason": None,
        }
        if index != baseline_index:
            if (
                row["mean_value_margin"]
                <= thresholds["minimum_mean_value_margin"] + 1e-12
            ):
                row["feasible"] = False
                row["rejection_reason"] = "insufficient_mean_improvement"
            elif (
                row["worst_value_margin"]
                < thresholds["minimum_worst_value_margin"] - 1e-12
            ):
                row["feasible"] = False
                row["rejection_reason"] = "worst_value_regression"
            elif (
                row["win_probability_margin"]
                < thresholds["minimum_win_probability_margin"] - 1e-12
            ):
                row["feasible"] = False
                row["rejection_reason"] = "win_probability_regression"
            elif (
                risk_mode == "strict"
                and row["paired_regret_min"]
                < thresholds["minimum_paired_regret_min"] - 1e-12
            ):
                row["feasible"] = False
                row["rejection_reason"] = "paired_world_regression"
            elif (
                risk_mode == "cvar"
                and row["paired_regret_cvar"]
                < thresholds["minimum_paired_cvar_margin"] - 1e-12
            ):
                row["feasible"] = False
                row["rejection_reason"] = "paired_tail_regret"
        rows.append(row)
    rank = {index: position for position, index in enumerate(order)}

    def best(indices: list[int]) -> int:
        best_mean = max((rows[rank[index]]["mean_value"] for index in indices))
        tied = [
            index
            for index in indices
            if math.isclose(
                rows[rank[index]]["mean_value"], best_mean, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        if baseline_index in tied:
            return baseline_index
        return min(tied, key=lambda index: rank[index])

    proposed_index = best(order)
    feasible = [int(row["action_index"]) for row in rows if row["feasible"]]
    selected_index = best(feasible)
    proposed_row = rows[rank[proposed_index]]
    selected_row = rows[rank[selected_index]]
    if selected_index == baseline_index and proposed_index == baseline_index:
        reason = "baseline_is_mean_optimal"
    elif selected_index == baseline_index:
        reason = "rejected_" + str(proposed_row["rejection_reason"])
    elif selected_index == proposed_index:
        reason = "risk_approved_override"
    else:
        reason = "selected_best_feasible_override"
    return {
        "selected_index": selected_index,
        "proposed_index": proposed_index,
        "selection_reason": reason,
        "metrics": {
            "risk_mode": risk_mode,
            "cvar_alpha": cvar_alpha,
            "baseline_mean_value": baseline_mean,
            "baseline_min_value": baseline_min,
            "baseline_win_probability": baseline_win_probability,
            **thresholds,
            **{
                key: selected_row[key]
                for key in (
                    "mean_value",
                    "mean_value_margin",
                    "min_value",
                    "worst_value_margin",
                    "paired_regret_min",
                    "paired_regret_cvar",
                    "win_probability",
                    "win_probability_margin",
                )
            },
        },
        "candidate_risk_rows": rows,
        "contract": RISK_SELECTION_CONTRACT,
        "risk_mode_contract": mode_contract,
    }


def _first_mismatch_path(left: Any, right: Any, path: str = "root") -> str:
    if type(left) is not type(right):
        return f"{path}:type({type(left).__name__}!={type(right).__name__})"
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}:len({len(left)}!={len(right)})"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            if left_item != right_item:
                return _first_mismatch_path(left_item, right_item, f"{path}[{index}]")
        return path
    return f"{path}:{left!r}!={right!r}"


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_nonnegative_weight(name: str, value: int | float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be non-negative and finite")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _incomplete(
    *,
    status: str,
    chosen_index: int,
    total_allocations: int,
    total_physical_allocations: int,
    completed_allocations: int,
    completed_physical_weight: int,
    root_action_count: int,
    nodes: int,
    policy_decisions: int = 0,
    started: float,
    reason: str,
    expanded_actions: int = 0,
    cutoffs: int = 0,
    cache_lookups: int = 0,
    cache_hits: int = 0,
    cache_entries: int = 0,
    belief_contract: str = UNIFORM_BELIEF_CONTRACT,
    belief_diagnostics: dict[str, float | int] | None = None,
    grouped_nodes: int = 0,
    grouped_worlds: int = 0,
    changed_groups: int = 0,
    feed_shape_dominance_groups: int = 0,
    grouped_nodes_by_depth: tuple[tuple[int, int], ...] = (),
    grouped_worlds_by_depth: tuple[tuple[int, int], ...] = (),
    frozen_nodes_by_depth: tuple[tuple[int, int], ...] = (),
    grouped_nodes_by_role: tuple[tuple[str, int], ...] = (),
    changed_groups_by_role: tuple[tuple[str, int], ...] = (),
    zero_weight_groups: int = 0,
    bound_pruned_actions: int = 0,
    bound_pruned_world_branches: int = 0,
) -> ExhaustiveDeterminizationResult:
    return ExhaustiveDeterminizationResult(
        status=status,
        selected_index=int(chosen_index),
        proposed_index=int(chosen_index),
        selection_reason="incomplete_preserves_v3pro",
        risk_contract=RISK_SELECTION_CONTRACT,
        risk_metrics={},
        belief_contract=str(belief_contract),
        belief_diagnostics=dict(belief_diagnostics or {}),
        total_allocations=int(total_allocations),
        total_physical_allocations=int(total_physical_allocations),
        completed_allocations=int(completed_allocations),
        completed_physical_weight=int(completed_physical_weight),
        root_action_count=int(root_action_count),
        nodes=int(nodes),
        policy_decisions=int(policy_decisions),
        expanded_actions=int(expanded_actions),
        cutoffs=int(cutoffs),
        cache_lookups=int(cache_lookups),
        cache_hits=int(cache_hits),
        cache_entries=int(cache_entries),
        elapsed_sec=time.perf_counter() - started,
        action_rows=(),
        candidate_risk_rows=(),
        incomplete_reason=reason,
        grouped_nodes=int(grouped_nodes),
        grouped_worlds=int(grouped_worlds),
        changed_groups=int(changed_groups),
        feed_shape_dominance_groups=int(feed_shape_dominance_groups),
        grouped_nodes_by_depth=tuple(grouped_nodes_by_depth),
        grouped_worlds_by_depth=tuple(grouped_worlds_by_depth),
        frozen_nodes_by_depth=tuple(frozen_nodes_by_depth),
        grouped_nodes_by_role=tuple(grouped_nodes_by_role),
        changed_groups_by_role=tuple(changed_groups_by_role),
        zero_weight_groups=int(zero_weight_groups),
        bound_pruned_actions=int(bound_pruned_actions),
        bound_pruned_world_branches=int(bound_pruned_world_branches),
    )


def exhaustive_determinization_action(
    *,
    env: Any,
    benchmark: Any,
    information_module: Any,
    reward_by_order: Any,
    actor_seat: int,
    actor_hand: list[str],
    played_cards: list[str],
    public_counts: list[int],
    chosen_index: int,
    v3pro_order: list[int],
    max_allocations: int,
    max_nodes_per_allocation: int,
    max_total_nodes: int,
    root_candidate_indices: list[int] | None = None,
    transposition_table: Any | None = None,
    risk_mode: str = "strict",
    risk_cvar_alpha: float = 1.0 / 3.0,
    minimum_mean_value_margin: float = 0.0,
    minimum_worst_value_margin: float = 0.0,
    minimum_paired_cvar_margin: float = 0.0,
    minimum_paired_regret_min: float = 0.0,
    minimum_win_probability_margin: float = 0.0,
    root_action_solver: Callable[..., dict[str, Any]] | None = None,
) -> ExhaustiveDeterminizationResult:
    """Solve every requested root action over every legal hidden allocation."""
    max_allocations = _validate_positive_integer("max_allocations", max_allocations)
    max_nodes_per_allocation = _validate_positive_integer(
        "max_nodes_per_allocation", max_nodes_per_allocation
    )
    max_total_nodes = _validate_positive_integer("max_total_nodes", max_total_nodes)
    if actor_seat not in range(4):
        raise ValueError("actor_seat must be in [0,3]")
    started = time.perf_counter()
    root_team = int(actor_seat) % 2
    before = benchmark.state_key(env, root_team)
    restoration_key = getattr(benchmark, "restoration_key", benchmark.state_key)
    restoration_before = restoration_key(env, root_team)
    root_indices = tuple(
        (int(index) for index in benchmark.unique_action_indices(env, False))
    )
    if not root_indices:
        raise RuntimeError("exhaustive search root has no legal action")
    chosen_index = int(chosen_index)
    chosen_representative = chosen_index
    representative_by_key: dict[str, int] = {}
    if chosen_index not in root_indices:
        try:
            representative_by_key = {
                benchmark.action_key(env.legal_moves[index]): index
                for index in root_indices
            }
            chosen_representative = representative_by_key[
                benchmark.action_key(env.legal_moves[chosen_index])
            ]
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "deployed V3Pro action has no semantically equivalent unique root action"
            ) from exc
    if root_candidate_indices is not None:
        if not root_candidate_indices:
            raise ValueError("root_candidate_indices must not be empty")
        if not representative_by_key and hasattr(env, "legal_moves"):
            representative_by_key = {
                benchmark.action_key(env.legal_moves[index]): index
                for index in root_indices
            }
        restricted: list[int] = []
        for raw_index in [chosen_index, *root_candidate_indices]:
            index = int(raw_index)
            if index in root_indices:
                representative = index
            else:
                try:
                    representative = representative_by_key[
                        benchmark.action_key(env.legal_moves[index])
                    ]
                except (AttributeError, IndexError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"requested root candidate has no legal representative: {index}"
                    ) from exc
            if representative not in restricted:
                restricted.append(representative)
        root_indices = tuple(restricted)
    try:
        enumeration = information_module.enumerate_hidden_hands_exhaustive(
            actor_seat,
            actor_hand,
            played_cards,
            public_counts,
            max_allocations=max_allocations,
        )
    except information_module.HiddenAllocationLimitExceeded as exc:
        return _incomplete(
            status="allocation_limit_exceeded",
            chosen_index=chosen_index,
            total_allocations=exc.total_allocations,
            total_physical_allocations=exc.total_physical_allocations,
            completed_allocations=0,
            completed_physical_weight=0,
            root_action_count=len(root_indices),
            nodes=0,
            started=started,
            reason=str(exc),
            belief_contract=UNIFORM_BELIEF_CONTRACT,
            belief_diagnostics={},
        )
    allocations = list(enumeration.allocations)
    allocation_weights = [int(value) for value in enumeration.allocation_weights]
    oracle_diagnostics: dict[str, float | int] = {}
    total_allocations = len(allocations)
    total_physical_allocations = sum(allocation_weights)
    log_likelihoods = [0.0 for allocation in allocations]
    belief = posterior_from_log_likelihoods(allocation_weights, log_likelihoods)
    decision_weights = list(belief.probabilities)
    belief_contract = UNIFORM_BELIEF_CONTRACT
    belief_diagnostics: dict[str, float | int] = {
        **oracle_diagnostics,
        "effective_sample_size": belief.effective_sample_size,
        "normalized_entropy": belief.normalized_entropy,
        "maximum_probability": belief.maximum_probability,
        "kl_from_prior": belief.kl_from_prior,
        "impossible_worlds": sum(
            (likelihood == -math.inf for likelihood in log_likelihoods)
        ),
    }
    table = (
        benchmark.ExactTranspositionTable()
        if transposition_table is None
        else transposition_table
    )
    values_by_allocation: list[dict[int, float]] = []
    completed_weights: list[int] = []
    nodes = 0
    policy_decisions = 0
    expanded_actions = cutoffs = cache_lookups = cache_hits = 0
    cache_entries = len(getattr(table, "entries", ()))
    grouped_nodes = grouped_worlds = changed_groups = 0
    feed_shape_dominance_groups = 0
    grouped_nodes_by_depth: tuple[tuple[int, int], ...] = ()
    grouped_worlds_by_depth: tuple[tuple[int, int], ...] = ()
    frozen_nodes_by_depth: tuple[tuple[int, int], ...] = ()
    grouped_nodes_by_role: tuple[tuple[str, int], ...] = ()
    changed_groups_by_role: tuple[tuple[str, int], ...] = ()
    zero_weight_groups = 0
    bound_pruned_actions = bound_pruned_world_branches = 0
    for allocation_index, (allocation, physical_weight) in enumerate(
        zip(allocations, allocation_weights, strict=True)
    ):
        remaining_nodes = max_total_nodes - nodes
        if remaining_nodes <= 0:
            return _incomplete(
                status="total_node_budget_exceeded",
                chosen_index=chosen_index,
                total_allocations=total_allocations,
                total_physical_allocations=total_physical_allocations,
                completed_allocations=len(values_by_allocation),
                completed_physical_weight=sum(completed_weights),
                root_action_count=len(root_indices),
                nodes=nodes,
                policy_decisions=policy_decisions,
                started=started,
                reason=f"total node budget exhausted before allocation {allocation_index}",
                expanded_actions=expanded_actions,
                cutoffs=cutoffs,
                cache_lookups=cache_lookups,
                cache_hits=cache_hits,
                cache_entries=cache_entries,
                belief_contract=UNIFORM_BELIEF_CONTRACT,
                belief_diagnostics=oracle_diagnostics,
            )
        allocation_budget = min(max_nodes_per_allocation, remaining_nodes)
        with information_module.temporary_hidden_hands(env, actor_seat, allocation):
            solver = (
                benchmark.solve_root_actions
                if root_action_solver is None
                else root_action_solver
            )
            solved = solver(
                env, root_indices, allocation_budget, table, reward_by_order, True
            )
        rule_after = benchmark.state_key(env, root_team)
        restoration_after = restoration_key(env, root_team)
        if rule_after != before or restoration_after != restoration_before:
            mismatch = (
                _first_mismatch_path(before, rule_after, "rule")
                if rule_after != before
                else _first_mismatch_path(
                    restoration_before, restoration_after, "restoration"
                )
            )
            raise RuntimeError(
                "exhaustive determinization search leaked engine state at " + mismatch
            )
        nodes += int(solved.get("nodes", 0))
        policy_decisions += int(solved.get("policy_decisions", 0))
        expanded_actions += int(solved.get("expanded_actions", 0))
        cutoffs += int(solved.get("cutoffs", 0))
        cache_lookups += int(solved.get("cache_lookups", 0))
        cache_hits += int(solved.get("cache_hits", 0))
        cache_entries = max(cache_entries, int(solved.get("cache_entries", 0)))
        if solved.get("status") != "ok":
            return _incomplete(
                status="node_budget_exceeded",
                chosen_index=chosen_index,
                total_allocations=total_allocations,
                total_physical_allocations=total_physical_allocations,
                completed_allocations=len(values_by_allocation),
                completed_physical_weight=sum(completed_weights),
                root_action_count=len(root_indices),
                nodes=nodes,
                policy_decisions=policy_decisions,
                started=started,
                reason=f"allocation {allocation_index} did not complete within {allocation_budget} nodes",
                expanded_actions=expanded_actions,
                cutoffs=cutoffs,
                cache_lookups=cache_lookups,
                cache_hits=cache_hits,
                cache_entries=cache_entries,
                belief_contract=UNIFORM_BELIEF_CONTRACT,
                belief_diagnostics=oracle_diagnostics,
            )
        solved_values = {
            int(index): float(value) for index, value in solved["values"].items()
        }
        if set(solved_values) != set(root_indices):
            raise RuntimeError("exact solver omitted one or more unique root actions")
        values_by_allocation.append(solved_values)
        completed_weights.append(int(physical_weight))
    action_rows: list[dict[str, Any]] = []
    means: dict[int, float] = {}
    for index in root_indices:
        values = [row[index] for row in values_by_allocation]
        mean = math.fsum(
            (
                value * weight
                for value, weight in zip(values, decision_weights, strict=True)
            )
        ) / math.fsum(decision_weights)
        means[index] = mean
        grade_distribution = None
        if all(
            (
                math.isfinite(float(value))
                and float(value).is_integer()
                and (int(value) in LEGAL_GRADES)
                for value in values
            )
        ):
            grade_distribution = weighted_grade_distribution(
                values, decision_weights, cvar_alpha=risk_cvar_alpha
            ).as_json()
        action_rows.append(
            {
                "action_index": index,
                "mean_value": mean,
                "min_value": min(values),
                "max_value": max(values),
                "lower_tail_cvar": _weighted_lower_tail_mean(
                    values, decision_weights, risk_cvar_alpha
                ),
                "grade_distribution": grade_distribution,
                "positive_unique_allocations": sum((value > 0 for value in values)),
                "negative_unique_allocations": sum((value < 0 for value in values)),
                "zero_unique_allocations": sum((value == 0 for value in values)),
                "positive_physical_weight": sum(
                    (
                        weight
                        for value, weight in zip(values, completed_weights, strict=True)
                        if value > 0
                    )
                ),
                "negative_physical_weight": sum(
                    (
                        weight
                        for value, weight in zip(values, completed_weights, strict=True)
                        if value < 0
                    )
                ),
                "zero_physical_weight": sum(
                    (
                        weight
                        for value, weight in zip(values, completed_weights, strict=True)
                        if value == 0
                    )
                ),
            }
        )
    normalized_order: list[int] = []
    if representative_by_key:
        for raw_index in v3pro_order:
            try:
                representative = representative_by_key[
                    benchmark.action_key(env.legal_moves[int(raw_index)])
                ]
            except (IndexError, KeyError, TypeError):
                continue
            if representative not in normalized_order:
                normalized_order.append(representative)
    else:
        for raw_index in v3pro_order:
            index = int(raw_index)
            if index in root_indices and index not in normalized_order:
                normalized_order.append(index)
    for index in root_indices:
        if index not in normalized_order:
            normalized_order.append(index)
    decision = select_risk_constrained_action(
        values_by_allocation=values_by_allocation,
        physical_weights=decision_weights,
        baseline_index=chosen_representative,
        candidate_order=normalized_order,
        risk_mode=risk_mode,
        cvar_alpha=risk_cvar_alpha,
        minimum_mean_value_margin=minimum_mean_value_margin,
        minimum_worst_value_margin=minimum_worst_value_margin,
        minimum_paired_cvar_margin=minimum_paired_cvar_margin,
        minimum_paired_regret_min=minimum_paired_regret_min,
        minimum_win_probability_margin=minimum_win_probability_margin,
    )
    proposed_representative = int(decision["proposed_index"])
    proposed = (
        chosen_index
        if proposed_representative == chosen_representative
        else proposed_representative
    )
    selected = (
        chosen_index
        if int(decision["selected_index"]) == chosen_representative
        else int(decision["selected_index"])
    )
    return ExhaustiveDeterminizationResult(
        status="complete",
        selected_index=selected,
        proposed_index=proposed,
        selection_reason=str(decision["selection_reason"]),
        risk_contract=RISK_SELECTION_CONTRACT,
        risk_metrics=dict(decision["metrics"]),
        belief_contract=belief_contract,
        belief_diagnostics=belief_diagnostics,
        total_allocations=total_allocations,
        total_physical_allocations=total_physical_allocations,
        completed_allocations=len(values_by_allocation),
        completed_physical_weight=sum(completed_weights),
        root_action_count=len(root_indices),
        nodes=nodes,
        policy_decisions=policy_decisions,
        expanded_actions=expanded_actions,
        cutoffs=cutoffs,
        cache_lookups=cache_lookups,
        cache_hits=cache_hits,
        cache_entries=cache_entries,
        elapsed_sec=time.perf_counter() - started,
        action_rows=tuple(action_rows),
        candidate_risk_rows=tuple(decision["candidate_risk_rows"]),
        grouped_nodes=grouped_nodes,
        grouped_worlds=grouped_worlds,
        changed_groups=changed_groups,
        feed_shape_dominance_groups=feed_shape_dominance_groups,
        grouped_nodes_by_depth=grouped_nodes_by_depth,
        grouped_worlds_by_depth=grouped_worlds_by_depth,
        frozen_nodes_by_depth=frozen_nodes_by_depth,
        grouped_nodes_by_role=grouped_nodes_by_role,
        changed_groups_by_role=changed_groups_by_role,
        zero_weight_groups=zero_weight_groups,
        bound_pruned_actions=bound_pruned_actions,
        bound_pruned_world_branches=bound_pruned_world_branches,
    )
