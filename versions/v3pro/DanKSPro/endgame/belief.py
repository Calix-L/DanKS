"""Numerically stable public-belief utilities for V3Pro endgame search.

This module is deliberately engine-independent.  It combines exact physical
deal multiplicities with externally supplied public-history likelihoods; it
never reads a held-out real hidden hand when constructing the posterior.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

from typing import Sequence

PUBLIC_BELIEF_CONTRACT = "history_conditioned_physical_posterior_v1"

UNIFORM_BELIEF_CONTRACT = "uniform_physical_prior_v1"

LEGAL_GRADES = (-3, -2, -1, 1, 2, 3)


def _positive_finite_weights(values: Sequence[int | float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("weights must be non-empty")
    weights = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("weights must be positive and finite")
    return weights


def _normalized_weights(values: Sequence[int | float]) -> tuple[float, ...]:
    weights = _positive_finite_weights(values)
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def _normalized_nonnegative_weights(
    values: Sequence[int | float],
) -> tuple[float, ...]:
    if not values:
        raise ValueError("weights must be non-empty")
    weights = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("weights must be non-negative and finite")
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return tuple(value / total for value in weights)


def _validate_probability_parameters(
    *,
    probability_floor: float,
    evidence_strength: float,
) -> tuple[float, float]:
    floor = float(probability_floor)
    strength = float(evidence_strength)
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("probability_floor must be finite and in (0,1]")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("evidence_strength must be finite and non-negative")
    return floor, strength


@dataclass(frozen=True)
class PublicBelief:
    probabilities: tuple[float, ...]
    prior_probabilities: tuple[float, ...]
    effective_sample_size: float
    normalized_entropy: float
    maximum_probability: float
    kl_from_prior: float

    def as_json(self) -> dict[str, object]:
        return {
            "contract": PUBLIC_BELIEF_CONTRACT,
            "probabilities": list(self.probabilities),
            "prior_probabilities": list(self.prior_probabilities),
            "effective_sample_size": self.effective_sample_size,
            "normalized_entropy": self.normalized_entropy,
            "maximum_probability": self.maximum_probability,
            "kl_from_prior": self.kl_from_prior,
        }


def posterior_from_log_likelihoods(
    physical_weights: Sequence[int | float],
    log_likelihoods: Sequence[float],
    *,
    likelihood_temperature: float = 1.0,
) -> PublicBelief:
    """Combine the exact physical prior and public-action log likelihoods."""

    if len(physical_weights) != len(log_likelihoods):
        raise ValueError("physical_weights and log_likelihoods must have equal length")
    prior = _normalized_weights(physical_weights)
    temperature = float(likelihood_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("likelihood_temperature must be positive and finite")

    likelihoods = tuple(float(value) for value in log_likelihoods)
    if any(math.isnan(value) for value in likelihoods):
        raise ValueError("log likelihoods must not contain NaN")
    if any(value == math.inf for value in likelihoods):
        raise ValueError("log likelihoods must not contain positive infinity")
    if all(value == -math.inf for value in likelihoods):
        raise ValueError("all hidden worlds are impossible")

    log_unnormalized = tuple(
        (
            -math.inf
            if likelihood == -math.inf
            else math.log(prior_probability) + likelihood / temperature
        )
        for prior_probability, likelihood in zip(prior, likelihoods, strict=True)
    )
    maximum = max(log_unnormalized)
    exponentials = tuple(
        0.0 if value == -math.inf else math.exp(value - maximum)
        for value in log_unnormalized
    )
    normalizer = math.fsum(exponentials)
    probabilities = tuple(value / normalizer for value in exponentials)
    effective_sample_size = 1.0 / math.fsum(
        probability * probability for probability in probabilities
    )
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    normalized_entropy = (
        entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    )
    kl_from_prior = math.fsum(
        probability * math.log(probability / prior_probability)
        for probability, prior_probability in zip(
            probabilities,
            prior,
            strict=True,
        )
        if probability > 0.0
    )
    return PublicBelief(
        probabilities=probabilities,
        prior_probabilities=prior,
        effective_sample_size=effective_sample_size,
        normalized_entropy=normalized_entropy,
        maximum_probability=max(probabilities),
        kl_from_prior=kl_from_prior,
    )


def update_log_likelihoods(
    current: Sequence[float],
    action_probabilities: Sequence[float],
    *,
    probability_floor: float = 1.0e-8,
    evidence_strength: float = 1.0,
) -> tuple[float, ...]:
    """Apply one observed public action to every surviving hidden world."""

    if len(current) != len(action_probabilities):
        raise ValueError("current and action_probabilities must have equal length")
    if not current:
        raise ValueError("likelihood vectors must be non-empty")
    floor, strength = _validate_probability_parameters(
        probability_floor=probability_floor,
        evidence_strength=evidence_strength,
    )
    output: list[float] = []
    for old_raw, probability_raw in zip(current, action_probabilities, strict=True):
        old = float(old_raw)
        probability = float(probability_raw)
        if math.isnan(old) or old == math.inf:
            raise ValueError("current log likelihoods must be finite or -inf")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("action probabilities must be finite and in [0,1]")
        if old == -math.inf:
            output.append(-math.inf)
        else:
            output.append(old + strength * math.log(max(floor, probability)))
    return tuple(output)


def _weighted_lower_tail_mean(
    values: Sequence[float],
    probabilities: Sequence[float],
    alpha: float,
) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("cvar_alpha must be finite and in (0,1]")
    remaining = alpha
    total = 0.0
    for value, probability in sorted(
        zip(values, probabilities, strict=True),
        key=lambda row: row[0],
    ):
        take = min(probability, remaining)
        total += float(value) * take
        remaining -= take
        if remaining <= 1.0e-15:
            break
    return total / alpha


def _grade_label(grade: int) -> str:
    return f"{grade:+d}"


@dataclass(frozen=True)
class GradeDistribution:
    probabilities: dict[str, float]
    expected_grade: float
    win_probability: float
    loss_probability: float
    lower_tail_cvar: float

    def as_json(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "expected_grade": self.expected_grade,
            "win_probability": self.win_probability,
            "loss_probability": self.loss_probability,
            "lower_tail_cvar": self.lower_tail_cvar,
        }


def weighted_grade_distribution(
    values: Sequence[int | float],
    weights: Sequence[int | float],
    *,
    cvar_alpha: float = 1.0 / 3.0,
) -> GradeDistribution:
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and have equal length")
    normalized = _normalized_nonnegative_weights(weights)
    grades: list[int] = []
    for value in values:
        number = float(value)
        grade = int(number)
        if not math.isfinite(number) or number != grade or grade not in LEGAL_GRADES:
            raise ValueError("terminal values must be one of the six legal grades")
        grades.append(grade)
    probabilities = {
        _grade_label(grade): math.fsum(
            probability
            for value, probability in zip(grades, normalized, strict=True)
            if value == grade
        )
        for grade in LEGAL_GRADES
    }
    return GradeDistribution(
        probabilities=probabilities,
        expected_grade=math.fsum(
            value * probability
            for value, probability in zip(grades, normalized, strict=True)
        ),
        win_probability=math.fsum(
            probability
            for value, probability in zip(grades, normalized, strict=True)
            if value > 0
        ),
        loss_probability=math.fsum(
            probability
            for value, probability in zip(grades, normalized, strict=True)
            if value < 0
        ),
        lower_tail_cvar=_weighted_lower_tail_mean(grades, normalized, cvar_alpha),
    )


@dataclass(frozen=True)
class PairedRegretDistribution:
    mean: float
    minimum: float
    lower_tail_cvar: float
    improvement_probability: float
    regression_probability: float

    def as_json(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "minimum": self.minimum,
            "lower_tail_cvar": self.lower_tail_cvar,
            "improvement_probability": self.improvement_probability,
            "regression_probability": self.regression_probability,
        }


def weighted_paired_regret_distribution(
    baseline_values: Sequence[int | float],
    candidate_values: Sequence[int | float],
    weights: Sequence[int | float],
    *,
    cvar_alpha: float = 1.0 / 3.0,
) -> PairedRegretDistribution:
    if not baseline_values or not (
        len(baseline_values) == len(candidate_values) == len(weights)
    ):
        raise ValueError(
            "baseline, candidate, and weights must be non-empty and aligned"
        )
    probabilities = _normalized_nonnegative_weights(weights)
    regrets: list[float] = []
    for baseline_raw, candidate_raw in zip(
        baseline_values,
        candidate_values,
        strict=True,
    ):
        baseline = float(baseline_raw)
        candidate = float(candidate_raw)
        if not math.isfinite(baseline) or not math.isfinite(candidate):
            raise ValueError("paired values must be finite")
        regrets.append(candidate - baseline)
    return PairedRegretDistribution(
        mean=math.fsum(
            regret * probability
            for regret, probability in zip(regrets, probabilities, strict=True)
        ),
        minimum=min(regrets),
        lower_tail_cvar=_weighted_lower_tail_mean(
            regrets,
            probabilities,
            cvar_alpha,
        ),
        improvement_probability=math.fsum(
            probability
            for regret, probability in zip(regrets, probabilities, strict=True)
            if regret > 0.0
        ),
        regression_probability=math.fsum(
            probability
            for regret, probability in zip(regrets, probabilities, strict=True)
            if regret < 0.0
        ),
    )
