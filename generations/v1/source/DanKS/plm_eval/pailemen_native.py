from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


PAILEMEN_NATIVE_OPPONENT = "pailemen_native_rule_ai"
DANZERO_RELEASE_RULEBOT = "danzero_rulebot_ai2"
EXPECTED_GDAI_SHA256 = os.environ.get("EXPECTED_GDAI_SHA256", "").strip()
TARGET_RULES: dict[str, Any] = {
    "mode": 1,
    "ending": 1,
    "level_up": 1,
    "tribute": 0,
    "one_way": False,
}
PLM_PROXY_OPPONENT = "plm_rulebot_proxy_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_target_rules(rules: Mapping[str, Any]) -> list[str]:
    failures = []
    for key, expected in TARGET_RULES.items():
        if key not in rules:
            failures.append(f"room_rule_{key}_missing")
            continue
        actual = rules[key]
        if isinstance(expected, bool):
            if isinstance(actual, str):
                normalized = actual.strip().lower()
                if normalized in {"true", "1"}:
                    actual = True
                elif normalized in {"false", "0"}:
                    actual = False
            elif isinstance(actual, (int, bool)):
                actual = bool(actual)
        elif isinstance(expected, int):
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                pass
        if actual != expected:
            failures.append(f"room_rule_{key}_mismatch:{actual!r}!={expected!r}")
    return failures


def validate_native_eval_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_matches: int | None = None,
) -> list[str]:
    failures: list[str] = []
    if manifest.get("version") != "pailemen_eval_manifest_v1":
        failures.append("manifest_version_mismatch")
    if manifest.get("opponent_id") != PAILEMEN_NATIVE_OPPONENT:
        failures.append("opponent_identity_mismatch")
    if EXPECTED_GDAI_SHA256 and manifest.get("gdai_binary_sha256") != EXPECTED_GDAI_SHA256:
        failures.append("gdai_binary_sha256_mismatch")
    failures.extend(validate_target_rules(manifest.get("room_rules") or {}))
    if manifest.get("common_random_numbers") is not False:
        failures.append("common_random_numbers_must_be_false")
    if manifest.get("evaluation_unit") != "full_upgrade_match":
        failures.append("evaluation_unit_mismatch")
    if expected_checkpoint_sha256 and manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        failures.append("checkpoint_sha256_mismatch")
    completed = int(manifest.get("matches_completed", 0) or 0)
    if expected_matches is not None and completed != int(expected_matches):
        failures.append(f"matches_completed_mismatch:{completed}!={int(expected_matches)}")
    tables = manifest.get("tables") or []
    if len(tables) != completed:
        failures.append("table_count_mismatch")
    replay_ids: set[str] = set()
    for index, table in enumerate(tables):
        prefix = f"table_{index}"
        desk_id = int(table.get("desk_id", 0) or 0)
        if desk_id <= 0:
            failures.append(f"{prefix}_desk_id_missing")
        if table.get("opponent_id") != PAILEMEN_NATIVE_OPPONENT:
            failures.append(f"{prefix}_opponent_identity_mismatch")
        if table.get("opposite_seats_verified") is not True:
            failures.append(f"{prefix}_opposite_seats_not_verified")
        create_chair = table.get("create_chair")
        join_chair = table.get("join_chair")
        if create_chair is None or join_chair is None:
            failures.append(f"{prefix}_chair_assignment_missing")
        elif abs(int(create_chair) - int(join_chair)) != 2:
            failures.append(f"{prefix}_chairs_not_opposite")
        if table.get("call_robot_verified") is not True:
            failures.append(f"{prefix}_call_robot_not_verified")
        if int(table.get("robot_count", 0) or 0) != 2:
            failures.append(f"{prefix}_robot_count_mismatch")
        if table.get("terminal_4136_verified") is not True:
            failures.append(f"{prefix}_terminal_4136_missing")
        replay_id = str(table.get("replay_id") or "")
        if not replay_id:
            failures.append(f"{prefix}_replay_id_missing")
        elif replay_id in replay_ids:
            failures.append(f"{prefix}_duplicate_replay_id")
        replay_ids.add(replay_id)
        if table.get("final_match_winner") not in (True, False):
            failures.append(f"{prefix}_final_match_winner_missing")
        failures.extend(f"{prefix}_{item}" for item in validate_target_rules(table.get("room_rules") or {}))
    gate = manifest.get("quality_gate") or {}
    recorded = list(gate.get("failures") or [])
    if recorded:
        failures.extend(f"recorded:{value}" for value in recorded)
    if gate.get("passed") is not True:
        failures.append("quality_gate_not_passed")
    return sorted(set(failures))


def validate_proxy_admission_report(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256s: list[str],
    minimum_top1: float = 0.70,
) -> list[str]:
    failures: list[str] = []
    if report.get("version") != "plm_proxy_admission_v1":
        failures.append("proxy_report_version_mismatch")
    if report.get("opponent_id") != PLM_PROXY_OPPONENT:
        failures.append("proxy_opponent_identity_mismatch")
    recorded = sorted(str(value) for value in report.get("checkpoint_sha256s") or [])
    if recorded != sorted(checkpoint_sha256s):
        failures.append("proxy_checkpoint_hashes_mismatch")
    if report.get("dataset_sha256") in (None, ""):
        failures.append("proxy_dataset_sha256_missing")
    if report.get("split") not in {"frozen_test", "frozen_validation"}:
        failures.append("proxy_split_not_frozen")
    metrics = report.get("metrics") or {}
    for bucket in ("overall", "lead", "nonpass"):
        value = (metrics.get(bucket) or {}).get("recall_at_1")
        if value is None:
            failures.append(f"proxy_{bucket}_top1_missing")
        elif float(value) + 1.0e-12 < minimum_top1:
            failures.append(f"proxy_{bucket}_top1_below_{minimum_top1:.2f}")
    if int(report.get("games", 0) or 0) <= 0:
        failures.append("proxy_games_missing")
    gate = report.get("quality_gate") or {}
    if gate.get("passed") is not True or gate.get("failures"):
        failures.append("proxy_quality_gate_failed")
    return sorted(set(failures))


def _rate(row: Mapping[str, Any], key: str) -> float:
    return float((row.get("metrics") or row).get(key, 0.0) or 0.0)


@dataclass(frozen=True)
class NativeRegressionDecision:
    version: str
    initial_negative: bool
    confirmation_negative: bool
    pooled_negative: bool
    sustained_decline: bool
    deltas: dict[str, float]
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeMilestoneDecision:
    version: str
    action: str
    qualified: bool
    needs_confirmation: bool
    reasons: list[str]
    failure_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_native_milestone(
    metrics: Mapping[str, Any],
    regression: NativeRegressionDecision,
) -> NativeMilestoneDecision:
    """Apply local safety gates around the only formal strength target.

    A single negative 10-match block pauses for confirmation. Strategy changes
    are authorized only after the preregistered two-block decline criterion.
    """

    failures: list[str] = []
    failure_types: set[str] = set()
    human = metrics.get("human_policy_recall") or {}
    human_baseline = metrics.get("human_policy_baseline") or {}
    for bucket in ("overall", "lead", "nonpass"):
        current = (human.get(bucket) or {}).get("recall_at_1")
        baseline = (human_baseline.get(bucket) or {}).get("recall_at_1")
        if current is None or baseline is None:
            failures.append(f"human_{bucket}_missing")
            failure_types.add("human_policy_forgetting")
        elif float(baseline) - float(current) > 0.02 + 1.0e-9:
            failures.append(f"human_{bucket}_drop_gt_2pp")
            failure_types.add("human_policy_forgetting")
    candidate = metrics.get("candidate_recall") or {}
    candidate_baseline = metrics.get("candidate_recall_baseline") or {}
    for key in ("p128", "fast_top10", "exact_top10"):
        current = candidate.get(key)
        baseline = candidate_baseline.get(key)
        if current is None or baseline is None:
            failures.append(f"candidate_{key}_missing")
            failure_types.add("candidate_support_regression")
        elif float(current) + 1.0e-12 < float(baseline):
            failures.append(f"candidate_{key}_regression")
            failure_types.add("candidate_support_regression")
    critic = metrics.get("critic_quality") or {}
    critic_baseline = metrics.get("critic_quality_baseline") or {}
    current_ev = critic.get("explained_variance")
    baseline_ev = critic_baseline.get("explained_variance")
    if current_ev is None or baseline_ev is None:
        failures.append("critic_ev_missing")
        failure_types.add("critic_underfit_or_bias")
    elif float(current_ev) + 1.0e-12 < float(baseline_ev):
        failures.append("critic_ev_regression")
        failure_types.add("critic_underfit_or_bias")
    if bool(critic.get("new_significant_phase_bias", False)):
        failures.append("critic_phase_bias")
        failure_types.add("critic_underfit_or_bias")
    if float(metrics.get("legal_action_rate", 0.0)) < 1.0:
        failures.append("illegal_action")
        failure_types.add("policy_collapse_or_legality")
    if failures:
        return NativeMilestoneDecision(
            version="pailemen_native_milestone_decision_v1",
            action="restore_previous_qualified",
            qualified=False,
            needs_confirmation=False,
            reasons=failures,
            failure_types=sorted(failure_types),
        )
    if regression.action == "request_confirmation":
        return NativeMilestoneDecision(
            version="pailemen_native_milestone_decision_v1",
            action="pause_and_confirm_native",
            qualified=False,
            needs_confirmation=True,
            reasons=["negative PaiLeMen native Rule AI 10-match block"],
            failure_types=["opponent_specific_vulnerability"],
        )
    if regression.sustained_decline:
        return NativeMilestoneDecision(
            version="pailemen_native_milestone_decision_v1",
            action="restore_and_read_eight_primary_papers",
            qualified=False,
            needs_confirmation=False,
            reasons=[
                "two negative native blocks, negative pooled match win rate, and negative secondary metric"
            ],
            failure_types=["opponent_specific_vulnerability"],
        )
    return NativeMilestoneDecision(
        version="pailemen_native_milestone_decision_v1",
        action=(
            "advance_one_unchanged_5000_window"
            if regression.confirmation_negative or regression.initial_negative
            else "advance"
        ),
        qualified=True,
        needs_confirmation=False,
        reasons=["PaiLeMen native target and local safety gates passed"],
        failure_types=[],
    )


def decide_sustained_native_decline(
    initial_current: Mapping[str, Any],
    initial_reference: Mapping[str, Any],
    confirmation_current: Mapping[str, Any] | None = None,
    confirmation_reference: Mapping[str, Any] | None = None,
) -> NativeRegressionDecision:
    primary = "final_match_win_rate"
    secondary = ("round_win_rate", "avg_gold_per_round")
    initial_delta = _rate(initial_current, primary) - _rate(initial_reference, primary)
    deltas = {f"initial_{primary}": initial_delta}
    initial_negative = initial_delta < 0.0
    if confirmation_current is None or confirmation_reference is None:
        return NativeRegressionDecision(
            version="pailemen_native_regression_v1",
            initial_negative=initial_negative,
            confirmation_negative=False,
            pooled_negative=False,
            sustained_decline=False,
            deltas=deltas,
            action="request_confirmation" if initial_negative else "continue",
        )

    confirmation_delta = _rate(confirmation_current, primary) - _rate(
        confirmation_reference, primary
    )
    deltas[f"confirmation_{primary}"] = confirmation_delta
    confirmation_negative = confirmation_delta < 0.0
    current_matches = int(initial_current.get("matches_completed", 0) or 0)
    reference_matches = int(initial_reference.get("matches_completed", 0) or 0)
    confirm_current_matches = int(confirmation_current.get("matches_completed", 0) or 0)
    confirm_reference_matches = int(confirmation_reference.get("matches_completed", 0) or 0)

    def pooled(left: Mapping[str, Any], right: Mapping[str, Any], key: str, n1: int, n2: int) -> float:
        total = n1 + n2
        if total <= 0:
            return 0.0
        return (_rate(left, key) * n1 + _rate(right, key) * n2) / total

    pooled_current = pooled(initial_current, confirmation_current, primary, current_matches, confirm_current_matches)
    pooled_reference = pooled(
        initial_reference,
        confirmation_reference,
        primary,
        reference_matches,
        confirm_reference_matches,
    )
    pooled_delta = pooled_current - pooled_reference
    deltas[f"pooled_{primary}"] = pooled_delta
    secondary_negative = False
    for key in secondary:
        current_value = pooled(initial_current, confirmation_current, key, current_matches, confirm_current_matches)
        reference_value = pooled(
            initial_reference,
            confirmation_reference,
            key,
            reference_matches,
            confirm_reference_matches,
        )
        deltas[f"pooled_{key}"] = current_value - reference_value
        secondary_negative = secondary_negative or current_value < reference_value
    pooled_negative = pooled_delta < 0.0
    sustained = initial_negative and confirmation_negative and pooled_negative and secondary_negative
    return NativeRegressionDecision(
        version="pailemen_native_regression_v1",
        initial_negative=initial_negative,
        confirmation_negative=confirmation_negative,
        pooled_negative=pooled_negative,
        sustained_decline=sustained,
        deltas=deltas,
        action=(
            "restore_and_read_eight_primary_papers"
            if sustained
            else "continue_one_unchanged_5000_window"
        ),
    )
