from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import ScoredAction
from .action_semantics import normalize_kind
from .scoring import ScoreWeights


STRUCTURAL_TERM_VERSION = "structural_score_terms_v1"
STRUCTURAL_TERM_NAMES = (
    "hand_count",
    "card_value",
    "retake",
    "residue",
    "current_control",
    "lead_action",
    "pass_pressure",
    "bomb_spend",
    "control_spend",
    "teammate_overcall",
    "break_group",
    "low_break_preference",
    "escape_risk",
    "tempo",
)
_PENALTY_TERMS = {
    "residue",
    "pass_pressure",
    "bomb_spend",
    "control_spend",
    "teammate_overcall",
    "break_group",
    "low_break_preference",
    "escape_risk",
}


def structural_score_terms(row: ScoredAction) -> np.ndarray:
    details = row.details
    is_bomb = normalize_kind(row.action.kind) in {"Bomb", "StraightFlush", "FourKings"}
    spend = float(details.get("spend_penalty", 0.0))
    raw = {
        "hand_count": float(row.hand_count_score),
        "card_value": float(row.card_value_score),
        "retake": float(row.retake_score),
        "residue": float(details.get("residue_penalty_score", 0.0)),
        "current_control": float(details.get("current_control_score", 0.0)),
        "lead_action": float(details.get("lead_action_score", 0.0)),
        "pass_pressure": float(details.get("pass_pressure_penalty", 0.0)),
        "bomb_spend": spend if is_bomb else 0.0,
        "control_spend": 0.0 if is_bomb else spend,
        "teammate_overcall": float(details.get("teammate_overcall_penalty", 0.0)),
        "break_group": float(details.get("break_group_penalty", 0.0)),
        "low_break_preference": float(details.get("low_break_preference_penalty", 0.0)),
        "escape_risk": float(details.get("escape_risk_penalty", 0.0)),
        "tempo": float(details.get("tempo_score", 0.0)),
    }
    return np.asarray(
        [-raw[name] if name in _PENALTY_TERMS else raw[name] for name in STRUCTURAL_TERM_NAMES],
        dtype=np.float32,
    )


def weights_to_vector(weights: ScoreWeights) -> np.ndarray:
    values = asdict(weights)
    return np.asarray([float(values[name]) for name in STRUCTURAL_TERM_NAMES], dtype=np.float32)


def score_from_terms(terms: np.ndarray, weights: ScoreWeights | Mapping[str, float]) -> np.ndarray:
    vector = (
        weights_to_vector(weights)
        if isinstance(weights, ScoreWeights)
        else np.asarray([float(weights[name]) for name in STRUCTURAL_TERM_NAMES], dtype=np.float32)
    )
    return np.asarray(terms, dtype=np.float32) @ vector


@dataclass(frozen=True)
class StructuralCalibrationProfile:
    version: str
    base: Mapping[str, float]
    lead_delta: Mapping[str, float]
    follow_delta: Mapping[str, float]
    partition_scope: str = "fixed_partition"
    metadata: Mapping[str, Any] | None = None

    def weights_for_kind(self, current_kind: str | None) -> dict[str, float]:
        delta = self.lead_delta if normalize_kind(current_kind) == "Lead" else self.follow_delta
        return {
            name: max(0.0, float(self.base[name]) + float(delta.get(name, 0.0)))
            for name in STRUCTURAL_TERM_NAMES
        }

    def score(self, row: ScoredAction, current_kind: str | None) -> float:
        return float(score_from_terms(structural_score_terms(row), self.weights_for_kind(current_kind)))


def load_structural_calibration_profile(path: str | Path) -> StructuralCalibrationProfile:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return structural_calibration_profile_from_mapping(payload)


def structural_calibration_profile_from_mapping(payload: Mapping[str, Any]) -> StructuralCalibrationProfile:
    if payload.get("version") != STRUCTURAL_TERM_VERSION:
        raise ValueError(
            f"unsupported structural calibration profile version {payload.get('version')!r}; "
            f"expected {STRUCTURAL_TERM_VERSION!r}"
        )
    for section in ("base", "lead_delta", "follow_delta"):
        if section not in payload:
            raise ValueError(f"structural calibration profile lacks {section}")
    missing = [name for name in STRUCTURAL_TERM_NAMES if name not in payload["base"]]
    if missing:
        raise ValueError(f"structural calibration profile base lacks: {', '.join(missing)}")
    return StructuralCalibrationProfile(
        version=str(payload["version"]),
        base=payload["base"],
        lead_delta=payload["lead_delta"],
        follow_delta=payload["follow_delta"],
        partition_scope=str(payload.get("partition_scope", "fixed_partition")),
        metadata=payload.get("metadata"),
    )
