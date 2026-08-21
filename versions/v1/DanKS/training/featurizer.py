from __future__ import annotations

from typing import Iterable

import numpy as np

from DanKS.retrieval.cards import ALL_CARDS, CARD_INDEX, RANKS, card_rank, normalize_cards, rank_strength
from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.models import ScoredAction
from DanKS.retrieval.action_semantics import normalize_kind
from DanKS.training.schema import (
    ACTION_KINDS,
    ACTION_KIND_DIM,
    CARD_DIM,
    CANDIDATE_DIM,
    CANDIDATE_SCALAR_DIM,
    GROUP_KINDS,
    RANK_DIM,
    STATE_DIM,
    TOPK,
)


KIND_INDEX = {kind: i for i, kind in enumerate(ACTION_KINDS)}
GROUP_INDEX = {kind: i for i, kind in enumerate(GROUP_KINDS)}
RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}


def _clip(value: float, scale: float) -> float:
    if scale == 0:
        return 0.0
    scaled = float(value) / scale
    if scaled > 5.0:
        return 5.0
    if scaled < -5.0:
        return -5.0
    return scaled


def _one_hot(index: int | None, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if index is not None and 0 <= index < dim:
        out[index] = 1.0
    return out


def _card_counts(cards: Iterable[str], *, normalize_by: float = 2.0) -> np.ndarray:
    out = np.zeros(len(ALL_CARDS), dtype=np.float32)
    for card in normalize_cards(cards):
        out[CARD_INDEX[card]] += 1.0 / normalize_by
    return out


def _rank_counts(cards: Iterable[str], *, normalize_by: float = 8.0) -> np.ndarray:
    out = np.zeros(len(RANKS), dtype=np.float32)
    for card in normalize_cards(cards):
        rank = card_rank(card)
        idx = RANK_INDEX.get(rank)
        if idx is not None:
            out[idx] += 1.0 / normalize_by
    return out


def _kind_index(kind: str | None) -> int:
    normalized = normalize_kind(kind or "Unknown")
    return KIND_INDEX.get(normalized, KIND_INDEX["Unknown"])


def _rank_index(rank: str | None) -> int | None:
    if not rank or rank == "PASS":
        return None
    return RANK_INDEX.get(rank)


def _last_player_relative(ctx: RetrievalContext) -> np.ndarray:
    # Slots: unknown, self, left opponent, teammate, right opponent.
    if ctx.last_player is None:
        return _one_hot(0, 5)
    rel = (int(ctx.last_player) - int(ctx.my_seat)) % 4
    return _one_hot(rel + 1, 5)


def state_features(hand: list[str], ctx: RetrievalContext) -> np.ndarray:
    features = [
        _card_counts(hand, normalize_by=2.0),
        _one_hot(_rank_index(ctx.cur_rank), len(RANKS)),
        _one_hot(_kind_index(ctx.current_kind), len(ACTION_KINDS)),
        _one_hot(_rank_index(ctx.current_rank), len(RANKS)),
        np.array([min(1.0, max(0.0, ctx.current_size / 10.0))], dtype=np.float32),
        np.array([min(1.0, max(0.0, x / 27.0)) for x in ctx.public_counts], dtype=np.float32),
        _last_player_relative(ctx),
        np.array([1.0 if normalize_kind(ctx.current_kind) == "Lead" else 0.0], dtype=np.float32),
        _remaining_rank_features(ctx),
    ]
    out = np.concatenate(features).astype(np.float32, copy=False)
    if out.shape[0] != STATE_DIM:
        raise ValueError(f"state feature dim mismatch: got {out.shape[0]}, expected {STATE_DIM}")
    return out


def _remaining_rank_features(ctx: RetrievalContext) -> np.ndarray:
    if not ctx.remaining_by_rank:
        return np.zeros(len(RANKS), dtype=np.float32)
    return np.array([min(1.0, max(0.0, ctx.remaining_by_rank.get(rank, 0) / 8.0)) for rank in RANKS], dtype=np.float32)


def candidate_features(
    row: ScoredAction, slot: int, *, original_hand_size: int,
    candidate_capacity: int = TOPK,
) -> np.ndarray:
    action = row.action
    kind = normalize_kind(action.kind)
    details = row.details
    after_hand = details.get("after_hand") or ()
    partition = row.partition
    strength = rank_strength(action.rank) if action.rank and action.rank != "PASS" else 0.0

    out = np.zeros(CANDIDATE_DIM, dtype=np.float32)
    offset = 0
    for card in normalize_cards(action.cards):
        out[offset + CARD_INDEX[card]] += 0.5
    offset += CARD_DIM

    kind_idx = _kind_index(kind)
    if 0 <= kind_idx < ACTION_KIND_DIM:
        out[offset + kind_idx] = 1.0
    offset += ACTION_KIND_DIM

    rank_idx = _rank_index(action.rank)
    if rank_idx is not None and 0 <= rank_idx < RANK_DIM:
        out[offset + rank_idx] = 1.0
    offset += RANK_DIM

    scalar_values = (
        min(1.0, (slot + 1) / candidate_capacity),
        _clip(row.score, 1000.0),
        _clip(row.hand_count_score, 20.0),
        _clip(row.card_value_score, 800.0),
        _clip(row.retake_score, 500.0),
        _clip(float(details.get("current_control_score", 0.0)), 200.0),
        _clip(float(details.get("lead_action_score", 0.0)), 900.0),
        _clip(float(details.get("spend_penalty", 0.0)), 5.0),
        _clip(float(details.get("break_group_penalty", 0.0)), 40.0),
        _clip(float(details.get("low_break_preference_penalty", 0.0)), 40.0),
        _clip(float(details.get("escape_risk_penalty", 0.0)), 2.0),
        _clip(float(details.get("pass_pressure_penalty", 0.0)), 2.0),
        _clip(float(details.get("teammate_overcall_penalty", 0.0)), 2.0),
        _clip(float(details.get("tempo_score", 0.0)), 500.0),
        min(1.0, max(0.0, float(details.get("my_min_steps", partition.hand_count)) / 12.0)),
        min(1.0, max(0.0, float(details.get("partner_min_steps", 0.0)) / 12.0)),
        min(1.0, max(0.0, float(details.get("opponent_min_steps_min", 0.0)) / 12.0)),
        min(1.0, max(0.0, float(details.get("opponent_short_pressure", 0.0)))),
        min(1.0, max(0.0, float(details.get("my_retake_count", 0.0)) / 8.0)),
        min(1.0, max(0.0, float(details.get("partner_follow_help", 0.0)))),
        min(1.0, max(0.0, float(details.get("must_block", 0.0)))),
        min(1.0, max(0.0, float(details.get("can_race", 0.0)))),
        min(1.0, max(0.0, action.size / 10.0)),
        min(1.0, max(0.0, len(after_hand) / max(1, original_hand_size))),
        min(1.0, max(0.0, partition.hand_count / 12.0)),
        strength,
    )
    if len(scalar_values) != CANDIDATE_SCALAR_DIM:
        raise ValueError("candidate scalar feature dim mismatch")
    out[offset : offset + CANDIDATE_SCALAR_DIM] = scalar_values
    offset += CANDIDATE_SCALAR_DIM

    for group in partition.groups:
        group_idx = GROUP_INDEX.get(normalize_kind(group.kind))
        if group_idx is not None:
            idx = offset + group_idx
            out[idx] = min(1.0, out[idx] + 0.25)
    return out


def featurize_topk(hand: list[str], ctx: RetrievalContext, ranked: list[ScoredAction], top_k: int = TOPK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0 or len(ranked) > top_k:
        raise ValueError("top_k must be positive and retain every supplied ranked action")
    state = state_features(hand, ctx)
    candidates = np.zeros((top_k, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros((top_k,), dtype=np.float32)
    for slot, row in enumerate(ranked):
        candidates[slot] = candidate_features(
            row, slot, original_hand_size=len(hand), candidate_capacity=top_k,
        )
        mask[slot] = 1.0
    return state, candidates, mask
