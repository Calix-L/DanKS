from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from DanKS.retrieval.cards import (
    ALL_CARDS,
    CARD_INDEX,
    RANKS,
    card_rank,
    normalize_card,
    normalize_cards,
    rank_strength,
)
from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.models import ScoredAction
from DanKS.retrieval.rules import normalize_kind
from DanKS.training.schema import (
    ACTION_KINDS,
    ACTION_KIND_DIM,
    CARD_DIM,
    CANDIDATE_DIM,
    CANDIDATE_SCALAR_DIM,
    GROUP_KINDS,
    HISTORY_EVENT_DIM,
    HISTORY_LENGTH,
    MAX_ACTION_CARDS,
    MAX_HAND_CARDS,
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
    for raw_card in cards:
        index = CARD_INDEX.get(raw_card)
        if index is None:
            index = CARD_INDEX[normalize_card(raw_card)]
        out[index] += 1.0 / normalize_by
    return out


def _rank_counts(cards: Iterable[str], *, normalize_by: float = 8.0) -> np.ndarray:
    out = np.zeros(len(RANKS), dtype=np.float32)
    for card in normalize_cards(cards):
        rank = card_rank(card)
        idx = RANK_INDEX.get(rank)
        if idx is not None:
            out[idx] += 1.0 / normalize_by
    return out


def _history_event_vector(
    *,
    actor: int,
    card_counts: np.ndarray,
    finish: bool,
    action_len: int,
    remaining_after: int,
    my_seat: int,
) -> np.ndarray:
    relative = (actor - int(my_seat)) % 4
    seat = _one_hot(relative, 4)
    passed = action_len == 0 and not finish
    extra = np.asarray(
        [
            *seat.tolist(),
            1.0 if passed else 0.0,
            1.0 if finish else 0.0,
            action_len / 27.0,
            remaining_after / 27.0,
            1.0 if relative in (0, 2) else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([card_counts, extra]).astype(np.float32, copy=False)


def history_features(
    history: Iterable[dict[str, object]] | None,
    *,
    my_seat: int,
) -> np.ndarray:
    """Encode the latest public actions as 15x64 Phase-3 history events."""

    events: list[np.ndarray] = []
    remaining = [27, 27, 27, 27]
    for raw in history or ():
        try:
            actor = int(raw.get("pos", -1))
        except (AttributeError, TypeError, ValueError):
            continue
        if actor < 0 or actor >= 4:
            continue
        cards = normalize_cards(raw.get("cards") or ())
        finish = bool(raw.get("finished", False))
        card_counts = np.zeros(CARD_DIM, dtype=np.float32)
        for card in cards:
            card_counts[CARD_INDEX[card]] += 1.0
        action_len = int(card_counts.sum())
        remaining[actor] = (
            0 if finish else max(0, remaining[actor] - action_len)
        )
        event = _history_event_vector(
            actor=actor,
            card_counts=card_counts,
            finish=finish,
            action_len=action_len,
            remaining_after=remaining[actor],
            my_seat=my_seat,
        )
        if event.shape != (HISTORY_EVENT_DIM,):
            raise ValueError(f"history event dim mismatch: {event.shape} != {(HISTORY_EVENT_DIM,)}")
        events.append(event)

    out = np.zeros((HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
    if events:
        recent = events[-HISTORY_LENGTH:]
        out[-len(recent):] = np.stack(recent, axis=0)
    return out


class IncrementalHistoryFeaturizer:
    """Encode an append-only tracker history while processing each event once."""

    def __init__(self) -> None:
        self._source: list[dict[str, Any]] | None = None
        self._raw_count = 0
        self._last_raw_object: dict[str, Any] | None = None
        self._remaining = [27, 27, 27, 27]
        self._events: deque[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = deque(maxlen=HISTORY_LENGTH)

    def reset(self) -> None:
        self._source = None
        self._raw_count = 0
        self._last_raw_object = None
        self._remaining = [27, 27, 27, 27]
        self._events.clear()

    def _can_extend(self, history: list[dict[str, Any]]) -> bool:
        if history is not self._source or len(history) < self._raw_count:
            return False
        if self._raw_count == 0:
            return True
        return history[self._raw_count - 1] is self._last_raw_object

    def features(
        self,
        history: Iterable[dict[str, object]] | None,
        *,
        my_seat: int,
    ) -> np.ndarray:
        if history is None:
            self.reset()
            return np.zeros((HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
        if not isinstance(history, list):
            self.reset()
            return history_features(history, my_seat=my_seat)
        if not self._can_extend(history):
            self.reset()
            self._source = history
        elif self._source is None:
            self._source = history

        for raw in history[self._raw_count :]:
            try:
                actor = int(raw.get("pos", -1))
            except (AttributeError, TypeError, ValueError):
                continue
            if actor < 0 or actor >= 4:
                continue
            cards = normalize_cards(raw.get("cards") or ())
            finish = bool(raw.get("finished", False))
            card_counts = np.zeros(CARD_DIM, dtype=np.float32)
            for card in cards:
                card_counts[CARD_INDEX[card]] += 1.0
            action_len = int(card_counts.sum())
            self._remaining[actor] = (
                0
                if finish
                else max(0, self._remaining[actor] - action_len)
            )
            # Cache the exact final event vector for each observer seat.
            # Public history is append-only, so rebuilding these vectors on
            # every decision only repeats allocation and one-hot work.
            self._events.append(
                tuple(
                    _history_event_vector(
                        actor=actor,
                        card_counts=card_counts,
                        finish=finish,
                        action_len=action_len,
                        remaining_after=self._remaining[actor],
                        my_seat=seat,
                    )
                    for seat in range(4)
                )
            )

        self._raw_count = len(history)
        self._last_raw_object = history[-1] if history else None
        out = np.zeros((HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
        if self._events:
            seat = int(my_seat) % 4
            encoded = [vectors[seat] for vectors in self._events]
            out[-len(encoded) :] = np.stack(encoded, axis=0)
        return out


def _kind_index(kind: str | None) -> int:
    direct = KIND_INDEX.get(kind) if kind is not None else None
    if direct is not None:
        return direct
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
    if len(hand) > MAX_HAND_CARDS:
        raise ValueError(
            f"official GuanDan hand cannot exceed {MAX_HAND_CARDS} cards: got {len(hand)}"
        )
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
    out = np.zeros(CANDIDATE_DIM, dtype=np.float32)
    return _fill_candidate_features(
        out,
        row,
        slot,
        original_hand_size=original_hand_size,
        candidate_capacity=candidate_capacity,
    )


def _fill_candidate_features(
    out: np.ndarray,
    row: ScoredAction,
    slot: int,
    *,
    original_hand_size: int,
    candidate_capacity: int,
) -> np.ndarray:
    if out.shape != (CANDIDATE_DIM,) or out.dtype != np.float32:
        raise ValueError(
            "candidate feature output must be a float32 CANDIDATE_DIM row"
        )
    action = row.action
    if action.size > MAX_ACTION_CARDS:
        raise ValueError(
            f"official GuanDan action cannot exceed {MAX_ACTION_CARDS} cards: "
            f"kind={action.kind} size={action.size}"
        )
    kind = normalize_kind(action.kind)
    details = row.details
    after_hand = details.get("after_hand") or ()
    partition = row.partition
    strength = rank_strength(action.rank) if action.rank and action.rank != "PASS" else 0.0

    offset = 0
    for raw_card in action.cards:
        index = CARD_INDEX.get(raw_card)
        if index is None:
            index = CARD_INDEX[normalize_card(raw_card)]
        out[offset + index] += 0.5
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
        group_idx = GROUP_INDEX.get(group.kind)
        if group_idx is None:
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
        _fill_candidate_features(
            candidates[slot],
            row,
            slot,
            original_hand_size=len(hand),
            candidate_capacity=top_k,
        )
        mask[slot] = 1.0
    return state, candidates, mask
