from __future__ import annotations

import os
from typing import Any


try:
    from .native_cpp import danks_actor_core
except Exception:  # pragma: no cover - exercised when the extension is absent.
    danks_actor_core = None


_TRUE_VALUES = {"1", "true", "yes", "on"}
_TUPLE_SIGNATURE_PAYLOADS = (
    os.environ.get("DANKS_NATIVE_TUPLE_SIGNATURE_PAYLOADS", "").strip().lower()
    in _TRUE_VALUES
)
_RAW_COVER_INPUTS = (
    os.environ.get("DANKS_NATIVE_RAW_COVER_INPUTS", "").strip().lower()
    in _TRUE_VALUES
)
_BUCKET_CAPSULES = (
    os.environ.get("DANKS_NATIVE_BUCKET_CAPSULES", "").strip().lower()
    in _TRUE_VALUES
)


def available() -> bool:
    disabled = os.environ.get("DANKS_DISABLE_NATIVE_ACTOR_CORE", "").strip().lower()
    return danks_actor_core is not None and disabled not in _TRUE_VALUES


def remove_cards_sorted(hand: list[str] | tuple[str, ...], action: list[str] | tuple[str, ...]) -> list[str]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    return list(danks_actor_core.remove_cards_sorted(hand, action))


def remove_cards_sorted_batch(
    hand: list[str] | tuple[str, ...],
    actions: list[list[str] | tuple[str, ...]],
) -> list[list[str]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "remove_cards_sorted_batch"):
        return [remove_cards_sorted(hand, action) for action in actions]
    return [list(after_hand) for after_hand in danks_actor_core.remove_cards_sorted_batch(hand, actions)]


def batch_action_static_features(actions: list[tuple[str, tuple[str, ...], str]], ctx: Any) -> list[tuple[float, ...]]:
    if danks_actor_core is None or not hasattr(danks_actor_core, "batch_action_static_features"):
        raise RuntimeError("C++ action-feature batch is not available")
    remaining_rj = int(ctx.remaining_detail.get("RJ", -1)) if ctx.remaining_detail else -1
    last_player = -1 if ctx.last_player is None else int(ctx.last_player)
    rows = danks_actor_core.batch_action_static_features(
        actions,
        ctx.cur_rank or "",
        ctx.current_rank or "",
        str(ctx.current_kind),
        tuple(int(value) for value in ctx.public_counts),
        int(ctx.my_seat),
        last_player,
        remaining_rj,
    )
    return [tuple(float(value) for value in row) for row in rows]


def batch_break_group_penalties(actions: list[Any], partitions: list[Any], profile: Any) -> list[float]:
    if danks_actor_core is None or not hasattr(danks_actor_core, "batch_break_group_penalties"):
        raise RuntimeError("C++ break-penalty batch is not available")
    action_rows = [(action.kind, action.cards) for action in actions]
    partition_rows = [
        [(group.kind, group.cards) for group in partition.groups]
        for partition in partitions
    ]
    return [
        float(value)
        for value in danks_actor_core.batch_break_group_penalties(
            action_rows,
            partition_rows,
            dict(profile.base_by_kind),
            float(profile.straight_flush_to_bomb),
            float(profile.bomb_break_size_bonus),
        )
    ]


def build_cover_inputs(
    hand: list[str] | tuple[str, ...],
    group_cards: list[tuple[str, ...] | list[str]],
) -> tuple[tuple[int, ...], list[list[list[int]]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "build_cover_inputs"):
        raise RuntimeError("C++ cover-input encoder is not available")
    if _BUCKET_CAPSULES and hasattr(danks_actor_core, "build_cover_inputs_capsule"):
        start, buckets = danks_actor_core.build_cover_inputs_capsule(hand, group_cards)
    else:
        start, buckets = danks_actor_core.build_cover_inputs(hand, group_cards)
    if _RAW_COVER_INPUTS:
        return tuple(start), buckets
    return tuple(int(value) for value in start), [
        [[int(value) for value in encoded] for encoded in bucket]
        for bucket in buckets
    ]


def build_cover_inputs_beam_order(
    hand: list[str] | tuple[str, ...],
    group_cards: list[tuple[str, ...] | list[str]],
) -> tuple[tuple[int, ...], list[list[list[int]]]]:
    if danks_actor_core is None or not hasattr(
        danks_actor_core, "build_cover_inputs_beam_order"
    ):
        raise RuntimeError("C++ beam-order cover-input encoder is not available")
    if _BUCKET_CAPSULES and hasattr(
        danks_actor_core,
        "build_cover_inputs_beam_order_capsule",
    ):
        start, buckets = danks_actor_core.build_cover_inputs_beam_order_capsule(
            hand,
            group_cards,
        )
    else:
        start, buckets = danks_actor_core.build_cover_inputs_beam_order(hand, group_cards)
    if _RAW_COVER_INPUTS:
        return tuple(start), buckets
    return tuple(int(value) for value in start), [
        [[int(value) for value in encoded] for encoded in bucket]
        for bucket in buckets
    ]


def generate_same_rank_group_signatures(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "generate_same_rank_group_signatures"):
        raise RuntimeError("C++ same-rank group generator is not available")
    return [
        (str(kind), str(rank), tuple(cards), tuple(wild_as))
        for kind, rank, cards, wild_as in danks_actor_core.generate_same_rank_group_signatures(hand, cur_rank or "")
    ]


def generate_sequence_group_signatures(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "generate_sequence_group_signatures"):
        raise RuntimeError("C++ sequence group generator is not available")
    return [
        (str(kind), str(rank), tuple(cards), tuple(wild_as))
        for kind, rank, cards, wild_as in danks_actor_core.generate_sequence_group_signatures(hand, cur_rank or "")
    ]


def generate_multi_sequence_group_signatures(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "generate_multi_sequence_group_signatures"):
        raise RuntimeError("C++ multi-sequence group generator is not available")
    return [
        (str(kind), str(rank), tuple(cards), tuple(wild_as))
        for kind, rank, cards, wild_as in danks_actor_core.generate_multi_sequence_group_signatures(hand, cur_rank or "")
    ]


def generate_triple_plus_group_signatures(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "generate_triple_plus_group_signatures"):
        raise RuntimeError("C++ triple-plus group generator is not available")
    return [
        (str(kind), str(rank), tuple(cards), tuple(wild_as))
        for kind, rank, cards, wild_as in danks_actor_core.generate_triple_plus_group_signatures(hand, cur_rank or "")
    ]


def generate_all_group_signatures(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "generate_all_group_signatures"):
        raise RuntimeError("C++ all-group generator is not available")
    if _TUPLE_SIGNATURE_PAYLOADS and hasattr(
        danks_actor_core,
        "generate_all_group_signatures_tupled",
    ):
        return list(danks_actor_core.generate_all_group_signatures_tupled(hand, cur_rank or ""))
    return [
        (str(kind), str(rank), tuple(cards), tuple(wild_as))
        for kind, rank, cards, wild_as in danks_actor_core.generate_all_group_signatures(hand, cur_rank or "")
    ]


def build_group_records_and_cover_inputs(
    hand: list[str] | tuple[str, ...],
    cur_rank: str | None,
) -> tuple[
    list[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[tuple[int, int], ...]]],
    tuple[int, ...],
    list[list[list[int]]],
]:
    if danks_actor_core is None:
        raise RuntimeError("C++ actor core is not available")
    if not hasattr(danks_actor_core, "build_group_records_and_cover_inputs"):
        raise RuntimeError("C++ group-record cover-input builder is not available")
    records, start, buckets = danks_actor_core.build_group_records_and_cover_inputs(hand, cur_rank or "")
    return (
        [
            (
                str(kind),
                str(rank),
                tuple(cards),
                tuple(wild_as),
                tuple((int(item[0]), int(item[1])) for item in key_items),
            )
            for kind, rank, cards, wild_as, key_items in records
        ],
        tuple(int(value) for value in start),
        [
            [[int(value) for value in encoded] for encoded in bucket]
            for bucket in buckets
        ],
    )


def module() -> Any:
    return danks_actor_core
