"""Public-information determinization helpers for V3Pro endgame experiments."""

from __future__ import annotations

import copy

import random

import statistics

import hashlib

import json

import math

import time

from collections import Counter

from contextlib import contextmanager

from dataclasses import dataclass

from functools import lru_cache

from typing import Any, Callable, Iterable

NORMAL_RANKS = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2")

SUITS = ("S", "H", "C", "D")

JOKERS = ("BJ", "RJ")

ALL_CARDS = tuple(f"{suit}{rank}" for rank in NORMAL_RANKS for suit in SUITS) + JOKERS


def canonical_double_deck() -> tuple[str, ...]:
    return ALL_CARDS + ALL_CARDS


def _to_retrieval_card(card: object) -> str:
    text = str(card).strip().upper().replace("10", "T")
    return {"SB": "BJ", "B": "BJ", "HR": "RJ", "R": "RJ"}.get(text, text)


def _to_engine_card(card: str) -> str:
    return {"BJ": "SB", "RJ": "HR"}.get(card, card)


def _validated_public_unseen(
    actor_seat: int,
    actor_hand: list[str],
    played_cards: list[str],
    public_counts: list[int],
) -> tuple[list[str], Counter[str]]:
    if actor_seat not in range(4):
        raise ValueError("actor_seat must be in [0,3]")
    if len(public_counts) != 4 or any(count < 0 for count in public_counts):
        raise ValueError("public_counts must contain four non-negative counts")
    normalized_actor = [_to_retrieval_card(card) for card in actor_hand]
    normalized_played = [_to_retrieval_card(card) for card in played_cards]
    if public_counts[actor_seat] != len(normalized_actor):
        raise ValueError("actor public count does not match actor hand")
    deck_counts = Counter(canonical_double_deck())
    visible = Counter(normalized_actor) + Counter(normalized_played)
    invalid = {
        card: count
        for card, count in visible.items()
        if card not in deck_counts or count > deck_counts[card]
    }
    if invalid:
        raise ValueError(f"public card multiplicity exceeds double deck: {invalid}")
    unseen = deck_counts - visible
    expected = sum(public_counts) - public_counts[actor_seat]
    if sum(unseen.values()) != expected:
        raise ValueError(
            "public unseen pool/count mismatch: "
            f"pool={sum(unseen.values())} expected={expected}"
        )
    return normalized_actor, unseen


def _card_distributions(total: int, capacities: tuple[int, ...]):
    def visit(position: int, remaining: int, prefix: tuple[int, ...]):
        if position == len(capacities) - 1:
            if remaining <= capacities[position]:
                yield prefix + (remaining,)
            return
        for count in range(min(remaining, capacities[position]) + 1):
            yield from visit(position + 1, remaining - count, prefix + (count,))

    yield from visit(0, int(total), ())


def _count_allocations(
    card_counts: tuple[tuple[str, int], ...],
    capacities: tuple[int, ...],
) -> int:
    @lru_cache(maxsize=None)
    def count(card_index: int, remaining: tuple[int, ...]) -> int:
        if card_index == len(card_counts):
            return int(all(value == 0 for value in remaining))
        multiplicity = card_counts[card_index][1]
        total = 0
        for distribution in _card_distributions(multiplicity, remaining):
            next_remaining = tuple(
                capacity - used
                for capacity, used in zip(remaining, distribution, strict=True)
            )
            total += count(card_index + 1, next_remaining)
        return total

    return count(0, capacities)


def count_hidden_hand_allocations(
    actor_seat: int,
    actor_hand: list[str],
    played_cards: list[str],
    public_counts: list[int],
) -> int:
    """Count distinct allocations to labeled hidden seats, treating duplicates equally."""
    _, unseen = _validated_public_unseen(
        actor_seat,
        actor_hand,
        played_cards,
        public_counts,
    )
    seats = tuple(seat for seat in range(4) if seat != actor_seat)
    capacities = tuple(int(public_counts[seat]) for seat in seats)
    return _count_allocations(tuple(unseen.items()), capacities)


class HiddenAllocationLimitExceeded(RuntimeError):
    def __init__(
        self,
        total_allocations: int,
        total_physical_allocations: int,
        limit: int,
    ) -> None:
        self.total_allocations = int(total_allocations)
        self.total_physical_allocations = int(total_physical_allocations)
        self.limit = int(limit)
        super().__init__(
            f"hidden allocation total {self.total_allocations} exceeds limit {self.limit}"
        )


@dataclass(frozen=True)
class HiddenHandEnumeration:
    total_allocations: int
    total_physical_allocations: int
    allocations: tuple[tuple[tuple[str, ...], ...], ...]
    allocation_weights: tuple[int, ...]


def _distribution_weight(multiplicity: int, distribution: tuple[int, ...]) -> int:
    remaining = int(multiplicity)
    weight = 1
    for count in distribution:
        weight *= math.comb(remaining, int(count))
        remaining -= int(count)
    return weight


def enumerate_hidden_hands_exhaustive(
    actor_seat: int,
    actor_hand: list[str],
    played_cards: list[str],
    public_counts: list[int],
    *,
    max_allocations: int,
) -> HiddenHandEnumeration:
    """Materialize every distinct legal hidden allocation or refuse the limit."""
    if (
        isinstance(max_allocations, bool)
        or int(max_allocations) != max_allocations
        or max_allocations <= 0
    ):
        raise ValueError("max_allocations must be a positive integer")
    normalized_actor, unseen = _validated_public_unseen(
        actor_seat,
        actor_hand,
        played_cards,
        public_counts,
    )
    card_counts = tuple(unseen.items())
    seats = tuple(seat for seat in range(4) if seat != actor_seat)
    capacities = tuple(int(public_counts[seat]) for seat in seats)
    total = _count_allocations(card_counts, capacities)
    total_cards = sum(capacities)
    total_physical = math.factorial(total_cards)
    for capacity in capacities:
        total_physical //= math.factorial(capacity)
    if total > int(max_allocations):
        raise HiddenAllocationLimitExceeded(
            total,
            total_physical,
            int(max_allocations),
        )

    hands: list[list[str]] = [[] for _ in range(4)]
    hands[actor_seat] = list(normalized_actor)
    allocations: list[tuple[tuple[str, ...], ...]] = []
    allocation_weights: list[int] = []

    def visit(card_index: int, remaining: tuple[int, ...], weight: int) -> None:
        if card_index == len(card_counts):
            if all(value == 0 for value in remaining):
                allocations.append(tuple(tuple(hand) for hand in hands))
                allocation_weights.append(int(weight))
            return
        card, multiplicity = card_counts[card_index]
        for distribution in _card_distributions(multiplicity, remaining):
            next_remaining = tuple(
                capacity - used
                for capacity, used in zip(remaining, distribution, strict=True)
            )
            for seat, count in zip(seats, distribution, strict=True):
                hands[seat].extend([card] * count)
            visit(
                card_index + 1,
                next_remaining,
                weight * _distribution_weight(multiplicity, distribution),
            )
            for seat, count in zip(seats, distribution, strict=True):
                if count:
                    del hands[seat][-count:]

    visit(0, capacities, 1)
    if len(allocations) != total:
        raise RuntimeError(
            f"hidden allocation enumeration mismatch: counted={total} "
            f"materialized={len(allocations)}"
        )
    if sum(allocation_weights) != total_physical:
        raise RuntimeError(
            "hidden allocation physical-weight mismatch: "
            f"expected={total_physical} actual={sum(allocation_weights)}"
        )
    return HiddenHandEnumeration(
        total,
        total_physical,
        tuple(allocations),
        tuple(allocation_weights),
    )


@contextmanager
def temporary_hidden_hands(env, actor_seat: int, sampled_hands: list[list[str]]):
    """Install a public-only determinization and restore exact engine objects."""

    if actor_seat not in range(4) or len(sampled_hands) != 4:
        raise ValueError("actor seat and sampled hands must cover four seats")
    original_hands = [player.hand_cards for player in env.players]
    original_hearts = [int(player.hearts_num) for player in env.players]
    actual_actor = Counter(
        _to_retrieval_card(card) for card in original_hands[actor_seat]
    )
    sampled_actor = Counter(sampled_hands[actor_seat])
    if sampled_actor != actual_actor:
        raise ValueError("sampled actor hand differs from the real actor hand")
    for seat in range(4):
        if len(sampled_hands[seat]) != len(original_hands[seat]):
            raise ValueError(f"sampled hand count differs at seat {seat}")
    try:
        current_rank = str(env.rank)
        heart_level = f"H{current_rank}"
        for seat, player in enumerate(env.players):
            if seat == actor_seat:
                continue
            player.hand_cards = []
            player.hearts_num = 0
            for card in sampled_hands[seat]:
                engine_card = _to_engine_card(card)
                player.add_card(engine_card, current_rank)
                if engine_card == heart_level:
                    player.hearts_num += 1
        yield
    finally:
        for player, hand, hearts in zip(
            env.players,
            original_hands,
            original_hearts,
            strict=True,
        ):
            player.hand_cards = hand
            player.hearts_num = hearts
