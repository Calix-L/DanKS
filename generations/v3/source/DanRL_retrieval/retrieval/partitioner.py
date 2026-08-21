from __future__ import annotations

import os
from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Iterator

from .cards import (
    ALL_CARDS,
    CARD_INDEX,
    NORMAL_RANKS,
    RANK_VALUE,
    STRAIGHT_RANKS,
    SUITS,
    card_rank,
    card_sort_key,
    card_suit,
    heart_level_card,
    normalize_cards,
    rank_strength,
    sorted_cards,
)
from .context import RetrievalContext
from .models import CardGroup, NativePartitionCovers, Partition, effective_group_cost
from . import native_cover
from . import native_actor_core


GROUP_PRIORITY_BASE = {
    "FourKings": 180.0,
    "StraightFlush": 130.0,
    "Bomb": 90.0,
    "StraightPair": 34.0,
    "StraightTriple": 40.0,
    "Straight": 24.0,
    "TriplePlus": 28.0,
    "Triple": 18.0,
    "Pair": 10.0,
    "Single": 4.0,
}

SEQUENCE_VALUE = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13}
SEQUENCE_RANK = {value: rank for rank, value in SEQUENCE_VALUE.items()}
RANK_POINTS = {"2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13, "BJ": 14, "RJ": 15}


def _make_group(
    kind: str,
    cards: tuple[str, ...],
    ctx: RetrievalContext,
    rank: str | None = None,
    meta: dict | None = None,
) -> CardGroup:
    if rank is None and cards:
        rank = max((card_rank(card) for card in cards), key=lambda r: RANK_VALUE.get(r, 0))
    strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail) if rank else 0.0
    return CardGroup(kind=kind, cards=sorted_cards(cards), rank=rank, strength=strength, meta=meta or {})


def _group_priority(group: CardGroup) -> float:
    base = GROUP_PRIORITY_BASE.get(group.kind, 0.0)
    if group.kind == "Bomb":
        base += (group.size - 4) * 14.0
    return base + group.strength * 12.0


def _follow_rank(level: str, lower: str, higher: str) -> bool:
    if lower == higher:
        return False
    if higher == level:
        return lower not in {"BJ", "RJ"}
    if lower == level:
        return higher in {"BJ", "RJ"}
    return RANK_POINTS.get(higher, 0) > RANK_POINTS.get(lower, 0)


def _sequence_value(cards: tuple[str, ...], wildcards: list[str] | tuple[str, ...], group_size: int) -> str | None:
    counts = [0] * 13
    wild_counter = Counter(wildcards)
    for card in cards:
        if wild_counter.get(card, 0) > 0:
            wild_counter[card] -= 1
            continue
        rank = card_rank(card)
        if rank not in SEQUENCE_VALUE:
            return None
        counts[SEQUENCE_VALUE[rank] - 1] += 1
    return _sequence_value_from_counts(counts, len(wildcards), group_size)


def _sequence_value_from_counts(counts: list[int], laizi: int, group_size: int) -> str | None:
    idx_2 = SEQUENCE_VALUE["2"] - 1
    idx_k = SEQUENCE_VALUE["K"] - 1
    idx_a = SEQUENCE_VALUE["A"] - 1
    min_idx = idx_2
    while min_idx <= idx_k and counts[min_idx] == 0:
        min_idx += 1
    if min_idx > idx_k:
        return None
    max_idx = idx_k
    while max_idx > min_idx and counts[max_idx] == 0:
        max_idx -= 1
    for idx in range(min_idx, max_idx + 1):
        short = group_size - counts[idx]
        if short == 0:
            continue
        if short < 0 or laizi < short:
            return None
        laizi -= short
    if counts[idx_a] > 0:
        short = group_size - counts[idx_a]
        if short < 0 or laizi < short:
            return None
        laizi -= short
        if (idx_k - max_idx) * group_size <= laizi:
            return "A"
        short = (min_idx - idx_2) * group_size
        if short > 0:
            if short > laizi:
                return None
            laizi -= short
        if laizi > 0:
            max_idx += laizi // group_size
    elif laizi > 0:
        short = (idx_k - max_idx + 1) * group_size
        if short <= laizi:
            return "A"
        max_idx += laizi // group_size
    return SEQUENCE_RANK.get(max_idx + 1)


def _triple_plus_value(cards: tuple[str, ...], wildcards: list[str] | tuple[str, ...], level: str) -> str | None:
    counts = Counter(card_rank(card) for card in cards)
    for card, count in Counter(wildcards).items():
        rank = card_rank(card)
        counts[rank] -= count
        if counts[rank] <= 0:
            del counts[rank]
    normal = [(rank, count) for rank, count in counts.items() if rank not in {"BJ", "RJ"} and count > 0]
    bj = counts.get("BJ", 0)
    rj = counts.get("RJ", 0)
    if bj + rj > 0:
        if bj > 0 and rj > 0:
            return None
        if bj not in {0, 2} or rj not in {0, 2}:
            return None
        if len(normal) != 1:
            return None
        return normal[0][0]
    if len(normal) != 2 or any(count > 3 for _rank, count in normal):
        return None
    normal.sort(key=lambda item: SEQUENCE_VALUE[item[0]])
    (r1, c1), (r2, c2) = normal
    if c1 == 3:
        return r1
    if c2 == 3:
        return r2
    return r2 if _follow_rank(level, r1, r2) else r1


def _counter_to_key(counter: Counter[str]) -> tuple[int, ...]:
    return tuple(int(counter.get(card, 0)) for card in ALL_CARDS)


def _group_sparse_key(cards: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for card in cards:
        idx = CARD_INDEX[card]
        counts[idx] = counts.get(idx, 0) + 1
    return tuple(sorted(counts.items()))


def _count_requirement_masks(counts: tuple[int, ...]) -> tuple[int, int]:
    require_one = 0
    require_two = 0
    for index, count in enumerate(counts):
        if count >= 1:
            require_one |= 1 << index
        if count >= 2:
            require_two |= 1 << index
    return require_one, require_two


def _sparse_requirement_masks(counts: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    require_one = 0
    require_two = 0
    for index, count in counts:
        require_one |= 1 << index
        if count >= 2:
            require_two |= 1 << index
    return require_one, require_two


def _group_local_sparse_key(
    cards: tuple[str, ...],
    local_index: dict[str, int],
) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for card in cards:
        idx = local_index[card]
        counts[idx] = counts.get(idx, 0) + 1
    return tuple(sorted(counts.items()))


def _subtract_sparse(state: tuple[int, ...], group: tuple[tuple[int, int], ...]) -> tuple[int, ...] | None:
    out = list(state)
    for idx, count in group:
        if count > out[idx]:
            return None
        out[idx] -= count
    return tuple(out)


def _subtract_local_sparse(state: tuple[int, ...], group: tuple[tuple[int, int], ...]) -> tuple[int, ...] | None:
    out = list(state)
    for idx, count in group:
        if count > out[idx]:
            return None
        out[idx] -= count
    return tuple(out)


def _rank_cards(cards: list[str], rank: str) -> list[str]:
    return sorted([card for card in cards if card_rank(card) == rank])


class FullSearchPartitioner:
    """Exact-cover partition search over generated Guandan card groups.

    The search considers all generated non-overlapping covers of the hand. It is
    efficient because recursion always branches on the first remaining physical
    card and memoizes the remaining-card state.
    """

    def __init__(
        self,
        exhaustive_top_threshold: int = 18,
        beam_width: int = 96,
        cache_size: int = 4096,
        use_native: bool = True,
        use_native_all: bool = False,
    ) -> None:
        self.exhaustive_top_threshold = exhaustive_top_threshold
        self.beam_width = beam_width
        self.cache_size = cache_size
        self.use_native = use_native
        self.use_native_all = use_native_all
        self._priority_profile_name = "default"
        self._priority_base_by_kind = GROUP_PRIORITY_BASE
        self._priority_bomb_size_bonus = 14.0
        self._group_cache: OrderedDict[tuple, tuple[CardGroup, ...]] = OrderedDict()
        self._all_cache: OrderedDict[tuple, tuple[Partition, ...]] = OrderedDict()
        self._top_cache: OrderedDict[tuple, tuple[Partition, ...]] = OrderedDict()
        self._minimum_effective_group_cache: OrderedDict[
            tuple, tuple[tuple[str, str | None, tuple[str, ...]], ...]
        ] = OrderedDict()
        self._native_score_input_cache: OrderedDict[tuple, tuple[tuple[CardGroup, ...], tuple[int, ...], list[list[list[int]]], str]] = OrderedDict()
        self._native_group_signature_cache: OrderedDict[
            tuple[tuple[str, ...], str | None],
            tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...],
        ] = OrderedDict()
        self._native_group_superset_cache: OrderedDict[
            tuple,
            tuple[CardGroup, ...],
        ] = OrderedDict()
        self._native_group_exact_cache: OrderedDict[
            tuple,
            tuple[CardGroup, ...],
        ] = OrderedDict()
        self._cover_input_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._bounded_partition_cache: OrderedDict[
            tuple, tuple[Partition, ...] | NativePartitionCovers
        ] = OrderedDict()
        self._bounded_partition_cache_gets = 0
        self._bounded_partition_cache_hits = 0

    def set_group_priority_profile(
        self,
        name: str,
        base_by_kind,
        bomb_size_bonus: float,
    ) -> None:
        """Keep bounded/beam partition ordering consistent with final scoring."""

        if (
            self._priority_profile_name == name
            and self._priority_base_by_kind is base_by_kind
            and self._priority_bomb_size_bonus == bomb_size_bonus
        ):
            return
        self._priority_profile_name = name
        self._priority_base_by_kind = base_by_kind
        self._priority_bomb_size_bonus = float(bomb_size_bonus)
        self._group_cache.clear()
        self._all_cache.clear()
        self._top_cache.clear()
        self._native_score_input_cache.clear()
        self._native_group_superset_cache.clear()
        self._native_group_exact_cache.clear()
        self._cover_input_cache.clear()

    def _group_priority_for_profile(self, group: CardGroup) -> float:
        base = self._priority_base_by_kind.get(group.kind, 0.0)
        if group.kind == "Bomb":
            base += (group.size - 4) * self._priority_bomb_size_bonus
        return base + group.strength * 12.0

    def _effective_group_score(self, group: CardGroup) -> float:
        return self._group_priority_for_profile(group) - 10.0 * effective_group_cost(group)

    def _native_penalty_adjusted_priority(self, group: CardGroup) -> float:
        """Compensate kernels that still subtract one fixed 10-point step."""

        return self._group_priority_for_profile(group) + 10.0 * (1 - effective_group_cost(group))

    def generate(self, hand_cards: Iterable[str], ctx: RetrievalContext, max_partitions: int | None = 8) -> list[Partition]:
        """Return partitions.

        `max_partitions=None` means return every exact-cover partition. A
        positive value returns the best prefix by the internal cheap ordering,
        which keeps existing ranker calls bounded.
        """

        if max_partitions is None:
            return self.generate_all(hand_cards, ctx)
        return self.generate_top(hand_cards, ctx, max_partitions)

    def generate_all(self, hand_cards: Iterable[str], ctx: RetrievalContext) -> list[Partition]:
        partition_key = self._partition_cache_key(hand_cards, ctx)
        key = ("all", partition_key)
        cached = self._cache_get(self._all_cache, key)
        if cached is not None:
            return list(cached)
        partitions = tuple(self.iter_partitions(hand_cards, ctx, partition_key))
        self._cache_put(self._all_cache, key, partitions)
        return list(partitions)

    def iter_partitions(self, hand_cards: Iterable[str], ctx: RetrievalContext, partition_key: tuple | None = None) -> Iterator[Partition]:
        cards = normalize_cards(hand_cards)
        if not cards:
            yield Partition(groups=(), mode="full_search_all")
            return

        group_entries, start, groups_by_first, native_buckets = self._cover_search_inputs(cards, ctx, partition_key)

        if self.use_native_all and native_cover.available():
            try:
                native_covers = native_cover.enumerate_covers(start, native_buckets)
                if not native_covers:
                    yield self._singles_partition(cards, ctx)
                    return
                groups_only = [entry[0] for entry in group_entries]
                for cover in native_covers:
                    yield Partition(tuple([groups_only[group_id] for group_id in cover]), "full_search_native")
                return
            except Exception:
                # Native code is an acceleration path only. Falling back preserves
                # exact Python semantics if the extension is absent or incompatible.
                pass

        @lru_cache(maxsize=None)
        def suffixes(state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
            if not any(state):
                return ((),)
            first = next(idx for idx, count in enumerate(state) if count)
            results: list[tuple[int, ...]] = []
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is None:
                    continue
                for tail in suffixes(next_state):
                    results.append((group_id,) + tail)
            return tuple(results)

        all_suffixes = suffixes(start)
        if not all_suffixes:
            yield self._singles_partition(cards, ctx)
            return
        groups_only = [entry[0] for entry in group_entries]
        for cover in all_suffixes:
            yield Partition(tuple([groups_only[group_id] for group_id in cover]), "full_search_all")

    def iter_group_covers(
        self,
        hand_cards: Iterable[str],
        ctx: RetrievalContext,
        partition_key: tuple | None = None,
    ) -> Iterator[tuple[tuple[CardGroup, ...], str]]:
        """Yield exact-cover group tuples without constructing every Partition.

        Full-search scoring only needs a Partition object for the final best
        cover. This preserves the same cover order as `iter_partitions()` while
        avoiding per-cover Partition allocation on the native all-covers path.
        """

        cards = normalize_cards(hand_cards)
        if not cards:
            yield (), "full_search_all"
            return

        if self.use_native_all and native_cover.available():
            try:
                groups_only, start, native_buckets, mode = self._native_score_inputs_cached(cards, ctx, partition_key)
                native_covers = native_cover.enumerate_covers(start, native_buckets)
                if not native_covers:
                    yield self._singles_partition(cards, ctx).groups, mode
                    return
                for cover in native_covers:
                    yield tuple([groups_only[group_id] for group_id in cover]), mode
                return
            except Exception:
                pass

        for partition in self.iter_partitions(cards, ctx, partition_key):
            yield partition.groups, partition.mode

    def native_cover_ids(
        self,
        hand_cards: Iterable[str],
        ctx: RetrievalContext,
        partition_key: tuple | None = None,
    ) -> tuple[list[CardGroup], list[list[int]], str] | None:
        if not (self.use_native_all and native_cover.available()):
            return None
        cards = list(hand_cards) if partition_key is not None else normalize_cards(hand_cards)
        if not cards:
            return [], [[]], "full_search_all"
        try:
            groups_only, start, native_buckets, mode = self._native_score_inputs_cached(cards, ctx, partition_key)
            native_covers = native_cover.enumerate_covers(start, native_buckets)
        except Exception:
            return None
        if not native_covers:
            partition = self._singles_partition(cards, ctx)
            return list(partition.groups), [list(range(len(partition.groups)))], partition.mode
        return list(groups_only), native_covers, mode

    def native_score_inputs(
        self,
        hand_cards: Iterable[str],
        ctx: RetrievalContext,
        partition_key: tuple | None = None,
    ) -> tuple[tuple[CardGroup, ...], tuple[int, ...], list[list[list[int]]], str] | None:
        if not (self.use_native_all and native_cover.available()):
            return None
        cards = list(hand_cards) if partition_key is not None else normalize_cards(hand_cards)
        if not cards:
            return (), (), [], "full_search_all"
        try:
            groups_only, start, native_buckets, mode = self._native_score_inputs_cached(cards, ctx, partition_key)
        except Exception:
            return None
        return groups_only, start, native_buckets, mode

    def _native_score_inputs_cached(
        self,
        cards: list[str],
        ctx: RetrievalContext,
        partition_key: tuple | None = None,
    ) -> tuple[tuple[CardGroup, ...], tuple[int, ...], list[list[list[int]]], str]:
        key = ("native_score_inputs", partition_key or self._partition_cache_key(cards, ctx))
        cached = self._cache_get(self._native_score_input_cache, key)
        if cached is not None:
            return cached
        group_entries, start, _groups_by_first, native_buckets = self._cover_search_inputs(
            cards,
            ctx,
            partition_key,
            include_python_buckets=False,
        )
        out = (tuple(entry[0] for entry in group_entries), start, native_buckets, "full_search_native")
        self._cache_put(self._native_score_input_cache, key, out)
        return out

    def iter_partitions_streaming(self, hand_cards: Iterable[str], ctx: RetrievalContext, partition_key: tuple | None = None) -> Iterator[Partition]:
        """Yield every exact-cover partition without materializing all covers first.

        This is intended for full-search scoring where the caller only needs the
        current best partition. It preserves the same cover space as
        `iter_partitions()` but avoids building a tuple/list containing every
        cover before scoring can begin.
        """

        cards = normalize_cards(hand_cards)
        if not cards:
            yield Partition(groups=(), mode="full_search_stream")
            return

        group_entries, start, groups_by_first, _native_buckets = self._cover_search_inputs(cards, ctx, partition_key)
        groups_only = [entry[0] for entry in group_entries]
        emitted = False

        def dfs(state: tuple[int, ...], cover: list[int]) -> Iterator[Partition]:
            nonlocal emitted
            if not any(state):
                emitted = True
                yield Partition(tuple([groups_only[group_id] for group_id in cover]), "full_search_stream")
                return
            first = next(idx for idx, count in enumerate(state) if count)
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is None:
                    continue
                cover.append(group_id)
                yield from dfs(next_state, cover)
                cover.pop()

        yield from dfs(start, [])
        if not emitted:
            yield self._singles_partition(cards, ctx)

    def _cover_search_inputs(
        self,
        cards: list[str],
        ctx: RetrievalContext,
        partition_key: tuple | None = None,
        include_priority: bool = False,
        include_python_buckets: bool = True,
        precomputed_groups: Iterable[CardGroup] | None = None,
    ) -> tuple[
        list[tuple[CardGroup, tuple[tuple[int, int], ...], float]],
        tuple[int, ...],
        dict[int, list[int]],
        list[list[list[int]]],
    ]:
        if (
            precomputed_groups is None
            and not include_python_buckets
            and self.use_native
            and os.environ.get("DANRL_USE_NATIVE_RECORD_COVER_INPUTS", "").strip().lower() in {"1", "true", "yes", "on"}
            and native_actor_core.available()
            and hasattr(native_actor_core.module(), "build_group_records_and_cover_inputs")
        ):
            return self._cover_search_inputs_native_records(cards, ctx, include_priority=include_priority)

        cover_cache_size = self._env_int(
            "DANRL_COVER_INPUT_CACHE_SIZE", 0, minimum=0,
        )
        cover_cache_key = None
        if cover_cache_size > 0:
            priority_profile = tuple(sorted(
                (str(kind), float(value))
                for kind, value in self._priority_base_by_kind.items()
            ))
            cover_cache_key = (
                tuple(cards),
                ctx.cur_rank,
                int(bool(ctx.remaining_detail)),
                int(bool(ctx.remaining_detail) and ctx.remaining_detail.get("RJ", 0) == 0),
                self._priority_profile_name,
                priority_profile,
                self._priority_bomb_size_bonus,
                int(include_priority),
                int(include_python_buckets),
            )
            cached_inputs = self._cover_input_cache.get(cover_cache_key)
            if cached_inputs is not None:
                self._cover_input_cache.move_to_end(cover_cache_key)
                return cached_inputs

        groups = (
            list(precomputed_groups)
            if precomputed_groups is not None
            else self._generate_groups(cards, ctx, partition_key)
        )
        native_cover_input_encoding = (
            not include_python_buckets
            and os.environ.get(
                "DANRL_NATIVE_COVER_INPUT_ENCODING",
                "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
            and native_actor_core.available()
            and hasattr(native_actor_core.module(), "build_cover_inputs")
        )
        if native_cover_input_encoding:
            start, native_buckets = native_actor_core.build_cover_inputs(
                cards,
                [group.cards for group in groups],
            )
            group_entries = [
                (
                    group,
                    (),
                    self._group_priority_for_profile(group) if include_priority else 0.0,
                )
                for group in groups
            ]
            out = (group_entries, start, {}, native_buckets)
            if cover_cache_key is not None:
                self._cover_input_cache[cover_cache_key] = out
                self._cover_input_cache.move_to_end(cover_cache_key)
                while len(self._cover_input_cache) > cover_cache_size:
                    self._cover_input_cache.popitem(last=False)
            return out

        hand_counter = Counter(cards)
        sparse_cover_inputs = os.environ.get(
            "DANRL_SPARSE_COVER_INPUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        local_cards = sorted(hand_counter, key=card_sort_key)
        local_index = {card: idx for idx, card in enumerate(local_cards)}
        global_to_local = None
        if sparse_cover_inputs:
            global_to_local = [-1] * len(ALL_CARDS)
            for card, index in local_index.items():
                global_to_local[CARD_INDEX[card]] = index
        start = tuple(int(hand_counter[card]) for card in local_cards)

        group_entries: list[tuple[CardGroup, tuple[tuple[int, int], ...], float]] = []
        groups_by_first: dict[int, list[int]] = {idx: [] for idx in range(len(local_cards))} if include_python_buckets else {}
        native_buckets: list[list[list[int]]] = [[] for _ in range(len(local_cards))]
        for group in groups:
            physical_counts = group.meta.get("__physical_counts") if sparse_cover_inputs else None
            if physical_counts is not None and global_to_local is not None:
                key = tuple(sorted(
                    (global_to_local[global_index], count)
                    for global_index, count in physical_counts
                ))
            else:
                key_items: list[tuple[int, int]] = []
                last_idx = -1
                for card in group.cards:
                    idx = local_index[card]
                    if idx == last_idx:
                        prev_idx, prev_count = key_items[-1]
                        key_items[-1] = (prev_idx, prev_count + 1)
                    else:
                        key_items.append((idx, 1))
                        last_idx = idx
                key = tuple(key_items)
            group_id = len(group_entries)
            group_entries.append((group, key, self._group_priority_for_profile(group) if include_priority else 0.0))
            first = key[0][0]
            if include_python_buckets:
                groups_by_first[first].append(group_id)
            encoded = [group_id]
            for idx, count in key:
                encoded.append(idx)
                encoded.append(count)
            native_buckets[first].append(encoded)

        out = (group_entries, start, groups_by_first, native_buckets)
        if cover_cache_key is not None:
            self._cover_input_cache[cover_cache_key] = out
            self._cover_input_cache.move_to_end(cover_cache_key)
            while len(self._cover_input_cache) > cover_cache_size:
                self._cover_input_cache.popitem(last=False)
        return out

    def _cover_search_inputs_native_records(
        self,
        cards: list[str],
        ctx: RetrievalContext,
        include_priority: bool = False,
    ) -> tuple[
        list[tuple[CardGroup, tuple[tuple[int, int], ...], float]],
        tuple[int, ...],
        dict[int, list[int]],
        list[list[list[int]]],
    ]:
        records, start, _native_buckets = native_actor_core.build_group_records_and_cover_inputs(cards, ctx.cur_rank)
        groups: dict[tuple[str, tuple[str, ...], str | None, tuple], CardGroup] = {}
        group_entries: list[tuple[CardGroup, tuple[tuple[int, int], ...], float]] = []
        strength_cache: dict[str | None, float] = {None: 0.0}
        for kind, rank, selected, wild_as, key_items in records:
            cards_key = sorted_cards(selected)
            meta = self._wild_meta(wild_as)
            wild_key = tuple(meta.get("wild_as", ()))
            key = (kind, cards_key, rank, wild_key)
            group = groups.get(key)
            if group is None:
                strength = strength_cache.get(rank)
                if strength is None:
                    strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail)
                    strength_cache[rank] = strength
                meta["__score_key"] = (kind, rank, len(cards_key), strength)
                group = CardGroup(kind=kind, cards=cards_key, rank=rank, strength=strength, meta=meta)
                groups[key] = group
            group_entries.append((group, key_items, self._group_priority_for_profile(group) if include_priority else 0.0))

        group_entries.sort(key=lambda item: (-self._group_priority_for_profile(item[0]), item[0].kind, item[0].cards))
        native_buckets: list[list[list[int]]] = [[] for _ in range(len(start))]
        for group_id, (_group, key_items, _score) in enumerate(group_entries):
            encoded = [group_id]
            for idx, count in key_items:
                encoded.append(idx)
                encoded.append(count)
            native_buckets[key_items[0][0]].append(encoded)
        return group_entries, start, {}, native_buckets

    def generate_top(self, hand_cards: Iterable[str], ctx: RetrievalContext, max_partitions: int = 8) -> list[Partition]:
        cards = normalize_cards(hand_cards)
        hand_count_window = self._env_int("DANRL_PARTITION_HAND_COUNT_WINDOW", -1)
        hand_count_window_min_hand = self._env_int("DANRL_PARTITION_HAND_COUNT_WINDOW_MIN_HAND", 0)
        effective_hand_count_window = hand_count_window if len(cards) >= hand_count_window_min_hand else -1
        key = (
            "top",
            "effective_hand_count_v1",
            max_partitions,
            self.exhaustive_top_threshold,
            self.beam_width,
            effective_hand_count_window,
            hand_count_window_min_hand,
            self._env_int("DANRL_PARTITION_HAND_COUNT_MAX_COVERS", 512),
            self._partition_cache_key(cards, ctx),
        )
        cached = self._cache_get(self._top_cache, key)
        if cached is not None:
            return list(cached)
        if effective_hand_count_window >= 0:
            bounded = self._generate_top_hand_count_window(cards, ctx, max_partitions, effective_hand_count_window)
            if bounded is not None:
                partitions = tuple(bounded)
                self._cache_put(self._top_cache, key, partitions)
                return list(partitions)
        if len(cards) > self.exhaustive_top_threshold:
            partitions = tuple(self.generate_beam_top(cards, ctx, max_partitions=max_partitions, beam_width=self.beam_width))
            self._cache_put(self._top_cache, key, partitions)
            return list(partitions)

        native_partitions = self._generate_top_native(cards, ctx, max_partitions)
        if native_partitions is not None:
            partitions = tuple(native_partitions)
            self._cache_put(self._top_cache, key, partitions)
            return list(partitions)

        scored = [
            (self._cheap_partition_score(partition), partition)
            for partition in self.iter_partitions(cards, ctx)
        ]
        scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
        partitions = tuple(partition for _, partition in scored[:max_partitions])
        self._cache_put(self._top_cache, key, partitions)
        return list(partitions)

    def _generate_top_hand_count_window(
        self,
        cards: list[str],
        ctx: RetrievalContext,
        max_partitions: int,
        hand_count_window: int,
    ) -> list[Partition] | None:
        """Return top covers from the minimum effective-hand-count layer plus a window.

        This keeps the exact-cover invariant for every returned partition, but
        avoids expanding cover layers whose effective hand count is already
        worse than the best available value. It is intentionally optional so it can be
        A/B tested against the older full/top-cover paths.
        """

        if not cards:
            return [Partition(groups=(), mode="hand_count_window")]
        if self.use_native and native_cover.available() and hasattr(native_cover.module(), "top_covers_effective_hand_count_window"):
            try:
                group_entries, start, _groups_by_first, native_buckets = self._cover_search_inputs(
                    cards,
                    ctx,
                    include_priority=True,
                    include_python_buckets=False,
                )
                groups_only = [entry[0] for entry in group_entries]
                group_scores = [self._effective_group_score(group) for group in groups_only]
                group_selection_priorities = [
                    self._native_penalty_adjusted_priority(group) for group in groups_only
                ]
                group_costs = [effective_group_cost(group) for group in groups_only]
                tie_keys = [self._native_tie_key(group) for group in groups_only]
                max_covers = self._env_int("DANRL_PARTITION_HAND_COUNT_MAX_COVERS", 512, minimum=max_partitions)
                native_select = os.environ.get("DANRL_NATIVE_SELECT_TOP_COVERS", "1").strip().lower() in {
                    "1", "true", "yes", "on",
                } and hasattr(native_cover.module(), "top_covers_effective_hand_count_window_selected")
                if native_select:
                    covers = native_cover.top_covers_effective_hand_count_window_selected(
                        start,
                        native_buckets,
                        group_scores,
                        group_selection_priorities,
                        tie_keys,
                        group_costs,
                        hand_count_window,
                        max_covers,
                        max_partitions,
                    )
                else:
                    covers = native_cover.top_covers_effective_hand_count_window(
                        start,
                        native_buckets,
                        group_scores,
                        tie_keys,
                        group_costs,
                        hand_count_window,
                        max_covers,
                    )
                if not covers:
                    return [self._singles_partition(cards, ctx)]
                if native_select:
                    return [
                        Partition(
                            tuple(groups_only[group_id] for group_id in cover),
                            "hand_count_window_native",
                        )
                        for cover in covers
                    ]
                id_postprocess = os.environ.get("DANRL_NATIVE_ID_POSTPROCESS", "1").strip().lower() in {
                    "1", "true", "yes", "on",
                }
                if id_postprocess:
                    group_sort_keys = [(group.kind, group.cards) for group in groups_only]
                    scored_covers = [
                        (sum(group_scores[group_id] for group_id in cover), cover)
                        for cover in covers
                    ]
                    scored_covers.sort(
                        key=lambda item: (-item[0], tuple(group_sort_keys[group_id] for group_id in item[1]))
                    )
                    return [
                        Partition(
                            tuple(groups_only[group_id] for group_id in cover),
                            "hand_count_window_native",
                        )
                        for _score, cover in scored_covers[:max_partitions]
                    ]
                partitions = [
                    Partition(tuple(groups_only[group_id] for group_id in cover), "hand_count_window_native")
                    for cover in covers
                ]
                scored = [(self._cheap_partition_score(partition), partition) for partition in partitions]
                scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
                return [partition for _score, partition in scored[:max_partitions]]
            except Exception:
                pass
        try:
            group_entries, start, groups_by_first, _native_buckets = self._cover_search_inputs(
                cards,
                ctx,
                include_priority=True,
                include_python_buckets=True,
            )
        except Exception:
            return None
        if not group_entries:
            return [self._singles_partition(cards, ctx)]

        groups_only = [entry[0] for entry in group_entries]
        group_scores = [self._effective_group_score(group) for group in groups_only]
        group_costs = [effective_group_cost(group) for group in groups_only]
        max_covers = self._env_int("DANRL_PARTITION_HAND_COUNT_MAX_COVERS", 512, minimum=max_partitions)

        for bucket in groups_by_first.values():
            bucket.sort(key=lambda group_id: (-group_scores[group_id], groups_only[group_id].kind, groups_only[group_id].cards))

        covers: list[tuple[float, tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()

        @lru_cache(maxsize=None)
        def min_effective_cost(state: tuple[int, ...]) -> int:
            if not any(state):
                return 0
            first = next(idx for idx, count in enumerate(state) if count)
            best = len(cards) + 1
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is None:
                    continue
                best = min(best, group_costs[group_id] + min_effective_cost(next_state))
            return best

        minimum_effective_cost = min_effective_cost(start)
        maximum_effective_cost = minimum_effective_cost + max(0, hand_count_window)

        def dfs(
            state: tuple[int, ...],
            chosen: list[int],
            score: float,
            effective_cost: int,
        ) -> None:
            if len(covers) >= max_covers:
                return
            if not any(state):
                if effective_cost <= maximum_effective_cost:
                    key = tuple(chosen)
                    if key not in seen:
                        seen.add(key)
                        covers.append((score, key))
                return
            first = next(idx for idx, count in enumerate(state) if count)
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is None:
                    continue
                next_cost = effective_cost + group_costs[group_id]
                if next_cost + min_effective_cost(next_state) > maximum_effective_cost:
                    continue
                chosen.append(group_id)
                dfs(next_state, chosen, score + group_scores[group_id], next_cost)
                chosen.pop()
                if len(covers) >= max_covers:
                    return

        dfs(start, [], 0.0, 0)

        if not covers:
            return [self._singles_partition(cards, ctx)]

        partitions = [
            Partition(tuple(groups_only[group_id] for group_id in cover), "hand_count_window")
            for _score, cover in covers
        ]
        scored = [(self._cheap_partition_score(partition), partition) for partition in partitions]
        scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
        return [partition for _score, partition in scored[:max_partitions]]

    def generate_top_hand_count_window_batch(
        self,
        super_hand_cards: Iterable[str],
        after_hands: Iterable[Iterable[str]],
        ctx: RetrievalContext,
        max_partitions: int,
        prepared_inputs=None,
    ) -> dict[tuple[str, ...], list[Partition] | NativePartitionCovers]:
        """Solve many sub-hands against one shared group universe."""

        window = self._env_int("DANRL_PARTITION_HAND_COUNT_WINDOW", -1)
        min_hand = self._env_int("DANRL_PARTITION_HAND_COUNT_WINDOW_MIN_HAND", 0)
        module = native_cover.module() if native_cover.available() else None
        if window < 0 or module is None or not hasattr(module, "top_covers_effective_hand_count_window_batch"):
            return {}
        super_cards = normalize_cards(super_hand_cards)
        keys = list(dict.fromkeys(tuple(normalize_cards(cards)) for cards in after_hands))
        eligible = [key for key in keys if len(key) >= min_hand]
        if not eligible:
            return {}
        max_covers = self._env_int(
            "DANRL_PARTITION_HAND_COUNT_MAX_COVERS", 512, minimum=max_partitions
        )
        cache_common = self._bounded_partition_cache_common_key(
            "effective_hand_count_window_v1", super_cards, ctx, max_partitions, window, max_covers
        )
        out, eligible = self._bounded_partition_cache_split(cache_common, eligible)
        if not eligible:
            return out
        try:
            if prepared_inputs is None:
                group_entries, _start, _groups_by_first, native_buckets = self._cover_search_inputs(
                    super_cards,
                    ctx,
                    include_priority=True,
                    include_python_buckets=False,
                )
            else:
                group_entries, _start, _groups_by_first, native_buckets = prepared_inputs
            groups_only = [entry[0] for entry in group_entries]
            group_scores = [self._effective_group_score(group) for group in groups_only]
            group_selection_priorities = [
                self._native_penalty_adjusted_priority(group) for group in groups_only
            ]
            group_costs = [effective_group_cost(group) for group in groups_only]
            tie_keys = [self._native_tie_key(group) for group in groups_only]
            local_cards = sorted(Counter(super_cards), key=card_sort_key)
            states = []
            for key in eligible:
                counts = Counter(key)
                states.append([int(counts[card]) for card in local_cards])
            native_select = os.environ.get("DANRL_NATIVE_SELECT_TOP_COVERS", "1").strip().lower() in {
                "1", "true", "yes", "on",
            } and hasattr(module, "top_covers_effective_hand_count_window_selected_batch")
            if native_select:
                covers_by_hand = native_cover.top_covers_effective_hand_count_window_selected_batch(
                    states,
                    native_buckets,
                    group_scores,
                    group_selection_priorities,
                    tie_keys,
                    group_costs,
                    window,
                    max_covers,
                    max_partitions,
                )
            else:
                covers_by_hand = native_cover.top_covers_effective_hand_count_window_batch(
                    states,
                    native_buckets,
                    group_scores,
                    tie_keys,
                    group_costs,
                    window,
                    max_covers,
                )
        except Exception:
            return {}

        id_postprocess = os.environ.get("DANRL_NATIVE_ID_POSTPROCESS", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        defer_partitions = self._defer_selected_partitions()
        shared_groups = tuple(groups_only) if defer_partitions and native_select else ()
        group_sort_keys = [(group.kind, group.cards) for group in groups_only] if id_postprocess else []
        for key, covers in zip(eligible, covers_by_hand):
            if not covers:
                out[key] = [self._singles_partition(list(key), ctx)]
                continue
            if native_select:
                if defer_partitions:
                    out[key] = NativePartitionCovers(
                        shared_groups,
                        tuple(tuple(int(group_id) for group_id in cover) for cover in covers),
                        "hand_count_window_native",
                    )
                else:
                    out[key] = [
                        Partition(
                            tuple(groups_only[group_id] for group_id in cover),
                            "hand_count_window_native",
                        )
                        for cover in covers
                    ]
                continue
            if id_postprocess:
                scored_covers = [
                    (sum(group_scores[group_id] for group_id in cover), cover)
                    for cover in covers
                ]
                scored_covers.sort(
                    key=lambda item: (-item[0], tuple(group_sort_keys[group_id] for group_id in item[1]))
                )
                out[key] = [
                    Partition(
                        tuple(groups_only[group_id] for group_id in cover),
                        "hand_count_window_native",
                    )
                    for _score, cover in scored_covers[:max_partitions]
                ]
                continue
            partitions = [
                Partition(tuple(groups_only[group_id] for group_id in cover), "hand_count_window_native")
                for cover in covers
            ]
            scored = [(self._cheap_partition_score(partition), partition) for partition in partitions]
            scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
            out[key] = [partition for _score, partition in scored[:max_partitions]]
        self._bounded_partition_cache_store(cache_common, out, eligible)
        return out

    def minimum_effective_groups(
        self,
        hand_cards: Iterable[str],
        ctx: RetrievalContext,
        groups=None,
    ) -> list[CardGroup]:
        """Return every group appearing in at least one global optimum."""

        cards = normalize_cards(hand_cards)
        current_groups = (
            list(groups) if groups is not None else self._generate_groups(cards, ctx)
        )
        cache_key = (
            "minimum_effective_groups_v1",
            tuple(sorted_cards(cards)),
            ctx.cur_rank,
        )
        cached = self._cache_get(self._minimum_effective_group_cache, cache_key)
        if cached is not None:
            selected_keys = set(cached)
            return [
                group for group in current_groups
                if (group.kind, group.rank, group.cards) in selected_keys
            ]
        module = native_cover.module() if native_cover.available() else None
        if module is not None and hasattr(module, "optimal_effective_group_ids"):
            try:
                group_entries, start, _groups_by_first, native_buckets = (
                    self._cover_search_inputs(
                        cards,
                        ctx,
                        include_priority=False,
                        include_python_buckets=False,
                        precomputed_groups=current_groups,
                    )
                )
                groups_only = [entry[0] for entry in group_entries]
                group_costs = [effective_group_cost(group) for group in groups_only]
                group_ids = module.optimal_effective_group_ids(
                    start, native_buckets, group_costs,
                )
                result = tuple(
                    (
                        groups_only[int(group_id)].kind,
                        groups_only[int(group_id)].rank,
                        groups_only[int(group_id)].cards,
                    )
                    for group_id in group_ids
                )
                self._cache_put(
                    self._minimum_effective_group_cache, cache_key, result,
                )
                selected_keys = set(result)
                return [
                    group for group in current_groups
                    if (group.kind, group.rank, group.cards) in selected_keys
                ]
            except Exception:
                pass

        group_entries, start, groups_by_first, _native_buckets = (
            self._cover_search_inputs(
                cards,
                ctx,
                include_priority=False,
                include_python_buckets=True,
                precomputed_groups=current_groups,
            )
        )
        group_costs = [effective_group_cost(entry[0]) for entry in group_entries]

        @lru_cache(maxsize=None)
        def minimum_cost(state: tuple[int, ...]) -> int:
            if not any(state):
                return 0
            first = next(index for index, count in enumerate(state) if count)
            best = len(cards) + 1
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is not None:
                    best = min(
                        best,
                        group_costs[group_id] + minimum_cost(next_state),
                    )
            return best

        optimal_group_ids: set[int] = set()
        visited: set[tuple[int, ...]] = set()

        def visit(state: tuple[int, ...]) -> None:
            if not any(state) or state in visited:
                return
            visited.add(state)
            optimum = minimum_cost(state)
            first = next(index for index, count in enumerate(state) if count)
            for group_id in groups_by_first.get(first, []):
                _group, group_state, _score = group_entries[group_id]
                next_state = _subtract_local_sparse(state, group_state)
                if next_state is None:
                    continue
                if group_costs[group_id] + minimum_cost(next_state) == optimum:
                    optimal_group_ids.add(group_id)
                    visit(next_state)

        visit(start)
        result = tuple(
            (
                group_entries[group_id][0].kind,
                group_entries[group_id][0].rank,
                group_entries[group_id][0].cards,
            )
            for group_id in sorted(optimal_group_ids)
        )
        self._cache_put(self._minimum_effective_group_cache, cache_key, result)
        selected_keys = set(result)
        return [
            group for group in current_groups
            if (group.kind, group.rank, group.cards) in selected_keys
        ]

    def generate_top_beam_batch(
        self,
        super_hand_cards: Iterable[str],
        after_hands: Iterable[Iterable[str]],
        ctx: RetrievalContext,
        max_partitions: int,
        prepared_inputs=None,
        beam_width: int | None = None,
    ) -> dict[tuple[str, ...], list[Partition] | NativePartitionCovers]:
        """Run the existing bounded beam ordering for many sub-hands in C++."""

        module = native_cover.module() if native_cover.available() else None
        if module is None or not hasattr(module, "top_covers_beam_batch"):
            return {}
        keys = list(dict.fromkeys(tuple(normalize_cards(cards)) for cards in after_hands))
        if not keys:
            return {}
        effective_beam_width = self.beam_width if beam_width is None else max(1, int(beam_width))
        cache_common = self._bounded_partition_cache_common_key(
            "beam_effective_hand_count_v1",
            normalize_cards(super_hand_cards),
            ctx,
            max_partitions,
            effective_beam_width,
        )
        out, keys = self._bounded_partition_cache_split(cache_common, keys)
        if not keys:
            return out
        super_batch = (
            prepared_inputs is not None
            and os.environ.get(
                "DANRL_NATIVE_BEAM_SUPER_BATCH", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        if super_batch:
            try:
                group_entries, _start, _groups_by_first, _prepared_native_buckets = prepared_inputs
                groups_only = [entry[0] for entry in group_entries]
                group_scores = [
                    self._native_penalty_adjusted_priority(entry[0]) for entry in group_entries
                ]
                tie_keys = [self._native_tie_key(group) for group in groups_only]
                local_cards = sorted(Counter(super_hand_cards), key=CARD_INDEX.__getitem__)
                group_sizes = [group.size for group in groups_only]
                cpp_buckets = os.environ.get(
                    "DANRL_NATIVE_BEAM_CPP_BUCKETS", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if cpp_buckets:
                    _beam_start, native_buckets = native_actor_core.build_cover_inputs_beam_order(
                        super_hand_cards,
                        [group.cards for group in groups_only],
                    )
                else:
                    local_index = {card: idx for idx, card in enumerate(local_cards)}
                    native_buckets = [[] for _ in local_cards]
                    bucket_ids: dict[int, list[int]] = {}
                    for group_id, group in enumerate(groups_only):
                        key_items = _group_local_sparse_key(group.cards, local_index)
                        encoded = [group_id]
                        for idx, count in key_items:
                            encoded.extend((idx, count))
                        bucket_ids.setdefault(key_items[0][0], []).append(group_id)
                        native_buckets[key_items[0][0]].append(encoded)
                    for first, ids in bucket_ids.items():
                        order = sorted(
                            range(len(ids)),
                            key=lambda pos: (
                                -group_scores[ids[pos]],
                                groups_only[ids[pos]].kind,
                                groups_only[ids[pos]].cards,
                            ),
                        )
                        native_buckets[first] = [native_buckets[first][pos] for pos in order]
                states = []
                for key in keys:
                    counts = Counter(key)
                    states.append([int(counts[card]) for card in local_cards])
                covers_by_hand = native_cover.top_covers_beam_batch(
                    states,
                    native_buckets,
                    group_scores,
                    group_sizes,
                    tie_keys,
                    effective_beam_width,
                    max_partitions,
                )
                shared_groups = tuple(groups_only) if self._defer_selected_partitions() else ()
                for key, covers in zip(keys, covers_by_hand):
                    if not covers:
                        out[key] = [self._singles_partition(list(key), ctx)]
                    elif shared_groups:
                        out[key] = NativePartitionCovers(
                            shared_groups,
                            tuple(tuple(int(group_id) for group_id in cover) for cover in covers),
                            "beam_top",
                        )
                    else:
                        out[key] = [
                            Partition(
                                tuple(groups_only[group_id] for group_id in cover),
                                "beam_top",
                            )
                            for cover in covers
                        ]
                self._bounded_partition_cache_store(cache_common, out, keys)
                return out
            except Exception:
                pass
        filter_super_universe = (
            prepared_inputs is not None
            and os.environ.get(
                "DANRL_NATIVE_BEAM_FILTER_SUPER_UNIVERSE", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        super_group_entries = prepared_inputs[0] if filter_super_universe else None
        for key in keys:
            try:
                # Equal-score wildcard interpretations rely on Python's stable
                # per-hand generation order. Keep that exact local universe and
                # move only beam expansion/sorting into native code.
                if super_group_entries is not None:
                    hand_counts_key = _counter_to_key(Counter(key))
                    group_entries = [
                        entry
                        for entry in super_group_entries
                        if all(
                            count <= hand_counts_key[index]
                            for index, count in entry[0].meta["__physical_counts"]
                        )
                    ]
                else:
                    group_entries, _start, _groups_by_first, _native_buckets = self._cover_search_inputs(
                        list(key),
                        ctx,
                        include_priority=True,
                        include_python_buckets=False,
                    )
                groups_only = [entry[0] for entry in group_entries]
                group_scores = [
                    self._native_penalty_adjusted_priority(entry[0]) for entry in group_entries
                ]
                group_sizes = [group.size for group in groups_only]
                tie_keys = [self._native_tie_key(group) for group in groups_only]
                local_cards = sorted(Counter(key), key=CARD_INDEX.__getitem__)
                local_index = {card: idx for idx, card in enumerate(local_cards)}
                native_buckets = [[] for _ in local_cards]
                bucket_ids: dict[int, list[int]] = {}
                for group_id, group in enumerate(groups_only):
                    key_items = _group_local_sparse_key(group.cards, local_index)
                    encoded = [group_id]
                    for idx, count in key_items:
                        encoded.extend((idx, count))
                    bucket_ids.setdefault(key_items[0][0], []).append(group_id)
                    native_buckets[key_items[0][0]].append(encoded)
                for first, ids in bucket_ids.items():
                    order = sorted(
                        range(len(ids)),
                        key=lambda pos: (
                            -group_scores[ids[pos]],
                            groups_only[ids[pos]].kind,
                            groups_only[ids[pos]].cards,
                        ),
                    )
                    native_buckets[first] = [native_buckets[first][pos] for pos in order]
                counts = Counter(key)
                state = [int(counts[card]) for card in local_cards]
                covers = native_cover.top_covers_beam_batch(
                    [state],
                    native_buckets,
                    group_scores,
                    group_sizes,
                    tie_keys,
                    effective_beam_width,
                    max_partitions,
                )[0]
            except Exception:
                continue
            if not covers:
                out[key] = [self._singles_partition(list(key), ctx)]
                continue
            if self._defer_selected_partitions():
                out[key] = NativePartitionCovers(
                    tuple(groups_only),
                    tuple(tuple(int(group_id) for group_id in cover) for cover in covers),
                    "beam_top",
                )
            else:
                out[key] = [
                    Partition(
                        tuple(groups_only[group_id] for group_id in cover),
                        "beam_top",
                    )
                    for cover in covers
                ]
        self._bounded_partition_cache_store(cache_common, out, keys)
        return out

    def generate_top_native_batch(
        self,
        super_hand_cards: Iterable[str],
        after_hands: Iterable[Iterable[str]],
        ctx: RetrievalContext,
        max_partitions: int,
        prepared_inputs=None,
    ) -> dict[tuple[str, ...], list[Partition] | NativePartitionCovers]:
        """Solve bounded top covers for many sub-hands in one native call."""

        module = native_cover.module() if native_cover.available() else None
        if module is None or not hasattr(module, "top_covers_batch"):
            return {}
        super_cards = normalize_cards(super_hand_cards)
        keys = list(dict.fromkeys(tuple(normalize_cards(cards)) for cards in after_hands))
        keys = [key for key in keys if key]
        if not keys:
            return {}
        native_limit = max(max_partitions * 8, max_partitions + 128)
        cache_common = self._bounded_partition_cache_common_key(
            "small_top_effective_hand_count_v1", super_cards, ctx, max_partitions, native_limit
        )
        out, keys = self._bounded_partition_cache_split(cache_common, keys)
        if not keys:
            return out
        try:
            if prepared_inputs is None:
                group_entries, _start, _groups_by_first, native_buckets = self._cover_search_inputs(
                    super_cards,
                    ctx,
                    include_priority=True,
                    include_python_buckets=False,
                )
            else:
                group_entries, _start, _groups_by_first, native_buckets = prepared_inputs
            groups_only = [entry[0] for entry in group_entries]
            group_scores = [self._effective_group_score(entry[0]) for entry in group_entries]
            group_priorities = [self._native_penalty_adjusted_priority(entry[0]) for entry in group_entries]
            tie_keys = [self._native_tie_key(group) for group in groups_only]
            local_cards = sorted(Counter(super_cards), key=card_sort_key)
            states = []
            for key in keys:
                counts = Counter(key)
                states.append([int(counts[card]) for card in local_cards])
            native_select = os.environ.get(
                "DANRL_NATIVE_SELECT_SMALL_TOP_COVERS", "0"
            ).strip().lower() in {"1", "true", "yes", "on"} and hasattr(
                module, "top_covers_selected_batch"
            )
            if native_select:
                covers_by_hand = native_cover.top_covers_selected_batch(
                    states,
                    native_buckets,
                    group_scores,
                    group_priorities,
                    tie_keys,
                    native_limit,
                    max_partitions,
                )
            else:
                covers_by_hand = native_cover.top_covers_batch(
                    states,
                    native_buckets,
                    group_scores,
                    tie_keys,
                    native_limit,
                )
        except Exception:
            return {}

        id_postprocess = os.environ.get("DANRL_NATIVE_ID_POSTPROCESS", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        defer_partitions = self._defer_selected_partitions()
        shared_groups = tuple(groups_only) if defer_partitions and native_select else ()
        group_sort_keys = [(group.kind, group.cards) for group in groups_only] if id_postprocess else []
        for key, covers in zip(keys, covers_by_hand):
            if not covers:
                out[key] = [self._singles_partition(list(key), ctx)]
                continue
            if native_select:
                if defer_partitions:
                    out[key] = NativePartitionCovers(
                        shared_groups,
                        tuple(tuple(int(group_id) for group_id in cover) for cover in covers),
                        "full_search_native_top",
                    )
                else:
                    out[key] = [
                        Partition(
                            tuple(groups_only[group_id] for group_id in cover),
                            "full_search_native_top",
                        )
                        for cover in covers
                    ]
                continue
            if id_postprocess:
                scored_covers = [
                    (sum(group_scores[group_id] for group_id in cover), cover)
                    for cover in covers
                ]
                scored_covers.sort(
                    key=lambda item: (-item[0], tuple(group_sort_keys[group_id] for group_id in item[1]))
                )
                selected_covers = [cover for _score, cover in scored_covers[:max_partitions]]
            else:
                partitions = [
                    Partition(tuple(groups_only[group_id] for group_id in cover), "full_search_native_top")
                    for cover in covers
                ]
                scored = [(self._cheap_partition_score(partition), partition) for partition in partitions]
                scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
                out[key] = [partition for _score, partition in scored[:max_partitions]]
                continue
            out[key] = [
                Partition(
                    tuple(groups_only[group_id] for group_id in cover),
                    "full_search_native_top",
                )
                for cover in selected_covers
            ]
        self._bounded_partition_cache_store(cache_common, out, keys)
        return out

    def clear_cache(self) -> None:
        self._group_cache.clear()
        self._all_cache.clear()
        self._top_cache.clear()
        self._native_score_input_cache.clear()
        self._native_group_signature_cache.clear()
        self._native_group_superset_cache.clear()
        self._native_group_exact_cache.clear()
        self._cover_input_cache.clear()
        self._bounded_partition_cache.clear()

    def _generate_top_native(
        self,
        cards: list[str],
        ctx: RetrievalContext,
        max_partitions: int,
    ) -> list[Partition] | None:
        if not self.use_native or not native_cover.available():
            return None
        try:
            group_entries, start, _groups_by_first, native_buckets = self._cover_search_inputs(
                cards,
                ctx,
                include_priority=True,
                include_python_buckets=False,
            )
            groups_only = [entry[0] for entry in group_entries]
            group_scores = [self._effective_group_score(entry[0]) for entry in group_entries]
            tie_keys = [self._native_tie_key(group) for group in groups_only]
            id_postprocess = os.environ.get("DANRL_NATIVE_ID_POSTPROCESS", "1").strip().lower() in {
                "1", "true", "yes", "on",
            }
            native_limit = max(max_partitions * 8, max_partitions + 128)
            covers = native_cover.top_covers(start, native_buckets, group_scores, tie_keys, native_limit)
        except Exception:
            return None
        if not covers:
            return [self._singles_partition(cards, ctx)]

        if id_postprocess:
            group_sort_keys = [(group.kind, group.cards) for group in groups_only]
            scored_covers = [
                (sum(group_scores[group_id] for group_id in cover), cover)
                for cover in covers
            ]
            scored_covers.sort(
                key=lambda item: (-item[0], tuple(group_sort_keys[group_id] for group_id in item[1]))
            )
            return [
                Partition(
                    tuple(groups_only[group_id] for group_id in cover),
                    "full_search_native_top",
                )
                for _score, cover in scored_covers[:max_partitions]
            ]

        partitions = [
            Partition(tuple([groups_only[group_id] for group_id in cover]), "full_search_native_top")
            for cover in covers
        ]
        scored = [(self._cheap_partition_score(partition), partition) for partition in partitions]
        scored.sort(key=lambda item: (-item[0], tuple((g.kind, g.cards) for g in item[1].groups)))
        return [partition for _score, partition in scored[:max_partitions]]

    def _bounded_partition_cache_common_key(
        self,
        mode: str,
        super_cards: Iterable[str],
        ctx: RetrievalContext,
        *parameters: int,
    ) -> tuple:
        # Group generation is a function of the physical hand and level rank.
        # Its cheap priority additionally changes only when all red jokers are
        # gone (the sole remaining_detail branch in rank_strength()). Dynamic
        # trick, pressure and history values are applied after this prefix.
        priority_profile = tuple(sorted(
            (str(kind), float(value))
            for kind, value in self._priority_base_by_kind.items()
        ))
        cache_by_target_hand = os.environ.get(
            "DANRL_BOUNDED_PARTITION_CACHE_BY_TARGET_HAND", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        universe_key = () if cache_by_target_hand else (tuple(normalize_cards(super_cards)),)
        return (
            mode,
            *universe_key,
            ctx.cur_rank,
            int(bool(ctx.remaining_detail)),
            int(ctx.remaining_detail.get("RJ", 0) == 0),
            self._priority_profile_name,
            priority_profile,
            self._priority_bomb_size_bonus,
            *parameters,
        )

    def _bounded_partition_cache_split(
        self,
        common_key: tuple,
        hands: list[tuple[str, ...]],
    ) -> tuple[
        dict[tuple[str, ...], list[Partition] | NativePartitionCovers],
        list[tuple[str, ...]],
    ]:
        capacity = self._env_int("DANRL_BOUNDED_PARTITION_CACHE_SIZE", 0, minimum=0)
        if capacity <= 0:
            return {}, hands
        out: dict[tuple[str, ...], list[Partition] | NativePartitionCovers] = {}
        missing: list[tuple[str, ...]] = []
        for hand in hands:
            self._bounded_partition_cache_gets += 1
            key = (*common_key, hand)
            cached = self._bounded_partition_cache.get(key)
            if cached is None:
                missing.append(hand)
                continue
            self._bounded_partition_cache_hits += 1
            self._bounded_partition_cache.move_to_end(key)
            out[hand] = cached if isinstance(cached, NativePartitionCovers) else list(cached)
        return out, missing

    def _bounded_partition_cache_store(
        self,
        common_key: tuple,
        out: dict[tuple[str, ...], list[Partition] | NativePartitionCovers],
        computed_hands: list[tuple[str, ...]],
    ) -> None:
        capacity = self._env_int("DANRL_BOUNDED_PARTITION_CACHE_SIZE", 0, minimum=0)
        if capacity <= 0:
            return
        for hand in computed_hands:
            partitions = out.get(hand)
            if partitions is None:
                continue
            key = (*common_key, hand)
            self._bounded_partition_cache[key] = (
                partitions
                if isinstance(partitions, NativePartitionCovers)
                else tuple(partitions)
            )
            self._bounded_partition_cache.move_to_end(key)
        while len(self._bounded_partition_cache) > capacity:
            self._bounded_partition_cache.popitem(last=False)

    def _native_tie_key(self, group: CardGroup) -> str:
        return f"{group.kind}\x1f{chr(0x1e).join(group.cards)}"

    @staticmethod
    def _defer_selected_partitions() -> bool:
        return os.environ.get(
            "DANRL_DEFER_SELECTED_PARTITIONS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def materialize_partitions(
        partitions: list[Partition] | NativePartitionCovers,
    ) -> list[Partition]:
        return partitions.materialize() if isinstance(partitions, NativePartitionCovers) else partitions

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                value = default
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _partition_cache_key(self, hand_cards: Iterable[str], ctx: RetrievalContext) -> tuple:
        cards = sorted_cards(normalize_cards(hand_cards))
        return self._partition_cache_key_from_sorted_cards(cards, ctx)

    def _partition_cache_key_from_sorted_cards(self, cards: Iterable[str], ctx: RetrievalContext) -> tuple:
        cards = tuple(cards)
        return (cards, *self._partition_context_key_suffix(ctx))

    @staticmethod
    def _partition_context_key_suffix(ctx: RetrievalContext) -> tuple:
        return (
            ctx.cur_rank,
            tuple((str(card), int(count)) for card, count in ctx.remaining_detail.items()),
            tuple((str(rank), int(count)) for rank, count in ctx.remaining_by_rank.items()),
        )

    def _cache_get(self, cache: OrderedDict[tuple, tuple[Partition, ...]], key: tuple) -> tuple[Partition, ...] | None:
        if self.cache_size <= 0:
            return None
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    def _cache_put(self, cache: OrderedDict[tuple, tuple[Partition, ...]], key: tuple, value: tuple[Partition, ...]) -> None:
        if self.cache_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def generate_beam_top(
        self,
        hand_cards: list[str],
        ctx: RetrievalContext,
        max_partitions: int = 8,
        beam_width: int = 96,
    ) -> list[Partition]:
        """Return bounded high-value covers for large hands.

        Full exact-cover enumeration can explode on 27-card opening hands. This
        method keeps the exact-cover invariant for each returned partition, but
        uses a beam over the first remaining physical card instead of materializing
        every cover.
        """

        cards = normalize_cards(hand_cards)
        if not cards:
            return [Partition(groups=(), mode="beam_top")]

        groups = self._generate_groups(cards, ctx)
        hand_counter = Counter(cards)
        # Compress absent cards while preserving the exact ALL_CARDS branch order.
        local_cards = sorted(hand_counter, key=CARD_INDEX.__getitem__)
        local_index = {card: idx for idx, card in enumerate(local_cards)}
        groups_by_first: dict[
            int,
            list[
                tuple[
                    CardGroup,
                    tuple[tuple[int, int], ...],
                    float,
                    tuple[str, tuple[str, ...]],
                ]
            ],
        ] = {}
        for group in groups:
            key = _group_local_sparse_key(group.cards, local_index)
            first = key[0][0]
            tie_key = (group.kind, group.cards)
            groups_by_first.setdefault(first, []).append(
                (group, key, self._group_priority_for_profile(group), tie_key)
            )
        for bucket in groups_by_first.values():
            bucket.sort(key=lambda item: (-item[2], item[3]))

        start = tuple(int(hand_counter[card]) for card in local_cards)
        beam: list[
            tuple[
                float,
                tuple[int, ...],
                int,
                tuple[CardGroup, ...],
                tuple[tuple[str, tuple[str, ...]], ...],
            ]
        ] = [(0.0, start, len(cards), (), ())]
        completed: list[
            tuple[
                float,
                tuple[CardGroup, ...],
                tuple[tuple[str, tuple[str, ...]], ...],
            ]
        ] = []
        max_steps = len(cards)
        for _ in range(max_steps):
            next_beam: list[
                tuple[
                    float,
                    tuple[int, ...],
                    int,
                    tuple[CardGroup, ...],
                    tuple[tuple[str, tuple[str, ...]], ...],
                ]
            ] = []
            for score, state, remaining, chosen, chosen_key in beam:
                first = next(i for i, count in enumerate(state) if count)
                for group, group_state, group_score, group_tie_key in groups_by_first.get(first, []):
                    next_state = _subtract_sparse(state, group_state)
                    if next_state is None:
                        continue
                    next_chosen = chosen + (group,)
                    next_chosen_key = chosen_key + (group_tie_key,)
                    next_score = score + group_score - 10.0 * effective_group_cost(group)
                    next_remaining = remaining - group.size
                    if next_remaining == 0:
                        completed.append((next_score, next_chosen, next_chosen_key))
                    else:
                        next_beam.append(
                            (
                                next_score,
                                next_state,
                                next_remaining,
                                next_chosen,
                                next_chosen_key,
                            )
                        )
            if not next_beam:
                break
            next_beam.sort(key=lambda item: (-item[0], len(item[3]), item[4]))
            beam = next_beam[:beam_width]

        if not completed:
            return [self._singles_partition(cards, ctx)]
        completed.sort(key=lambda item: (-item[0], item[2]))
        out: list[Partition] = []
        seen = set()
        for _score, groups, _tie_key in completed:
            key = tuple((group.kind, group.rank, group.cards, tuple(group.meta.get("wild_as", ()))) for group in groups)
            if key in seen:
                continue
            seen.add(key)
            out.append(Partition(groups=groups, mode="beam_top"))
            if len(out) >= max_partitions:
                break
        return out

    def _cheap_partition_score(self, partition: Partition) -> float:
        return sum(self._group_priority_for_profile(group) for group in partition.groups) - 10.0 * partition.effective_hand_count

    def _generate_groups(self, cards: list[str], ctx: RetrievalContext, partition_key: tuple | None = None) -> list[CardGroup]:
        cache_key = ("groups", partition_key if partition_key is not None else self._partition_cache_key(cards, ctx))
        cached = self._cache_get(self._group_cache, cache_key)
        if cached is not None:
            return list(cached)

        if (
            self.use_native
            and native_actor_core.available()
            and hasattr(native_actor_core.module(), "generate_all_group_signatures")
        ):
            native_groups = self._generate_groups_native(cards, ctx)
            self._cache_put(self._group_cache, cache_key, native_groups)
            return list(native_groups)

        groups: dict[tuple[str, tuple[str, ...], str | None, tuple], CardGroup] = {}
        wild_card = heart_level_card(ctx.cur_rank)
        natural_by_rank: dict[str, list[str]] = {rank: [] for rank in NORMAL_RANKS}
        sequence_cards: list[str] = []
        sequence_by_rank: dict[str, list[str]] = {rank: [] for rank in NORMAL_RANKS}
        sequence_by_suit_rank: dict[tuple[str, str], list[str]] = {}
        wildcards: list[str] = []
        hand_counter: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for card in cards:
            hand_counter[card] += 1
            rank = card_rank(card)
            counts[rank] += 1
            is_wild = bool(wild_card and card == wild_card)
            if is_wild:
                wildcards.append(card)
            elif rank in NORMAL_RANKS:
                natural_by_rank[rank].append(card)
            if card in {"BJ", "RJ"}:
                continue
            sequence_cards.append(card)
            suit = card[0]
            sequence_by_rank.setdefault(rank, []).append(card)
            sequence_by_suit_rank.setdefault((suit, rank), []).append(card)
        strength_cache: dict[str | None, float] = {None: 0.0}

        def add(kind: str, selected: tuple[str, ...], rank: str | None = None, meta: dict | None = None) -> None:
            cards_key = sorted_cards(selected)
            if rank is None and cards_key:
                rank = max((card_rank(card) for card in cards_key), key=lambda r: RANK_VALUE.get(r, 0))
            meta = meta or {}
            key = (kind, cards_key, rank, tuple(meta.get("wild_as", ())))
            if key in groups:
                return
            strength = strength_cache.get(rank)
            if strength is None:
                strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail)
                strength_cache[rank] = strength
            groups[key] = CardGroup(kind=kind, cards=cards_key, rank=rank, strength=strength, meta=meta)

        # Singles, pairs, triples, same-rank bombs.
        for card in cards:
            add("Single", (card,), card_rank(card))
        for rank in NORMAL_RANKS:
            natural_rank_cards = natural_by_rank.get(rank, [])
            self._add_same_rank_groups(add, natural_rank_cards, wildcards, rank)
        if counts.get("BJ", 0) >= 2:
            add("Pair", ("BJ", "BJ"), "BJ")
        if counts.get("RJ", 0) >= 2:
            add("Pair", ("RJ", "RJ"), "RJ")

        # Four kings.
        if counts.get("BJ", 0) >= 2 and counts.get("RJ", 0) >= 2:
            add("FourKings", ("BJ", "BJ", "RJ", "RJ"), "RJ")

        # Five-card straights and straight flushes.
        for start in range(0, len(STRAIGHT_RANKS) - 4):
            ranks = tuple(STRAIGHT_RANKS[start : start + 5])
            for straight, wild_as in self._rank_sequence_options(sequence_cards, [], ranks, rank_cards_by_rank=sequence_by_rank):
                if self._is_plain_straight_flush_interpretation(straight):
                    continue
                add("Straight", straight, ranks[-1], self._wild_meta(wild_as))
            for suit in SUITS:
                for flush, wild_as in self._rank_sequence_options(
                    sequence_cards,
                    [],
                    ranks,
                    suit,
                    suit_rank_cards=sequence_by_suit_rank,
                ):
                    add("StraightFlush", flush, ranks[-1], self._wild_meta(wild_as))

        # Three consecutive pairs.
        for start in range(0, len(STRAIGHT_RANKS) - 2):
            ranks = tuple(STRAIGHT_RANKS[start : start + 3])
            for selected, wild_as in self._multi_rank_sequence_options(sequence_cards, [], ranks, 2, sequence_by_rank):
                add("StraightPair", selected, ranks[-1], self._wild_meta(wild_as))

        # Two consecutive triples.
        for start in range(0, len(STRAIGHT_RANKS) - 1):
            ranks = tuple(STRAIGHT_RANKS[start : start + 2])
            for selected, wild_as in self._multi_rank_sequence_options(sequence_cards, [], ranks, 3, sequence_by_rank):
                add("StraightTriple", selected, ranks[-1], self._wild_meta(wild_as))

        # Three plus pair.
        triple_groups = [g for g in groups.values() if g.kind == "Triple"]
        pair_groups = [g for g in groups.values() if g.kind == "Pair"]
        triple_infos = [(group, Counter(group.cards), tuple(group.meta.get("wild_as", ()))) for group in triple_groups]
        pair_infos = [(group, Counter(group.cards), tuple(group.meta.get("wild_as", ()))) for group in pair_groups]
        for triple, triple_count, triple_wild_as in triple_infos:
            for pair, pair_count, pair_wild_as in pair_infos:
                if triple.rank == pair.rank:
                    continue
                fits = True
                for card, count in triple_count.items():
                    if count + pair_count.get(card, 0) > hand_counter[card]:
                        fits = False
                        break
                if fits:
                    for card, count in pair_count.items():
                        if card not in triple_count and count > hand_counter[card]:
                            fits = False
                            break
                if fits:
                    selected = tuple(triple.cards + pair.cards)
                    add("TriplePlus", selected, triple.rank, self._wild_meta(triple_wild_as + pair_wild_as))

        out = tuple(sorted(groups.values(), key=lambda g: (-self._group_priority_for_profile(g), g.kind, g.cards)))
        self._cache_put(self._group_cache, cache_key, out)
        return list(out)

    def _generate_groups_native(self, cards: list[str], ctx: RetrievalContext) -> tuple[CardGroup, ...]:
        exact_cache_size = self._env_int(
            "DANRL_NATIVE_GROUP_EXACT_CACHE_SIZE", 0, minimum=0,
        )
        superset_cache_size = self._env_int(
            "DANRL_NATIVE_GROUP_SUPERSET_CACHE_SIZE", 0, minimum=0,
        )
        mask_filter = os.environ.get(
            "DANRL_NATIVE_GROUP_MASK_FILTER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        hand_counts_key = _counter_to_key(Counter(cards))
        hand_require_one, hand_require_two = (
            _count_requirement_masks(hand_counts_key) if mask_filter else (0, 0)
        )
        strength_context = (
            ctx.cur_rank,
            bool(ctx.remaining_detail),
            bool(ctx.remaining_detail) and ctx.remaining_detail.get("RJ", 0) == 0,
        )
        superset_key = (*strength_context, hand_counts_key)
        if mask_filter:
            superset_key = (*superset_key, hand_require_one, hand_require_two)
        if exact_cache_size > 0:
            cached_exact = self._native_group_exact_cache.get(superset_key)
            if cached_exact is not None:
                self._native_group_exact_cache.move_to_end(superset_key)
                return cached_exact

        def remember_exact(value: tuple[CardGroup, ...]) -> tuple[CardGroup, ...]:
            if exact_cache_size <= 0:
                return value
            self._native_group_exact_cache[superset_key] = value
            self._native_group_exact_cache.move_to_end(superset_key)
            while len(self._native_group_exact_cache) > exact_cache_size:
                self._native_group_exact_cache.popitem(last=False)
            return value

        if superset_cache_size > 0:
            cached_exact = self._native_group_superset_cache.get(superset_key)
            if cached_exact is not None:
                self._native_group_superset_cache.move_to_end(superset_key)
                return remember_exact(cached_exact)
            best_groups: tuple[CardGroup, ...] | None = None
            best_size = 1 << 30
            for cached_key, cached_groups in reversed(self._native_group_superset_cache.items()):
                if cached_key[:3] != strength_context:
                    continue
                cached_counts = cached_key[3]
                cached_size = sum(cached_counts)
                if cached_size >= best_size:
                    continue
                is_superset = (
                    hand_require_one & ~cached_key[4] == 0
                    and hand_require_two & ~cached_key[5] == 0
                    if mask_filter
                    else all(have >= need for have, need in zip(cached_counts, hand_counts_key))
                )
                if is_superset:
                    best_groups = cached_groups
                    best_size = cached_size
            if best_groups is not None:
                if mask_filter:
                    filtered = tuple(
                        group
                        for group in best_groups
                        if group.meta["__require_one"] & ~hand_require_one == 0
                        and group.meta["__require_two"] & ~hand_require_two == 0
                    )
                else:
                    filtered = tuple(
                        group
                        for group in best_groups
                        if all(
                            count <= hand_counts_key[index]
                            for index, count in group.meta["__physical_counts"]
                        )
                    )
                self._native_group_superset_cache[superset_key] = filtered
                self._native_group_superset_cache.move_to_end(superset_key)
                while len(self._native_group_superset_cache) > superset_cache_size:
                    self._native_group_superset_cache.popitem(last=False)
                return remember_exact(filtered)
        groups: dict[tuple[str, tuple[str, ...], str | None, tuple], CardGroup] = {}
        strength_cache: dict[str | None, float] = {None: 0.0}
        signature_cache_size = self._env_int(
            "DANRL_NATIVE_GROUP_SIGNATURE_CACHE_SIZE", 0, minimum=0,
        )
        signature_key = (sorted_cards(cards), ctx.cur_rank)
        signatures = (
            self._native_group_signature_cache.get(signature_key)
            if signature_cache_size > 0
            else None
        )
        if signatures is not None:
            self._native_group_signature_cache.move_to_end(signature_key)
        else:
            signatures = tuple(
                native_actor_core.generate_all_group_signatures(cards, ctx.cur_rank)
            )
            if signature_cache_size > 0:
                self._native_group_signature_cache[signature_key] = signatures
                self._native_group_signature_cache.move_to_end(signature_key)
                while len(self._native_group_signature_cache) > signature_cache_size:
                    self._native_group_signature_cache.popitem(last=False)
        for kind, rank, selected, wild_as in signatures:
            cards_key = selected
            meta = self._wild_meta(wild_as)
            physical_counts = _group_sparse_key(cards_key)
            meta["__physical_counts"] = physical_counts
            if mask_filter:
                meta["__require_one"], meta["__require_two"] = _sparse_requirement_masks(
                    physical_counts
                )
            wild_key = tuple(meta.get("wild_as", ()))
            key = (kind, cards_key, rank, wild_key)
            if key in groups:
                continue
            strength = strength_cache.get(rank)
            if strength is None:
                strength = rank_strength(rank, ctx.cur_rank, ctx.remaining_detail)
                strength_cache[rank] = strength
            meta["__score_key"] = (kind, rank, len(cards_key), strength)
            groups[key] = CardGroup(kind=kind, cards=cards_key, rank=rank, strength=strength, meta=meta)
        out = tuple(sorted(groups.values(), key=lambda g: (-self._group_priority_for_profile(g), g.kind, g.cards)))
        if superset_cache_size > 0:
            self._native_group_superset_cache[superset_key] = out
            self._native_group_superset_cache.move_to_end(superset_key)
            while len(self._native_group_superset_cache) > superset_cache_size:
                self._native_group_superset_cache.popitem(last=False)
        return remember_exact(out)

    def _wild_meta(self, wild_as) -> dict:
        if not wild_as:
            return {}
        return {
            "wild_count": len(wild_as),
            "wild_as": tuple(wild_as),
        }

    def _is_plain_straight_flush_interpretation(self, selected: tuple[str, ...]) -> bool:
        first_suit: str | None = None
        for card in selected:
            if card in {"BJ", "RJ"}:
                continue
            suit = card[0]
            if first_suit is None:
                first_suit = suit
            elif suit != first_suit:
                return False
        return first_suit is not None

    def _add_same_rank_groups(self, add, natural_rank_cards: list[str], wildcards: list[str], rank: str) -> None:
        max_total = len(natural_rank_cards) + len(wildcards)
        wild_prefixes = tuple(tuple(wildcards[:count]) for count in range(len(wildcards) + 1))
        wild_as_by_count = tuple((rank,) * count for count in range(len(wildcards) + 1))
        for size, kind in ((2, "Pair"), (3, "Triple")):
            if max_total >= size:
                for n_natural in range(max(0, size - len(wildcards)), min(size, len(natural_rank_cards)) + 1):
                    n_wild = size - n_natural
                    wild_prefix = wild_prefixes[n_wild]
                    wild_as = wild_as_by_count[n_wild]
                    for naturals in combinations(natural_rank_cards, n_natural):
                        selected = naturals + wild_prefix
                        add(kind, selected, rank, self._wild_meta(wild_as))
        for size in range(4, max_total + 1):
            for n_natural in range(max(0, size - len(wildcards)), min(size, len(natural_rank_cards)) + 1):
                n_wild = size - n_natural
                wild_prefix = wild_prefixes[n_wild]
                wild_as = wild_as_by_count[n_wild]
                for naturals in combinations(natural_rank_cards, n_natural):
                    selected = naturals + wild_prefix
                    add("Bomb", selected, rank, self._wild_meta(wild_as))

    def _rank_sequence_options(
        self,
        natural_cards: list[str],
        wildcards: list[str],
        ranks: tuple[str, ...],
        suit: str | None = None,
        rank_cards_by_rank: dict[str, list[str]] | None = None,
        suit_rank_cards: dict[tuple[str, str], list[str]] | None = None,
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        options: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((), ())]
        for rank in ranks:
            if suit is None and rank_cards_by_rank is not None:
                candidates = rank_cards_by_rank.get(rank, [])
            elif suit is not None and suit_rank_cards is not None:
                candidates = suit_rank_cards.get((suit, rank), [])
            else:
                candidates = [
                    card for card in natural_cards
                    if card_rank(card) == rank and (suit is None or card_suit(card) == suit)
                ]
            rebuilt = []
            for selected, wild_as in options:
                used = len(wild_as)
                for card in candidates:
                    rebuilt.append((selected + (card,), wild_as))
                if used < len(wildcards):
                    rebuilt.append((selected + (wildcards[used],), wild_as + (rank,)))
            options = rebuilt
        return options

    def _multi_rank_sequence_options(
        self,
        natural_cards: list[str],
        wildcards: list[str],
        ranks: tuple[str, ...],
        need_per_rank: int,
        rank_cards_by_rank: dict[str, list[str]] | None = None,
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        options: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((), ())]
        for rank in ranks:
            rank_cards = rank_cards_by_rank.get(rank, []) if rank_cards_by_rank is not None else _rank_cards(natural_cards, rank)
            rebuilt = []
            for selected, wild_as in options:
                used = len(wild_as)
                for n_natural in range(max(0, need_per_rank - (len(wildcards) - used)), min(need_per_rank, len(rank_cards)) + 1):
                    n_wild = need_per_rank - n_natural
                    if used + n_wild > len(wildcards):
                        continue
                    for naturals in combinations(rank_cards, n_natural):
                        rebuilt.append((
                            selected + tuple(naturals) + tuple(wildcards[used:used + n_wild]),
                            wild_as + (rank,) * n_wild,
                        ))
            options = rebuilt
        return options

    def _singles_partition(self, cards: list[str], ctx: RetrievalContext) -> Partition:
        return Partition(
            groups=tuple(_make_group("Single", (card,), ctx, card_rank(card)) for card in sorted_cards(cards)),
            mode="singles_fallback",
        )


# Backward-compatible name used by existing ranker code.
HeuristicPartitioner = FullSearchPartitioner
