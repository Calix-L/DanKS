from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import replace
import heapq
import os
from typing import Any, Iterable

from .cards import card_sort_key, normalize_cards, rank_strength
from .action_semantics import normalize_kind, normalize_rank
from .context import RetrievalContext, build_context
from .models import ActionCandidate, CardGroup, NativePartitionCovers, Partition, ScoredAction
from . import native_cover
from . import native_actor_core
from .partitioner import FullSearchPartitioner, HeuristicPartitioner
from .scoring import (
    BREAK_PENALTY_PROFILES,
    BreakPenaltyProfile,
    PartitionScorer,
    ScoreWeights,
    action_spend_penalty,
    current_control_value,
    escape_risk_penalty_from_current_control,
    lead_action_value,
    pass_pressure_penalty,
    break_group_penalty,
    break_group_penalty_indexed,
    break_group_penalty_table,
    break_group_penalty_from_groups,
    prepare_break_group_penalty_index,
    score_partition,
    update_break_group_penalty_index,
    teammate_overcall_penalty,
)
from .tempo import build_tempo_context, tempo_action_value_from_context, tempo_features_from_context


def normalize_action(action: Any, fallback_index: int) -> ActionCandidate:
    if isinstance(action, ActionCandidate):
        return action
    if not isinstance(action, dict):
        raise TypeError(f"action must be dict or ActionCandidate, got {type(action).__name__}")
    cards = tuple(normalize_cards(action.get("cards") or []))
    kind = normalize_kind(str(action.get("kind") or action.get("type") or ("PASS" if not cards else "Unknown")))
    return ActionCandidate(
        index=int(action.get("index", fallback_index)),
        kind=kind,
        cards=cards,
        rank="PASS" if kind == "PASS" else normalize_rank(str(action.get("rank"))) if action.get("rank") is not None else None,
    )


def remove_cards_from_counts(hand_counts: Counter[str], action_cards: tuple[str, ...]) -> list[str]:
    action_counts = Counter(action_cards)
    for card, count in action_counts.items():
        if count > hand_counts.get(card, 0):
            raise ValueError(f"action card {card} not in hand")
    out: list[str] = []
    for card, count in hand_counts.items():
        remaining = count - action_counts.get(card, 0)
        if remaining > 0:
            out.extend([card] * remaining)
    return sorted(out, key=card_sort_key)


def remove_cards_from_hand(hand: list[str] | tuple[str, ...], action_cards: tuple[str, ...], hand_counts: Counter[str]) -> list[str]:
    if native_actor_core.available():
        return native_actor_core.remove_cards_sorted(hand, action_cards)
    return remove_cards_from_counts(hand_counts, action_cards)


def batch_after_hands_by_action_cards(
    hand: list[str] | tuple[str, ...],
    actions: list[ActionCandidate],
    hand_counts: Counter[str],
) -> dict[tuple[str, ...], tuple[str, ...]]:
    action_cards: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for action in actions:
        if action.kind == "PASS" or action.cards in seen:
            continue
        seen.add(action.cards)
        action_cards.append(action.cards)
    if not action_cards:
        return {}
    if native_actor_core.available():
        try:
            after_hands = native_actor_core.remove_cards_sorted_batch(hand, action_cards)
            return {cards: tuple(after_hand) for cards, after_hand in zip(action_cards, after_hands)}
        except Exception:
            pass
    return {cards: tuple(remove_cards_from_counts(hand_counts, cards)) for cards in action_cards}


class StructuralCandidateRanker:
    def __init__(
        self,
        partitioner: HeuristicPartitioner | None = None,
        weights: ScoreWeights = ScoreWeights(),
        break_profile: BreakPenaltyProfile | None = None,
        max_partitions: int | None = 8,
        lead_max_partitions: int | None = None,
        follow_max_partitions: int | None = None,
        exact_best_cache_size: int = 8192,
    ) -> None:
        self.partitioner = partitioner or FullSearchPartitioner(use_native_all=True)
        implicit_profile = break_profile is None
        raw_profile_name = os.environ.get("DANKS_BREAK_PROFILE", "").strip()
        profile_selected_by_env = implicit_profile and bool(raw_profile_name)
        if break_profile is None:
            profile_name = raw_profile_name or "default"
            try:
                break_profile = BREAK_PENALTY_PROFILES[profile_name]
            except KeyError as exc:
                choices = ", ".join(sorted(BREAK_PENALTY_PROFILES))
                raise ValueError(f"unknown DANKS_BREAK_PROFILE={profile_name!r}; expected one of: {choices}") from exc
        if implicit_profile:
            raw_break_weight = os.environ.get("DANKS_BREAK_GROUP_WEIGHT", "").strip()
            if raw_break_weight:
                weights = replace(weights, break_group=float(raw_break_weight))
            elif profile_selected_by_env:
                weights = replace(weights, break_group=break_profile.recommended_group_weight)
        self.weights = weights
        self.break_profile = break_profile
        if hasattr(self.partitioner, "set_group_priority_profile"):
            self.partitioner.set_group_priority_profile(
                break_profile.name,
                break_profile.group_value_base,
                break_profile.bomb_size_bonus,
            )
        self.max_partitions = max_partitions
        self.lead_max_partitions = lead_max_partitions
        self.follow_max_partitions = follow_max_partitions
        self.exact_best_cache_size = exact_best_cache_size
        self._exact_best_cache: OrderedDict[tuple, tuple[Partition, float, dict[str, float]]] = OrderedDict()
        self._break_penalty_cache: OrderedDict[tuple, dict[int, float]] = OrderedDict()
        self._flat_score_entry_cache: OrderedDict[tuple, list[list[float | int]]] = OrderedDict()
        self.last_rank_stats: dict[str, Any] = {}
        self._exact_best_cache_gets = 0
        self._exact_best_cache_hits = 0
        self._break_penalty_cache_gets = 0
        self._break_penalty_cache_hits = 0
        self._flat_score_entry_cache_gets = 0
        self._flat_score_entry_cache_hits = 0
        self._native_selected_input_cache: dict[int, tuple[object, list[CardGroup], list[list[int]], list[list[float | int]]]] = {}
        self._stable_search_hands: dict[int, tuple[tuple, tuple[str, ...]]] = {}

    def rank(
        self,
        hand_cards: list[str],
        legal_actions: list[dict[str, Any] | ActionCandidate] | None,
        context: RetrievalContext | dict[str, Any],
        top_k: int | None = None,
        approximate_top_k: bool = False,
    ) -> list[ScoredAction]:
        self._native_selected_input_cache.clear()
        hand = normalize_cards(hand_cards)
        hand_counts: Counter[str] = Counter(hand)
        ctx = self._context_with_hand(context, hand)
        search_hand = self._stable_search_hand(tuple(hand), ctx)
        if legal_actions is None:
            raise ValueError("legal_actions must be supplied by Dan_platform")
        actions = [normalize_action(raw_action, i) for i, raw_action in enumerate(legal_actions)]
        action_static_features = self._batch_action_static_features(actions, ctx)
        original_action_count = len(actions)
        partition_limit = self._partition_limit(ctx)
        prepared_batch_inputs = None
        reuse_cover_inputs = os.environ.get(
            "DANKS_NATIVE_BATCH_REUSE_COVER_INPUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            partition_limit is not None
            and reuse_cover_inputs
            and isinstance(self.partitioner, FullSearchPartitioner)
        ):
            try:
                prepared_batch_inputs = self.partitioner._cover_search_inputs(
                    list(search_hand),
                    ctx,
                    include_priority=True,
                    include_python_buckets=False,
                )
            except Exception:
                prepared_batch_inputs = None
        approximate_action_limit = self._approx_action_limit()
        approximated_actions = False
        if approximate_action_limit > 0 and len(actions) > approximate_action_limit:
            actions = self._prefilter_actions_approx(
                actions,
                tuple(hand),
                hand_counts,
                ctx,
                approximate_action_limit,
                top_k,
                action_static_features,
                prepared_batch_inputs,
            )
            approximated_actions = True
        shared_scorer = PartitionScorer(ctx, self.weights, self.break_profile)
        before_primary_kind = self._primary_retake_kind(ctx, ActionCandidate(-1, "PASS", (), "PASS"))
        tempo_ctx = build_tempo_context(ctx)
        partition_context_suffix = (
            self.partitioner._partition_context_key_suffix(ctx)
            if isinstance(self.partitioner, FullSearchPartitioner)
            else None
        )
        exact_cache_ctx_suffix = self._exact_best_cache_context_suffix(ctx) if partition_limit is None else ()
        hand_key = tuple(hand)
        after_hand_by_action_cards = batch_after_hands_by_action_cards(hand, actions, hand_counts)
        action_after_primary: dict[int, tuple[tuple[str, ...], str]] = {}
        for action in actions:
            if action.kind == "PASS":
                after_key = hand_key
            else:
                after_key = after_hand_by_action_cards.get(action.cards)
                if after_key is None:
                    after_key = tuple(remove_cards_from_hand(hand, action.cards, hand_counts))
                    after_hand_by_action_cards[action.cards] = after_key
            action_after_primary[action.index] = (after_key, self._primary_retake_kind(ctx, action))

        partitions_by_after_hand: dict[
            tuple[str, ...], list[Partition] | NativePartitionCovers
        ] = {}
        batch_includes_before = False
        include_before_in_batch = os.environ.get(
            "DANKS_NATIVE_BATCH_INCLUDE_BEFORE_HAND", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            partition_limit is not None
            and include_before_in_batch
            and isinstance(self.partitioner, FullSearchPartitioner)
        ):
            batch_hands = list(dict.fromkeys(
                [hand_key, *(key for key, _kind in action_after_primary.values())]
            ))
            if reuse_cover_inputs and prepared_batch_inputs is None:
                try:
                    prepared_batch_inputs = self.partitioner._cover_search_inputs(
                        list(search_hand),
                        ctx,
                        include_priority=True,
                        include_python_buckets=False,
                    )
                except Exception:
                    prepared_batch_inputs = None
            partitions_by_after_hand.update(
                self.partitioner.generate_top_hand_count_window_batch(
                    search_hand,
                    batch_hands,
                    ctx,
                    partition_limit,
                    prepared_inputs=prepared_batch_inputs,
                )
            )
            small_hand_batch = os.environ.get(
                "DANKS_NATIVE_SMALL_HAND_BATCH", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if small_hand_batch:
                hand_count_window = self.partitioner._env_int(
                    "DANKS_PARTITION_HAND_COUNT_WINDOW", -1,
                )
                window_min_hand = self.partitioner._env_int(
                    "DANKS_PARTITION_HAND_COUNT_WINDOW_MIN_HAND", 0,
                )
                missing_hands = [
                    key
                    for key in batch_hands
                    if key not in partitions_by_after_hand
                    and len(key) <= self.partitioner.exhaustive_top_threshold
                    and (hand_count_window < 0 or len(key) < window_min_hand)
                ]
                if missing_hands:
                    partitions_by_after_hand.update(
                        self.partitioner.generate_top_native_batch(
                            search_hand,
                            missing_hands,
                            ctx,
                            partition_limit,
                            prepared_inputs=prepared_batch_inputs,
                        )
                    )
                native_beam_batch = os.environ.get(
                    "DANKS_NATIVE_BEAM_BATCH", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                beam_hands = [
                    key
                    for key in batch_hands
                    if key not in partitions_by_after_hand
                    and len(key) > self.partitioner.exhaustive_top_threshold
                    and (hand_count_window < 0 or len(key) < window_min_hand)
                ]
                if native_beam_batch and beam_hands:
                    partitions_by_after_hand.update(
                        self.partitioner.generate_top_beam_batch(
                            search_hand,
                            beam_hands,
                            ctx,
                            partition_limit,
                            prepared_inputs=prepared_batch_inputs,
                        )
                    )
            batch_includes_before = True
        if partition_limit is None:
            before_partition, _before_score, _before_parts, break_penalties = self._full_before_stats(
                hand,
                ctx,
                actions,
                before_primary_kind,
                shared_scorer,
                exact_cache_ctx_suffix,
            )
            before_partitions = None
        else:
            before_partitions = partitions_by_after_hand.get(hand_key)
            if before_partitions is None:
                before_partitions = self.partitioner.generate(hand, ctx, partition_limit)
            else:
                before_partitions = self.partitioner.materialize_partitions(before_partitions)
            partitions_by_after_hand[hand_key] = before_partitions
            before_partition, _before_score, _before_parts = self._best_partition(before_partitions, ctx, before_primary_kind, shared_scorer)
            batch_break_penalty = os.environ.get("DANKS_BATCH_BREAK_PENALTY", "1").strip().lower() in {
                "1", "true", "yes", "on",
            }
            indexed_break_penalty = os.environ.get("DANKS_INDEXED_BREAK_PENALTY", "1").strip().lower() in {
                "1", "true", "yes", "on",
            }
            if batch_break_penalty and indexed_break_penalty:
                break_penalties = self._batch_break_group_penalties(actions, before_partitions)
            elif batch_break_penalty:
                break_penalties = break_group_penalty_table(actions, before_partitions, self.break_profile)
            else:
                break_penalties = None
        before_min_steps = float(before_partition.hand_count)
        scored: list[ScoredAction] = []
        exact_gets_before = self._exact_best_cache_gets
        exact_hits_before = self._exact_best_cache_hits
        break_gets_before = self._break_penalty_cache_gets
        break_hits_before = self._break_penalty_cache_hits
        flat_gets_before = self._flat_score_entry_cache_gets
        flat_hits_before = self._flat_score_entry_cache_hits
        if before_partitions is not None:
            partitions_by_after_hand[hand_key] = before_partitions
        best_by_after_and_kind: dict[tuple[tuple[str, ...], str], tuple[Partition, float, dict[str, float]]] = {}
        tempo_features_by_best_key = {}
        native_super_universe = self._native_super_hand_universe(hand, ctx, partition_context_suffix) if partition_limit is None else None
        best_by_after_and_kind[(hand_key, before_primary_kind)] = (before_partition, _before_score, _before_parts)
        if (
            partition_limit is not None
            and isinstance(self.partitioner, FullSearchPartitioner)
            and not batch_includes_before
        ):
            batch_hands = [key for key, _kind in dict.fromkeys(action_after_primary.values()) if key != hand_key]
            if batch_hands:
                if reuse_cover_inputs and prepared_batch_inputs is None:
                    try:
                        prepared_batch_inputs = self.partitioner._cover_search_inputs(
                            list(search_hand),
                            ctx,
                            include_priority=True,
                            include_python_buckets=False,
                        )
                    except Exception:
                        prepared_batch_inputs = None
                partitions_by_after_hand.update(
                    self.partitioner.generate_top_hand_count_window_batch(
                        search_hand,
                        batch_hands,
                        ctx,
                        partition_limit,
                        prepared_inputs=prepared_batch_inputs,
                    )
                )
                small_hand_batch = os.environ.get("DANKS_NATIVE_SMALL_HAND_BATCH", "1").strip().lower() in {
                    "1", "true", "yes", "on",
                }
                if small_hand_batch:
                    hand_count_window = self.partitioner._env_int("DANKS_PARTITION_HAND_COUNT_WINDOW", -1)
                    window_min_hand = self.partitioner._env_int(
                        "DANKS_PARTITION_HAND_COUNT_WINDOW_MIN_HAND", 0,
                    )
                    missing_hands = [
                        key
                        for key in batch_hands
                        if key not in partitions_by_after_hand
                        and len(key) <= self.partitioner.exhaustive_top_threshold
                        and (hand_count_window < 0 or len(key) < window_min_hand)
                    ]
                    if missing_hands:
                        partitions_by_after_hand.update(
                            self.partitioner.generate_top_native_batch(
                                search_hand,
                                missing_hands,
                                ctx,
                                partition_limit,
                                prepared_inputs=prepared_batch_inputs,
                            )
                        )
        if partition_limit is None and native_super_universe is not None and partition_context_suffix is not None:
            needed_best_keys = [
                best_key
                for best_key in dict.fromkeys(action_after_primary.values())
                if best_key not in best_by_after_and_kind
            ]
            best_by_after_and_kind.update(
                self._batch_best_partitions_from_native_super_universe(
                    needed_best_keys,
                    ctx,
                    shared_scorer,
                    exact_cache_ctx_suffix,
                    partition_context_suffix,
                    native_super_universe,
                )
            )
        if partition_limit is not None:
            needed_best_keys = [
                best_key
                for best_key in dict.fromkeys(action_after_primary.values())
                if best_key not in best_by_after_and_kind
                and best_key[0] in partitions_by_after_hand
            ]
            best_by_after_and_kind.update(
                self._batch_best_selected_partitions(
                    needed_best_keys,
                    partitions_by_after_hand,
                    shared_scorer,
                )
            )
        local_best_hits = 0
        local_best_misses = 0
        for action in actions:
            after_key, primary_kind = action_after_primary[action.index]
            best_key = (after_key, primary_kind)
            best = best_by_after_and_kind.get(best_key)
            if best is None:
                local_best_misses += 1
                if partition_limit is None:
                    after_partition_key = (
                        (after_key, *partition_context_suffix)
                        if partition_context_suffix is not None
                        else None
                    )
                    best = self._best_partition_from_native_super_universe(
                        after_key,
                        ctx,
                        primary_kind,
                        shared_scorer,
                        exact_cache_ctx_suffix,
                        after_partition_key,
                        native_super_universe,
                    )
                    if best is None:
                        best = self._best_partition_streaming(
                            after_key,
                            ctx,
                            primary_kind,
                            shared_scorer,
                            exact_cache_ctx_suffix,
                            after_partition_key,
                        )
                else:
                    partitions = partitions_by_after_hand.get(after_key)
                    if partitions is None:
                        partitions = self.partitioner.generate(after_key, ctx, partition_limit)
                        partitions_by_after_hand[after_key] = partitions
                    elif isinstance(partitions, NativePartitionCovers):
                        partitions = partitions.materialize()
                        partitions_by_after_hand[after_key] = partitions
                    best = self._best_partition(partitions, ctx, primary_kind, shared_scorer)
                best_by_after_and_kind[best_key] = best
            else:
                local_best_hits += 1
            best_partition, best_score, best_parts = best
            static_features = action_static_features.get(action.index)
            if static_features is None:
                current_control = current_control_value(action, ctx)
                lead_action = lead_action_value(action, ctx)
                spend_penalty = action_spend_penalty(action, ctx)
                escape_penalty = escape_risk_penalty_from_current_control(action, ctx, current_control)
                overcall_penalty = teammate_overcall_penalty(action, ctx)
            else:
                _rank_score, current_control, lead_action, spend_penalty, escape_penalty, overcall_penalty = static_features
            best_score += self.weights.current_control * current_control
            best_score += self.weights.lead_action * lead_action
            best_score -= self.weights.bomb_spend * spend_penalty if action.kind in {"Bomb", "StraightFlush", "FourKings"} else self.weights.control_spend * spend_penalty
            best_score -= self.weights.teammate_overcall * overcall_penalty
            pass_penalty = 0.0
            if action.kind == "PASS":
                pass_penalty = pass_pressure_penalty(ctx)
                best_score -= self.weights.pass_pressure * pass_penalty
            break_penalty = (
                break_penalties.get(action.index, 0.0)
                if break_penalties is not None
                else break_group_penalty(action, before_partitions, self.break_profile)
            )
            best_score -= self.weights.break_group * break_penalty
            best_score -= self.weights.escape_risk * escape_penalty
            cached_tempo_features = tempo_features_by_best_key.get(best_key)
            if cached_tempo_features is None:
                retake_count = best_parts.get("retake_count_value")
                if retake_count is None:
                    retake_count = shared_scorer.retake_count_value(best_partition)
                cached_tempo_features = tempo_features_from_context(
                    best_partition,
                    tempo_ctx,
                    retake_count=retake_count,
                )
                tempo_features_by_best_key[best_key] = cached_tempo_features
            tempo_score, action_tempo_features = tempo_action_value_from_context(
                action,
                cached_tempo_features,
                tempo_ctx,
                before_min_steps=before_min_steps,
                current_control=current_control / 100.0,
            )
            best_score += self.weights.tempo * tempo_score
            scored.append(
                ScoredAction(
                    action=action,
                    score=best_score,
                    partition=best_partition,
                    hand_count_score=best_parts["hand_count_score"],
                    card_value_score=best_parts["card_value_score"],
                    retake_score=best_parts["retake_score"],
                    details={
                        "after_hand": after_key,
                        "partition_mode": best_partition.mode,
                        "residue_penalty_score": best_parts["residue_penalty_score"],
                        "current_control_score": current_control,
                        "lead_action_score": lead_action,
                        "spend_penalty": spend_penalty,
                        "teammate_overcall_penalty": overcall_penalty,
                        "pass_pressure_penalty": pass_penalty,
                        "break_group_penalty": break_penalty,
                        "escape_risk_penalty": escape_penalty,
                        "tempo_score": tempo_score,
                        "my_min_steps": action_tempo_features.my_min_steps,
                        "partner_min_steps": action_tempo_features.partner_min_steps,
                        "opponent_min_steps_min": action_tempo_features.opponent_min_steps_min,
                        "opponent_short_pressure": action_tempo_features.opponent_short_pressure,
                        "my_retake_count": action_tempo_features.my_retake_count,
                        "partner_follow_help": action_tempo_features.partner_follow_help,
                        "must_block": action_tempo_features.must_block,
                        "can_race": action_tempo_features.can_race,
                    },
                )
            )
        if approximate_top_k and top_k is not None and top_k > 0 and normalize_kind(ctx.current_kind) != "Lead":
            return heapq.nlargest(top_k, scored, key=lambda row: row.score)

        self._apply_low_break_preference(scored)
        if normalize_kind(ctx.current_kind) == "Lead":
            if top_k is not None and top_k > 0:
                scored = self._apply_lead_top5_coverage_topk(scored, ctx, top_k)
            else:
                scored.sort(key=lambda row: row.score, reverse=True)
                scored = self._apply_lead_top5_coverage(scored, ctx)
        elif top_k is not None and top_k > 0 and len(scored) > top_k:
            scored = heapq.nlargest(top_k, scored, key=lambda row: row.score)
        else:
            scored.sort(key=lambda row: row.score, reverse=True)
        unique_after_hands = {key[0] for key in best_by_after_and_kind}
        self.last_rank_stats = {
            "actions": len(actions),
            "original_actions": original_action_count,
            "approx_action_limit": approximate_action_limit,
            "approximated_actions": approximated_actions,
            "scored": len(scored),
            "top_k": top_k,
            "partition_limit": "all" if partition_limit is None else partition_limit,
            "hand_size": len(hand),
            "unique_action_card_sets": len(after_hand_by_action_cards),
            "unique_after_hands": len(unique_after_hands),
            "unique_best_keys": len(best_by_after_and_kind),
            "local_best_hits": local_best_hits,
            "local_best_misses": local_best_misses,
            "exact_best_cache_gets": self._exact_best_cache_gets - exact_gets_before,
            "exact_best_cache_hits": self._exact_best_cache_hits - exact_hits_before,
            "break_penalty_cache_gets": self._break_penalty_cache_gets - break_gets_before,
            "break_penalty_cache_hits": self._break_penalty_cache_hits - break_hits_before,
            "flat_score_entry_cache_gets": self._flat_score_entry_cache_gets - flat_gets_before,
            "flat_score_entry_cache_hits": self._flat_score_entry_cache_hits - flat_hits_before,
            "native_super_universe": native_super_universe is not None,
        }
        return scored if top_k is None else scored[:top_k]

    def _approx_action_limit(self) -> int:
        raw = os.environ.get("DANKS_APPROX_ACTION_LIMIT", "").strip()
        if not raw:
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            return 0

    def _prefilter_actions_approx(
        self,
        actions: list[ActionCandidate],
        hand: tuple[str, ...],
        hand_counts: Counter[str],
        ctx: RetrievalContext,
        limit: int,
        top_k: int | None,
        action_static_features: dict[int, tuple[float, float, float, float, float, float]] | None = None,
        prepared_inputs=None,
    ) -> list[ActionCandidate]:
        if limit <= 0 or len(actions) <= limit:
            return actions
        target = min(len(actions), max(limit, (top_k or 0) * 4, 32))
        hand_window = self._env_int("DANKS_APPROX_HAND_WINDOW", 2, minimum=0)
        beam_width = self._env_int("DANKS_APPROX_BEAM_WIDTH", 32, minimum=1)
        bomb_keep_size = self._env_int("DANKS_APPROX_BOMB_KEEP_SIZE", 6, minimum=4)
        keep: dict[int, ActionCandidate] = {}
        essential: set[int] = set()
        approx_rows: list[tuple[ActionCandidate, int, float, float]] = []
        before_parts = None
        try:
            native_before = self.partitioner.generate_top_beam_batch(
                hand,
                [hand],
                ctx,
                max_partitions=4,
                prepared_inputs=prepared_inputs,
                beam_width=beam_width,
            ).get(hand)
            if isinstance(native_before, NativePartitionCovers):
                before_parts = self.partitioner.materialize_partitions(native_before)
            elif native_before:
                before_parts = native_before
        except Exception:
            before_parts = None
        if before_parts is None:
            before_parts = self.partitioner.generate_beam_top(
                list(hand), ctx, max_partitions=4, beam_width=beam_width
            )
        before_steps = min((part.hand_count for part in before_parts), default=len(hand))
        batch_prefilter_break = os.environ.get(
            "DANKS_APPROX_BATCH_BREAK_PENALTY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        break_proxy_cache: dict[int, float] = (
            self._batch_break_group_penalties(actions, before_parts)
            if batch_prefilter_break
            else {}
        )
        proxy_cache: dict[int, tuple[int, float]] = {}
        prefilter_score_cache: dict[int, float] = {}
        current_control_cache: dict[int, float] = {}
        lead_action_cache: dict[int, float] = {}
        spend_penalty_cache: dict[int, float] = {}
        escape_penalty_cache: dict[int, float] = {}
        rank_score_cache: dict[int, float] = {}
        natural_lead_cards: set[tuple[str, ...]] = set()
        if normalize_kind(ctx.current_kind) == "Lead":
            for partition in before_parts:
                for group in partition.groups:
                    kind = normalize_kind(group.kind)
                    if kind in {"Bomb", "StraightFlush", "FourKings"}:
                        continue
                    if group.cards:
                        natural_lead_cards.add(tuple(sorted(group.cards, key=card_sort_key)))

        def add(action: ActionCandidate, *, must: bool = False) -> None:
            keep.setdefault(action.index, action)
            if must:
                essential.add(action.index)

        def proxy(action: ActionCandidate) -> tuple[int, float]:
            cached_proxy = proxy_cache.get(action.index)
            if cached_proxy is not None:
                return cached_proxy
            if action.kind == "PASS":
                cached_proxy = (before_steps, 0.0)
            else:
                cached_break = break_proxy_cache.get(action.index)
                if cached_break is None:
                    cached_break = float(break_group_penalty(action, before_parts, self.break_profile))
                    break_proxy_cache[action.index] = cached_break
                # Playing a whole current group usually reduces one hand. Splitting a
                # valuable group pays an approximate hand-count debt before exact
                # ranking is run on the retained candidates.
                break_debt = min(4, int(cached_break / 2.5))
                steps = max(0, int(before_steps) - 1 + break_debt)
                asset_proxy = -100.0 * cached_break + 2.0 * float(action.size)
                cached_proxy = (steps, asset_proxy)
            proxy_cache[action.index] = cached_proxy
            return cached_proxy

        for action in actions:
            steps, asset = proxy(action)
            static_features = (action_static_features or {}).get(action.index)
            if static_features is None:
                local_score = self._approx_action_score(action, ctx)
                current_control = current_control_value(action, ctx)
                lead_action = lead_action_value(action, ctx)
                spend_penalty = action_spend_penalty(action, ctx)
                escape_penalty = escape_risk_penalty_from_current_control(action, ctx, current_control)
                rank_score = self._rank_score(action, ctx)
            else:
                rank_score, current_control, lead_action, spend_penalty, escape_penalty, overcall_penalty = static_features
                local_score = self._approx_action_score_from_features(
                    action,
                    rank_score,
                    current_control,
                    lead_action,
                    spend_penalty,
                    escape_penalty,
                    overcall_penalty,
                    ctx,
                )
            approx_rows.append((action, steps, asset, local_score))
            prefilter_score_cache[action.index] = self._approx_prefilter_score(
                action,
                (steps, asset),
                ctx,
                local_score=local_score,
            )
            current_control_cache[action.index] = current_control
            lead_action_cache[action.index] = lead_action
            spend_penalty_cache[action.index] = spend_penalty
            escape_penalty_cache[action.index] = escape_penalty
            rank_score_cache[action.index] = rank_score
            if action.kind == "PASS":
                add(action, must=True)
            elif action.kind in {"FourKings", "StraightFlush"}:
                add(action, must=True)
            elif action.kind == "Bomb" and len(action.cards) >= bomb_keep_size:
                add(action, must=True)
            elif current_control_cache[action.index] >= 75.0:
                add(action, must=True)
            elif lead_action_cache[action.index] >= 12.0:
                add(action, must=True)
            elif (
                natural_lead_cards
                and normalize_kind(ctx.current_kind) == "Lead"
                and tuple(sorted(action.cards, key=card_sort_key)) in natural_lead_cards
            ):
                add(action)

        non_pass_steps = [steps for action, steps, _asset, _local_score in approx_rows if action.kind != "PASS"]
        best_steps = min(non_pass_steps) if non_pass_steps else 0
        for action, steps, _asset, _local_score in approx_rows:
            if action.kind != "PASS" and steps <= best_steps + hand_window:
                add(action)

        by_kind: dict[str, list[ActionCandidate]] = {}
        for action in actions:
            by_kind.setdefault(action.kind, []).append(action)
        per_kind = max(2, min(24, max(1, target // max(1, len(by_kind)))))
        for same_kind in by_kind.values():
            same_kind.sort(key=lambda action: prefilter_score_cache[action.index], reverse=True)
            for action in same_kind[:per_kind]:
                add(action)
            diverse_keep = max(2, min(12, per_kind // 2))
            low_spend = sorted(
                same_kind,
                key=lambda action: (
                    spend_penalty_cache[action.index],
                    escape_penalty_cache[action.index],
                    len(action.cards),
                    -rank_score_cache[action.index],
                ),
            )
            for action in low_spend[:diverse_keep]:
                add(action)
            low_rank = sorted(
                same_kind,
                key=lambda action: (
                    rank_score_cache[action.index],
                    spend_penalty_cache[action.index],
                    len(action.cards),
                ),
            )
            for action in low_rank[: max(1, diverse_keep // 2)]:
                add(action)

        ranked = sorted(
            actions,
            key=lambda action: prefilter_score_cache[action.index],
            reverse=True,
        )
        for action in ranked:
            if len(keep) >= target:
                break
            add(action)
        if len(keep) <= target:
            return list(keep.values())
        essential_actions = [keep[idx] for idx in essential if idx in keep]
        essential_actions.sort(key=lambda action: prefilter_score_cache[action.index], reverse=True)
        out: list[ActionCandidate] = []
        seen: set[int] = set()
        for action in essential_actions[:target]:
            out.append(action)
            seen.add(action.index)
        for action in ranked:
            if len(out) >= target:
                break
            if action.index not in keep or action.index in seen:
                continue
            out.append(action)
            seen.add(action.index)
        return out

    def _approx_prefilter_score(
        self,
        action: ActionCandidate,
        proxy: tuple[int, float],
        ctx: RetrievalContext,
        *,
        local_score: float | None = None,
    ) -> float:
        steps, asset = proxy
        # Hand count is the main pruning axis; asset/control terms are escape
        # lanes for structurally valuable or tempo-critical actions.
        lead_bonus = 0.0
        if normalize_kind(ctx.current_kind) == "Lead" and action.kind != "PASS":
            break_estimate = max(0.0, -float(asset) / 100.0)
            kind = normalize_kind(action.kind)
            if kind not in {"Bomb", "StraightFlush", "FourKings"}:
                lead_bonus += max(0.0, 6.0 - break_estimate * 1.5)
            if kind in {"Straight", "StraightPair", "StraightTriple", "TriplePlus"}:
                lead_bonus += 4.0
            elif kind in {"Single", "Pair", "Triple"}:
                lead_bonus += 2.0
        if local_score is None:
            local_score = self._approx_action_score(action, ctx)
        return -18.0 * float(steps) + 0.04 * asset + local_score + lead_bonus

    def _env_int(self, name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(minimum, int(raw))
        except ValueError:
            return default

    def _approx_action_score(self, action: ActionCandidate, ctx: RetrievalContext) -> float:
        if action.kind == "PASS":
            return -1000.0 - self.weights.pass_pressure * pass_pressure_penalty(ctx)
        rank_score = self._rank_score(action, ctx)
        size = float(len(action.cards))
        kind_bonus = {
            "FourKings": 40.0,
            "StraightFlush": 28.0,
            "Bomb": 18.0 + size,
            "StraightTriple": 9.0,
            "StraightPair": 8.0,
            "Straight": 7.0,
            "TriplePair": 6.0,
            "Triple": 4.0,
            "Pair": 3.0,
            "Single": 1.0,
        }.get(action.kind, 0.0)
        current_control = current_control_value(action, ctx)
        lead_action = lead_action_value(action, ctx)
        spend_penalty = action_spend_penalty(action, ctx)
        escape_penalty = escape_risk_penalty_from_current_control(action, ctx, current_control)
        overcall_penalty = teammate_overcall_penalty(action, ctx)
        bomb_weight = self.weights.bomb_spend if action.kind in {"Bomb", "StraightFlush", "FourKings"} else self.weights.control_spend
        return (
            kind_bonus
            + 0.35 * rank_score
            + 0.08 * size
            + self.weights.current_control * current_control
            + self.weights.lead_action * lead_action
            - bomb_weight * spend_penalty
            - self.weights.escape_risk * escape_penalty
            - self.weights.teammate_overcall * overcall_penalty
        )

    def _approx_action_score_from_features(
        self,
        action: ActionCandidate,
        rank_score: float,
        current_control: float,
        lead_action: float,
        spend_penalty: float,
        escape_penalty: float,
        overcall_penalty: float,
        ctx: RetrievalContext,
    ) -> float:
        if action.kind == "PASS":
            return -1000.0 - self.weights.pass_pressure * pass_pressure_penalty(ctx)
        size = float(len(action.cards))
        kind_bonus = {
            "FourKings": 40.0,
            "StraightFlush": 28.0,
            "Bomb": 18.0 + size,
            "StraightTriple": 9.0,
            "StraightPair": 8.0,
            "Straight": 7.0,
            "TriplePair": 6.0,
            "Triple": 4.0,
            "Pair": 3.0,
            "Single": 1.0,
        }.get(action.kind, 0.0)
        bomb_weight = self.weights.bomb_spend if action.kind in {"Bomb", "StraightFlush", "FourKings"} else self.weights.control_spend
        return (
            kind_bonus
            + 0.35 * rank_score
            + 0.08 * size
            + self.weights.current_control * current_control
            + self.weights.lead_action * lead_action
            - bomb_weight * spend_penalty
            - self.weights.escape_risk * escape_penalty
            - self.weights.teammate_overcall * overcall_penalty
        )

    def _batch_action_static_features(
        self,
        actions: list[ActionCandidate],
        ctx: RetrievalContext,
    ) -> dict[int, tuple[float, float, float, float, float, float]]:
        enabled = os.environ.get("DANKS_NATIVE_ACTION_FEATURE_BATCH", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if not enabled or not actions or not native_actor_core.available():
            return {}
        try:
            rows = native_actor_core.batch_action_static_features(
                [(action.kind, action.cards, action.rank or "") for action in actions],
                replace(ctx, current_kind=normalize_kind(ctx.current_kind)),
            )
            if len(rows) != len(actions):
                return {}
            return {action.index: row for action, row in zip(actions, rows)}
        except Exception:
            return {}

    def _batch_break_group_penalties(
        self,
        actions: list[ActionCandidate],
        partitions: list[Partition],
    ) -> dict[int, float]:
        enabled = os.environ.get("DANKS_NATIVE_BREAK_PENALTY_BATCH", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if enabled and actions and partitions and native_actor_core.available():
            try:
                values = native_actor_core.batch_break_group_penalties(
                    actions, partitions, self.break_profile,
                )
                if len(values) == len(actions):
                    return {action.index: value for action, value in zip(actions, values)}
            except Exception:
                pass
        return break_group_penalty_indexed(actions, partitions, self.break_profile)

    def _rank_score(self, action: ActionCandidate, ctx: RetrievalContext) -> float:
        if not action.rank:
            return 0.0
        try:
            return float(rank_strength(action.rank, ctx.current_rank))
        except Exception:
            return 0.0

    def _partition_limit(self, ctx: RetrievalContext) -> int:
        if normalize_kind(ctx.current_kind) == "Lead":
            return self.lead_max_partitions or self.max_partitions
        return self.follow_max_partitions or self.max_partitions

    def _iter_full_partitions(self, hand: Iterable[str], ctx: RetrievalContext, partition_key: tuple | None = None):
        if (
            isinstance(self.partitioner, FullSearchPartitioner)
            and self.partitioner.use_native_all
            and native_cover.available()
        ):
            return self.partitioner.iter_partitions(hand, ctx, partition_key)
        if hasattr(self.partitioner, "iter_partitions_streaming"):
            if isinstance(self.partitioner, FullSearchPartitioner):
                return self.partitioner.iter_partitions_streaming(hand, ctx, partition_key)
            return self.partitioner.iter_partitions_streaming(hand, ctx)
        return iter(self.partitioner.generate(hand, ctx, None))

    def _iter_full_group_covers(self, hand: Iterable[str], ctx: RetrievalContext, partition_key: tuple | None = None):
        if isinstance(self.partitioner, FullSearchPartitioner):
            return self.partitioner.iter_group_covers(hand, ctx, partition_key)
        return ((partition.groups, partition.mode) for partition in self._iter_full_partitions(hand, ctx, partition_key))

    def _context_with_hand(self, context: RetrievalContext | dict[str, Any], hand: list[str]) -> RetrievalContext:
        if isinstance(context, RetrievalContext):
            return context
        payload = dict(context)
        seat = str(int(payload.get("my_seat", 0)))
        known = dict(payload.get("known_hand_cards") or {})
        known.setdefault(seat, hand)
        payload["known_hand_cards"] = known
        return build_context(payload)

    def _stable_search_hand(
        self,
        hand: tuple[str, ...],
        ctx: RetrievalContext,
    ) -> tuple[str, ...]:
        enabled = os.environ.get("DANKS_STABLE_SEARCH_UNIVERSE", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if not enabled:
            return hand
        context_key = (
            ctx.cur_rank,
            bool(ctx.remaining_detail),
            bool(ctx.remaining_detail) and ctx.remaining_detail.get("RJ", 0) == 0,
        )
        cached = self._stable_search_hands.get(ctx.my_seat)
        if cached is not None and cached[0] == context_key:
            root = cached[1]
            root_counts = Counter(root)
            if all(count <= root_counts[card] for card, count in Counter(hand).items()):
                return root
        self._stable_search_hands[ctx.my_seat] = (context_key, hand)
        return hand

    def _primary_retake_kind(self, ctx: RetrievalContext, action: ActionCandidate) -> str:
        if ctx.current_kind and ctx.current_kind != "Lead":
            return ctx.current_kind
        return action.kind

    def _best_partition(
        self,
        partitions: list[Partition],
        ctx: RetrievalContext,
        primary_kind: str,
        scorer: PartitionScorer | None = None,
    ) -> tuple[Partition, float, dict[str, float]]:
        best: tuple[Partition, float, tuple[float, float, float, float]] | None = None
        scorer = scorer or PartitionScorer(ctx, self.weights, self.break_profile)
        pressure_values = scorer._pressure_values_for_primary(primary_kind)
        native_selected_score = os.environ.get(
            "DANKS_NATIVE_SELECTED_PARTITION_SCORE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            native_selected_score
            and partitions
            and native_cover.available()
            and hasattr(native_cover.module(), "best_selected_cover_by_score_entries")
        ):
            try:
                cache_key = id(partitions)
                cached_inputs = self._native_selected_input_cache.get(cache_key)
                if cached_inputs is not None and cached_inputs[0] is partitions:
                    _owner, groups, covers, flat_entries = cached_inputs
                else:
                    groups = []
                    group_id_by_identity: dict[int, int] = {}
                    covers = []
                    for partition in partitions:
                        cover: list[int] = []
                        for group in partition.groups:
                            identity = id(group)
                            group_id = group_id_by_identity.get(identity)
                            if group_id is None:
                                group_id = len(groups)
                                group_id_by_identity[identity] = group_id
                                groups.append(group)
                            cover.append(group_id)
                        covers.append(cover)
                    flat_entries = scorer.flat_cover_score_entries(groups)
                    self._native_selected_input_cache[cache_key] = (
                        partitions,
                        groups,
                        covers,
                        flat_entries,
                    )
                native_best = native_cover.best_selected_cover_by_score_entries(
                    covers,
                    flat_entries,
                    scorer.native_weight_values(),
                    pressure_values,
                )
                if native_best is not None:
                    best_index, total, parts_values, retake_count = native_best
                    parts = scorer.parts_from_values(*parts_values)
                    parts["retake_count_value"] = retake_count
                    return partitions[best_index], total, parts
            except Exception:
                pass
        for partition in partitions:
            total, hand_count_score, card_value_score, retake_score, residue_score = scorer.score_values(partition, primary_kind, pressure_values)
            if best is None or total > best[1]:
                best = (partition, total, (hand_count_score, card_value_score, retake_score, residue_score))
        if best is None:
            empty = Partition(groups=())
            total, parts = scorer.score(empty, primary_kind)
            return empty, total, parts
        return best[0], best[1], scorer.parts_from_values(*best[2])

    def _batch_best_selected_partitions(
        self,
        best_keys: list[tuple[tuple[str, ...], str]],
        partitions_by_after_hand: dict[
            tuple[str, ...], list[Partition] | NativePartitionCovers
        ],
        scorer: PartitionScorer,
    ) -> dict[tuple[tuple[str, ...], str], tuple[Partition, float, dict[str, float]]]:
        enabled = os.environ.get("DANKS_NATIVE_SELECTED_PARTITION_BATCH", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if (
            not enabled
            or not best_keys
            or not native_cover.available()
            or not hasattr(native_cover.module(), "best_selected_covers_by_score_entries_batch")
        ):
            return {}
        out: dict[tuple[tuple[str, ...], str], tuple[Partition, float, dict[str, float]]] = {}
        shared_native_batches: dict[int, tuple[tuple[CardGroup, ...], list[int]]] = {}
        for position, (after_hand, _primary_kind) in enumerate(best_keys):
            partitions = partitions_by_after_hand[after_hand]
            if not isinstance(partitions, NativePartitionCovers):
                continue
            bucket = shared_native_batches.get(id(partitions.groups))
            if bucket is None:
                shared_native_batches[id(partitions.groups)] = (partitions.groups, [position])
            elif bucket[0] is partitions.groups:
                bucket[1].append(position)

        handled_positions: set[int] = set()
        for source_groups, positions in shared_native_batches.values():
            if len(positions) < 2:
                continue
            cover_batches = []
            pressure_values_by_batch = []
            partition_batches = []
            for position in positions:
                after_hand, primary_kind = best_keys[position]
                partitions = partitions_by_after_hand[after_hand]
                if not isinstance(partitions, NativePartitionCovers):
                    break
                cover_batches.append(partitions.covers)
                pressure_values_by_batch.append(scorer._pressure_values_for_primary(primary_kind))
                partition_batches.append(partitions)
            else:
                try:
                    results = native_cover.best_selected_covers_by_score_entries_batch(
                        cover_batches,
                        scorer.flat_cover_score_entries(source_groups),
                        scorer.native_weight_values(),
                        pressure_values_by_batch,
                    )
                except Exception:
                    continue
                for position, partitions, result in zip(positions, partition_batches, results):
                    if result is None:
                        continue
                    best_index, total, parts_values, retake_count = result
                    parts = scorer.parts_from_values(*parts_values)
                    parts["retake_count_value"] = retake_count
                    out[best_keys[position]] = (
                        partitions.materialize_one(best_index),
                        total,
                        parts,
                    )
                    handled_positions.add(position)

        if handled_positions:
            best_keys = [
                best_key
                for position, best_key in enumerate(best_keys)
                if position not in handled_positions
            ]
        if not best_keys:
            return out
        groups: list[CardGroup] = []
        group_id_by_identity: dict[int, int] = {}
        cover_batches: list[list[list[int]]] = []
        pressure_values_by_batch: list[tuple[float, ...]] = []
        partition_batches: list[list[Partition] | NativePartitionCovers] = []
        for after_hand, primary_kind in best_keys:
            partitions = partitions_by_after_hand[after_hand]
            covers: list[list[int]] = []
            if isinstance(partitions, NativePartitionCovers):
                source_groups = partitions.groups
                source_covers = partitions.covers
            else:
                source_groups_list: list[CardGroup] = []
                source_group_id_by_identity: dict[int, int] = {}
                source_covers_list: list[list[int]] = []
                for partition in partitions:
                    source_cover: list[int] = []
                    for group in partition.groups:
                        identity = id(group)
                        source_group_id = source_group_id_by_identity.get(identity)
                        if source_group_id is None:
                            source_group_id = len(source_groups_list)
                            source_group_id_by_identity[identity] = source_group_id
                            source_groups_list.append(group)
                        source_cover.append(source_group_id)
                    source_covers_list.append(source_cover)
                source_groups = tuple(source_groups_list)
                source_covers = tuple(tuple(cover) for cover in source_covers_list)
            local_to_batch: list[int] = []
            for group in source_groups:
                identity = id(group)
                group_id = group_id_by_identity.get(identity)
                if group_id is None:
                    group_id = len(groups)
                    group_id_by_identity[identity] = group_id
                    groups.append(group)
                local_to_batch.append(group_id)
            covers = [
                [local_to_batch[group_id] for group_id in cover]
                for cover in source_covers
            ]
            cover_batches.append(covers)
            pressure_values_by_batch.append(scorer._pressure_values_for_primary(primary_kind))
            partition_batches.append(partitions)
        try:
            results = native_cover.best_selected_covers_by_score_entries_batch(
                cover_batches,
                scorer.flat_cover_score_entries(groups),
                scorer.native_weight_values(),
                pressure_values_by_batch,
            )
        except Exception:
            return {}
        for best_key, partitions, result in zip(best_keys, partition_batches, results):
            if result is None:
                continue
            best_index, total, parts_values, retake_count = result
            parts = scorer.parts_from_values(*parts_values)
            parts["retake_count_value"] = retake_count
            best_partition = (
                partitions.materialize_one(best_index)
                if isinstance(partitions, NativePartitionCovers)
                else partitions[best_index]
            )
            out[best_key] = best_partition, total, parts
        return out

    def _best_partition_streaming(
        self,
        hand: Iterable[str],
        ctx: RetrievalContext,
        primary_kind: str,
        scorer: PartitionScorer | None = None,
        exact_cache_ctx_suffix: tuple | None = None,
        partition_key: tuple | None = None,
    ) -> tuple[Partition, float, dict[str, float]]:
        partition_key = (
            partition_key
            if partition_key is not None
            else self.partitioner._partition_cache_key(hand, ctx) if isinstance(self.partitioner, FullSearchPartitioner) else ()
        )
        cache_key = self._exact_best_cache_key_from_partition_key(partition_key, ctx, primary_kind, exact_cache_ctx_suffix) if partition_key else ()
        cached = self._exact_best_cache_get(cache_key)
        if cached is not None:
            return cached
        scorer = scorer or PartitionScorer(ctx, self.weights, self.break_profile)
        pressure_values = scorer._pressure_values_for_primary(primary_kind)
        native_best = self._best_partition_native_cover_ids(hand, ctx, partition_key, primary_kind, scorer, pressure_values)
        if native_best is not None:
            self._exact_best_cache_put(cache_key, native_best)
            return native_best
        best: tuple[tuple, str, float, tuple[float, float, float, float]] | None = None
        for groups, mode in self._iter_full_group_covers(hand, ctx, partition_key):
            total, hand_count_score, card_value_score, retake_score, residue_score = scorer.score_group_values(groups, primary_kind, pressure_values)
            if best is None or total > best[2]:
                best = (groups, mode, total, (hand_count_score, card_value_score, retake_score, residue_score))
        if best is None:
            empty = Partition(groups=())
            total, parts = scorer.score(empty, primary_kind)
            self._exact_best_cache_put(cache_key, (empty, total, parts))
            return empty, total, parts
        out = (Partition(best[0], best[1]), best[2], scorer.parts_from_values(*best[3]))
        self._exact_best_cache_put(cache_key, out)
        return out

    def _best_partition_native_cover_ids(
        self,
        hand: Iterable[str],
        ctx: RetrievalContext,
        partition_key: tuple,
        primary_kind: str,
        scorer: PartitionScorer,
        pressure_values: tuple[float, ...],
    ) -> tuple[Partition, float, dict[str, float]] | None:
        if not isinstance(self.partitioner, FullSearchPartitioner):
            return None
        native_scored = self._best_partition_native_score_entries(hand, ctx, partition_key, scorer, pressure_values)
        if native_scored is not None:
            return native_scored
        native = self.partitioner.native_cover_ids(hand, ctx, partition_key)
        if native is None:
            return None
        groups_only, covers, mode = native
        group_entries = [scorer.cover_score_entry(group) for group in groups_only]
        best = scorer.best_cover_values(covers, group_entries, primary_kind, pressure_values)
        if best is None:
            return None
        groups = tuple(groups_only[group_id] for group_id in best[0])
        return Partition(groups, mode), best[1], scorer.parts_from_values(*best[2])

    def _best_partition_native_score_entries(
        self,
        hand: Iterable[str],
        ctx: RetrievalContext,
        partition_key: tuple,
        scorer: PartitionScorer,
        pressure_values: tuple[float, ...],
    ) -> tuple[Partition, float, dict[str, float]] | None:
        if not isinstance(self.partitioner, FullSearchPartitioner):
            return None
        native_inputs = self.partitioner.native_score_inputs(hand, ctx, partition_key)
        if native_inputs is None:
            return None
        groups_only, start, native_buckets, mode = native_inputs
        flat_entries = self._flat_score_entries_cached(partition_key, ctx, scorer, groups_only)
        try:
            native_best = self._native_best_cover_by_score_entries(
                start,
                native_buckets,
                flat_entries,
                scorer,
                pressure_values,
            )
            if native_best is None:
                return None
            cover, total, parts_values, retake_count = native_best
        except Exception:
            return None
        groups = tuple(groups_only[group_id] for group_id in cover)
        parts = scorer.parts_from_values(*parts_values)
        if retake_count is not None:
            parts["retake_count_value"] = retake_count
        return Partition(groups, mode), total, parts

    def _native_super_hand_universe(
        self,
        hand: tuple[str, ...],
        ctx: RetrievalContext,
        partition_context_suffix: tuple | None,
    ) -> tuple[tuple, tuple[str, ...], tuple[CardGroup, ...], list[list[list[int]]], str] | None:
        if (
            os.environ.get("DANKS_DISABLE_NATIVE_SUPER_HAND_UNIVERSE", "").strip().lower() in {"1", "true", "yes", "on"}
            or partition_context_suffix is None
            or not isinstance(self.partitioner, FullSearchPartitioner)
            or not native_cover.available()
        ):
            return None
        sorted_hand = tuple(sorted(hand, key=card_sort_key))
        partition_key = (sorted_hand, *partition_context_suffix)
        native_inputs = self.partitioner.native_score_inputs(sorted_hand, ctx, partition_key)
        if native_inputs is None:
            return None
        groups_only, _start, native_buckets, mode = native_inputs
        local_cards = tuple(sorted(Counter(sorted_hand), key=card_sort_key))
        return partition_key, local_cards, groups_only, native_buckets, mode

    def _best_partition_from_native_super_universe(
        self,
        hand: tuple[str, ...],
        ctx: RetrievalContext,
        primary_kind: str,
        scorer: PartitionScorer,
        exact_cache_ctx_suffix: tuple | None,
        partition_key: tuple | None,
        native_super_universe: tuple[tuple, tuple[str, ...], tuple[CardGroup, ...], list[list[list[int]]], str] | None,
    ) -> tuple[Partition, float, dict[str, float]] | None:
        if native_super_universe is None or partition_key is None:
            return None
        cache_key = self._exact_best_cache_key_from_partition_key(partition_key, ctx, primary_kind, exact_cache_ctx_suffix)
        cached = self._exact_best_cache_get(cache_key)
        if cached is not None:
            return cached
        if not hand:
            empty = Partition(groups=())
            total, parts = scorer.score(empty, primary_kind)
            out = (empty, total, parts)
            self._exact_best_cache_put(cache_key, out)
            return out
        base_partition_key, local_cards, groups_only, native_buckets, mode = native_super_universe
        counts = Counter(hand)
        if any(card not in local_cards for card in counts):
            return None
        state = tuple(int(counts.get(card, 0)) for card in local_cards)
        flat_entries = self._flat_score_entries_cached(base_partition_key, ctx, scorer, groups_only)
        pressure_values = scorer._pressure_values_for_primary(primary_kind)
        try:
            native_best = self._native_best_cover_by_score_entries(
                state,
                native_buckets,
                flat_entries,
                scorer,
                pressure_values,
            )
            if native_best is None:
                return None
            cover, total, parts_values, retake_count = native_best
        except Exception:
            return None
        groups = tuple(groups_only[group_id] for group_id in cover)
        parts = scorer.parts_from_values(*parts_values)
        if retake_count is not None:
            parts["retake_count_value"] = retake_count
        out = (Partition(groups, mode), total, parts)
        self._exact_best_cache_put(cache_key, out)
        return out

    def _batch_best_partitions_from_native_super_universe(
        self,
        best_keys: list[tuple[tuple[str, ...], str]],
        ctx: RetrievalContext,
        scorer: PartitionScorer,
        exact_cache_ctx_suffix: tuple | None,
        partition_context_suffix: tuple,
        native_super_universe: tuple[tuple, tuple[str, ...], tuple[CardGroup, ...], list[list[list[int]]], str],
    ) -> dict[tuple[tuple[str, ...], str], tuple[Partition, float, dict[str, float]]]:
        if (
            not best_keys
            or os.environ.get("DANKS_DISABLE_NATIVE_SUPER_HAND_BATCH", "").strip().lower() in {"1", "true", "yes", "on"}
            or not hasattr(native_cover.module(), "best_covers_by_score_entries_with_retake_batch")
        ):
            return {}
        base_partition_key, local_cards, groups_only, native_buckets, mode = native_super_universe
        weights = scorer.native_weight_values()
        flat_entries = self._flat_score_entries_cached(base_partition_key, ctx, scorer, groups_only)
        pending: list[tuple[tuple[tuple[str, ...], str], tuple, tuple[str, ...]]] = []
        states: list[list[int]] = []
        pressure_values_by_state: list[list[float]] = []
        out: dict[tuple[tuple[str, ...], str], tuple[Partition, float, dict[str, float]]] = {}
        for best_key in best_keys:
            after_key, primary_kind = best_key
            partition_key = (after_key, *partition_context_suffix)
            cache_key = self._exact_best_cache_key_from_partition_key(partition_key, ctx, primary_kind, exact_cache_ctx_suffix)
            cached = self._exact_best_cache_get(cache_key)
            if cached is not None:
                out[best_key] = cached
                continue
            if not after_key:
                empty = Partition(groups=())
                total, parts = scorer.score(empty, primary_kind)
                value = (empty, total, parts)
                self._exact_best_cache_put(cache_key, value)
                out[best_key] = value
                continue
            counts = Counter(after_key)
            if any(card not in local_cards for card in counts):
                continue
            state = [int(counts.get(card, 0)) for card in local_cards]
            pending.append((best_key, cache_key, after_key))
            states.append(state)
            pressure_values_by_state.append(list(scorer._pressure_values_for_primary(primary_kind)))
        if not states:
            return out
        try:
            batch = native_cover.best_covers_by_score_entries_with_retake_batch(
                states,
                native_buckets,
                flat_entries,
                weights,
                pressure_values_by_state,
            )
        except Exception:
            return out
        for (best_key, cache_key, _after_key), native_best in zip(pending, batch):
            if native_best is None:
                continue
            cover, total, parts_values, retake_count = native_best
            groups = tuple(groups_only[group_id] for group_id in cover)
            parts = scorer.parts_from_values(*parts_values)
            parts["retake_count_value"] = retake_count
            value = (Partition(groups, mode), total, parts)
            self._exact_best_cache_put(cache_key, value)
            out[best_key] = value
        return out

    def _native_best_cover_by_score_entries(
        self,
        state,
        native_buckets,
        flat_entries,
        scorer: PartitionScorer,
        pressure_values: tuple[float, ...],
    ) -> tuple[list[int], float, tuple[float, float, float, float], float | None] | None:
        weights = scorer.native_weight_values()
        module = native_cover.module()
        use_retake = (
            hasattr(module, "best_cover_by_score_entries_with_retake")
            and os.environ.get("DANKS_DISABLE_NATIVE_RETAKE_SUMMARY", "").strip().lower() not in {"1", "true", "yes", "on"}
        )
        use_dp = os.environ.get("DANKS_ENABLE_NATIVE_SCORE_DP", "").strip().lower() in {"1", "true", "yes", "on"}
        frontier_limit = int(os.environ.get("DANKS_NATIVE_SCORE_DP_FRONTIER_LIMIT", "200000"))
        if use_dp and use_retake and hasattr(module, "best_cover_by_score_entries_dp_with_retake"):
            best_with_retake = native_cover.best_cover_by_score_entries_dp_with_retake(
                state,
                native_buckets,
                flat_entries,
                weights,
                pressure_values,
                frontier_limit=frontier_limit,
            )
            if best_with_retake is not None:
                cover, total, parts_values, retake_count = best_with_retake
                return cover, total, parts_values, retake_count
        if use_retake:
            best_with_retake = native_cover.best_cover_by_score_entries_with_retake(
                state,
                native_buckets,
                flat_entries,
                weights,
                pressure_values,
            )
            if best_with_retake is not None:
                cover, total, parts_values, retake_count = best_with_retake
                return cover, total, parts_values, retake_count
        if use_dp and hasattr(module, "best_cover_by_score_entries_dp"):
            best = native_cover.best_cover_by_score_entries_dp(
                state,
                native_buckets,
                flat_entries,
                weights,
                pressure_values,
                frontier_limit=frontier_limit,
            )
            if best is not None:
                cover, total, parts_values = best
                return cover, total, parts_values, None
        best = native_cover.best_cover_by_score_entries(
            state,
            native_buckets,
            flat_entries,
            weights,
            pressure_values,
        )
        if best is None:
            return None
        cover, total, parts_values = best
        return cover, total, parts_values, None

    def _flat_score_entries_cached(
        self,
        partition_key: tuple,
        ctx: RetrievalContext,
        scorer: PartitionScorer,
        groups_only,
    ) -> list[list[float | int]]:
        if os.environ.get("DANKS_DISABLE_FLAT_SCORE_ENTRY_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return scorer.flat_cover_score_entries(groups_only)
        key = (
            partition_key,
            tuple(int(count) for count in ctx.public_counts),
            normalize_kind(ctx.current_kind),
            ctx.current_rank,
            ctx.current_size,
        )
        self._flat_score_entry_cache_gets += 1
        cached = self._flat_score_entry_cache.get(key)
        if cached is not None:
            self._flat_score_entry_cache_hits += 1
            self._flat_score_entry_cache.move_to_end(key)
            return cached
        entries = scorer.flat_cover_score_entries(groups_only)
        if self.exact_best_cache_size > 0:
            self._flat_score_entry_cache[key] = entries
            self._flat_score_entry_cache.move_to_end(key)
            while len(self._flat_score_entry_cache) > self.exact_best_cache_size:
                self._flat_score_entry_cache.popitem(last=False)
        return entries

    def _exact_best_cache_key(self, hand: Iterable[str], ctx: RetrievalContext, primary_kind: str | None) -> tuple:
        if not isinstance(self.partitioner, FullSearchPartitioner):
            return ()
        return self._exact_best_cache_key_from_partition_key(
            self.partitioner._partition_cache_key(hand, ctx),
            ctx,
            primary_kind,
        )

    def _exact_best_cache_key_from_partition_key(
        self,
        partition_key: tuple,
        ctx: RetrievalContext,
        primary_kind: str | None,
        ctx_suffix: tuple | None = None,
    ) -> tuple:
        suffix = ctx_suffix if ctx_suffix is not None else self._exact_best_cache_context_suffix(ctx)
        return (primary_kind, partition_key, *suffix)

    def _exact_best_cache_context_suffix(self, ctx: RetrievalContext) -> tuple:
        return (
            ctx.my_seat,
            tuple(int(count) for count in ctx.public_counts),
            normalize_kind(ctx.current_kind),
            ctx.current_rank,
            ctx.current_size,
            self.weights,
        )

    def _exact_best_cache_get(self, key: tuple) -> tuple[Partition, float, dict[str, float]] | None:
        if not key or self.exact_best_cache_size <= 0:
            return None
        self._exact_best_cache_gets += 1
        cached = self._exact_best_cache.get(key)
        if cached is None:
            return None
        self._exact_best_cache_hits += 1
        self._exact_best_cache.move_to_end(key)
        return cached

    def _exact_best_cache_put(self, key: tuple, value: tuple[Partition, float, dict[str, float]]) -> None:
        if not key or self.exact_best_cache_size <= 0:
            return
        self._exact_best_cache[key] = value
        self._exact_best_cache.move_to_end(key)
        while len(self._exact_best_cache) > self.exact_best_cache_size:
            self._exact_best_cache.popitem(last=False)

    def _break_penalty_cache_key(self, hand: list[str], ctx: RetrievalContext, actions: list[ActionCandidate]) -> tuple:
        if not isinstance(self.partitioner, FullSearchPartitioner):
            return ()
        return self._break_penalty_cache_key_from_partition_key(
            self.partitioner._partition_cache_key(hand, ctx),
            actions,
        )

    @staticmethod
    def _break_penalty_cache_key_from_partition_key(partition_key: tuple, actions: list[ActionCandidate]) -> tuple:
        return (
            partition_key,
            tuple((action.index, action.kind, action.size, action.cards) for action in actions),
        )

    def _break_penalty_cache_get(self, key: tuple) -> dict[int, float] | None:
        if not key or self.exact_best_cache_size <= 0:
            return None
        self._break_penalty_cache_gets += 1
        cached = self._break_penalty_cache.get(key)
        if cached is None:
            return None
        self._break_penalty_cache_hits += 1
        self._break_penalty_cache.move_to_end(key)
        return cached

    def _break_penalty_cache_put(self, key: tuple, value: dict[int, float]) -> None:
        if not key or self.exact_best_cache_size <= 0:
            return
        self._break_penalty_cache[key] = value
        self._break_penalty_cache.move_to_end(key)
        while len(self._break_penalty_cache) > self.exact_best_cache_size:
            self._break_penalty_cache.popitem(last=False)

    def _full_before_stats(
        self,
        hand: list[str],
        ctx: RetrievalContext,
        actions: list[ActionCandidate],
        primary_kind: str,
        scorer: PartitionScorer | None = None,
        exact_cache_ctx_suffix: tuple | None = None,
    ) -> tuple[Partition, float, dict[str, float], dict[int, float]]:
        best: tuple[tuple, str, float, tuple[float, float, float, float]] | None = None
        partition_key = self.partitioner._partition_cache_key(hand, ctx) if isinstance(self.partitioner, FullSearchPartitioner) else ()
        cache_key = self._exact_best_cache_key_from_partition_key(partition_key, ctx, primary_kind, exact_cache_ctx_suffix) if partition_key else ()
        if hasattr(self.partitioner, "_generate_groups"):
            break_cache_key = self._break_penalty_cache_key_from_partition_key(partition_key, actions) if partition_key else ()
            break_penalties = self._break_penalty_cache_get(break_cache_key)
            if break_penalties is None:
                groups = (
                    self.partitioner._generate_groups(hand, ctx, partition_key)
                    if isinstance(self.partitioner, FullSearchPartitioner)
                    else self.partitioner._generate_groups(hand, ctx)
                )
                break_penalties = break_group_penalty_from_groups(
                    actions,
                    groups,
                    self.break_profile,
                )
                self._break_penalty_cache_put(break_cache_key, break_penalties)
            cached = self._exact_best_cache_get(cache_key)
            if cached is not None:
                return cached[0], cached[1], cached[2], break_penalties
            active_break_actions = None
            group_counter_cache = None
        else:
            active_break_actions, break_penalties = prepare_break_group_penalty_index(actions)
            group_counter_cache: dict[int, Counter[str]] = {}
        scorer = scorer or PartitionScorer(ctx, self.weights, self.break_profile)
        pressure_values = scorer._pressure_values_for_primary(primary_kind)
        if active_break_actions is None:
            native_best = self._best_partition_native_cover_ids(hand, ctx, partition_key, primary_kind, scorer, pressure_values)
            if native_best is not None:
                out = native_best
                self._exact_best_cache_put(cache_key, out)
                return out[0], out[1], out[2], break_penalties
            else:
                for groups, mode in self._iter_full_group_covers(hand, ctx, partition_key):
                    total, hand_count_score, card_value_score, retake_score, residue_score = scorer.score_group_values(groups, primary_kind, pressure_values)
                    if best is None or total > best[2]:
                        best = (groups, mode, total, (hand_count_score, card_value_score, retake_score, residue_score))
        else:
            for partition in self._iter_full_partitions(hand, ctx, partition_key):
                update_break_group_penalty_index(
                    active_break_actions,
                    break_penalties,
                    partition,
                    group_counter_cache,
                    self.break_profile,
                )
                total, hand_count_score, card_value_score, retake_score, residue_score = scorer.score_values(partition, primary_kind, pressure_values)
                if best is None or total > best[2]:
                    best = (partition.groups, partition.mode, total, (hand_count_score, card_value_score, retake_score, residue_score))
        if best is None:
            empty = Partition(groups=())
            total, parts = scorer.score(empty, primary_kind)
            self._exact_best_cache_put(cache_key, (empty, total, parts))
            return empty, total, parts, break_penalties
        out = (Partition(best[0], best[1]), best[2], scorer.parts_from_values(*best[3]))
        self._exact_best_cache_put(cache_key, out)
        return out[0], out[1], out[2], break_penalties

    def _apply_low_break_preference(self, scored: list[ScoredAction]) -> None:
        buckets: dict[tuple[str, int], float] = {}
        for row in scored:
            if row.action.kind == "PASS":
                continue
            key = (row.action.kind, row.action.size)
            penalty = float(row.details.get("break_group_penalty", 0.0))
            if key not in buckets or penalty < buckets[key]:
                buckets[key] = penalty
        for row in scored:
            if row.action.kind == "PASS":
                row.details["low_break_preference_penalty"] = 0.0
                continue
            key = (row.action.kind, row.action.size)
            penalty = max(0.0, float(row.details.get("break_group_penalty", 0.0)) - buckets.get(key, 0.0))
            row.details["low_break_preference_penalty"] = penalty
            if penalty:
                object.__setattr__(row, "score", row.score - self.weights.low_break_preference * penalty)

    def _apply_lead_top5_coverage(self, scored: list[ScoredAction], ctx: RetrievalContext) -> list[ScoredAction]:
        """Keep lead top candidates diverse across common first-play intents.

        Lead decisions are not only "highest scalar value"; humans commonly
        consider several intents: unload low singles, start a pair/small group,
        start a sequence, or play a triple-family shape. We keep the top-scored
        action first, then reserve the rest of the top-5 shortlist for these
        general intents when available.
        """

        if len(scored) <= 5:
            return scored

        selected: list[ScoredAction] = []
        used: set[tuple[str, str | None, tuple[str, ...]]] = set()

        def key(row: ScoredAction) -> tuple[str, str | None, tuple[str, ...]]:
            return normalize_kind(row.action.kind), row.action.rank, tuple(row.action.cards)

        def add(row: ScoredAction | None) -> None:
            if row is None or len(selected) >= 5:
                return
            row_key = key(row)
            if row_key in used:
                return
            selected.append(row)
            used.add(row_key)

        add(scored[0])

        def strength(row: ScoredAction) -> float:
            return rank_strength(row.action.rank, ctx.cur_rank, ctx.remaining_detail) if row.action.rank else 0.0

        def break_cost(row: ScoredAction) -> float:
            return float(row.details.get("break_group_penalty", 0.0)) + float(row.details.get("low_break_preference_penalty", 0.0))

        singles = [row for row in scored if normalize_kind(row.action.kind) == "Single"]
        singles_by_break = sorted(singles, key=lambda row: (break_cost(row), strength(row), -row.score))
        seen_single_ranks: set[str | None] = set()
        for row in singles_by_break:
            if row.action.rank in seen_single_ranks:
                continue
            add(row)
            seen_single_ranks.add(row.action.rank)
            if sum(1 for item in selected if normalize_kind(item.action.kind) == "Single") >= 2:
                break

        pairs = [row for row in scored if normalize_kind(row.action.kind) == "Pair"]
        pairs_by_break = sorted(pairs, key=lambda row: (break_cost(row), strength(row), -row.score))
        seen_pair_ranks: set[str | None] = set()
        for row in pairs_by_break:
            if row.action.rank in seen_pair_ranks:
                continue
            add(row)
            seen_pair_ranks.add(row.action.rank)
            break

        sequences = [row for row in scored if normalize_kind(row.action.kind) in {"Straight", "StraightPair", "StraightTriple"}]
        sequences.sort(key=lambda row: row.score, reverse=True)
        add(sequences[0] if sequences else None)

        triples = [row for row in scored if normalize_kind(row.action.kind) in {"Triple", "TriplePlus"}]
        triples.sort(key=lambda row: row.score, reverse=True)
        add(triples[0] if triples else None)

        for row in scored:
            add(row)
            if len(selected) >= 5:
                break

        selected_keys = {key(row) for row in selected}
        return selected + [row for row in scored if key(row) not in selected_keys]

    def _apply_lead_top5_coverage_topk(self, scored: list[ScoredAction], ctx: RetrievalContext, top_k: int) -> list[ScoredAction]:
        """Return the same lead-covered top-k prefix without sorting every scored action."""

        if len(scored) <= top_k:
            scored.sort(key=lambda row: row.score, reverse=True)
            return self._apply_lead_top5_coverage(scored, ctx)[:top_k]

        selected: list[ScoredAction] = []
        used: set[tuple[str, str | None, tuple[str, ...]]] = set()
        coverage_n = min(5, top_k)

        def key(row: ScoredAction) -> tuple[str, str | None, tuple[str, ...]]:
            return normalize_kind(row.action.kind), row.action.rank, tuple(row.action.cards)

        def add(row: ScoredAction | None) -> None:
            if row is None or len(selected) >= coverage_n:
                return
            row_key = key(row)
            if row_key in used:
                return
            selected.append(row)
            used.add(row_key)

        def strength(row: ScoredAction) -> float:
            return rank_strength(row.action.rank, ctx.cur_rank, ctx.remaining_detail) if row.action.rank else 0.0

        def break_cost(row: ScoredAction) -> float:
            return float(row.details.get("break_group_penalty", 0.0)) + float(row.details.get("low_break_preference_penalty", 0.0))

        add(max(scored, key=lambda row: row.score))

        singles = [row for row in scored if normalize_kind(row.action.kind) == "Single"]
        singles_by_break = sorted(singles, key=lambda row: (break_cost(row), strength(row), -row.score))
        seen_single_ranks: set[str | None] = set()
        for row in singles_by_break:
            if row.action.rank in seen_single_ranks:
                continue
            add(row)
            seen_single_ranks.add(row.action.rank)
            if sum(1 for item in selected if normalize_kind(item.action.kind) == "Single") >= 2:
                break

        pairs = [row for row in scored if normalize_kind(row.action.kind) == "Pair"]
        pairs_by_break = sorted(pairs, key=lambda row: (break_cost(row), strength(row), -row.score))
        seen_pair_ranks: set[str | None] = set()
        for row in pairs_by_break:
            if row.action.rank in seen_pair_ranks:
                continue
            add(row)
            seen_pair_ranks.add(row.action.rank)
            break

        sequence = max(
            (row for row in scored if normalize_kind(row.action.kind) in {"Straight", "StraightPair", "StraightTriple"}),
            key=lambda row: row.score,
            default=None,
        )
        add(sequence)

        triple = max(
            (row for row in scored if normalize_kind(row.action.kind) in {"Triple", "TriplePlus"}),
            key=lambda row: row.score,
            default=None,
        )
        add(triple)

        selected_keys = {key(row) for row in selected}
        fill = heapq.nlargest(top_k + len(selected), scored, key=lambda row: row.score)
        out = list(selected)
        for row in fill:
            if key(row) in selected_keys:
                continue
            out.append(row)
            if len(out) >= top_k:
                break
        return out
