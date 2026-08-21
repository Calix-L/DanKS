from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

from .models import ActionCandidate, ScoredAction
from .partitioner import FullSearchPartitioner
from .ranker import StructuralCandidateRanker, normalize_action


@dataclass(frozen=True)
class B5RankTrace:
    legal_action_indices: tuple[int, ...]
    prefilter_action_indices: tuple[int, ...]
    cover_discovered_action_indices: tuple[int, ...]
    top_k_action_indices: tuple[int, ...]
    shadow_oracle_budget: int | None = None
    cover_score_epsilon: float | None = None
    cover_score_gaps: tuple[tuple[int, float], ...] = ()

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "legal_action_indices": list(self.legal_action_indices),
            "prefilter_action_indices": list(self.prefilter_action_indices),
            "cover_discovered_action_indices": list(self.cover_discovered_action_indices),
            "top_k_action_indices": list(self.top_k_action_indices),
            "shadow_oracle_budget": self.shadow_oracle_budget,
            "cover_score_epsilon": self.cover_score_epsilon,
            "cover_score_gaps": [list(item) for item in self.cover_score_gaps],
        }


class B5TraceRanker(StructuralCandidateRanker):
    """Opt-in offline trace around the production Cython ranker.

    The first pass is byte-for-byte the production Top-K call. A hook records
    the exact post-prefilter actions. The second pass disables only the
    approximate prefilter and ranks that already-filtered set without Top-K
    truncation, exposing which actions obtained a structural Cover result.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        shadow_oracle_budget = kwargs.pop("shadow_oracle_budget", None)
        cover_score_epsilon = kwargs.pop("cover_score_epsilon", 0.0)
        super().__init__(*args, **kwargs)
        if (
            shadow_oracle_budget is not None
            and (
                isinstance(shadow_oracle_budget, bool)
                or not isinstance(shadow_oracle_budget, int)
                or shadow_oracle_budget <= 0
            )
        ):
            raise ValueError("shadow_oracle_budget must be a positive integer or None")
        if (
            isinstance(cover_score_epsilon, bool)
            or not isinstance(cover_score_epsilon, (int, float))
            or not math.isfinite(float(cover_score_epsilon))
            or float(cover_score_epsilon) < 0
        ):
            raise ValueError("cover_score_epsilon must be finite and nonnegative")
        self.shadow_oracle_budget = shadow_oracle_budget
        self.cover_score_epsilon = float(cover_score_epsilon)
        self.shadow_ranker = None
        if shadow_oracle_budget is not None:
            self.shadow_ranker = StructuralCandidateRanker(
                partitioner=FullSearchPartitioner(use_native_all=True, cache_size=0),
                weights=self.weights,
                break_profile=self.break_profile,
                max_partitions=shadow_oracle_budget,
                lead_max_partitions=shadow_oracle_budget,
                follow_max_partitions=shadow_oracle_budget,
                exact_best_cache_size=0,
            )
        self._captured_prefilter_actions: tuple[ActionCandidate, ...] | None = None

    def _prefilter_actions_approx(self, *args: Any, **kwargs: Any) -> list[ActionCandidate]:
        actions = super()._prefilter_actions_approx(*args, **kwargs)
        self._captured_prefilter_actions = tuple(actions)
        return actions

    def rank_with_b5_trace(
        self,
        hand_cards: list[str],
        legal_actions: list[dict[str, Any] | ActionCandidate],
        context: Any,
        *,
        top_k: int,
        approximate_top_k: bool = True,
    ) -> tuple[list[ScoredAction], B5RankTrace]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = [normalize_action(action, index) for index, action in enumerate(legal_actions)]
        if not normalized:
            raise ValueError("legal_actions must not be empty")
        self._captured_prefilter_actions = None
        top_rows = super().rank(
            hand_cards,
            legal_actions,
            context,
            top_k=top_k,
            approximate_top_k=approximate_top_k,
        )
        filtered = list(self._captured_prefilter_actions or normalized)
        previous_limit = os.environ.get("DANRL_APPROX_ACTION_LIMIT")
        try:
            os.environ["DANRL_APPROX_ACTION_LIMIT"] = "0"
            cover_rows = super().rank(
                hand_cards,
                filtered,
                context,
                top_k=None,
                approximate_top_k=False,
            )
        finally:
            if previous_limit is None:
                os.environ.pop("DANRL_APPROX_ACTION_LIMIT", None)
            else:
                os.environ["DANRL_APPROX_ACTION_LIMIT"] = previous_limit
        top_indices = tuple(int(row.action.index) for row in top_rows)
        cover_indices = tuple(int(row.action.index) for row in cover_rows)
        score_gaps: tuple[tuple[int, float], ...] = ()
        if self.shadow_ranker is not None:
            previous_limit = os.environ.get("DANRL_APPROX_ACTION_LIMIT")
            try:
                os.environ["DANRL_APPROX_ACTION_LIMIT"] = "0"
                shadow_rows = self.shadow_ranker.rank(
                    hand_cards,
                    filtered,
                    context,
                    top_k=None,
                    approximate_top_k=False,
                )
            finally:
                if previous_limit is None:
                    os.environ.pop("DANRL_APPROX_ACTION_LIMIT", None)
                else:
                    os.environ["DANRL_APPROX_ACTION_LIMIT"] = previous_limit
            bounded_scores = {int(row.action.index): float(row.score) for row in cover_rows}
            shadow_scores = {int(row.action.index): float(row.score) for row in shadow_rows}
            expected = {int(action.index) for action in filtered}
            if set(bounded_scores) != expected or set(shadow_scores) != expected:
                raise RuntimeError("B5 bounded/shadow Cover did not score every prefiltered action")
            gap_by_action = {
                action: shadow_scores[action] - bounded_scores[action]
                for action in expected
            }
            score_gaps = tuple(sorted(gap_by_action.items()))
            # A production Top-K action necessarily passed the live Cover
            # stage.  For all remaining actions, require its bounded search
            # score to be within epsilon of the larger frozen shadow budget.
            cover_set = set(top_indices)
            cover_set.update(
                action
                for action, gap in gap_by_action.items()
                if gap <= self.cover_score_epsilon
            )
            cover_indices = tuple(
                int(action.index) for action in filtered if int(action.index) in cover_set
            )
        trace = B5RankTrace(
            legal_action_indices=tuple(int(action.index) for action in normalized),
            prefilter_action_indices=tuple(int(action.index) for action in filtered),
            cover_discovered_action_indices=cover_indices,
            top_k_action_indices=top_indices,
            shadow_oracle_budget=self.shadow_oracle_budget,
            cover_score_epsilon=(
                self.cover_score_epsilon if self.shadow_ranker is not None else None
            ),
            cover_score_gaps=score_gaps,
        )
        if not set(trace.prefilter_action_indices) <= set(trace.legal_action_indices):
            raise RuntimeError("B5 prefilter trace is not a subset of legal actions")
        if not set(trace.cover_discovered_action_indices) <= set(trace.prefilter_action_indices):
            raise RuntimeError("B5 Cover trace is not a subset of prefilter actions")
        if not set(trace.top_k_action_indices) <= set(trace.cover_discovered_action_indices):
            raise RuntimeError("B5 Top-K trace is not a subset of Cover-discovered actions")
        return top_rows, trace
