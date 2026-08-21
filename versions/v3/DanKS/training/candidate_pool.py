from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from DanKS.retrieval.structural_calibration import (
    StructuralCalibrationProfile,
    score_from_terms,
)
from DanKS.training.schema import ACTION_KIND_DIM, CARD_DIM, RANK_DIM, TOPK

POOL_VERSION = "ragged_candidate_pool_v1"


@dataclass(frozen=True)
class CandidatePoolSample:
    state: np.ndarray
    candidates: np.ndarray
    retrieval_score: np.ndarray
    retrieval_rank: np.ndarray
    positive_indices: np.ndarray
    sample_id: str
    replay_id: str
    split: str
    current_kind: str
    human_kind: str
    structural_terms: np.ndarray | None = None


class CandidatePoolShard:
    """Read-only view over one ragged candidate-pool NPZ shard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # NpzFile.__getitem__ rereads a ZIP member on every access.  Training
        # samples rows randomly, so cache each array once per shard.
        with np.load(self.path, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        self.offsets = self.data["candidate_offsets"].astype(np.int64, copy=False)
        self.positive_offsets = self.data["positive_offsets"].astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def sample(self, index: int) -> CandidatePoolSample:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        pos_start = int(self.positive_offsets[index])
        pos_end = int(self.positive_offsets[index + 1])
        return CandidatePoolSample(
            state=self.data["state"][index],
            candidates=self.data["candidates"][start:end],
            retrieval_score=self.data["retrieval_score"][start:end],
            retrieval_rank=self.data["retrieval_rank"][start:end],
            positive_indices=self.data["positive_indices"][pos_start:pos_end],
            sample_id=str(self.data["sample_id"][index]),
            replay_id=str(self.data["replay_id"][index]),
            split=str(self.data["split"][index]),
            current_kind=str(self.data["current_kind"][index]),
            human_kind=str(self.data["human_kind"][index]),
            structural_terms=(
                self.data["structural_terms"][start:end]
                if "structural_terms" in self.data
                else None
            ),
        )


def truncate_candidate_pool_sample(
    sample: CandidatePoolSample,
    candidate_limit: int | None,
) -> CandidatePoolSample:
    """Return the retrieval prefix used by a smaller online candidate budget.

    P192 shards retain the exact retrieval rank for every row, so P128/P96
    experiments can reuse the expensive human-state ranking cache.  Positive
    indices are remapped rather than silently treated as positions in the
    truncated tensor.
    """

    if candidate_limit is None:
        return sample
    limit = int(candidate_limit)
    if limit <= 0:
        raise ValueError("candidate_limit must be positive")
    keep = np.flatnonzero(np.asarray(sample.retrieval_rank) < limit)
    if keep.size == len(sample.candidates):
        return sample
    old_to_new = np.full((len(sample.candidates),), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.size, dtype=np.int64)
    positives = np.asarray(sample.positive_indices, dtype=np.int64)
    valid_positives = positives[(positives >= 0) & (positives < len(old_to_new))]
    remapped = old_to_new[valid_positives]
    remapped = remapped[remapped >= 0]
    return replace(
        sample,
        candidates=sample.candidates[keep],
        retrieval_score=sample.retrieval_score[keep],
        retrieval_rank=sample.retrieval_rank[keep],
        positive_indices=remapped,
        structural_terms=(
            sample.structural_terms[keep]
            if sample.structural_terms is not None
            else None
        ),
    )


def rescore_candidate_pool_sample(
    sample: CandidatePoolSample,
    profile: StructuralCalibrationProfile,
) -> CandidatePoolSample:
    """Apply a fixed-partition structural profile and rebuild retrieval rank features."""

    if sample.structural_terms is None:
        raise ValueError("candidate sample lacks structural terms")
    weights = profile.weights_for_kind(sample.current_kind)
    raw_scores = np.asarray(score_from_terms(sample.structural_terms, weights), dtype=np.float32)
    order = np.argsort(-raw_scores, kind="stable")
    old_to_new = np.empty(len(order), dtype=np.int64)
    old_to_new[order] = np.arange(len(order), dtype=np.int64)
    positives = np.asarray(sample.positive_indices, dtype=np.int64)
    valid = positives[(positives >= 0) & (positives < len(order))]
    remapped = np.sort(old_to_new[valid])
    candidates = np.asarray(sample.candidates[order], dtype=np.float32).copy()
    scalar_offset = CARD_DIM + ACTION_KIND_DIM + RANK_DIM
    for rank in range(len(candidates)):
        candidates[rank, scalar_offset] = min(1.0, (rank + 1) / TOPK)
        candidates[rank, scalar_offset + 1] = np.clip(raw_scores[order[rank]] / 1000.0, -5.0, 5.0)
    return replace(
        sample,
        candidates=candidates,
        retrieval_score=np.clip(raw_scores[order] / 1000.0, -5.0, 5.0),
        retrieval_rank=np.arange(len(order), dtype=np.int16),
        positive_indices=remapped,
        structural_terms=sample.structural_terms[order],
    )


def shortlist_indices(
    rerank_scores: Sequence[float] | np.ndarray,
    retrieval_ranks: Sequence[int] | np.ndarray,
    *,
    top_k: int = 10,
    lock_retrieval_top: int = 3,
) -> np.ndarray:
    """Keep retrieval's safest prefix, then fill remaining slots by reranker.

    The returned order retains the original retrieval order for locked rows and
    uses descending reranker score for the learned portion.
    """

    scores = np.asarray(rerank_scores, dtype=np.float64)
    ranks = np.asarray(retrieval_ranks, dtype=np.int64)
    if scores.ndim != 1 or ranks.ndim != 1 or scores.shape != ranks.shape:
        raise ValueError("scores and retrieval_ranks must be same-length vectors")
    if top_k <= 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    target = min(int(top_k), int(scores.size))
    retrieval_order = np.argsort(ranks, kind="stable")
    locked = list(retrieval_order[: min(lock_retrieval_top, target)])
    used = set(locked)
    learned_order = np.argsort(-scores, kind="stable")
    out = list(locked)
    for index in learned_order:
        value = int(index)
        if value in used:
            continue
        out.append(value)
        used.add(value)
        if len(out) >= target:
            break
    return np.asarray(out, dtype=np.int64)


def pool_paths(values: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            out.extend(sorted(path.glob("*.npz")))
        else:
            out.append(path)
    if not out:
        raise FileNotFoundError("no candidate-pool shards found")
    return out
