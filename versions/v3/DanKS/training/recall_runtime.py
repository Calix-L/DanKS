from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Any, Sequence

import numpy as np
import torch

from DanKS.retrieval.context import RetrievalContext
from DanKS.retrieval.candidate_coverage import tactical_coverage_indices
from DanKS.retrieval.models import ScoredAction
from DanKS.training.candidate_pool import shortlist_indices
from DanKS.training.featurizer import candidate_features, state_features
from DanKS.training.recall_model import CandidateRecallReranker
from DanKS.training.schema import CANDIDATE_DIM, FEATURE_VERSION, STATE_DIM, TOPK
from DanKS.training.accelerator import initialize_device


ORIGINAL_RANK_DETAIL = "danks_original_retrieval_rank"


def featurize_with_original_ranks(
    hand: list[str],
    ctx: RetrievalContext,
    rows: Sequence[ScoredAction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = state_features(hand, ctx)
    candidates = np.zeros((TOPK, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros((TOPK,), dtype=np.float32)
    for slot, row in enumerate(rows[:TOPK]):
        original_rank = int(row.details.get(ORIGINAL_RANK_DETAIL, slot))
        candidates[slot] = candidate_features(
            row,
            min(original_rank, TOPK - 1),
            original_hand_size=len(hand),
            context=ctx,
        )
        mask[slot] = 1.0
    return state, candidates, mask


class CandidateRecallRuntime:
    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "auto") -> None:
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = initialize_device(str(device))
        payload = torch.load(self.checkpoint, map_location="cpu")
        pool_config = payload.get("pool_config") or {}
        if pool_config.get("pool_limit") is not None:
            os.environ["DANKS_APPROX_ACTION_LIMIT"] = str(int(pool_config["pool_limit"]))
        if pool_config.get("retrieval_profile"):
            os.environ["DANKS_BREAK_PROFILE"] = str(pool_config["retrieval_profile"])
        if pool_config.get("break_group_weight") is not None:
            os.environ["DANKS_BREAK_GROUP_WEIGHT"] = str(float(pool_config["break_group_weight"]))
        if pool_config.get("approx_prefilter_version"):
            os.environ["DANKS_APPROX_PREFILTER_VERSION"] = str(pool_config["approx_prefilter_version"])
        if pool_config.get("prefilter_checkpoint"):
            prefilter_path = Path(str(pool_config["prefilter_checkpoint"])).expanduser().resolve()
            expected_hash = pool_config.get("prefilter_checkpoint_sha256")
            if expected_hash:
                digest = hashlib.sha256()
                with prefilter_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != str(expected_hash):
                    raise RuntimeError("prefilter checkpoint SHA256 mismatch")
            os.environ["DANKS_PREFILTER_CHECKPOINT"] = str(prefilter_path)
        selection = pool_config.get("prefilter_selection") or {}
        if selection.get("safe_slots") is not None:
            os.environ["DANKS_PREFILTER_SAFE_SLOTS"] = str(int(selection["safe_slots"]))
        if selection.get("diversity_slots") is not None:
            os.environ["DANKS_PREFILTER_DIVERSITY_SLOTS"] = str(int(selection["diversity_slots"]))
        if selection.get("learned_slots") is not None:
            os.environ["DANKS_PREFILTER_LEARNED_SLOTS"] = str(int(selection["learned_slots"]))
        if pool_config.get("structural_calibration"):
            os.environ["DANKS_STRUCTURAL_CALIBRATION_JSON"] = json.dumps(
                pool_config["structural_calibration"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        self.pool_config = pool_config
        config = payload.get("model_config") or {}
        if int(config.get("state_dim", -1)) != STATE_DIM or int(config.get("candidate_dim", -1)) != CANDIDATE_DIM:
            raise RuntimeError("recall checkpoint feature dimensions do not match")
        if str(payload.get("feature_version")) != FEATURE_VERSION:
            raise RuntimeError("recall checkpoint feature version does not match")
        self.model = CandidateRecallReranker(
            STATE_DIM,
            CANDIDATE_DIM,
            hidden_dim=int(config.get("hidden_dim", 96)),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        shortlist = payload.get("shortlist") or {}
        self.top_k = int(shortlist.get("top_k", TOPK))
        self.lock_retrieval_top = int(shortlist.get("lock_retrieval_top", 3))
        if self.top_k != TOPK:
            raise RuntimeError(f"recall checkpoint top_k={self.top_k}, expected {TOPK}")

    def shortlist(
        self,
        hand: list[str],
        ctx: RetrievalContext,
        ranked: Sequence[ScoredAction],
    ) -> tuple[list[ScoredAction], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not ranked:
            return [], state_features(hand, ctx), np.zeros((TOPK, CANDIDATE_DIM), dtype=np.float32), np.zeros(TOPK, dtype=np.float32), np.empty(0)
        pool_candidates = np.stack(
            [
                candidate_features(
                    row,
                    min(index, TOPK - 1),
                    original_hand_size=len(hand),
                    context=ctx,
                )
                for index, row in enumerate(ranked)
            ]
        ).astype(np.float32)
        state = state_features(hand, ctx)
        scores = np.asarray([np.clip(row.score / 1000.0, -5.0, 5.0) for row in ranked], dtype=np.float32)
        ranks = np.arange(len(ranked), dtype=np.int64)
        with torch.inference_mode():
            logits, _ = self.model(
                torch.from_numpy(state).unsqueeze(0).to(self.device),
                torch.from_numpy(pool_candidates).unsqueeze(0).to(self.device),
                torch.ones((1, len(ranked)), dtype=torch.bool, device=self.device),
                torch.from_numpy(scores).unsqueeze(0).to(self.device),
                torch.from_numpy(ranks).unsqueeze(0).to(self.device),
            )
        logits_np = logits.squeeze(0).float().cpu().numpy()
        preference_indices = shortlist_indices(
            logits_np,
            ranks,
            top_k=len(ranked),
            lock_retrieval_top=self.lock_retrieval_top,
        )
        selected_indices = tactical_coverage_indices(
            [row.action for row in ranked],
            preference_indices,
            top_k=TOPK,
        )
        selected: list[ScoredAction] = []
        candidates = np.zeros((TOPK, CANDIDATE_DIM), dtype=np.float32)
        mask = np.zeros((TOPK,), dtype=np.float32)
        for slot, index in enumerate(selected_indices):
            row = ranked[int(index)]
            row.details[ORIGINAL_RANK_DETAIL] = int(index)
            row.details["recall_logit"] = float(logits_np[int(index)])
            selected.append(row)
            candidates[slot] = pool_candidates[int(index)]
            mask[slot] = 1.0
        return selected, state, candidates, mask, logits_np
