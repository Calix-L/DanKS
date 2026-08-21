from __future__ import annotations

import torch
from torch import nn

from DanKS.training.model import mlp


class CandidateRecallReranker(nn.Module):
    """Low-capacity residual reranker for a ragged retrieval candidate pool.

    The last layer is zero initialized, so a new model exactly preserves the
    monotonic retrieval-score ordering before learning from human decisions.
    """

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        *,
        hidden_dim: int = 96,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.state_encoder = mlp([state_dim, hidden_dim, hidden_dim])
        self.candidate_encoder = mlp([candidate_dim, hidden_dim, hidden_dim])
        # state, candidate, masked mean, masked max, retrieval-top1, score/rank
        self.residual_head = mlp([hidden_dim * 5 + 2, hidden_dim * 2, hidden_dim, 1])
        last = self.residual_head[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("residual head must end in Linear")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        retrieval_score: torch.Tensor,
        retrieval_rank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, _ = candidates.shape
        valid = mask > 0
        z_state = self.state_encoder(state)
        z_candidate = self.candidate_encoder(candidates.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(z_candidate.dtype)
        z_mean = (z_candidate * valid[:, :, None]).sum(dim=1) / denom
        lowest = torch.finfo(z_candidate.dtype).min
        z_max = z_candidate.masked_fill(~valid[:, :, None], lowest).max(dim=1).values
        z_max = torch.where(torch.isfinite(z_max), z_max, torch.zeros_like(z_max))
        top1_index = retrieval_rank.masked_fill(~valid, torch.iinfo(retrieval_rank.dtype).max).argmin(dim=1)
        z_top1 = z_candidate[torch.arange(batch, device=state.device), top1_index]
        rank_feature = retrieval_rank.to(z_candidate.dtype) / 192.0
        relation = torch.cat(
            [
                z_state[:, None, :].expand(-1, slots, -1),
                z_candidate,
                z_mean[:, None, :].expand(-1, slots, -1),
                z_max[:, None, :].expand(-1, slots, -1),
                z_top1[:, None, :].expand(-1, slots, -1),
                retrieval_score[:, :, None],
                rank_feature[:, :, None],
            ],
            dim=-1,
        )
        residual = self.residual_head(relation).squeeze(-1)
        logits = retrieval_score + self.residual_scale * residual
        mask_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~valid, mask_value), residual.masked_fill(~valid, 0.0)


def multi_positive_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_rows = positive_mask.any(dim=1) & valid_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return logits.new_tensor(0.0)
    selected = logits[valid_rows]
    positives = positive_mask[valid_rows]
    valid = valid_mask[valid_rows]
    all_logsumexp = torch.logsumexp(selected.masked_fill(~valid, torch.finfo(selected.dtype).min), dim=1)
    pos_logsumexp = torch.logsumexp(selected.masked_fill(~positives, torch.finfo(selected.dtype).min), dim=1)
    losses = all_logsumexp - pos_logsumexp
    if row_weight is None:
        return losses.mean()
    weights = row_weight[valid_rows].to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def recall_boundary_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    boundary_k: int = 10,
    margin: float = 0.25,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = valid_mask.bool()
    positive = positive_mask.bool() & valid
    negative = ~positive_mask.bool() & valid
    valid_rows = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid_rows.any()):
        return logits.new_tensor(0.0)

    lowest = torch.finfo(logits.dtype).min
    positive_max = logits.masked_fill(~positive, lowest).max(dim=1).values
    max_k = min(max(1, int(boundary_k)), int(logits.shape[1]))
    negative_top = torch.topk(logits.masked_fill(~negative, lowest), k=max_k, dim=1).values
    boundary_index = negative.sum(dim=1).clamp(min=1, max=max_k).sub(1).unsqueeze(1)
    boundary = negative_top.gather(1, boundary_index).squeeze(1)
    losses = torch.relu(logits.new_tensor(margin) - positive_max + boundary)[valid_rows]
    if row_weight is None:
        return losses.mean()
    weights = row_weight[valid_rows].to(logits.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)
