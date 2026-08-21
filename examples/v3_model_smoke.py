#!/usr/bin/env python3
"""Run a deterministic forward pass through the public DanKS V3 network."""

from __future__ import annotations

import DanKS
import torch

from DanKS.training.model import EfficientTeamBeliefTop10Selector
from DanKS.training.schema import (
    CANDIDATE_DIM,
    HISTORY_EVENT_DIM,
    HISTORY_LENGTH,
    STATE_DIM,
    TEAM_BELIEF_SEAT_COUNT,
    TEAM_BELIEF_TARGET_DIM,
    TOPK,
)


def main() -> None:
    if DanKS.GENERATION != "v3":
        raise RuntimeError("this example requires the danks-v3 package")
    torch.manual_seed(2026)
    model = EfficientTeamBeliefTop10Selector(
        STATE_DIM,
        CANDIDATE_DIM,
        hidden_dim=64,
        candidate_hidden_dim=48,
    ).eval()
    state = torch.zeros(1, STATE_DIM)
    candidates = torch.zeros(1, TOPK, CANDIDATE_DIM)
    mask = torch.zeros(1, TOPK)
    mask[:, 0] = 1
    history = torch.zeros(1, HISTORY_LENGTH, HISTORY_EVENT_DIM)

    with torch.no_grad():
        logits, value, belief = model.forward_with_team_belief(
            state,
            candidates,
            mask,
            history,
        )

    expected_belief = (1, TOPK, TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM)
    if logits.shape != (1, TOPK) or value.shape != (1,) or belief.shape != expected_belief:
        raise RuntimeError(
            f"unexpected output shapes: logits={tuple(logits.shape)}, "
            f"value={tuple(value.shape)}, belief={tuple(belief.shape)}"
        )
    if not all(torch.isfinite(tensor).all() for tensor in (logits, value, belief)):
        raise RuntimeError("model produced non-finite output")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "DanKS V3 model ready: "
        f"logits={tuple(logits.shape)}, value={tuple(value.shape)}, "
        f"belief={tuple(belief.shape)}, parameters={parameters:,}"
    )


if __name__ == "__main__":
    main()
