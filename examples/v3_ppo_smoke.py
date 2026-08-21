#!/usr/bin/env python3
"""Run one synthetic PPO update through the public DanKS V3 learner."""

from __future__ import annotations

import math
import DanKS
import torch

from DanKS.training.model import EfficientTeamBeliefTop10Selector
from DanKS.training.ppo import PPOConfig, Top10PPOAgent, masked_categorical
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
        hidden_dim=32,
        candidate_hidden_dim=24,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    agent = Top10PPOAgent(
        model,
        optimizer,
        PPOConfig(train_iters=1, target_kl=1.0),
    )

    batch_size = 2
    state = torch.randn(batch_size, STATE_DIM) * 0.01
    candidates = torch.randn(batch_size, TOPK, CANDIDATE_DIM) * 0.01
    mask = torch.zeros(batch_size, TOPK)
    mask[:, :2] = 1
    history = torch.randn(batch_size, HISTORY_LENGTH, HISTORY_EVENT_DIM) * 0.01
    action = torch.tensor([0, 1])
    with torch.no_grad():
        old_logits = model.policy_forward(state, candidates, mask, history)
        old_logp = masked_categorical(old_logits, mask).log_prob(action)

    team_belief_labels = torch.zeros(
        batch_size,
        TOPK,
        TEAM_BELIEF_SEAT_COUNT,
        TEAM_BELIEF_TARGET_DIM,
    )
    team_belief_mask = torch.zeros(batch_size, TOPK, TEAM_BELIEF_SEAT_COUNT)
    team_belief_mask[:, :2] = 1
    before = [parameter.detach().clone() for parameter in model.parameters()]
    info = agent.update(
        {
            "state": state,
            "candidates": candidates,
            "mask": mask,
            "history": history,
            "action": action,
            "logp": old_logp,
            "advantage": torch.tensor([1.0, -1.0]),
            "returns": torch.tensor([0.5, -0.5]),
            "team_belief_labels": team_belief_labels,
            "team_belief_mask": team_belief_mask,
        }
    )

    if info["iters"] != 1.0 or not all(math.isfinite(value) for value in info.values()):
        raise RuntimeError(f"invalid PPO metrics: {info}")
    if not any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    ):
        raise RuntimeError("the PPO update did not change any model parameter")

    print(f"DanKS V3 PPO ready: iters={int(info['iters'])}, loss={info['loss']:.6f}")


if __name__ == "__main__":
    main()
