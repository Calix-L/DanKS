from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from DanKS.training.model import Top10Selector
from DanKS.training.accelerator import ACCELERATOR_TYPES
from DanKS.training.schema import (
    TEAM_BELIEF_TARGET_DIM,
    TEAM_BELIEF_TARGET_NAMES,
)


@dataclass(frozen=True)
class PPOConfig:
    clip_ratio: float = 0.08
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    team_belief_coef: float = 0.05
    team_belief_pos_weights: tuple[float, ...] = (
        1.0, 4.0, 12.0, 1.5, 1.0, 1.0, 6.0,
    )
    max_grad_norm: float = 5.0
    target_kl: float = 0.03
    train_iters: int = 4
    kl_check_interval: int = 1
    dual_clip: float | None = 3.0
    amp: bool = False
    amp_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if len(self.team_belief_pos_weights) != TEAM_BELIEF_TARGET_DIM or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.team_belief_pos_weights
        ):
            raise ValueError(
                "team_belief_pos_weights must match every team-belief target "
                "with finite positive values"
            )


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    mask_value = torch.finfo(logits.dtype).min
    return torch.distributions.Categorical(logits=logits.masked_fill(mask <= 0, mask_value))


def team_belief_classification_metrics(
    belief_logits: torch.Tensor,
    labels: torch.Tensor,
    effective: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return zero-safe per-target diagnostics for masked belief labels."""

    target_metric_names = tuple(
        target.removesuffix("_response")
        for target in TEAM_BELIEF_TARGET_NAMES
    )
    predicted = (belief_logits >= 0).to(belief_logits.dtype) * effective
    positive = (labels >= 0.5).to(belief_logits.dtype) * effective
    correct = (
        (belief_logits >= 0) == (labels >= 0.5)
    ).to(belief_logits.dtype)
    effective_count = effective.sum(dim=(0, 1, 2)).expand(
        belief_logits.shape[-1],
    )
    positive_count = positive.sum(dim=(0, 1, 2))
    predicted_positive_count = predicted.sum(dim=(0, 1, 2))
    true_positive = (predicted * positive).sum(dim=(0, 1, 2))
    correct_count = (correct * effective).sum(dim=(0, 1, 2))
    squared_error_sum = (
        (belief_logits.sigmoid() - labels).square() * effective
    ).sum(dim=(0, 1, 2))
    values = {
        "accuracy": correct_count / effective_count.clamp_min(1.0),
        "positive_recall": true_positive / positive_count.clamp_min(1.0),
        "positive_precision": (
            true_positive / predicted_positive_count.clamp_min(1.0)
        ),
        "predicted_positive_rate": (
            predicted_positive_count / effective_count.clamp_min(1.0)
        ),
        "brier": squared_error_sum / effective_count.clamp_min(1.0),
        "correct_count": correct_count,
        "positive_count": positive_count,
        "predicted_positive_count": predicted_positive_count,
        "true_positive_count": true_positive,
        "effective_count": effective_count,
        "squared_error_sum": squared_error_sum,
    }
    return {
        f"team_belief_{target}_{metric}": value[index]
        for metric, value in values.items()
        for index, target in enumerate(target_metric_names)
    }


@torch.no_grad()
def policy_step(
    model: Top10Selector,
    state: torch.Tensor,
    candidates: torch.Tensor,
    mask: torch.Tensor,
    *,
    history: torch.Tensor | None = None,
    deterministic: bool = False,
) -> dict[str, torch.Tensor]:
    logits, value = model(state, candidates, mask, history)
    pi = masked_categorical(logits, mask)
    action = torch.argmax(logits, dim=1) if deterministic else pi.sample()
    logp = pi.log_prob(action)
    return {"action": action, "logp": logp, "value": value, "logits": logits}


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = float(last_value)
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


class Top10PPOAgent:
    def __init__(self, model: Top10Selector, optimizer: torch.optim.Optimizer, config: PPOConfig | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or PPOConfig()

    def loss(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        old_logp: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
        history: torch.Tensor | None = None,
        team_belief_labels: torch.Tensor | None = None,
        team_belief_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        belief_logits = None
        if team_belief_labels is not None:
            if team_belief_mask is None:
                raise ValueError("team_belief_mask is required with team_belief_labels")
            forward_with_belief = getattr(self.model, "forward_with_team_belief", None)
            if forward_with_belief is None:
                raise ValueError("team-belief labels require a team-belief selector")
            logits, value, belief_logits = forward_with_belief(
                state, candidates, mask, history,
            )
        else:
            if (
                self.config.team_belief_coef > 0
                and bool(getattr(self.model, "requires_team_belief_labels", False))
            ):
                raise ValueError("team-tactics selector requires team-belief rollout labels")
            logits, value = self.model(state, candidates, mask, history)
        pi = masked_categorical(logits, mask)
        logp = pi.log_prob(action.long())
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantage
        surrogate = torch.min(ratio * advantage, clipped)
        if self.config.dual_clip is not None:
            surrogate = torch.where(advantage < 0, torch.max(surrogate, self.config.dual_clip * advantage), surrogate)
        policy_loss = -surrogate.mean()
        value_loss = 0.5 * F.mse_loss(value, returns)
        entropy = pi.entropy().mean()
        team_belief_loss = logits.new_zeros(())
        team_belief_accuracy = logits.new_zeros(())
        target_metric_names = tuple(
            target.removesuffix("_response")
            for target in TEAM_BELIEF_TARGET_NAMES
        )
        target_metrics = {
            f"team_belief_{target}_{metric}": logits.new_zeros(())
            for target in target_metric_names
            for metric in (
                "accuracy",
                "positive_recall",
                "positive_precision",
                "predicted_positive_rate",
                "brier",
                "correct_count",
                "positive_count",
                "predicted_positive_count",
                "true_positive_count",
                "effective_count",
                "squared_error_sum",
            )
        }
        if belief_logits is not None:
            assert team_belief_labels is not None and team_belief_mask is not None
            if belief_logits.shape != team_belief_labels.shape:
                raise ValueError(
                    "team-belief label shape mismatch: "
                    f"{tuple(team_belief_labels.shape)} != {tuple(belief_logits.shape)}"
                )
            effective = team_belief_mask.to(belief_logits.dtype).unsqueeze(-1)
            labels = team_belief_labels.to(belief_logits.dtype)
            pos_weight = belief_logits.new_tensor(
                self.config.team_belief_pos_weights,
            )
            element_loss = F.binary_cross_entropy_with_logits(
                belief_logits,
                labels,
                reduction="none",
                pos_weight=pos_weight,
            )
            denominator = (effective.sum() * belief_logits.shape[-1]).clamp_min(1.0)
            team_belief_loss = (element_loss * effective).sum() / denominator
            with torch.no_grad():
                correct = (
                    (belief_logits >= 0)
                    == (labels >= 0.5)
                ).to(belief_logits.dtype)
                team_belief_accuracy = (correct * effective).sum() / denominator
                target_metrics = team_belief_classification_metrics(
                    belief_logits, labels, effective,
                )
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
            + self.config.team_belief_coef * team_belief_loss
        )

        with torch.no_grad():
            approx_kl = (old_logp - logp).mean()
            clip_rate = ((ratio < 1.0 - self.config.clip_ratio) | (ratio > 1.0 + self.config.clip_ratio)).float().mean()
        info = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "kl": approx_kl.detach(),
            "clip_rate": clip_rate.detach(),
            "team_belief_loss": team_belief_loss.detach(),
            "team_belief_accuracy": team_belief_accuracy.detach(),
            **{key: value.detach() for key, value in target_metrics.items()},
        }
        return loss, info

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        adv = batch["advantage"]
        batch = dict(batch)
        batch["advantage"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.0e-8)

        last_info: dict[str, torch.Tensor] | None = None
        iters_done = 0
        for iteration in range(self.config.train_iters):
            if type(self.optimizer).__module__.startswith("torch_npu.optim"):
                # torch_npu fused optimizers reject set_to_none=True.
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=batch["state"].device.type,
                dtype=self.config.amp_dtype,
                enabled=self.config.amp and batch["state"].device.type in ACCELERATOR_TYPES,
            ):
                loss, info = self.loss(
                    batch["state"],
                    batch["candidates"],
                    batch["mask"],
                    batch["action"],
                    batch["logp"],
                    batch["advantage"],
                    batch["returns"],
                    batch.get("history"),
                    batch.get("team_belief_labels"),
                    batch.get("team_belief_mask"),
                )
            should_check_kl = (
                (iteration + 1) % max(1, self.config.kl_check_interval) == 0
                or iteration + 1 == self.config.train_iters
            )
            if should_check_kl and float(info["kl"].detach().cpu()) > 1.5 * self.config.target_kl:
                last_info = info
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            last_info = info
            iters_done += 1

        assert last_info is not None
        out = {key: float(value.detach().cpu().item()) for key, value in last_info.items()}
        out["iters"] = float(iters_done)
        return out


def tensor_batch(data: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    specs = {
        "state": torch.float32,
        "candidates": torch.float32,
        "mask": torch.float32,
        "history": torch.float32,
        "action": torch.long,
        "logp": torch.float32,
        "advantage": torch.float32,
        "returns": torch.float32,
        "team_belief_labels": torch.float32,
        "team_belief_mask": torch.float32,
    }
    return {
        key: torch.as_tensor(data[key], dtype=dtype, device=device)
        for key, dtype in specs.items()
        if key in data
    }
