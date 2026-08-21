from __future__ import annotations

import os
from itertools import chain
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from DanKS.training.schema import (
    ACTION_KINDS,
    ACTION_KIND_DIM,
    CARD_DIM,
    CANDIDATE_TACTICAL_SCALAR_OFFSET,
    CARD_MEMORY_DIM,
    GROUP_KIND_DIM,
    HISTORY_EVENT_DIM,
    HISTORY_LENGTH,
    LAST_PLAYER_DIM,
    LEVEL_RANK_DIM,
    MAX_ACTION_CARDS,
    MAX_HAND_CARDS,
    PRESSURE_STATE_DIM,
    RANK_DIM,
    CARD_MEMORY_PLAYED_EXACT_OFFSET,
    CARD_MEMORY_STAT_FIELD_COUNT,
    CARD_MEMORY_STAT_OFFSET,
    FEATURE_VERSION,
    STATE_CARD_MEMORY_OFFSET,
    STATE_LAST_PLAYER_OFFSET,
    STATE_PUBLIC_COUNTS_OFFSET,
    STATE_TRICK_KIND_OFFSET,
    TEAM_BELIEF_PUBLIC_SEAT_DIM,
    TEAM_BELIEF_POS_WEIGHTS,
    TEAM_BELIEF_RELATIVE_SEATS,
    TEAM_BELIEF_SEAT_COUNT,
    TEAM_BELIEF_TARGET_DIM,
    TEAM_TACTICAL_INTERACTION_DIM,
    TRICK_KIND_DIM,
    TRICK_KINDS,
)


def calibrated_team_belief_probabilities(
    logits: torch.Tensor,
    log_pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Undo weighted-BCE prior tilt before probabilities drive tactics."""

    if log_pos_weight is None:
        log_pos_weight = logits.new_tensor(TEAM_BELIEF_POS_WEIGHTS).log()
    return torch.sigmoid(logits - log_pos_weight)


def mlp(dims: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP used by the fixed Phase-1 backbone."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(self.norm(inputs))


class CardTokenEncoder(nn.Module):
    """Encode physical-card count vectors as structured card tokens."""

    def __init__(self, card_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        token_dim = max(16, min(64, (output_dim // 4 // 4) * 4))
        self.card_identity = nn.Parameter(torch.empty(card_dim, token_dim))
        nn.init.normal_(self.card_identity, std=0.02)
        self.count_projection = nn.Linear(1, token_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=token_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        counts: torch.Tensor,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        original_shape = counts.shape[:-1]
        flat_counts = counts.reshape(-1, counts.shape[-1])
        active_indices = torch.nonzero(
            flat_counts.abs().sum(dim=-1) > 0, as_tuple=False,
        ).squeeze(-1)
        counts = flat_counts.index_select(0, active_indices)
        tracing = torch.jit.is_tracing()
        if not tracing and active_indices.numel() == 0:
            return flat_counts.new_zeros((*original_shape, self.output_dim))
        if tracing:
            # A traced graph must also accept a later all-PASS batch.  Keep one
            # private non-empty row so MultiheadAttention never sees batch=0;
            # it is discarded before scattering public outputs.
            sentinel = F.one_hot(
                torch.zeros(1, dtype=torch.long, device=flat_counts.device),
                num_classes=flat_counts.shape[-1],
            ).to(flat_counts.dtype) * 0.5
            counts = torch.cat([counts, sentinel], dim=0)
        weights = counts.abs()
        if max_tokens is not None:
            if max_tokens <= 0 or max_tokens > counts.shape[-1]:
                raise ValueError(
                    f"max_tokens must be in [1,{counts.shape[-1]}], got {max_tokens}"
                )
            # Stable compaction preserves physical-card order. Tokens after the
            # official hand/action limit were padding keys in the dense
            # attention graph and cannot affect any pooled valid-token output.
            positions = torch.arange(
                counts.shape[-1], device=counts.device,
            ).expand_as(counts)
            sort_keys = torch.where(
                weights > 0, positions, positions + counts.shape[-1],
            )
            indices = sort_keys.topk(
                max_tokens, dim=-1, largest=False, sorted=True,
            ).indices
            counts = counts.gather(-1, indices)
            weights = weights.gather(-1, indices)
            identities = self.card_identity[indices]
        else:
            identities = self.card_identity.unsqueeze(0)
        tokens = identities + self.count_projection(counts.unsqueeze(-1))
        padding_mask = weights <= 0
        all_empty = padding_mask.all(dim=-1)
        # Transformer attention requires at least one unmasked key. Keep this
        # branch-free so a traced CPU actor remains valid when PASS moves to a
        # different Top10 slot. The final multiplier still maps empties to zero.
        padding_mask = padding_mask.clone()
        padding_mask[..., 0] = padding_mask[..., 0] & ~all_empty
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=-2) / denom
        nonempty = (weights.sum(dim=-1, keepdim=True) > 0).to(pooled.dtype)
        encoded_rows = self.output(pooled) * nonempty
        if tracing:
            encoded_rows = encoded_rows[:-1]
        output = encoded_rows.new_zeros((flat_counts.shape[0], encoded_rows.shape[-1]))
        output = output.index_copy(0, active_indices, encoded_rows)
        return output.reshape(*original_shape, encoded_rows.shape[-1])


class Phase14Enhancer(nn.Module):
    """Cumulative Phase 1-4 representation on the existing Top10 tensors.

    Phase 1 adds residual refinement. Phase 2 enriches separately encoded
    state/action features and the retrieval-ranked candidate set. Phase 3
    recurrently summarizes the actual public action history. Phase 4 applies a
    structured card-token encoder to the hand and every candidate action.
    """

    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int) -> None:
        super().__init__()
        expected_state_dim = (
            CARD_DIM
            + LEVEL_RANK_DIM
            + TRICK_KIND_DIM
            + RANK_DIM
            + 1
            + 4
            + LAST_PLAYER_DIM
            + RANK_DIM
            + CARD_MEMORY_DIM
            + PRESSURE_STATE_DIM
        )
        if state_dim != expected_state_dim:
            raise ValueError(f"Phase1-4 state layout mismatch: got {state_dim}, expected {expected_state_dim}")
        if candidate_dim <= CARD_DIM + GROUP_KIND_DIM:
            raise ValueError("Phase1-4 candidate layout is missing structured action features")

        self.state_residual = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
        self.candidate_residual = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))

        recurrent_dim = max(16, min(128, hidden_dim // 2))
        self.history_encoder = nn.GRU(HISTORY_EVENT_DIM, recurrent_dim, batch_first=True)
        self.history_projection = nn.Linear(recurrent_dim, hidden_dim)
        self.candidate_context_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        candidate_tail_dim = candidate_dim - CARD_DIM
        self.rich_candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_tail_dim),
            nn.Linear(candidate_tail_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.card_encoder = CardTokenEncoder(CARD_DIM, hidden_dim)
        self.compact_card_tokens = os.environ.get(
            "DANKS_PHASE14_COMPACT_TOKENS", "1",
        ).lower() not in {"0", "false", "no", "off"}
        self.state_structure_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.candidate_structure_projection = nn.Sequential(
            nn.Linear(hidden_dim + GROUP_KIND_DIM, hidden_dim),
            nn.SiLU(),
        )

        self.state_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for fusion in (self.state_fusion, self.candidate_fusion):
            output = fusion[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("Phase fusion must end in Linear")
            nn.init.orthogonal_(output.weight, gain=0.1)
            nn.init.zeros_(output.bias)

    def _history_context(
        self,
        history: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if history is None:
            history = reference.new_zeros((reference.shape[0], HISTORY_LENGTH, HISTORY_EVENT_DIM))
        if history.dim() != 3 or history.shape[1:] != (HISTORY_LENGTH, HISTORY_EVENT_DIM):
            raise ValueError(
                f"history must have shape [B,{HISTORY_LENGTH},{HISTORY_EVENT_DIM}], got {tuple(history.shape)}"
            )
        _, hidden = self.history_encoder(history.to(dtype=reference.dtype))
        return self.history_projection(hidden[-1])

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        z_state: torch.Tensor,
        z_candidate: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (mask > 0).to(z_candidate.dtype)
        batch, slots, _ = candidates.shape

        phase1_state = self.state_residual(z_state)
        phase1_candidate = self.candidate_residual(z_candidate)

        state_context = self._history_context(history, state)
        candidate_sequence, _ = self.candidate_context_gru(phase1_candidate * valid[:, :, None])
        candidate_sequence = candidate_sequence * valid[:, :, None]

        phase2_candidate = self.rich_candidate_encoder(candidates[..., CARD_DIM:])
        compact_hand_tokens = (
            MAX_HAND_CARDS
            if self.compact_card_tokens
            else None
        )
        compact_action_tokens = (
            MAX_ACTION_CARDS
            if self.compact_card_tokens
            else None
        )
        hand_structure = self.state_structure_projection(
            self.card_encoder(
                state[..., :CARD_DIM],
                max_tokens=compact_hand_tokens,
            )
        )
        action_cards = candidates[..., :CARD_DIM].reshape(batch * slots, CARD_DIM)
        action_structure = self.card_encoder(
            action_cards,
            max_tokens=compact_action_tokens,
        ).reshape(batch, slots, -1)
        group_structure = candidates[..., -GROUP_KIND_DIM:]
        action_structure = self.candidate_structure_projection(
            torch.cat([action_structure, group_structure], dim=-1)
        )

        state_delta = self.state_fusion(
            torch.cat([phase1_state, state_context, hand_structure], dim=-1)
        )
        state_context_slots = state_context[:, None, :].expand(-1, slots, -1)
        candidate_delta = self.candidate_fusion(
            torch.cat(
                [
                    phase1_candidate,
                    phase2_candidate,
                    candidate_sequence,
                    action_structure,
                    state_context_slots,
                ],
                dim=-1,
            )
        )
        return z_state + state_delta, (z_candidate + candidate_delta) * valid[:, :, None]


class Top10Selector(nn.Module):
    """Legacy shared state/candidate encoder for one frozen retrieval Top10."""

    requires_history = False

    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 256, candidate_hidden_dim: int = 192) -> None:
        super().__init__()
        self.state_encoder = mlp([state_dim, hidden_dim, hidden_dim])
        self.candidate_encoder = mlp([candidate_dim, candidate_hidden_dim, hidden_dim])
        self.policy_head = mlp([hidden_dim * 2, hidden_dim, 1])
        self.value_head = mlp([hidden_dim * 2, hidden_dim, 1])

    def _encode_public(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_state = self.state_encoder(state)
        batch, slots, _ = candidates.shape
        z_cand = self.candidate_encoder(candidates.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        return z_state, z_cand

    def _policy_from_encoded(
        self,
        z_state: torch.Tensor,
        z_cand: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        slots = z_cand.shape[1]
        z_state_slots = z_state[:, None, :].expand(-1, slots, -1)
        logits = self.policy_head(torch.cat([z_state_slots, z_cand], dim=-1)).squeeze(-1)
        return logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)

    def _value_from_encoded(
        self,
        z_state: torch.Tensor,
        z_cand: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(self._pooled_public(z_state, z_cand, mask)).squeeze(-1)

    @staticmethod
    def _pooled_public(
        z_state: torch.Tensor,
        z_cand: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (z_cand * mask[:, :, None]).sum(dim=1) / denom
        return torch.cat([z_state, pooled], dim=-1)

    def policy_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return masked actor logits from public observations only."""

        z_state, z_cand = self._encode_public(state, candidates, mask, history)
        return self._policy_from_encoded(z_state, z_cand, mask)

    def value_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the public-observation value estimate."""

        z_state, z_cand = self._encode_public(state, candidates, mask, history)
        return self._value_from_encoded(z_state, z_cand, mask)

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked policy logits and value without changing the legacy computation.

        Shapes:
            state: [B, S]
            candidates: [B, K, C]
            mask: [B, K], 1 for valid slots.
        """

        z_state, z_cand = self._encode_public(state, candidates, mask, history)
        return self._policy_from_encoded(z_state, z_cand, mask), self._value_from_encoded(z_state, z_cand, mask)

    def distribution(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        logits = self.policy_forward(state, candidates, mask, history)
        return torch.distributions.Categorical(logits=logits)

    def policy_parameters(self):
        return chain(self.state_encoder.parameters(), self.candidate_encoder.parameters(), self.policy_head.parameters())

    def value_parameters(self):
        return chain(self.state_encoder.parameters(), self.candidate_encoder.parameters(), self.value_head.parameters())

    @property
    def actor_critic_parameters_disjoint(self) -> bool:
        return False


class RelativeTop10Selector(Top10Selector):
    """Backward-compatible selector with a zero-init candidate-relative head.

    The inherited policy and value paths stay intact.  At initialization the
    residual is identically zero, which makes logits/value exactly equal to an
    existing Top10Selector checkpoint.  Training can therefore freeze the old
    PPO model and learn only explicit candidate-set comparisons from humans.
    """

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        hidden_dim: int = 256,
        candidate_hidden_dim: int = 192,
        relative_hidden_dim: int = 96,
    ) -> None:
        super().__init__(state_dim, candidate_dim, hidden_dim, candidate_hidden_dim)
        # state, candidate, masked mean/max, retrieval-top1, base-logit margin
        self.relative_head = mlp([hidden_dim * 5 + 1, relative_hidden_dim * 2, relative_hidden_dim, 1])
        last = self.relative_head[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("relative head must end in Linear")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.relative_hidden_dim = int(relative_hidden_dim)

    def policy_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = mask > 0
        z_state, z_candidate = self._encode_public(state, candidates, mask, history)
        base_logits = self._policy_from_encoded(z_state, z_candidate, mask)
        slots = candidates.shape[1]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(z_candidate.dtype)
        z_mean = (z_candidate * valid[:, :, None]).sum(dim=1) / denom
        lowest = torch.finfo(z_candidate.dtype).min
        z_max = z_candidate.masked_fill(~valid[:, :, None], lowest).max(dim=1).values
        # Retrieval input is always rank ordered; row zero is the frozen Top1.
        z_top1 = z_candidate[:, 0]
        valid_base = base_logits.masked_fill(~valid, lowest)
        best_base = valid_base.max(dim=1, keepdim=True).values
        base_margin = base_logits - best_base
        relation = torch.cat(
            [
                z_state[:, None, :].expand(-1, slots, -1),
                z_candidate,
                z_mean[:, None, :].expand(-1, slots, -1),
                z_max[:, None, :].expand(-1, slots, -1),
                z_top1[:, None, :].expand(-1, slots, -1),
                base_margin[:, :, None],
            ],
            dim=-1,
        )
        residual = self.relative_head(relation).squeeze(-1)
        return (base_logits + residual).masked_fill(~valid, torch.finfo(base_logits.dtype).min)

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.policy_forward(state, candidates, mask, history),
            self.value_forward(state, candidates, mask, history),
        )

    def freeze_base(self) -> None:
        for module in (
            self.state_encoder,
            self.candidate_encoder,
            self.policy_head,
            self.value_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def relative_parameters(self):
        return self.relative_head.parameters()

    def policy_parameters(self):
        return chain(super().policy_parameters(), self.relative_head.parameters())


class DecoupledTop10Selector(Top10Selector):
    """Top10 selector whose public actor and critic have disjoint encoders.

    The value head keeps its historical name so a legacy shared checkpoint can
    be migrated without changing the value output.  Only the two critic
    encoders are new; they are initialized as exact copies of the shared
    encoders.
    """

    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 256, candidate_hidden_dim: int = 192) -> None:
        super().__init__(state_dim, candidate_dim, hidden_dim, candidate_hidden_dim)
        self.critic_state_encoder = mlp([state_dim, hidden_dim, hidden_dim])
        self.critic_candidate_encoder = mlp([candidate_dim, candidate_hidden_dim, hidden_dim])
        self.copy_public_encoders_to_critic()
        self.human_outcome_head = mlp([hidden_dim * 2, hidden_dim, 1])

    def copy_public_encoders_to_critic(self) -> None:
        self.critic_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.critic_candidate_encoder.load_state_dict(self.candidate_encoder.state_dict())

    def value_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.critic_features(state, candidates, mask, history)
        return self.value_head(features).squeeze(-1)

    def critic_features(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_state = self.critic_state_encoder(state)
        batch, slots, _ = candidates.shape
        z_cand = self.critic_candidate_encoder(candidates.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        return self._pooled_public(z_state, z_cand, mask)

    def human_outcome_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.human_outcome_head(self.critic_features(state, candidates, mask, history)).squeeze(-1)

    def human_outcome_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.human_outcome_head.parameters(),
        )

    def critic_joint_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.value_head.parameters(),
            self.human_outcome_head.parameters(),
        )

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.policy_forward(state, candidates, mask, history),
            self.value_forward(state, candidates, mask, history),
        )

    def value_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.value_head.parameters(),
        )

    @property
    def actor_critic_parameters_disjoint(self) -> bool:
        return True


class DecoupledRelativeTop10Selector(RelativeTop10Selector):
    """Relative selector with a separately trainable public critic encoder."""

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        hidden_dim: int = 256,
        candidate_hidden_dim: int = 192,
        relative_hidden_dim: int = 96,
    ) -> None:
        super().__init__(state_dim, candidate_dim, hidden_dim, candidate_hidden_dim, relative_hidden_dim)
        self.critic_state_encoder = mlp([state_dim, hidden_dim, hidden_dim])
        self.critic_candidate_encoder = mlp([candidate_dim, candidate_hidden_dim, hidden_dim])
        self.copy_public_encoders_to_critic()
        self.human_outcome_head = mlp([hidden_dim * 2, hidden_dim, 1])

    def copy_public_encoders_to_critic(self) -> None:
        self.critic_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.critic_candidate_encoder.load_state_dict(self.candidate_encoder.state_dict())

    def value_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.critic_features(state, candidates, mask, history)
        return self.value_head(features).squeeze(-1)

    def critic_features(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_state = self.critic_state_encoder(state)
        batch, slots, _ = candidates.shape
        z_cand = self.critic_candidate_encoder(candidates.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        return self._pooled_public(z_state, z_cand, mask)

    def human_outcome_forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.human_outcome_head(self.critic_features(state, candidates, mask, history)).squeeze(-1)

    def human_outcome_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.human_outcome_head.parameters(),
        )

    def critic_joint_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.value_head.parameters(),
            self.human_outcome_head.parameters(),
        )

    def value_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.value_head.parameters(),
        )

    @property
    def actor_critic_parameters_disjoint(self) -> bool:
        return True


class Phase14Top10Selector(DecoupledTop10Selector):
    """Top10 PPO selector with the cumulative Phase 1-4 backbone.

    Actor and critic keep independent encoders. The actor receives the frozen
    retrieval Top10 and the public history tensor; the critic sees the same
    public information without sharing trainable representation parameters.
    """

    requires_history = True

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        hidden_dim: int = 256,
        candidate_hidden_dim: int = 192,
    ) -> None:
        super().__init__(state_dim, candidate_dim, hidden_dim, candidate_hidden_dim)
        self.phase_encoder = Phase14Enhancer(state_dim, candidate_dim, hidden_dim)
        self.critic_phase_encoder = Phase14Enhancer(state_dim, candidate_dim, hidden_dim)
        self.critic_phase_encoder.load_state_dict(self.phase_encoder.state_dict())

    def _encode_public(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_state = self.state_encoder(state)
        batch, slots, _ = candidates.shape
        z_candidate = self.candidate_encoder(
            candidates.reshape(batch * slots, -1)
        ).reshape(batch, slots, -1)
        return self.phase_encoder(
            state, candidates, mask, z_state, z_candidate, history
        )

    def critic_features(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_state = self.critic_state_encoder(state)
        batch, slots, _ = candidates.shape
        z_candidate = self.critic_candidate_encoder(
            candidates.reshape(batch * slots, -1)
        ).reshape(batch, slots, -1)
        z_state, z_candidate = self.critic_phase_encoder(
            state, candidates, mask, z_state, z_candidate, history
        )
        return self._pooled_public(z_state, z_candidate, mask)

    def policy_parameters(self):
        return chain(super().policy_parameters(), self.phase_encoder.parameters())

    def value_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.critic_phase_encoder.parameters(),
            self.value_head.parameters(),
        )

    def human_outcome_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.critic_phase_encoder.parameters(),
            self.human_outcome_head.parameters(),
        )

    def critic_joint_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.critic_phase_encoder.parameters(),
            self.value_head.parameters(),
            self.human_outcome_head.parameters(),
        )


class EfficientPhaseEnhancer(Phase14Enhancer):
    """Phase encoder with symmetric masked candidate-set attention."""

    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int) -> None:
        super().__init__(state_dim, candidate_dim, hidden_dim)
        del self.candidate_context_gru
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.candidate_set_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True,
        )

    def forward(self, state, candidates, mask, z_state, z_candidate, history):
        valid_bool = mask > 0
        valid = valid_bool.to(z_candidate.dtype)
        batch, slots, _ = candidates.shape
        phase1_state = self.state_residual(z_state)
        phase1_candidate = self.candidate_residual(z_candidate)
        state_context = self._history_context(history, state)
        candidate_context, _ = self.candidate_set_attention(
            phase1_candidate,
            phase1_candidate,
            phase1_candidate,
            key_padding_mask=~valid_bool,
            need_weights=False,
        )
        candidate_context = candidate_context * valid[:, :, None]
        phase2_candidate = self.rich_candidate_encoder(candidates[..., CARD_DIM:])
        hand_structure = self.state_structure_projection(
            self.card_encoder(state[..., :CARD_DIM], max_tokens=MAX_HAND_CARDS)
        )
        action_cards = candidates[..., :CARD_DIM].reshape(batch * slots, CARD_DIM)
        action_structure = self.card_encoder(
            action_cards, max_tokens=MAX_ACTION_CARDS,
        ).reshape(batch, slots, -1)
        action_structure = self.candidate_structure_projection(
            torch.cat([action_structure, candidates[..., -GROUP_KIND_DIM:]], dim=-1)
        )
        state_delta = self.state_fusion(
            torch.cat([phase1_state, state_context, hand_structure], dim=-1)
        )
        candidate_delta = self.candidate_fusion(
            torch.cat(
                [
                    phase1_candidate,
                    phase2_candidate,
                    candidate_context,
                    action_structure,
                    state_context[:, None, :].expand(-1, slots, -1),
                ],
                dim=-1,
            )
        )
        return z_state + state_delta, (z_candidate + candidate_delta) * valid[:, :, None]


class EfficientTeamBeliefTop10Selector(Top10Selector):
    """V12 selector with calibrated response-quality supervision."""

    requires_history = True
    requires_team_belief_labels = True

    def __init__(self, state_dim, candidate_dim, hidden_dim=256, candidate_hidden_dim=192):
        super().__init__(state_dim, candidate_dim, hidden_dim, candidate_hidden_dim)
        self.critic_state_encoder = mlp([state_dim, hidden_dim, hidden_dim])
        self.critic_candidate_encoder = mlp([candidate_dim, candidate_hidden_dim, hidden_dim])
        self.actor_phase_encoder = EfficientPhaseEnhancer(state_dim, candidate_dim, hidden_dim)
        self.critic_phase_encoder = EfficientPhaseEnhancer(state_dim, candidate_dim, hidden_dim)
        self.critic_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.critic_candidate_encoder.load_state_dict(self.candidate_encoder.state_dict())
        self.critic_phase_encoder.load_state_dict(self.actor_phase_encoder.state_dict())
        self.team_public_seat_encoder = mlp([TEAM_BELIEF_PUBLIC_SEAT_DIM, 64, 64])
        self.team_belief_trunk = mlp([hidden_dim * 2 + 64, 128, 64])
        self.team_belief_head = nn.Linear(64, TEAM_BELIEF_TARGET_DIM)
        self.register_buffer(
            "team_belief_log_pos_weight",
            torch.tensor(TEAM_BELIEF_POS_WEIGHTS, dtype=torch.float32).log(),
        )
        self.team_belief_fusion = nn.Linear(
            TEAM_BELIEF_SEAT_COUNT * TEAM_BELIEF_TARGET_DIM
            + TEAM_TACTICAL_INTERACTION_DIM,
            hidden_dim,
        )
        nn.init.zeros_(self.team_belief_fusion.weight)
        nn.init.zeros_(self.team_belief_fusion.bias)

    def set_team_belief_pos_weights(self, values) -> None:
        weights = torch.as_tensor(
            values,
            dtype=self.team_belief_log_pos_weight.dtype,
            device=self.team_belief_log_pos_weight.device,
        )
        if weights.shape != (TEAM_BELIEF_TARGET_DIM,) or not bool(
            torch.isfinite(weights).all() and (weights > 0).all()
        ):
            raise ValueError("team-belief positive weights must be finite and positive")
        self.team_belief_log_pos_weight.copy_(weights.log())

    @staticmethod
    def _public_other_seats(state: torch.Tensor) -> torch.Tensor:
        rows = []
        for seat_index, relative_seat in enumerate(TEAM_BELIEF_RELATIVE_SEATS):
            played_start = (
                STATE_CARD_MEMORY_OFFSET + CARD_MEMORY_PLAYED_EXACT_OFFSET
                + relative_seat * CARD_DIM
            )
            played = state[:, played_start:played_start + CARD_DIM]
            stat_start = (
                STATE_CARD_MEMORY_OFFSET + CARD_MEMORY_STAT_OFFSET
                + relative_seat * CARD_MEMORY_STAT_FIELD_COUNT
            )
            stats = state[:, stat_start:stat_start + CARD_MEMORY_STAT_FIELD_COUNT]
            seat_id = F.one_hot(
                torch.full(
                    (state.shape[0],), seat_index,
                    dtype=torch.long, device=state.device,
                ),
                num_classes=TEAM_BELIEF_SEAT_COUNT,
            ).to(state.dtype)
            rows.append(torch.cat([played, stats, seat_id], dim=-1))
        return torch.stack(rows, dim=1)

    def _actor_with_team_belief(self, state, candidates, mask, history):
        z_state = self.state_encoder(state)
        batch, slots, _ = candidates.shape
        z_candidate = self.candidate_encoder(
            candidates.reshape(batch * slots, -1)
        ).reshape(batch, slots, -1)
        z_state, z_candidate = self.actor_phase_encoder(
            state, candidates, mask, z_state, z_candidate, history,
        )
        seat = self.team_public_seat_encoder(self._public_other_seats(state))
        seat = seat[:, None].expand(-1, slots, -1, -1)
        state4 = z_state[:, None, None].expand(-1, slots, TEAM_BELIEF_SEAT_COUNT, -1)
        cand4 = z_candidate[:, :, None].expand(-1, -1, TEAM_BELIEF_SEAT_COUNT, -1)
        belief = self.team_belief_head(
            self.team_belief_trunk(torch.cat([state4, cand4, seat], dim=-1))
        )
        tactical = self._team_tactical_interactions(state, candidates, belief)
        belief_probabilities = calibrated_team_belief_probabilities(
            belief, self.team_belief_log_pos_weight,
        )
        belief_and_tactics = torch.cat(
            [belief_probabilities.flatten(2), tactical], dim=-1,
        )
        fused = z_candidate + self.team_belief_fusion(belief_and_tactics)
        fused = fused * (mask > 0).to(fused.dtype)[:, :, None]
        return z_state, fused, belief

    def _team_tactical_interactions(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        belief_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compose decision-aligned public tactics from learned seat beliefs."""

        belief = calibrated_team_belief_probabilities(
            belief_logits, self.team_belief_log_pos_weight,
        )
        partner = belief[:, :, 1]
        opponent_any = torch.maximum(
            belief[:, :, (0, 2), 0],
            belief[:, :, (0, 2), 1],
        ).amax(dim=2)
        partner_any = torch.maximum(partner[:, :, 0], partner[:, :, 1])
        action_kinds = candidates[
            :, :, CARD_DIM:CARD_DIM + ACTION_KIND_DIM
        ]
        is_pass = action_kinds[:, :, ACTION_KINDS.index("PASS")]
        is_bomb = action_kinds[
            :, :, (
                ACTION_KINDS.index("Bomb"),
                ACTION_KINDS.index("StraightFlush"),
                ACTION_KINDS.index("FourKings"),
            )
        ].sum(dim=-1).clamp(max=1.0)
        trick_kinds = state[
            :, STATE_TRICK_KIND_OFFSET:STATE_TRICK_KIND_OFFSET + TRICK_KIND_DIM
        ]
        special_kind = (
            0.70 * trick_kinds[:, TRICK_KINDS.index("Straight")]
            + 0.85 * trick_kinds[:, TRICK_KINDS.index("StraightPair")]
            + trick_kinds[:, TRICK_KINDS.index("StraightTriple")]
        )
        last_player = state[
            :, STATE_LAST_PLAYER_OFFSET:STATE_LAST_PLAYER_OFFSET + LAST_PLAYER_DIM
        ]
        holder_is_teammate = last_player[:, 2]
        holder_is_opponent = (last_player[:, 1] + last_player[:, 3]).clamp(max=1.0)
        special_yield = (
            is_pass * special_kind[:, None] * holder_is_teammate[:, None]
        )
        only_bomb_response = candidates[
            :, :, CANDIDATE_TACTICAL_SCALAR_OFFSET
        ]
        bomb_interrupt = (
            is_bomb * holder_is_opponent[:, None] * only_bomb_response
        )
        return torch.stack(
            (
                partner[:, :, 2],
                is_pass * partner_any,
                special_yield * (1.0 - opponent_any),
                bomb_interrupt * (1.0 - partner[:, :, 0]),
            ),
            dim=-1,
        )

    def policy_forward(self, state, candidates, mask, history=None):
        z_state, z_candidate, _ = self._actor_with_team_belief(
            state, candidates, mask, history,
        )
        return self._policy_from_encoded(z_state, z_candidate, mask)

    def critic_features(self, state, candidates, mask, history=None):
        z_state = self.critic_state_encoder(state)
        batch, slots, _ = candidates.shape
        z_candidate = self.critic_candidate_encoder(
            candidates.reshape(batch * slots, -1)
        ).reshape(batch, slots, -1)
        z_state, z_candidate = self.critic_phase_encoder(
            state, candidates, mask, z_state, z_candidate, history,
        )
        return self._pooled_public(z_state, z_candidate, mask)

    def value_forward(self, state, candidates, mask, history=None):
        return self.value_head(
            self.critic_features(state, candidates, mask, history)
        ).squeeze(-1)

    def forward_with_team_belief(self, state, candidates, mask, history=None):
        z_state, z_candidate, belief = self._actor_with_team_belief(
            state, candidates, mask, history,
        )
        logits = self._policy_from_encoded(z_state, z_candidate, mask)
        value = self.value_forward(state, candidates, mask, history)
        return logits, value, belief

    def forward(self, state, candidates, mask, history=None):
        logits, value, _ = self.forward_with_team_belief(
            state, candidates, mask, history,
        )
        return logits, value

    @property
    def actor_critic_parameters_disjoint(self) -> bool:
        return True

    def policy_parameters(self):
        return chain(
            self.state_encoder.parameters(),
            self.candidate_encoder.parameters(),
            self.policy_head.parameters(),
            self.actor_phase_encoder.parameters(),
            self.team_public_seat_encoder.parameters(),
            self.team_belief_trunk.parameters(),
            self.team_belief_head.parameters(),
            self.team_belief_fusion.parameters(),
        )

    def value_parameters(self):
        return chain(
            self.critic_state_encoder.parameters(),
            self.critic_candidate_encoder.parameters(),
            self.critic_phase_encoder.parameters(),
            self.value_head.parameters(),
        )


TOP10_SELECTOR_TYPE = "top10_selector_v3"
RELATIVE_SELECTOR_TYPE = "relative_top10_selector_v4"
DECOUPLED_TOP10_SELECTOR_TYPE = "top10_selector_v3_decoupled_critic_v1"
DECOUPLED_RELATIVE_SELECTOR_TYPE = "relative_top10_selector_v4_decoupled_critic_v1"
PHASE14_SELECTOR_TYPE = "top10_selector_phase14_history_v1"
EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE = FEATURE_VERSION


def _validate_phase14_history_contract(payload: dict[str, Any], model_type: str) -> None:
    if model_type not in {PHASE14_SELECTOR_TYPE, EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE}:
        return
    from DanKS.training.schema import (
        HISTORY_EVENT_DIM,
        HISTORY_LENGTH,
        HISTORY_PROTOCOL,
        TEAM_BELIEF_PROTOCOL,
    )

    history_contract = (
        payload.get("history_protocol"),
        int(payload.get("history_length", -1)),
        int(payload.get("history_event_dim", -1)),
    )
    expected_history_contract = (
        HISTORY_PROTOCOL,
        HISTORY_LENGTH,
        HISTORY_EVENT_DIM,
    )
    if history_contract != expected_history_contract:
        raise RuntimeError(
            "Phase1-4 checkpoint history protocol mismatch: "
            f"checkpoint={history_contract} expected={expected_history_contract}"
        )
    if (
        model_type == EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE
        and payload.get("team_belief_protocol") != TEAM_BELIEF_PROTOCOL
    ):
        raise RuntimeError(
            "team-belief protocol mismatch: "
            f"checkpoint={payload.get('team_belief_protocol')!r} "
            f"expected={TEAM_BELIEF_PROTOCOL!r}"
        )


def selector_model_type(model: Top10Selector) -> str:
    if isinstance(model, EfficientTeamBeliefTop10Selector):
        return EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE
    if isinstance(model, Phase14Top10Selector):
        return PHASE14_SELECTOR_TYPE
    if isinstance(model, DecoupledRelativeTop10Selector):
        return DECOUPLED_RELATIVE_SELECTOR_TYPE
    if isinstance(model, DecoupledTop10Selector):
        return DECOUPLED_TOP10_SELECTOR_TYPE
    if isinstance(model, RelativeTop10Selector):
        return RELATIVE_SELECTOR_TYPE
    return TOP10_SELECTOR_TYPE


def selector_config_from_checkpoint(payload: dict[str, Any]) -> dict[str, int]:
    """Return architecture config saved in a selector checkpoint.

    Older checkpoints only stored train args. Fall back to the historical
    defaults so they remain loadable.
    """

    config = dict(payload.get("model_config") or {})
    args = payload.get("args") or {}
    out = {
        "hidden_dim": int(config.get("hidden_dim", args.get("hidden_dim", 256))),
        "candidate_hidden_dim": int(config.get("candidate_hidden_dim", args.get("candidate_hidden_dim", 192))),
    }
    if str(payload.get("model_type") or "") in {RELATIVE_SELECTOR_TYPE, DECOUPLED_RELATIVE_SELECTOR_TYPE}:
        out["relative_hidden_dim"] = int(config.get("relative_hidden_dim", 96))
    return out


def build_selector_from_checkpoint(payload: dict[str, Any], *, device: torch.device | str | None = None) -> Top10Selector:
    from DanKS.training.schema import TOPK, normalize_candidate_contract

    normalize_candidate_contract(
        payload.get("candidate_capacity", TOPK),
        payload.get("action_support", "structured_topk"),
    )
    state_dim = int(payload.get("state_dim", -1))
    candidate_dim = int(payload.get("candidate_dim", -1))
    config = selector_config_from_checkpoint(payload)
    model_type = str(payload.get("model_type") or TOP10_SELECTOR_TYPE)
    known_model_types = {
        TOP10_SELECTOR_TYPE,
        RELATIVE_SELECTOR_TYPE,
        DECOUPLED_TOP10_SELECTOR_TYPE,
        DECOUPLED_RELATIVE_SELECTOR_TYPE,
        PHASE14_SELECTOR_TYPE,
        EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE,
    }
    if model_type not in known_model_types:
        raise RuntimeError(f"unsupported selector model_type: {model_type!r}")
    _validate_phase14_history_contract(payload, model_type)
    if model_type == EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE:
        model = EfficientTeamBeliefTop10Selector(
            state_dim,
            candidate_dim,
            hidden_dim=config["hidden_dim"],
            candidate_hidden_dim=config["candidate_hidden_dim"],
        )
    elif model_type == PHASE14_SELECTOR_TYPE:
        model = Phase14Top10Selector(
            state_dim,
            candidate_dim,
            hidden_dim=config["hidden_dim"],
            candidate_hidden_dim=config["candidate_hidden_dim"],
        )
    elif model_type in {RELATIVE_SELECTOR_TYPE, DECOUPLED_RELATIVE_SELECTOR_TYPE}:
        relative_hidden_dim = int((payload.get("model_config") or {}).get("relative_hidden_dim", 96))
        model_class = DecoupledRelativeTop10Selector if model_type == DECOUPLED_RELATIVE_SELECTOR_TYPE else RelativeTop10Selector
        model = model_class(
            state_dim,
            candidate_dim,
            hidden_dim=config["hidden_dim"],
            candidate_hidden_dim=config["candidate_hidden_dim"],
            relative_hidden_dim=relative_hidden_dim,
        )
    else:
        model_class = DecoupledTop10Selector if model_type == DECOUPLED_TOP10_SELECTOR_TYPE else Top10Selector
        model = model_class(
            state_dim,
            candidate_dim,
            hidden_dim=config["hidden_dim"],
            candidate_hidden_dim=config["candidate_hidden_dim"],
        )
    if device is not None:
        model = model.to(device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = (
        "human_outcome_head.",
    ) if model_type in {
        DECOUPLED_TOP10_SELECTOR_TYPE,
        DECOUPLED_RELATIVE_SELECTOR_TYPE,
        PHASE14_SELECTOR_TYPE,
        EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE,
    } else ()
    missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing)]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"selector checkpoint is incompatible: missing={missing} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return model


def build_decoupled_selector_from_checkpoint(
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> Top10Selector:
    """Migrate a shared selector to a disjoint public critic with exact outputs."""

    model_type = str(payload.get("model_type") or TOP10_SELECTOR_TYPE)
    if model_type in {
        DECOUPLED_TOP10_SELECTOR_TYPE,
        DECOUPLED_RELATIVE_SELECTOR_TYPE,
        PHASE14_SELECTOR_TYPE,
        EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE,
    }:
        return build_selector_from_checkpoint(payload, device=device)
    config = selector_config_from_checkpoint(payload)
    common = {
        "state_dim": int(payload.get("state_dim", -1)),
        "candidate_dim": int(payload.get("candidate_dim", -1)),
        "hidden_dim": config["hidden_dim"],
        "candidate_hidden_dim": config["candidate_hidden_dim"],
    }
    if model_type == RELATIVE_SELECTOR_TYPE:
        model: Top10Selector = DecoupledRelativeTop10Selector(
            **common,
            relative_hidden_dim=int(config.get("relative_hidden_dim", 96)),
        )
    else:
        model = DecoupledTop10Selector(**common)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("critic_state_encoder.", "critic_candidate_encoder.", "human_outcome_head."))
    ]
    if unexpected or missing:
        raise RuntimeError(f"shared checkpoint is incompatible with decoupled critic: missing={missing} unexpected={unexpected}")
    model.copy_public_encoders_to_critic()
    if device is not None:
        model = model.to(device)
    return model


def build_relative_selector_from_base_checkpoint(
    payload: dict[str, Any],
    *,
    relative_hidden_dim: int = 96,
    device: torch.device | str | None = None,
) -> RelativeTop10Selector:
    config = selector_config_from_checkpoint(payload)
    model = RelativeTop10Selector(
        int(payload.get("state_dim", -1)),
        int(payload.get("candidate_dim", -1)),
        hidden_dim=config["hidden_dim"],
        candidate_hidden_dim=config["candidate_hidden_dim"],
        relative_hidden_dim=relative_hidden_dim,
    )
    base_state = payload["model_state_dict"]
    incompatible = model.load_state_dict(base_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.startswith("relative_head.")]
    if unexpected or missing:
        raise RuntimeError(f"base checkpoint is incompatible: missing={missing} unexpected={unexpected}")
    if device is not None:
        model = model.to(device)
    return model


def reload_selector_from_checkpoint(
    model: Top10Selector | None,
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> Top10Selector:
    """Reuse an architecture-compatible selector while loading new weights."""

    config = selector_config_from_checkpoint(payload)
    wanted_type = str(payload.get("model_type") or TOP10_SELECTOR_TYPE)
    _validate_phase14_history_contract(payload, wanted_type)
    compatible = model is not None and selector_model_type(model) == wanted_type and (
        model.state_encoder[0].in_features == int(payload.get("state_dim", -1))
        and model.candidate_encoder[0].in_features == int(payload.get("candidate_dim", -1))
        and model.state_encoder[0].out_features == config["hidden_dim"]
        and model.candidate_encoder[0].out_features == config["candidate_hidden_dim"]
    )
    if not compatible:
        return build_selector_from_checkpoint(payload, device=device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("human_outcome_head.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"selector checkpoint reload is incompatible: missing={missing} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return model


def pairwise_margin_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    chosen = logits.gather(1, labels[:, None])
    loss = torch.relu(margin - chosen + logits)
    valid = mask > 0
    valid.scatter_(1, labels[:, None], False)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    return loss.masked_select(valid).mean()
