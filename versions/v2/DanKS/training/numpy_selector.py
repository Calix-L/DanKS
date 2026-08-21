from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyTop10Selector:
    """Exact NumPy inference runtime for the released legacy Top10Selector."""

    format_version = "danks_numpy_top10_v1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.state_dim = int(payload["state_dim"])
            self.candidate_dim = int(payload["candidate_dim"])
            self.feature_version = str(payload["feature_version"])
            self.source_checkpoint_sha256 = str(payload["source_checkpoint_sha256"])
            if str(payload["format_version"]) != self.format_version:
                raise RuntimeError("unsupported NumPy selector format")
            self.weights = {
                key: np.asarray(payload[key], dtype=np.float32)
                for key in payload.files
                if "__" in key
            }

    def _linear(self, x: np.ndarray, prefix: str) -> np.ndarray:
        weight = self.weights[f"{prefix}__weight"]
        bias = self.weights[f"{prefix}__bias"]
        return np.matmul(x, weight.T) + bias

    def infer(
        self,
        state: np.ndarray,
        candidates: np.ndarray,
        mask: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        state_batch = np.asarray(state, dtype=np.float32).reshape(1, self.state_dim)
        candidate_batch = np.asarray(candidates, dtype=np.float32).reshape(
            1, -1, self.candidate_dim
        )
        mask_batch = np.asarray(mask, dtype=np.float32).reshape(1, -1)
        slots = candidate_batch.shape[1]

        z_state = np.maximum(self._linear(state_batch, "state_encoder__0"), 0.0)
        z_state = self._linear(z_state, "state_encoder__2")
        flat_candidates = candidate_batch.reshape(slots, self.candidate_dim)
        z_candidate = np.maximum(
            self._linear(flat_candidates, "candidate_encoder__0"), 0.0
        )
        z_candidate = self._linear(z_candidate, "candidate_encoder__2").reshape(
            1, slots, -1
        )

        state_slots = np.broadcast_to(z_state[:, None, :], (1, slots, z_state.shape[1]))
        policy_input = np.concatenate((state_slots, z_candidate), axis=-1)
        policy_hidden = np.maximum(
            self._linear(policy_input, "policy_head__0"), 0.0
        )
        logits = self._linear(policy_hidden, "policy_head__2").reshape(1, slots)
        logits = np.where(mask_batch > 0, logits, np.finfo(np.float32).min)

        denominator = np.maximum(mask_batch.sum(axis=1, keepdims=True), 1.0)
        pooled = (z_candidate * mask_batch[:, :, None]).sum(axis=1) / denominator
        value_input = np.concatenate((z_state, pooled), axis=-1)
        value_hidden = np.maximum(
            self._linear(value_input, "value_head__0"), 0.0
        )
        value = self._linear(value_hidden, "value_head__2").reshape(-1)

        scaled = logits / max(float(temperature), 1.0e-3)
        scaled -= np.max(scaled, axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= np.maximum(
            probabilities.sum(axis=1, keepdims=True), 1.0e-12
        )
        return logits[0], probabilities[0], float(value[0])
