from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from DanKS.training.schema import (
    CANDIDATE_DIM,
    FEATURE_VERSION,
    HISTORY_EVENT_DIM,
    HISTORY_LENGTH,
    STATE_DIM,
    TOPK,
)


class OnnxPhase14Selector:
    """Fixed-shape CPU runtime for the released Phase1-4 Top10 actor."""

    model_type = "top10_selector_phase14_history_v1"
    requires_history = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state_dim = STATE_DIM
        self.candidate_dim = CANDIDATE_DIM
        self.feature_version = FEATURE_VERSION
        self.history_length = HISTORY_LENGTH
        self.history_event_dim = HISTORY_EVENT_DIM

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual_inputs = {
            item.name: tuple(item.shape) for item in self.session.get_inputs()
        }
        expected_inputs = {
            "state": (1, STATE_DIM),
            "candidates": (1, TOPK, CANDIDATE_DIM),
            "mask": (1, TOPK),
            "history": (1, HISTORY_LENGTH, HISTORY_EVENT_DIM),
        }
        if actual_inputs != expected_inputs:
            raise RuntimeError(
                f"Phase1-4 ONNX input contract mismatch: "
                f"actual={actual_inputs} expected={expected_inputs}"
            )

    def infer(
        self,
        state: np.ndarray,
        candidates: np.ndarray,
        mask: np.ndarray,
        history: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        inputs = {
            "state": np.asarray(state, dtype=np.float32).reshape(1, STATE_DIM),
            "candidates": np.asarray(candidates, dtype=np.float32).reshape(
                1, TOPK, CANDIDATE_DIM
            ),
            "mask": np.asarray(mask, dtype=np.float32).reshape(1, TOPK),
            "history": np.asarray(history, dtype=np.float32).reshape(
                1, HISTORY_LENGTH, HISTORY_EVENT_DIM
            ),
        }
        logits_batch, value_batch = self.session.run(
            ["logits", "value"], inputs
        )
        logits = np.asarray(logits_batch[0], dtype=np.float32)
        scaled = logits / max(float(temperature), 1.0e-3)
        scaled -= float(np.max(scaled))
        probabilities = np.exp(scaled)
        probabilities /= max(float(probabilities.sum()), 1.0e-12)
        return logits, probabilities, float(value_batch.reshape(-1)[0])
