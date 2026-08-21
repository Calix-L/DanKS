#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import onnxruntime as ort

from DanKS.gdai_adapter.gdai_payload import (
    build_dan_platform_action_list,
    ogd_label_to_plm_tile,
)
from DanKS.retrieval import native_actor_core, native_cover
from DanKS.training.featurizer import history_features
from DanKS.training.onnx_phase14_selector import OnnxPhase14Selector
from DanKS.training.schema import (
    CANDIDATE_DIM,
    FEATURE_VERSION,
    HISTORY_EVENT_DIM,
    HISTORY_EVENT_SEMANTICS,
    HISTORY_LENGTH,
    HISTORY_PROTOCOL,
    STATE_DIM,
    TOPK,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    os.environ.get(
        "CHECKPOINT", ROOT / "models" / "policy.pt"
    )
)
ONNX_MODEL = Path(
    os.environ.get(
        "ONNX_MODEL", ROOT / "models" / "policy.onnx"
    )
)
GDAI_BIN = Path(os.environ.get("GDAI_BIN", ROOT / "bin" / "gdai_linux_local"))
EXPECTED_CHECKPOINT_SHA256 = os.environ.get("EXPECTED_CHECKPOINT_SHA256", "").strip()
EXPECTED_ONNX_SHA256 = os.environ.get("EXPECTED_ONNX_SHA256", "").strip()
EXPECTED_GDAI_SHA256 = os.environ.get("EXPECTED_GDAI_SHA256", "").strip()
EXPECTED_NATIVE_COVER_SHA256 = os.environ.get("EXPECTED_NATIVE_COVER_SHA256", "").strip()
EXPECTED_NATIVE_ACTOR_SHA256 = os.environ.get("EXPECTED_NATIVE_ACTOR_SHA256", "").strip()
EXPECTED_FAST_PATH_ENV = {
    "DANRL_NATIVE_COVER_INPUT_ENCODING": "1",
    "DANRL_NATIVE_BUCKET_CAPSULES": "1",
    "DANRL_NATIVE_RAW_COVER_INPUTS": "1",
    "DANRL_APPROX_BATCH_BREAK_PENALTY": "1",
    "DANRL_NATIVE_BATCH_REUSE_COVER_INPUTS": "1",
    "DANRL_NATIVE_ACTION_FEATURE_BATCH": "1",
    "DANRL_NATIVE_COMPACT_WINDOW_DP": "1",
    "DANRL_NATIVE_COMPACT_WINDOW_MIN_STATES": "1",
    "DANRL_NATIVE_LAZY_COMPACT_WINDOW_DP": "1",
    "DANRL_NATIVE_LAZY_SELECTED_BOUND": "1",
    "DANRL_NATIVE_DEPTH_WINDOW_UPPER_BOUND": "1",
    "DANRL_NATIVE_BATCH_SHARED_TOP_MEMO": "1",
    "DANRL_NATIVE_BATCH_PACKED_TOP_MEMO": "1",
    "DANRL_NATIVE_COMPACT_TOP_DP": "1",
    "DANRL_NATIVE_LAZY_COMPACT_TOP_DP": "1",
    "DANRL_NATIVE_FLAT_DEPTH_MEMO": "1",
    "DANRL_NATIVE_GROUP_SUPERSET_CACHE_SIZE": "32",
    "DANRL_NATIVE_GROUP_MASK_FILTER": "1",
    "DANRL_NATIVE_BREAK_PENALTY_BATCH": "1",
    "DANRL_BOUNDED_PARTITION_CACHE_SIZE": "256",
    "DANRL_BOUNDED_PARTITION_CACHE_BY_TARGET_HAND": "1",
    "DANRL_DEFER_SELECTED_PARTITIONS": "1",
    "DANRL_NATIVE_SELECTED_PARTITION_BATCH": "1",
    "DANRL_NATIVE_SELECT_SMALL_TOP_COVERS": "1",
    "DANRL_NATIVE_BEAM_BATCH": "1",
    "DANRL_NATIVE_BEAM_SUPER_BATCH": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(sys.version_info[:2] == (3, 11), f"Python 3.11 required: {sys.version}")
    require(platform.system() == "Linux", f"Linux required: {platform.system()}")
    require(platform.machine() == "x86_64", f"x86_64 required: {platform.machine()}")
    require(CHECKPOINT.is_file(), f"checkpoint missing: {CHECKPOINT}")
    require(ONNX_MODEL.is_file(), f"ONNX model missing: {ONNX_MODEL}")
    require(
        GDAI_BIN.is_file() and os.access(GDAI_BIN, os.X_OK),
        f"gdai binary missing or not executable: {GDAI_BIN}",
    )
    for name, value in (
        ("EXPECTED_CHECKPOINT_SHA256", EXPECTED_CHECKPOINT_SHA256),
        ("EXPECTED_ONNX_SHA256", EXPECTED_ONNX_SHA256),
        ("EXPECTED_GDAI_SHA256", EXPECTED_GDAI_SHA256),
        ("EXPECTED_NATIVE_COVER_SHA256", EXPECTED_NATIVE_COVER_SHA256),
        ("EXPECTED_NATIVE_ACTOR_SHA256", EXPECTED_NATIVE_ACTOR_SHA256),
    ):
        require(bool(value), f"{name} must be supplied by the authorized runtime")
    require(
        sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint SHA256 mismatch",
    )
    require(
        sha256(ONNX_MODEL) == EXPECTED_ONNX_SHA256,
        "ONNX SHA256 mismatch",
    )
    require(
        sha256(GDAI_BIN) == EXPECTED_GDAI_SHA256,
        "gdai binary SHA256 mismatch",
    )

    native_dir = ROOT / "DanKS" / "retrieval" / "native_cpp"
    require(
        sha256(native_dir / "danrl_cover.cpython-311-x86_64-linux-gnu.so")
        == EXPECTED_NATIVE_COVER_SHA256,
        "native cover SHA256 mismatch",
    )
    require(
        sha256(native_dir / "danrl_actor_core.cpython-311-x86_64-linux-gnu.so")
        == EXPECTED_NATIVE_ACTOR_SHA256,
        "native actor-core SHA256 mismatch",
    )
    require(native_cover.available(), "native cover extension failed to import")
    require(native_actor_core.available(), "native actor-core extension failed to import")
    for name, expected in EXPECTED_FAST_PATH_ENV.items():
        require(
            os.environ.get(name) == expected,
            f"lossless native fast path mismatch: {name}="
            f"{os.environ.get(name)!r}, expected {expected!r}",
        )

    model = OnnxPhase14Selector(ONNX_MODEL)
    require(model.state_dim == STATE_DIM, "selector state_dim mismatch")
    require(model.candidate_dim == CANDIDATE_DIM, "selector candidate_dim mismatch")
    require(model.feature_version == FEATURE_VERSION, "feature_version mismatch")
    history = history_features(
        [
            {"pos": 1, "cards": ["S3"], "finished": False},
            {"pos": 2, "cards": [], "finished": False},
        ],
        my_seat=0,
    )
    require(
        history.shape == (HISTORY_LENGTH, HISTORY_EVENT_DIM),
        "history tensor shape mismatch",
    )
    logits, probabilities, value = model.infer(
        np.zeros(STATE_DIM, dtype=np.float32),
        np.zeros((TOPK, CANDIDATE_DIM), dtype=np.float32),
        np.ones(TOPK, dtype=np.float32),
        history,
    )
    require(tuple(logits.shape) == (TOPK,), "selector logits shape mismatch")
    require(
        tuple(probabilities.shape) == (TOPK,),
        "selector probabilities shape mismatch",
    )
    require(np.isfinite(logits).all(), "selector logits are not finite")
    require(np.isfinite(value), "selector value is not finite")

    tiles = [ogd_label_to_plm_tile(card) for card in ("S3", "H4", "C5")]
    actions = build_dan_platform_action_list(
        {
            "level_value": 2,
            "must_discard": True,
            "self_hand": tiles,
            "play_history": [],
        }
    )
    require(bool(actions), "Dan_platform produced no legal lead actions")
    require(
        not any(action[0] == "PASS" for action in actions),
        "Dan_platform produced PASS on lead",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "onnxruntime": ort.__version__,
                "selector_runtime": "onnx-cpu",
                "checkpoint": str(CHECKPOINT.resolve()),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "onnx_model": str(ONNX_MODEL.resolve()),
                "onnx_sha256": EXPECTED_ONNX_SHA256,
                "gdai_sha256": EXPECTED_GDAI_SHA256,
                "feature_version": FEATURE_VERSION,
                "state_dim": STATE_DIM,
                "candidate_dim": CANDIDATE_DIM,
                "top_k": TOPK,
                "history_protocol": HISTORY_PROTOCOL,
                "history_event_semantics": HISTORY_EVENT_SEMANTICS,
                "history_shape": [HISTORY_LENGTH, HISTORY_EVENT_DIM],
                "native_cover": True,
                "native_actor_core": True,
                "lossless_native_fast_paths": len(EXPECTED_FAST_PATH_ENV),
                "default_policy_workers": int(
                    os.environ.get("DANKS_POLICY_WORKERS", "1")
                ),
                "legal_lead_actions": len(actions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
