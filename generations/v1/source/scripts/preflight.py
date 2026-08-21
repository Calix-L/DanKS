#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from DanKS.plm_adapter.gdai_payload import build_dan_platform_action_list, ogd_label_to_plm_tile
from DanKS.retrieval import native_actor_core, native_cover
from DanKS.training.numpy_selector import NumpyTop10Selector
from DanKS.training.schema import CANDIDATE_DIM, FEATURE_VERSION, STATE_DIM, TOPK


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("CHECKPOINT", ROOT / "models" / "policy.pt"))
GDAI_BIN = Path(os.environ.get("GDAI_BIN", ROOT / "bin" / "gdai_linux_local"))
NUMPY_WEIGHTS = CHECKPOINT.with_suffix(".npz")
EXPECTED_CHECKPOINT_SHA256 = os.environ.get("EXPECTED_CHECKPOINT_SHA256", "").strip()
EXPECTED_GDAI_SHA256 = os.environ.get("EXPECTED_GDAI_SHA256", "").strip()
EXPECTED_NUMPY_SHA256 = os.environ.get("EXPECTED_NUMPY_SHA256", "").strip()
EXPECTED_NATIVE_COVER_SHA256 = os.environ.get("EXPECTED_NATIVE_COVER_SHA256", "").strip()
EXPECTED_NATIVE_ACTOR_SHA256 = os.environ.get("EXPECTED_NATIVE_ACTOR_SHA256", "").strip()


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
    require(sys.version_info[:2] == (3, 11), f"Python 3.11 required, found {sys.version}")
    require(platform.system() == "Linux", f"Linux required, found {platform.system()}")
    require(platform.machine() == "x86_64", f"x86_64 required, found {platform.machine()}")
    require(CHECKPOINT.is_file(), f"checkpoint missing: {CHECKPOINT}")
    require(GDAI_BIN.is_file() and os.access(GDAI_BIN, os.X_OK), f"gdai binary missing or not executable: {GDAI_BIN}")
    for name, value in (
        ("EXPECTED_CHECKPOINT_SHA256", EXPECTED_CHECKPOINT_SHA256),
        ("EXPECTED_GDAI_SHA256", EXPECTED_GDAI_SHA256),
        ("EXPECTED_NUMPY_SHA256", EXPECTED_NUMPY_SHA256),
        ("EXPECTED_NATIVE_COVER_SHA256", EXPECTED_NATIVE_COVER_SHA256),
        ("EXPECTED_NATIVE_ACTOR_SHA256", EXPECTED_NATIVE_ACTOR_SHA256),
    ):
        require(bool(value), f"{name} must be supplied by the authorized runtime")
    require(sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")
    require(sha256(GDAI_BIN) == EXPECTED_GDAI_SHA256, "gdai binary SHA256 mismatch")
    require(NUMPY_WEIGHTS.is_file(), f"NumPy selector weights missing: {NUMPY_WEIGHTS}")
    require(sha256(NUMPY_WEIGHTS) == EXPECTED_NUMPY_SHA256, "NumPy selector SHA256 mismatch")
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

    model = NumpyTop10Selector(NUMPY_WEIGHTS)
    require(model.state_dim == STATE_DIM, "selector state_dim mismatch")
    require(model.candidate_dim == CANDIDATE_DIM, "selector candidate_dim mismatch")
    require(model.feature_version == FEATURE_VERSION, "selector feature_version mismatch")
    require(model.source_checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256, "selector source checkpoint mismatch")
    logits, probabilities, value = model.infer(
        np.zeros(STATE_DIM, dtype=np.float32),
        np.zeros((TOPK, CANDIDATE_DIM), dtype=np.float32),
        np.ones(TOPK, dtype=np.float32),
    )
    require(tuple(logits.shape) == (TOPK,), "selector logits shape mismatch")
    require(tuple(probabilities.shape) == (TOPK,), "selector probabilities shape mismatch")
    require(np.isfinite(value), "selector value is not finite")

    tiles = [ogd_label_to_plm_tile(card) for card in ("S3", "H4", "C5")]
    actions = build_dan_platform_action_list(
        {"level_value": 2, "must_discard": True, "self_hand": tiles, "play_history": []}
    )
    require(bool(actions), "Dan_platform produced no legal lead actions")
    require(not any(action[0] == "PASS" for action in actions), "Dan_platform produced PASS on lead")

    result = {
        "ok": True,
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "selector_runtime": "numpy-cpu",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "numpy_weights": str(NUMPY_WEIGHTS.resolve()),
        "numpy_weights_sha256": EXPECTED_NUMPY_SHA256,
        "gdai_sha256": EXPECTED_GDAI_SHA256,
        "feature_version": FEATURE_VERSION,
        "state_dim": STATE_DIM,
        "candidate_dim": CANDIDATE_DIM,
        "top_k": TOPK,
        "native_cover": True,
        "native_actor_core": True,
        "legal_lead_actions": len(actions),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
