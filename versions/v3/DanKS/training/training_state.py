from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


TRAINING_STATE_KIND = "top10_ppo_training_state_v1"
TRAINING_STATE_SUFFIX = "_trainer"


def training_state_path(checkpoint: str | Path) -> Path:
    """Return the optimizer sidecar path paired with a model checkpoint."""

    checkpoint = Path(checkpoint)
    return checkpoint.with_name(
        f"{checkpoint.stem}{TRAINING_STATE_SUFFIX}{checkpoint.suffix or '.pt'}"
    )


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def snapshot_optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Take an independent CPU snapshot suitable for atomic serialization."""

    return _to_cpu(optimizer.state_dict())


def zero_optimizer_grad(optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer).__module__.startswith("torch_npu.optim"):
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)


def save_optimizer_training_state(
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    *,
    model_type: str,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
) -> Path:
    """Atomically publish optimizer state before its paired model checkpoint."""

    checkpoint = Path(checkpoint)
    sidecar = training_state_path(checkpoint)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    payload = {
        "kind": TRAINING_STATE_KIND,
        "model_checkpoint": checkpoint.name,
        "model_type": model_type,
        "model_config": dict(model_config),
        "optimizer_type": (
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        "optimizer_config": dict(optimizer_config),
        "optimizer_state_dict": snapshot_optimizer_state_to_cpu(optimizer),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def resolve_training_state_path(
    checkpoint: str | Path,
    checkpoint_payload: dict[str, Any],
) -> tuple[Path, bool]:
    checkpoint = Path(checkpoint)
    declared_name = checkpoint_payload.get("training_state_file")
    if declared_name is None:
        return training_state_path(checkpoint), False
    if not isinstance(declared_name, str) or Path(declared_name).name != declared_name:
        raise ValueError(
            f"invalid training_state_file in checkpoint {checkpoint}: "
            f"{declared_name!r}"
        )
    return checkpoint.with_name(declared_name), True


def load_optimizer_training_state(
    checkpoint: str | Path,
    checkpoint_payload: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    expected_model_type: str,
    expected_model_config: dict[str, Any],
    expected_optimizer_config: dict[str, Any],
) -> Path | None:
    """Restore Adam state, accepting legacy model-only checkpoints."""

    checkpoint = Path(checkpoint)
    sidecar, declared = resolve_training_state_path(checkpoint, checkpoint_payload)
    if not sidecar.is_file():
        if declared:
            raise FileNotFoundError(
                f"checkpoint {checkpoint} declares missing optimizer state {sidecar}"
            )
        return None

    payload = torch.load(
        sidecar,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("kind") != TRAINING_STATE_KIND:
        raise ValueError(
            f"optimizer state kind mismatch in {sidecar}: "
            f"{payload.get('kind')!r}"
        )
    if payload.get("model_checkpoint") != checkpoint.name:
        raise ValueError(
            f"optimizer state/model mismatch: sidecar={sidecar} "
            f"records={payload.get('model_checkpoint')!r} "
            f"expected={checkpoint.name!r}"
        )
    if payload.get("model_type") != expected_model_type:
        raise ValueError(
            f"optimizer model_type mismatch in {sidecar}: "
            f"{payload.get('model_type')!r} != {expected_model_type!r}"
        )
    if payload.get("model_config") != expected_model_config:
        raise ValueError(
            f"optimizer model_config mismatch in {sidecar}: "
            f"{payload.get('model_config')!r} != {expected_model_config!r}"
        )
    if payload.get("optimizer_config") != expected_optimizer_config:
        raise ValueError(
            f"optimizer configuration mismatch in {sidecar}: "
            f"{payload.get('optimizer_config')!r} != "
            f"{expected_optimizer_config!r}"
        )
    state_dict = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"optimizer_state_dict is missing from {sidecar}")
    optimizer.load_state_dict(state_dict)
    zero_optimizer_grad(optimizer)
    return sidecar
