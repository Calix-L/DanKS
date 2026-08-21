from __future__ import annotations

import importlib
from typing import Any

import torch


ACCELERATOR_TYPES = frozenset({"cuda", "npu"})


def load_torch_npu(*, required: bool = False) -> Any | None:
    """Register the Ascend backend without making it a CUDA-host dependency."""

    try:
        return importlib.import_module("torch_npu")
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "NPU requested, but torch_npu is not installed. Install the torch/torch_npu "
                "pair matching this host's CANN release."
            ) from exc
        return None


def npu_available() -> bool:
    load_torch_npu(required=False)
    try:
        return bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        return False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if npu_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # With backend auto-loading disabled, torch does not recognize the private
    # ``npu`` device string until torch_npu has registered it.
    requested_type = str(value).split(":", 1)[0].lower()
    if requested_type == "npu":
        load_torch_npu(required=True)
    requested = torch.device(value)
    if requested.type == "npu":
        if not npu_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is false.")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return requested


def initialize_device(value: str | torch.device) -> torch.device:
    """Resolve an accelerator and make its index current for this process."""

    device = resolve_device(str(value))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        load_torch_npu(required=True)
        torch.npu.set_device(device)
    return device


def is_accelerator(device: torch.device | torch.Tensor) -> bool:
    device_type = device.device.type if isinstance(device, torch.Tensor) else device.type
    return device_type in ACCELERATOR_TYPES


def seed_accelerator(device: torch.device, seed: int) -> None:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "npu":
        load_torch_npu(required=True)
        torch.npu.manual_seed_all(seed)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "npu":
        load_torch_npu(required=True)
        return torch.npu.get_device_name(device)
    return str(device)


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Take a stable, backend-neutral snapshot suitable for checkpoints."""

    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
