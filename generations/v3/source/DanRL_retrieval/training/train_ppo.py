#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__" and os.environ.get("DANRL_DISABLE_PERSISTENT_PPO", "").lower() not in {"1", "true", "yes"}:
    from DanRL_retrieval.training.persistent_ppo_transport import maybe_run_client

    persistent_returncode = maybe_run_client()
    if persistent_returncode is not None:
        raise SystemExit(persistent_returncode)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


from DanRL_retrieval.training.model import (  # noqa: E402
    EfficientTeamBeliefTop10Selector,
    Phase14Top10Selector,
    Top10Selector,
    build_selector_from_checkpoint,
    reload_selector_from_checkpoint,
    selector_config_from_checkpoint,
    selector_model_type,
)
from DanRL_retrieval.training.ppo import PPOConfig, Top10PPOAgent, tensor_batch  # noqa: E402
from DanRL_retrieval.training.schema import (  # noqa: E402
    ACTION_KINDS, CARD_DIM, CANDIDATE_DIM, CANDIDATE_TACTICAL_SCALAR_OFFSET,
    FEATURE_VERSION, HISTORY_EVENT_DIM, HISTORY_LENGTH,
    HISTORY_EVENT_SEMANTICS, HISTORY_PROTOCOL, STATE_DIM, TOPK, FULL_LEGAL,
    STATE_LAST_PLAYER_OFFSET, STATE_TRICK_KIND_OFFSET, TRICK_KINDS,
    TEAM_BELIEF_PROTOCOL, TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM,
    TEAM_BELIEF_TARGET_NAMES,
    normalize_candidate_contract,
)
from DanRL_retrieval.training.accelerator import device_name, initialize_device, is_accelerator, seed_accelerator  # noqa: E402
from DanRL_retrieval.training.training_state import (  # noqa: E402
    load_optimizer_training_state,
    save_optimizer_training_state,
    training_state_path,
    zero_optimizer_grad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO top10 policy on frozen retrieval-baseline top10 rollout data.")
    parser.add_argument("--rollout", default=str(ROOT / "DanRL_retrieval" / "data" / "ppo_rollout_bc_smoke.npz"))
    parser.add_argument(
        "--rollout-shards",
        nargs="*",
        default=None,
        help="Read multiple shard NPZ files directly and concatenate in memory, skipping a pre-combined rollout.npz.",
    )
    parser.add_argument("--init-mode", choices=("retrieval", "scratch"), default="retrieval", help="Initialize a fresh top10 policy when --init-checkpoint is not set.")
    parser.add_argument(
        "--architecture",
        choices=("legacy", "phase14", "team_belief"),
        default="team_belief",
        help="team_belief selects the V11 calibrated tactical-quality network with pressure state and hidden-hand auxiliary supervision.",
    )
    parser.add_argument("--init-checkpoint", default="", help="Warm-start from a schema-compatible Top10 selector/PPO checkpoint.")
    parser.add_argument("--output", default=str(ROOT / "DanRL_retrieval" / "checkpoints" / "top10_selector_ppo_smoke.pt"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--candidate-hidden-dim", type=int, default=192)
    parser.add_argument("--behavior-pretrain-epochs", type=int, default=1, help="Supervised warmup on rollout behavior actions before PPO; useful for retrieval-baseline rollouts.")
    parser.add_argument("--behavior-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--behavior-pretrain-weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.08)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--train-iters", type=int, default=4)
    parser.add_argument("--kl-check-interval", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--team-belief-coef", type=float, default=0.05)
    parser.add_argument(
        "--team-belief-pos-weights",
        type=float,
        nargs=TEAM_BELIEF_TARGET_DIM,
        metavar=(
            "NON_BOMB", "BOMB", "FINISHING", "LOW_COST",
            "BOMB_PRESERVE", "CONTROL_PRESERVE", "FIVE_CARD",
        ),
        default=(1.0, 4.0, 12.0, 1.5, 1.0, 1.0, 6.0),
        help="Positive-class weights for all hidden-response and quality targets.",
    )
    parser.add_argument("--dual-clip", type=float, default=3.0)
    parser.add_argument("--allow-zero-advantage", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use accelerator autocast during PPO updates.")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--compile-mode", default=None, choices=("default", "reduce-overhead", "max-autotune"), help="Optional torch.compile mode.")
    parser.add_argument("--fused-optimizer", action="store_true", help="Use fused AdamW on CUDA when supported; ignored on NPU.")
    parser.add_argument("--preload-to-device", action="store_true", help="Move the full rollout to the training device and batch with device-side indices.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run policy evaluation every N epochs.")
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Use at most this many fixed samples for policy evaluation; 0 means full rollout.")
    parser.add_argument("--skip-before-eval", action="store_true", help="Skip policy evaluation before training.")
    parser.add_argument("--skip-first-eval", action="store_true", help="Do not force policy evaluation after epoch 1.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Do not force policy evaluation on the final epoch.")
    parser.add_argument(
        "--disable-tactical-resampling",
        action="store_true",
        help="Disable the V11 partner/enemy finishing, special-yield, and bomb-interrupt epoch mix.",
    )
    parser.add_argument(
        "--tactical-max-repeat",
        type=int,
        default=8,
        help="Maximum times one rollout row may appear in a V11 resampled epoch; overflow is redistributed.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


PPO_KEYS = ("state", "candidates", "mask", "history", "action", "logp", "advantage", "returns")
TEAM_BELIEF_KEYS = ("team_belief_labels", "team_belief_mask")
TACTICAL_ORDINARY = 0
TACTICAL_PARTNER_FINISHING = 1
TACTICAL_OPPONENT_FINISHING = 2
TACTICAL_SPECIAL_YIELD = 3
TACTICAL_BOMB_INTERRUPT = 4
TACTICAL_CATEGORY_NAMES = (
    "ordinary",
    "partner_finishing",
    "opponent_finishing",
    "special_yield",
    "bomb_interrupt",
)
TACTICAL_TARGET_SHARES = np.asarray(
    (0.60, 0.10, 0.05, 0.10, 0.15), dtype=np.float64,
)
TACTICAL_MAX_REPEAT = 8


def tactical_row_categories(
    state: np.ndarray,
    candidates: np.ndarray,
    mask: np.ndarray,
    team_belief_labels: np.ndarray,
) -> np.ndarray:
    """Assign one priority tactical category to each rollout row."""

    rows = int(state.shape[0])
    if (
        candidates.shape[:2] != mask.shape
        or candidates.shape[0] != rows
        or team_belief_labels.shape[:2] != mask.shape
    ):
        raise ValueError("tactical category inputs do not share row/candidate shapes")
    valid = mask > 0
    finishing = team_belief_labels[..., 2] > 0.5
    partner_finishing = (
        finishing[:, :, 1] & valid
    ).any(axis=1)
    opponent_finishing = (
        finishing[:, :, (0, 2)] & valid[..., None]
    ).any(axis=(1, 2))

    trick_special = state[:, [
        STATE_TRICK_KIND_OFFSET + TRICK_KINDS.index("Straight"),
        STATE_TRICK_KIND_OFFSET + TRICK_KINDS.index("StraightPair"),
        STATE_TRICK_KIND_OFFSET + TRICK_KINDS.index("StraightTriple"),
    ]].sum(axis=1) > 0
    holder_teammate = state[:, STATE_LAST_PLAYER_OFFSET + 2] > 0
    is_pass = candidates[
        :, :, CARD_DIM + ACTION_KINDS.index("PASS")
    ] > 0
    special_yield = trick_special & holder_teammate & (is_pass & valid).any(axis=1)

    holder_opponent = (
        state[:, STATE_LAST_PLAYER_OFFSET + 1]
        + state[:, STATE_LAST_PLAYER_OFFSET + 3]
    ) > 0
    bomb_indices = [
        CARD_DIM + ACTION_KINDS.index(kind)
        for kind in ("Bomb", "StraightFlush", "FourKings")
    ]
    is_bomb = candidates[:, :, bomb_indices].sum(axis=2) > 0
    only_bomb = candidates[:, :, CANDIDATE_TACTICAL_SCALAR_OFFSET] > 0
    bomb_interrupt = holder_opponent & (is_bomb & only_bomb & valid).any(axis=1)

    categories = np.full(rows, TACTICAL_ORDINARY, dtype=np.int64)
    categories[bomb_interrupt] = TACTICAL_BOMB_INTERRUPT
    categories[special_yield] = TACTICAL_SPECIAL_YIELD
    categories[opponent_finishing] = TACTICAL_OPPONENT_FINISHING
    categories[partner_finishing] = TACTICAL_PARTNER_FINISHING
    return categories


def stratified_epoch_indices(
    categories: np.ndarray,
    num_samples: int,
    *,
    seed: int,
    target_shares: np.ndarray = TACTICAL_TARGET_SHARES,
    max_repeat: int = TACTICAL_MAX_REPEAT,
) -> np.ndarray:
    """Sample one deterministic epoch with bounded, redistributed quotas."""

    categories = np.asarray(categories, dtype=np.int64)
    if categories.ndim != 1 or categories.size == 0 or num_samples <= 0:
        raise ValueError("stratified sampling requires rows and positive num_samples")
    if isinstance(max_repeat, bool) or int(max_repeat) != max_repeat or max_repeat <= 0:
        raise ValueError("tactical max_repeat must be a positive integer")
    max_repeat = int(max_repeat)
    category_count = len(TACTICAL_CATEGORY_NAMES)
    pools = [
        np.flatnonzero(categories == category)
        for category in range(category_count)
    ]
    available = np.asarray([pool.size > 0 for pool in pools], dtype=bool)
    shares = np.asarray(target_shares, dtype=np.float64)
    if (
        shares.shape != (category_count,)
        or not np.isfinite(shares).all()
        or (shares < 0).any()
    ):
        raise ValueError(
            "tactical target shares must match every finite nonnegative category"
        )
    shares = np.where(available, shares, 0.0)
    if shares.sum() <= 0:
        shares = available.astype(np.float64)
    shares /= shares.sum()
    capacities = np.asarray(
        [pool.size * max_repeat for pool in pools], dtype=np.int64,
    )
    if int(capacities.sum()) < int(num_samples):
        raise ValueError(
            "tactical max_repeat capacity is smaller than num_samples"
        )

    def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
        normalized = np.asarray(weights, dtype=np.float64)
        normalized = normalized / normalized.sum()
        exact = normalized * int(total)
        result = np.floor(exact).astype(np.int64)
        remainder = int(total) - int(result.sum())
        if remainder:
            order = np.argsort(-(exact - result), kind="stable")
            result[order[:remainder]] += 1
        return result

    quotas = np.minimum(largest_remainder(int(num_samples), shares), capacities)
    while int(quotas.sum()) < int(num_samples):
        deficit = int(num_samples) - int(quotas.sum())
        room = capacities - quotas
        active = room > 0
        redistribution_weights = np.where(active, shares, 0.0)
        if redistribution_weights.sum() <= 0:
            redistribution_weights = active.astype(np.float64)
        addition = np.minimum(
            largest_remainder(deficit, redistribution_weights), room,
        )
        if not addition.any():
            addition[int(np.flatnonzero(active)[0])] = 1
        quotas += addition

    rng = np.random.default_rng(int(seed))
    sampled = []
    for pool, quota in zip(pools, quotas, strict=True):
        quota = int(quota)
        if quota <= 0:
            continue
        full_cycles, remainder = divmod(quota, int(pool.size))
        pieces = [rng.permutation(pool) for _ in range(full_cycles)]
        if remainder:
            pieces.append(rng.permutation(pool)[:remainder])
        sampled.append(np.concatenate(pieces))
    out = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(out)
    return out


def tactical_distribution(categories: np.ndarray) -> str:
    counts = np.bincount(
        np.asarray(categories, dtype=np.int64),
        minlength=len(TACTICAL_CATEGORY_NAMES),
    )
    return " ".join(
        f"{name}={int(count)}" for name, count in zip(
            TACTICAL_CATEGORY_NAMES, counts, strict=True,
        )
    )


def summarize_epoch_stats(stats: dict[str, list[float]]) -> dict[str, float]:
    """Average scalar losses but aggregate belief ratios from exact counts."""

    summary = {key: float(np.mean(values)) for key, values in stats.items()}

    def summed(key: str) -> float:
        return float(np.sum(stats.get(key, ()), dtype=np.float64))

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    total_correct = 0.0
    total_effective = 0.0
    for target_name in TEAM_BELIEF_TARGET_NAMES:
        target = target_name.removesuffix("_response")
        prefix = f"team_belief_{target}"
        correct = summed(f"{prefix}_correct_count")
        positive = summed(f"{prefix}_positive_count")
        predicted_positive = summed(f"{prefix}_predicted_positive_count")
        true_positive = summed(f"{prefix}_true_positive_count")
        effective = summed(f"{prefix}_effective_count")
        squared_error = summed(f"{prefix}_squared_error_sum")
        if not any(
            key.startswith(f"{prefix}_") for key in stats
        ):
            continue
        summary[f"{prefix}_accuracy"] = ratio(correct, effective)
        summary[f"{prefix}_positive_recall"] = ratio(true_positive, positive)
        summary[f"{prefix}_positive_precision"] = ratio(
            true_positive, predicted_positive,
        )
        summary[f"{prefix}_predicted_positive_rate"] = ratio(
            predicted_positive, effective,
        )
        summary[f"{prefix}_brier"] = ratio(squared_error, effective)
        total_correct += correct
        total_effective += effective
    if total_effective > 0:
        summary["team_belief_accuracy"] = total_correct / total_effective
    return summary
_PINNED_STATE_BUFFERS: dict[
    tuple[str, tuple[int, ...], torch.dtype], torch.Tensor
] = {}
_FLAT_STATE_BUFFERS: dict[
    tuple[bool, tuple[tuple[str, tuple[int, ...], torch.dtype], ...]],
    dict[torch.dtype, torch.Tensor],
] = {}
_PERSISTENT_MODELS: dict[
    tuple[str, str, int, int], Top10Selector
] = {}


@dataclass
class PersistentOptimizerEntry:
    optimizer: torch.optim.Optimizer
    source_path: str
    source_fingerprint: tuple[int, int, int, int] | None


_PERSISTENT_OPTIMIZERS: dict[
    tuple[int, float, float, float, bool], PersistentOptimizerEntry
] = {}
_PERSISTENT_CHECKPOINTS: dict[
    str, tuple[tuple[int, int, int, int], Top10Selector, dict, str]
] = {}
_SHARD_LOAD_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def env_enabled(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).lower() not in {"0", "false", "no", "off"}


def checkpoint_fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def take_persistent_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[Top10Selector, dict] | None:
    """Consume an exact in-memory successor checkpoint when the chain is intact."""

    if not env_enabled("DANRL_PERSISTENT_LEARNER_REUSE"):
        return None
    resolved = str(path.resolve())
    cached = _PERSISTENT_CHECKPOINTS.pop(resolved, None)
    if cached is None:
        return None
    fingerprint, model, metadata, cached_device = cached
    try:
        unchanged = fingerprint == checkpoint_fingerprint(path)
    except FileNotFoundError:
        unchanged = False
    if not unchanged or cached_device != str(device):
        return None
    return model, dict(metadata)


def publish_persistent_checkpoint(
    path: Path,
    model: Top10Selector,
    payload: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Remember a successfully published checkpoint without retaining CPU weights."""

    if not env_enabled("DANRL_PERSISTENT_LEARNER_REUSE"):
        return
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    resolved = str(path.resolve())
    _PERSISTENT_CHECKPOINTS.clear()
    _PERSISTENT_CHECKPOINTS[resolved] = (
        checkpoint_fingerprint(path),
        model,
        metadata,
        str(device),
    )
    if optimizer is not None:
        fingerprint = checkpoint_fingerprint(path)
        for entry in _PERSISTENT_OPTIMIZERS.values():
            if entry.optimizer is optimizer:
                entry.source_path = resolved
                entry.source_fingerprint = fingerprint
                break


def shard_load_executor(path_count: int) -> concurrent.futures.ThreadPoolExecutor | None:
    global _SHARD_LOAD_EXECUTOR
    if path_count <= 1 or not env_enabled("DANRL_PARALLEL_SHARD_LOAD", default=True):
        return None
    if _SHARD_LOAD_EXECUTOR is None:
        workers = min(path_count, max(1, int(os.environ.get("DANRL_SHARD_LOAD_THREADS", "8"))))
        _SHARD_LOAD_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ppo-shard-load",
        )
    return _SHARD_LOAD_EXECUTOR


def load_rollout_shard(path: Path) -> tuple[dict[str, np.ndarray], dict | None, int]:
    if path.is_dir():
        arrays = {}
        history_present = (path / "history.npy").exists()
        missing = [
            key for key in PPO_KEYS
            if key != "history" and not (path / f"{key}.npy").exists()
        ]
        if missing:
            raise ValueError(f"{path} is missing rollout .npy files: {missing}")
        for key in PPO_KEYS:
            key_path = path / f"{key}.npy"
            if key == "history" and not key_path.exists():
                rows = int(np.load(path / "action.npy", allow_pickle=False, mmap_mode="r").shape[0])
                arrays[key] = np.zeros(
                    (rows, HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32
                )
            else:
                arrays[key] = np.load(key_path, allow_pickle=False, mmap_mode="r")
        belief_paths = [path / f"{key}.npy" for key in TEAM_BELIEF_KEYS]
        if any(item.exists() for item in belief_paths):
            if not all(item.exists() for item in belief_paths):
                raise ValueError(f"{path} has an incomplete team-belief rollout contract")
            for key, key_path in zip(TEAM_BELIEF_KEYS, belief_paths, strict=True):
                arrays[key] = np.load(key_path, allow_pickle=False, mmap_mode="r")
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        metadata = dict(metadata or {})
        metadata["history_synthesized"] = bool(
            metadata.get("history_synthesized", False) or not history_present
        )
        return arrays, metadata, int(arrays["action"].shape[0])
    with np.load(path, allow_pickle=False) as data:
        history_present = "history" in data.files
        missing = [key for key in PPO_KEYS if key != "history" and key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing rollout keys: {missing}")
        rows = int(data["action"].shape[0])
        arrays = {
            key: (
                data[key]
                if key in data.files
                else np.zeros((rows, HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
            )
            for key in PPO_KEYS
        }
        belief_present = [key in data.files for key in TEAM_BELIEF_KEYS]
        if any(belief_present):
            if not all(belief_present):
                raise ValueError(f"{path} has an incomplete team-belief rollout contract")
            for key in TEAM_BELIEF_KEYS:
                arrays[key] = data[key]
        metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
        metadata = dict(metadata or {})
        metadata["history_synthesized"] = bool(
            metadata.get("history_synthesized", False) or not history_present
        )
        return arrays, metadata, rows


def load_rollout_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    paths = [Path(path) for path in (args.rollout_shards or [])]
    paths = [path for path in paths if path.exists()]
    if not paths:
        paths = [Path(args.rollout)]

    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in (*PPO_KEYS, *TEAM_BELIEF_KEYS)
    }
    metadata_items = []
    contracts: list[tuple[int, str]] = []
    storage_capacities: list[int] = []
    history_protocols: set[str | None] = set()
    history_event_semantics: set[str | None] = set()
    history_synthesized = False
    team_belief_presence: list[bool] = []
    team_belief_protocols: set[str | None] = set()
    kept_paths = []
    executor = shard_load_executor(len(paths))
    loaded = (
        executor.map(load_rollout_shard, paths)
        if executor is not None
        else map(load_rollout_shard, paths)
    )
    for path, (shard_arrays, shard_metadata, rows) in zip(paths, loaded, strict=True):
        if rows <= 0:
            continue
        source_metadata = shard_metadata or {}
        history_synthesized = bool(
            history_synthesized or source_metadata.get("history_synthesized", False)
        )
        history_protocols.add(
            str(source_metadata["history_protocol"])
            if source_metadata.get("history_protocol")
            else None
        )
        history_event_semantics.add(
            str(source_metadata["history_event_semantics"])
            if source_metadata.get("history_event_semantics")
            else None
        )
        has_team_belief = all(key in shard_arrays for key in TEAM_BELIEF_KEYS)
        team_belief_presence.append(has_team_belief)
        team_belief_protocols.add(
            str(source_metadata["team_belief_protocol"])
            if source_metadata.get("team_belief_protocol")
            else None
        )
        contract = normalize_candidate_contract(
            source_metadata.get("candidate_capacity", TOPK),
            source_metadata.get("action_support", "structured_topk"),
        )
        storage_capacity = int(
            source_metadata.get("storage_candidate_capacity", shard_arrays["candidates"].shape[1])
        )
        if (storage_capacity <= 0 or shard_arrays["candidates"].shape[1] != storage_capacity
                or shard_arrays["mask"].shape[1] != storage_capacity):
            raise ValueError(f"{path} candidate storage metadata/shape mismatch")
        if contract[1] != FULL_LEGAL and storage_capacity != contract[0]:
            raise ValueError(f"{path} structured_topk width does not match candidate_capacity")
        contracts.append(contract)
        storage_capacities.append(storage_capacity)
        for key in PPO_KEYS:
            arrays[key].append(shard_arrays[key])
        if has_team_belief:
            for key in TEAM_BELIEF_KEYS:
                arrays[key].append(shard_arrays[key])
        if shard_metadata is not None:
            metadata_items.append(shard_metadata)
        kept_paths.append(str(path))

    if not arrays["state"]:
        raise RuntimeError("no rollout rows found")

    if len(set(contracts)) != 1:
        raise ValueError("rollout shards mix candidate_capacity/action_support contracts")
    if any(team_belief_presence) and not all(team_belief_presence):
        raise ValueError("rollout shards mix team-belief supervision contracts")
    candidate_capacity, action_support = contracts[0]
    storage_candidate_capacity = max(storage_capacities)
    for index, storage_capacity in enumerate(storage_capacities):
        missing = storage_candidate_capacity - storage_capacity
        if missing:
            arrays["candidates"][index] = np.pad(
                arrays["candidates"][index], ((0, 0), (0, missing), (0, 0)),
            )
            arrays["mask"][index] = np.pad(
                arrays["mask"][index], ((0, 0), (0, missing)),
            )
            if team_belief_presence and team_belief_presence[index]:
                arrays["team_belief_labels"][index] = np.pad(
                    arrays["team_belief_labels"][index],
                    ((0, 0), (0, missing), (0, 0), (0, 0)),
                )
                arrays["team_belief_mask"][index] = np.pad(
                    arrays["team_belief_mask"][index],
                    ((0, 0), (0, missing), (0, 0)),
                )

    out = {
        key: np.concatenate(values, axis=0) if len(values) > 1 else values[0]
        for key, values in arrays.items()
        if values
    }
    metadata = {
        "source": "train_ppo_direct_shards" if len(kept_paths) > 1 or args.rollout_shards else "train_ppo_rollout",
        "inputs": kept_paths,
        "input_count": len(kept_paths),
        "rows": int(out["action"].shape[0]),
        "source_metadata": metadata_items,
        "candidate_capacity": candidate_capacity,
        "storage_candidate_capacity": storage_candidate_capacity,
        "action_support": action_support,
        "history_protocol": (
            next(iter(history_protocols))
            if len(history_protocols) == 1 and not history_synthesized
            else None
        ),
        "history_event_semantics": (
            next(iter(history_event_semantics))
            if len(history_event_semantics) == 1 and not history_synthesized
            else None
        ),
        "history_synthesized": history_synthesized,
        "team_belief_protocol": (
            next(iter(team_belief_protocols))
            if team_belief_presence and all(team_belief_presence)
            and len(team_belief_protocols) == 1
            else None
        ),
    }
    return out, metadata


def validate_rollout_history_contract(architecture: str, metadata: dict) -> None:
    if architecture not in {"phase14", "team_belief"}:
        return
    if metadata.get("history_synthesized"):
        raise ValueError(
            f"{architecture} requires rollout data with real public history; "
            "one or more shards synthesized an all-zero history tensor"
        )
    if metadata.get("history_protocol") != HISTORY_PROTOCOL:
        raise ValueError(
            "phase14 rollout history protocol mismatch: "
            f"rollout={metadata.get('history_protocol')!r} expected={HISTORY_PROTOCOL!r}"
        )
    if metadata.get("history_event_semantics") != HISTORY_EVENT_SEMANTICS:
        raise ValueError(
            "phase14 rollout history event semantics mismatch: "
            f"rollout={metadata.get('history_event_semantics')!r} "
            f"expected={HISTORY_EVENT_SEMANTICS!r}"
        )
    if (
        architecture == "team_belief"
        and metadata.get("team_belief_protocol") != TEAM_BELIEF_PROTOCOL
    ):
        raise ValueError(
            "team_belief rollout supervision protocol mismatch: "
            f"rollout={metadata.get('team_belief_protocol')!r} "
            f"expected={TEAM_BELIEF_PROTOCOL!r}"
        )


def amp_dtype_from_arg(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_adamw(parameters, *, lr: float, weight_decay: float, eps: float, fused: bool, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "weight_decay": weight_decay, "eps": eps}
    if fused and device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except TypeError:
            print("warning: fused AdamW not supported by this torch build; falling back to AdamW", flush=True)
    if fused and device.type == "npu":
        from torch_npu.optim import NpuFusedAdamW

        return NpuFusedAdamW(parameters, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)


def optimizer_config(
    *,
    lr: float,
    weight_decay: float,
    eps: float,
    fused: bool,
) -> dict[str, float | bool]:
    return {
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "eps": float(eps),
        "fused": bool(fused),
    }


def make_continuous_adamw(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    eps: float,
    fused: bool,
    device: torch.device,
    init_checkpoint: str | Path = "",
    init_payload: dict | None = None,
    model_type: str = "",
    model_config: dict | None = None,
) -> tuple[torch.optim.Optimizer, str]:
    """Keep Adam moments across updates and restore them after a restart."""

    continuous = env_enabled("DANRL_CONTINUOUS_OPTIMIZER", default=True)
    reuse = continuous and env_enabled("DANRL_PERSISTENT_LEARNER_REUSE")
    key = (id(model), float(lr), float(weight_decay), float(eps), bool(fused))
    checkpoint = Path(init_checkpoint) if init_checkpoint else None
    resolved = str(checkpoint.resolve()) if checkpoint is not None else ""
    fingerprint = (
        checkpoint_fingerprint(checkpoint)
        if checkpoint is not None and checkpoint.is_file()
        else None
    )
    entry = _PERSISTENT_OPTIMIZERS.get(key) if reuse else None
    if (
        entry is not None
        and entry.source_path == resolved
        and entry.source_fingerprint == fingerprint
    ):
        zero_optimizer_grad(entry.optimizer)
        return entry.optimizer, "resident"

    optimizer = make_adamw(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        fused=fused,
        device=device,
    )
    source = "fresh"
    if continuous and checkpoint is not None:
        restored = load_optimizer_training_state(
            checkpoint,
            init_payload or {},
            optimizer,
            expected_model_type=model_type,
            expected_model_config=dict(model_config or {}),
            expected_optimizer_config=optimizer_config(
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                fused=fused,
            ),
        )
        source = f"sidecar:{restored}" if restored is not None else "legacy_fresh"
    if reuse:
        _PERSISTENT_OPTIMIZERS.clear()
        _PERSISTENT_OPTIMIZERS[key] = PersistentOptimizerEntry(
            optimizer=optimizer,
            source_path=resolved,
            source_fingerprint=fingerprint,
        )
    return optimizer, source


def iter_device_batches(
    arrays: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    allow_direct_full_batch: bool = True,
):
    # A full-batch update is permutation invariant. Avoid allocating a random
    # permutation and eight index_select copies when every row already forms
    # the sole batch (the production 3.2k rows use batch_size=8192).
    if (
        allow_direct_full_batch
        and os.environ.get("DANRL_DIRECT_FULL_BATCH", "1").lower()
        not in {"0", "false", "no", "off"}
        and batch_size >= indices.numel()
        and all(array.shape[0] == indices.numel() for array in arrays)
    ):
        yield arrays
        return
    if shuffle:
        order = indices[torch.randperm(indices.numel(), device=indices.device)]
    else:
        order = indices
    for start in range(0, order.numel(), batch_size):
        batch_idx = order[start : start + batch_size]
        yield tuple(array.index_select(0, batch_idx) for array in arrays)


def snapshot_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a CUDA state dict through reusable pinned buffers with one sync."""

    state = model.state_dict()
    tensors = list(state.values())
    flat_storage = env_enabled("DANRL_FLAT_CHECKPOINT_STORAGE")
    if (
        not tensors
        or tensors[0].device.type != "cuda"
        or os.environ.get("DANRL_ASYNC_CHECKPOINT_SNAPSHOT", "1").lower()
        in {"0", "false", "no", "off"}
    ):
        snapshot = {key: value.detach().cpu() for key, value in state.items()}
        return (
            copy_state_dict_to_flat_cpu(snapshot, pin_memory=False)
            if flat_storage and snapshot
            else snapshot
        )

    if flat_storage:
        snapshot = copy_state_dict_to_flat_cpu(state, pin_memory=True)
        torch.cuda.synchronize(tensors[0].device)
        return snapshot

    snapshot: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        detached = value.detach()
        buffer_key = (name, tuple(detached.shape), detached.dtype)
        buffer = _PINNED_STATE_BUFFERS.get(buffer_key)
        if buffer is None:
            buffer = torch.empty(
                detached.shape,
                dtype=detached.dtype,
                device="cpu",
                pin_memory=True,
            )
            _PINNED_STATE_BUFFERS[buffer_key] = buffer
        buffer.copy_(detached, non_blocking=True)
        snapshot[name] = buffer
    torch.cuda.synchronize(tensors[0].device)
    return snapshot


def copy_state_dict_to_flat_cpu(
    state: dict[str, torch.Tensor],
    *,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    """Copy tensors into one reusable CPU storage per dtype.

    The returned mapping keeps the original names, shapes, dtypes, and values.
    PyTorch checkpoint serialization then writes one large storage instead of
    hundreds of small storage records, which makes mmap actor reloads cheaper.
    """

    signature = tuple(
        (name, tuple(value.shape), value.dtype)
        for name, value in state.items()
    )
    cache_key = (bool(pin_memory), signature)
    buffers = _FLAT_STATE_BUFFERS.get(cache_key)
    if buffers is None:
        totals: dict[torch.dtype, int] = {}
        for value in state.values():
            totals[value.dtype] = totals.get(value.dtype, 0) + value.numel()
        buffers = {
            dtype: torch.empty(
                total,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for dtype, total in totals.items()
        }
        _FLAT_STATE_BUFFERS[cache_key] = buffers

    offsets = {dtype: 0 for dtype in buffers}
    snapshot: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        detached = value.detach()
        offset = offsets[detached.dtype]
        end = offset + detached.numel()
        target = buffers[detached.dtype][offset:end].view(detached.shape)
        target.copy_(detached, non_blocking=pin_memory)
        snapshot[name] = target
        offsets[detached.dtype] = end
    return snapshot


@torch.no_grad()
def evaluate_policy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    mean_slot = 0.0
    entropy_sum = 0.0
    for batch in loader:
        state, candidates, mask, history, action, _logp, _advantage, _returns = batch
        state = state.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        logits, _value = model(state, candidates, mask, history)
        pred = torch.argmax(logits, dim=1)
        top_indices = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
        pi = torch.distributions.Categorical(logits=logits)
        total += int(action.numel())
        top1 += int((pred == action).sum().item())
        top3 += int((top_indices[:, : min(3, top_indices.shape[1])] == action[:, None]).any(dim=1).sum().item())
        top5 += int((top_indices == action[:, None]).any(dim=1).sum().item())
        mean_slot += float((pred.float() + 1.0).sum().item())
        entropy_sum += float(pi.entropy().sum().item())
    denom = max(1, total)
    return {
        "top1": top1 / denom,
        "top3": top3 / denom,
        "top5": top5 / denom,
        "mean_slot": mean_slot / denom,
        "entropy": entropy_sum / denom,
    }


@torch.no_grad()
def evaluate_policy_device(
    model: torch.nn.Module,
    arrays: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    mean_slot = 0.0
    entropy_sum = 0.0
    for batch in iter_device_batches(arrays[:8], indices, batch_size, shuffle=False):
        state, candidates, mask, history, action, _logp, _advantage, _returns = batch
        logits, _value = model(state, candidates, mask, history)
        pred = torch.argmax(logits, dim=1)
        top_indices = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
        pi = torch.distributions.Categorical(logits=logits)
        total += int(action.numel())
        top1 += int((pred == action).sum().item())
        top3 += int((top_indices[:, : min(3, top_indices.shape[1])] == action[:, None]).any(dim=1).sum().item())
        top5 += int((top_indices == action[:, None]).any(dim=1).sum().item())
        mean_slot += float((pred.float() + 1.0).sum().item())
        entropy_sum += float(pi.entropy().sum().item())
    denom = max(1, total)
    return {
        "top1": top1 / denom,
        "top3": top3 / denom,
        "top5": top5 / denom,
        "mean_slot": mean_slot / denom,
        "entropy": entropy_sum / denom,
    }


def format_eval(metrics: dict[str, float]) -> str:
    return (
        f"top1={metrics['top1']:.1%} top3={metrics['top3']:.1%} "
        f"top5={metrics['top5']:.1%} mean_slot={metrics['mean_slot']:.2f} "
        f"entropy={metrics['entropy']:.3f}"
    )


def build_initial_model(args: argparse.Namespace, device: torch.device) -> tuple[Top10Selector, dict[str, int], dict]:
    if args.init_checkpoint:
        checkpoint = Path(args.init_checkpoint)
        resident = take_persistent_checkpoint(checkpoint, device)
        if resident is None:
            # Normalize checkpoint storage on CPU so CUDA and NPU runs can exchange
            # the same selector checkpoint without preserving backend device tags.
            load_kwargs = (
                {"weights_only": False}
                if env_enabled("DANRL_TRUSTED_CHECKPOINT_FAST_LOAD")
                else {}
            )
            payload = torch.load(checkpoint, map_location="cpu", **load_kwargs)
            resident_model = None
        else:
            resident_model, payload = resident
            print(f"init_checkpoint_load=resident path={checkpoint}", flush=True)
        ckpt_state_dim = int(payload.get("state_dim", -1))
        ckpt_candidate_dim = int(payload.get("candidate_dim", -1))
        ckpt_feature_version = payload.get("feature_version")
        if ckpt_state_dim != STATE_DIM or ckpt_candidate_dim != CANDIDATE_DIM:
            raise ValueError(
                "init checkpoint feature shape mismatch: "
                f"checkpoint state_dim={ckpt_state_dim} candidate_dim={ckpt_candidate_dim}, "
                f"current state_dim={STATE_DIM} candidate_dim={CANDIDATE_DIM}"
            )
        if ckpt_feature_version != FEATURE_VERSION:
            raise ValueError(
                "init checkpoint feature_version mismatch: "
                f"checkpoint={ckpt_feature_version!r} current={FEATURE_VERSION!r}"
            )
        config = selector_config_from_checkpoint(payload)
        cache_key = (
            str(device),
            str(payload.get("model_type") or ""),
            int(config["hidden_dim"]),
            int(config["candidate_hidden_dim"]),
        )
        reuse = (
            os.environ.get("DANRL_PERSISTENT_LEARNER_REUSE", "0").lower()
            not in {"0", "false", "no", "off"}
        )
        model = resident_model
        if model is None:
            model = _PERSISTENT_MODELS.get(cache_key) if reuse else None
        if model is None:
            model = build_selector_from_checkpoint(payload, device=device)
            if reuse:
                _PERSISTENT_MODELS.clear()
                _PERSISTENT_MODELS[cache_key] = model
        elif resident_model is None:
            reused = reload_selector_from_checkpoint(model, payload, device=device)
            if reused is not model:
                raise RuntimeError("persistent selector cache changed architecture")
            model.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        checkpoint_architecture = (
            "team_belief"
            if isinstance(model, EfficientTeamBeliefTop10Selector)
            else "phase14"
            if isinstance(model, Phase14Top10Selector)
            else "legacy"
        )
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                f"--architecture={args.architecture} does not match init checkpoint "
                f"architecture={checkpoint_architecture}"
            )
        if (
            checkpoint_architecture in {"phase14", "team_belief"}
            and payload.get("history_event_semantics") != HISTORY_EVENT_SEMANTICS
        ):
            raise ValueError(
                "Phase1-4 init checkpoint predates the corrected history event semantics: "
                f"checkpoint={payload.get('history_event_semantics')!r} "
                f"expected={HISTORY_EVENT_SEMANTICS!r}"
            )
        return model, config, payload
    model_class = (
        EfficientTeamBeliefTop10Selector
        if args.architecture == "team_belief"
        else Phase14Top10Selector
        if args.architecture == "phase14"
        else Top10Selector
    )
    model = model_class(
        STATE_DIM,
        CANDIDATE_DIM,
        hidden_dim=args.hidden_dim,
        candidate_hidden_dim=args.candidate_hidden_dim,
    ).to(device)
    return model, {"hidden_dim": args.hidden_dim, "candidate_hidden_dim": args.candidate_hidden_dim}, {}


def pretrain_on_behavior(
    model: torch.nn.Module,
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    args: argparse.Namespace,
    device_tensors: tuple[torch.Tensor, ...] | None = None,
) -> None:
    if args.behavior_pretrain_epochs <= 0:
        return
    use_device_batches = bool(args.preload_to_device and is_accelerator(device))
    if use_device_batches:
        pretrain_tensors = device_tensors[:5] if device_tensors is not None else tuple(t.to(device, non_blocking=True) for t in tensors[:5])
        pretrain_indices = torch.arange(pretrain_tensors[0].shape[0], dtype=torch.long, device=device)
        pretrain_loader = None
    else:
        pretrain_tensors = ()
        pretrain_indices = None
        pretrain_loader = DataLoader(
            TensorDataset(*tensors[:5]),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
    optimizer = make_adamw(
        model.parameters(),
        lr=args.behavior_pretrain_lr,
        weight_decay=args.behavior_pretrain_weight_decay,
        eps=1e-5,
        fused=args.fused_optimizer,
        device=device,
    )
    model.train()
    for epoch in range(1, args.behavior_pretrain_epochs + 1):
        total = 0
        loss_sum = 0.0
        acc = 0
        if use_device_batches:
            assert pretrain_indices is not None
            batch_iter = iter_device_batches(pretrain_tensors, pretrain_indices, args.batch_size, shuffle=True)
        else:
            assert pretrain_loader is not None
            batch_iter = pretrain_loader
        for state, candidates, mask, history, action in batch_iter:
            if not use_device_batches:
                state = state.to(device, non_blocking=True)
                candidates = candidates.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                history = history.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _value = model(state, candidates, mask, history)
            loss = F.cross_entropy(logits, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_n = int(action.numel())
            total += batch_n
            loss_sum += float(loss.detach().cpu().item()) * batch_n
            acc += int((torch.argmax(logits, dim=1) == action).sum().detach().cpu().item())
        denom = max(1, total)
        print(f"behavior_pretrain epoch={epoch} loss={loss_sum / denom:.4f} top1={acc / denom:.1%}", flush=True)


def main() -> None:
    job_started = time.perf_counter()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = initialize_device(args.device)
    npu_jit_compile = False
    if device.type == "npu":
        npu_jit_compile = os.environ.get("TRAIN_NPU_JIT_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
        torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    torch.set_float32_matmul_precision(args.matmul_precision)
    use_amp = bool(args.amp and is_accelerator(device))
    amp_dtype = amp_dtype_from_arg(args.amp_dtype)
    if is_accelerator(device):
        seed_accelerator(device, args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if is_accelerator(device):
        print(
            f"device={device} accelerator_name={device_name(device)} amp={use_amp} "
            f"amp_dtype={args.amp_dtype} matmul_precision={args.matmul_precision} "
            f"npu_jit_compile={npu_jit_compile}",
            flush=True,
        )
    else:
        print(f"device={device}", flush=True)

    rollout_arrays, rollout_metadata = load_rollout_arrays(args)
    rollout_loaded_at = time.perf_counter()
    validate_rollout_history_contract(args.architecture, rollout_metadata)
    state = rollout_arrays["state"].astype(np.float32, copy=False)
    candidates = rollout_arrays["candidates"].astype(np.float32, copy=False)
    mask = rollout_arrays["mask"].astype(np.float32, copy=False)
    history = rollout_arrays["history"].astype(np.float32, copy=False)
    action = rollout_arrays["action"].astype(np.int64, copy=False)
    logp = rollout_arrays["logp"].astype(np.float32, copy=False)
    advantage = rollout_arrays["advantage"].astype(np.float32, copy=False)
    returns = rollout_arrays["returns"].astype(np.float32, copy=False)
    team_belief_labels = (
        rollout_arrays["team_belief_labels"].astype(np.float32, copy=False)
        if "team_belief_labels" in rollout_arrays
        else None
    )
    team_belief_mask = (
        rollout_arrays["team_belief_mask"].astype(np.float32, copy=False)
        if "team_belief_mask" in rollout_arrays
        else None
    )
    dims_ok = (
        state.ndim == 2
        and candidates.ndim == 3
        and mask.ndim == 2
        and history.ndim == 3
        and state.shape[0] == candidates.shape[0] == mask.shape[0]
        and state.shape[1] == STATE_DIM
        and candidates.shape[1] == int(rollout_metadata["storage_candidate_capacity"])
        and mask.shape[1] == int(rollout_metadata["storage_candidate_capacity"])
        and candidates.shape[2] == CANDIDATE_DIM
        and history.shape[1:] == (HISTORY_LENGTH, HISTORY_EVENT_DIM)
    )
    if not dims_ok:
        raise ValueError(
            f"rollout dim mismatch: state={state.shape}, candidates={candidates.shape}, "
            f"mask={mask.shape}, history={history.shape}; "
            f"expected state_dim={STATE_DIM}, storage_candidate_capacity="
            f"{rollout_metadata['storage_candidate_capacity']}, candidate_dim={CANDIDATE_DIM}"
        )
    if args.architecture == "team_belief":
        expected_labels = (
            state.shape[0], mask.shape[1],
            TEAM_BELIEF_SEAT_COUNT, TEAM_BELIEF_TARGET_DIM,
        )
        expected_belief_mask = (
            state.shape[0], mask.shape[1], TEAM_BELIEF_SEAT_COUNT,
        )
        if team_belief_labels is None or team_belief_mask is None:
            raise ValueError("team_belief architecture requires privileged rollout labels")
        if (
            team_belief_labels.shape != expected_labels
            or team_belief_mask.shape != expected_belief_mask
        ):
            raise ValueError(
                "team-belief rollout dim mismatch: "
                f"labels={team_belief_labels.shape} mask={team_belief_mask.shape}; "
                f"expected={expected_labels}/{expected_belief_mask}"
            )
    if (
        not np.isfinite(state).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(mask).all()
        or not np.isfinite(history).all()
    ):
        raise ValueError("rollout contains non-finite state/candidate/mask/history features")
    if not np.isfinite(logp).all() or not np.isfinite(advantage).all() or not np.isfinite(returns).all():
        raise ValueError("rollout contains non-finite logp/advantage/returns")
    if (action < 0).any() or (action >= mask.shape[1]).any():
        raise ValueError("rollout action index outside candidate mask width")
    if not bool((mask[np.arange(len(action)), action] > 0).all()):
        raise ValueError("rollout contains actions that are masked out")
    if float(np.std(advantage)) < 1.0e-8 and not args.allow_zero_advantage:
        raise ValueError("rollout advantage is nearly constant; check reward attribution or pass --allow-zero-advantage")

    use_tactical_resampling = bool(
        args.architecture == "team_belief"
        and not args.disable_tactical_resampling
    )
    tactical_categories = None
    if use_tactical_resampling:
        assert team_belief_labels is not None
        tactical_categories = tactical_row_categories(
            state, candidates, mask, team_belief_labels,
        )
        print(
            f"tactical_source {tactical_distribution(tactical_categories)}",
            flush=True,
        )

    raw_model, model_config, init_payload = build_initial_model(args, device)
    model_loaded_at = time.perf_counter()
    rollout_contract = normalize_candidate_contract(
        rollout_metadata["candidate_capacity"], rollout_metadata["action_support"],
    )
    if init_payload:
        init_contract = normalize_candidate_contract(
            init_payload.get("candidate_capacity", TOPK),
            init_payload.get("action_support", "structured_topk"),
        )
        if init_contract != rollout_contract:
            raise ValueError("init checkpoint and rollout candidate contracts do not match")
    print(f"init_mode={args.init_mode} init_checkpoint={args.init_checkpoint or 'none'} model_config={model_config}", flush=True)
    model: torch.nn.Module = raw_model

    tensor_arrays = [state, candidates, mask, history, action, logp, advantage, returns]
    if team_belief_labels is not None and team_belief_mask is not None:
        tensor_arrays.extend([team_belief_labels, team_belief_mask])
    tensors = tuple(torch.from_numpy(x) for x in tensor_arrays)
    device_tensors = None
    all_indices = None
    eval_indices = None
    if args.preload_to_device:
        device_tensors = tuple(t.to(device, non_blocking=True) for t in tensors)
        all_indices = torch.arange(state.shape[0], dtype=torch.long, device=device)
        eval_indices = all_indices
        if args.eval_max_samples > 0 and eval_indices.numel() > args.eval_max_samples:
            eval_indices = eval_indices[: args.eval_max_samples]
        print(
            f"preload_to_device=true rows={state.shape[0]} "
            f"approx_mb={sum(array.nbytes for array in tensor_arrays) / 1024 / 1024:.1f}",
            flush=True,
        )
    data_ready_at = time.perf_counter()

    pretrain_on_behavior(raw_model, tensors, device, args, device_tensors=device_tensors)

    if args.compile:
        if hasattr(torch, "compile"):
            compile_mode = None if args.compile_mode in (None, "default") else args.compile_mode
            model = torch.compile(model, mode=compile_mode)
        else:
            raise RuntimeError("torch.compile requested, but this torch version does not provide it.")
    if args.preload_to_device:
        loader = None
        train_dataset = None
        eval_loader = None
    else:
        train_dataset = TensorDataset(*tensors)
        loader = None if use_tactical_resampling else DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=False, pin_memory=device.type == "cuda",
        )
        eval_tensors = tensors[:8]
        if args.eval_max_samples > 0 and state.shape[0] > args.eval_max_samples:
            eval_tensors = tuple(t[: args.eval_max_samples] for t in eval_tensors)
        eval_loader = DataLoader(TensorDataset(*eval_tensors), batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.skip_before_eval:
        before = {"top1": float("nan"), "top3": float("nan"), "top5": float("nan"), "mean_slot": float("nan"), "entropy": float("nan")}
        print("before eval=skip", flush=True)
    elif args.preload_to_device:
        assert device_tensors is not None and eval_indices is not None
        before = evaluate_policy_device(model, device_tensors, eval_indices, args.batch_size)
        print(f"before {format_eval(before)}", flush=True)
    else:
        assert eval_loader is not None
        before = evaluate_policy(model, eval_loader, device)
        print(f"before {format_eval(before)}", flush=True)

    config = PPOConfig(
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        team_belief_coef=args.team_belief_coef,
        team_belief_pos_weights=tuple(args.team_belief_pos_weights),
        target_kl=args.target_kl,
        train_iters=args.train_iters,
        kl_check_interval=args.kl_check_interval,
        dual_clip=args.dual_clip if args.dual_clip > 0 else None,
        amp=use_amp,
        amp_dtype=amp_dtype,
    )
    current_model_type = selector_model_type(raw_model)
    current_optimizer_config = optimizer_config(
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
        fused=args.fused_optimizer,
    )
    optimizer, optimizer_source = make_continuous_adamw(
        raw_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
        fused=args.fused_optimizer,
        device=device,
        init_checkpoint=args.init_checkpoint,
        init_payload=init_payload,
        model_type=current_model_type,
        model_config=model_config,
    )
    print(
        f"optimizer_continuity={int(env_enabled('DANRL_CONTINUOUS_OPTIMIZER', default=True))} "
        f"optimizer_state={optimizer_source}",
        flush=True,
    )
    agent = Top10PPOAgent(model, optimizer, config)

    last_info: dict[str, float] = {}
    after = before
    model.train()
    for epoch in range(1, args.epochs + 1):
        stats: dict[str, list[float]] = {}
        tactical_epoch_indices = None
        if use_tactical_resampling:
            assert tactical_categories is not None
            tactical_epoch_indices = stratified_epoch_indices(
                tactical_categories,
                state.shape[0],
                seed=args.seed + epoch,
                max_repeat=args.tactical_max_repeat,
            )
            print(
                f"tactical_sampled epoch={epoch} "
                f"{tactical_distribution(tactical_categories[tactical_epoch_indices])}",
                flush=True,
            )
        if args.preload_to_device:
            assert device_tensors is not None and all_indices is not None
            if tactical_epoch_indices is None:
                train_indices = all_indices
            else:
                train_indices = torch.as_tensor(
                    tactical_epoch_indices,
                    dtype=torch.long,
                    device=device,
                )
            train_iter = iter_device_batches(
                device_tensors,
                train_indices,
                args.batch_size,
                shuffle=tactical_epoch_indices is None,
                allow_direct_full_batch=tactical_epoch_indices is None,
            )
        else:
            if tactical_epoch_indices is None:
                assert loader is not None
                train_iter = loader
            else:
                assert train_dataset is not None
                train_iter = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=tactical_epoch_indices.tolist(),
                    drop_last=False,
                    pin_memory=device.type == "cuda",
                )
        for batch in train_iter:
            batch_np = {
                "state": batch[0],
                "candidates": batch[1],
                "mask": batch[2],
                "history": batch[3],
                "action": batch[4],
                "logp": batch[5],
                "advantage": batch[6],
                "returns": batch[7],
            }
            if len(batch) > 8:
                batch_np["team_belief_labels"] = batch[8]
                batch_np["team_belief_mask"] = batch[9]
            info = agent.update(tensor_batch(batch_np, device))
            last_info = info
            for key, value in info.items():
                stats.setdefault(key, []).append(value)
        should_eval = (
            (epoch == 1 and not args.skip_first_eval)
            or (epoch == args.epochs and not args.skip_final_eval)
            or args.eval_every <= 1
            or epoch % args.eval_every == 0
        )
        if should_eval:
            if args.preload_to_device:
                assert device_tensors is not None and eval_indices is not None
                after = evaluate_policy_device(model, device_tensors, eval_indices, args.batch_size)
            else:
                assert eval_loader is not None
                after = evaluate_policy(model, eval_loader, device)
        summary = summarize_epoch_stats(stats)
        eval_text = "eval=run" if should_eval else "eval=skip"
        if should_eval:
            metric_text = format_eval(after)
        else:
            metric_text = "top1=NA top3=NA top5=NA mean_slot=NA entropy=NA"
        print(
            f"epoch={epoch} {eval_text} {metric_text} "
            f"pg={summary.get('policy_loss', 0.0):.4f} vf={summary.get('value_loss', 0.0):.4f} "
            f"belief={summary.get('team_belief_loss', 0.0):.4f} "
            f"belief_acc={summary.get('team_belief_accuracy', 0.0):.3f} "
            f"finish_recall={summary.get('team_belief_finishing_positive_recall', 0.0):.3f} "
            f"finish_precision={summary.get('team_belief_finishing_positive_precision', 0.0):.3f} "
            f"finish_pred_rate={summary.get('team_belief_finishing_predicted_positive_rate', 0.0):.3f} "
            f"finish_brier={summary.get('team_belief_finishing_brier', 0.0):.3f} "
            f"low_cost_recall={summary.get('team_belief_low_cost_positive_recall', 0.0):.3f} "
            f"low_cost_precision={summary.get('team_belief_low_cost_positive_precision', 0.0):.3f} "
            f"low_cost_pred_rate={summary.get('team_belief_low_cost_predicted_positive_rate', 0.0):.3f} "
            f"bomb_preserve_recall={summary.get('team_belief_bomb_preserving_positive_recall', 0.0):.3f} "
            f"bomb_preserve_precision={summary.get('team_belief_bomb_preserving_positive_precision', 0.0):.3f} "
            f"bomb_preserve_pred_rate={summary.get('team_belief_bomb_preserving_predicted_positive_rate', 0.0):.3f} "
            f"control_preserve_recall={summary.get('team_belief_control_preserving_positive_recall', 0.0):.3f} "
            f"control_preserve_precision={summary.get('team_belief_control_preserving_positive_precision', 0.0):.3f} "
            f"control_preserve_pred_rate={summary.get('team_belief_control_preserving_predicted_positive_rate', 0.0):.3f} "
            f"five_card_recall={summary.get('team_belief_five_card_positive_recall', 0.0):.3f} "
            f"five_card_precision={summary.get('team_belief_five_card_positive_precision', 0.0):.3f} "
            f"five_card_pred_rate={summary.get('team_belief_five_card_predicted_positive_rate', 0.0):.3f} "
            f"kl={summary.get('kl', 0.0):.5f} clip={summary.get('clip_rate', 0.0):.3f}",
            flush=True,
        )
    optimized_at = time.perf_counter()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    checkpoint_payload = {
        "model_state_dict": {
            key: value
            for key, value in snapshot_state_dict_to_cpu(raw_model).items()
        },
        "state_dim": STATE_DIM,
        "candidate_dim": CANDIDATE_DIM,
        "feature_version": FEATURE_VERSION,
        "history_protocol": (
            HISTORY_PROTOCOL if args.architecture in {"phase14", "team_belief"} else None
        ),
        "history_event_semantics": (
            HISTORY_EVENT_SEMANTICS
            if args.architecture in {"phase14", "team_belief"}
            else None
        ),
        "history_length": (
            HISTORY_LENGTH if args.architecture in {"phase14", "team_belief"} else 0
        ),
        "history_event_dim": (
            HISTORY_EVENT_DIM if args.architecture in {"phase14", "team_belief"} else 0
        ),
        "team_belief_protocol": (
            TEAM_BELIEF_PROTOCOL if args.architecture == "team_belief" else None
        ),
        "candidate_capacity": rollout_contract[0],
        "action_support": rollout_contract[1],
        "model_type": current_model_type,
        "model_config": model_config,
        "args": vars(args),
        "init_checkpoint": str(args.init_checkpoint),
        "init_mode": args.init_mode,
        "rollout_metadata": rollout_metadata,
        "before": before,
        "last_info": last_info,
        "kind": "top10_ppo",
        "init_checkpoint_metadata": init_payload.get("metadata", {}) if init_payload else {},
    }
    continuous_optimizer = env_enabled("DANRL_CONTINUOUS_OPTIMIZER", default=True)
    if continuous_optimizer:
        checkpoint_payload["training_state_file"] = training_state_path(output).name
        checkpoint_payload["optimizer_continuity"] = "adamw_sidecar_v1"
        save_optimizer_training_state(
            output,
            optimizer,
            model_type=current_model_type,
            model_config=model_config,
            optimizer_config=current_optimizer_config,
        )
    else:
        training_state_path(output).unlink(missing_ok=True)
    torch.save(checkpoint_payload, tmp_output)
    tmp_output.replace(output)
    publish_persistent_checkpoint(
        output,
        raw_model,
        checkpoint_payload,
        device,
        optimizer=optimizer if continuous_optimizer else None,
    )
    saved_at = time.perf_counter()
    print(f"saved={output}", flush=True)
    print(
        "timing_sec "
        f"rollout_load={rollout_loaded_at - job_started:.4f} "
        f"model_load={model_loaded_at - rollout_loaded_at:.4f} "
        f"data_prepare={data_ready_at - model_loaded_at:.4f} "
        f"optimize={optimized_at - data_ready_at:.4f} "
        f"checkpoint={saved_at - optimized_at:.4f} "
        f"total={saved_at - job_started:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
