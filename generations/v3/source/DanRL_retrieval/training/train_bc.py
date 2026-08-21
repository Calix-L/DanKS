#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.training.accelerator import (  # noqa: E402
    cpu_state_dict,
    device_name,
    initialize_device,
    is_accelerator,
    seed_accelerator,
)
from DanRL_retrieval.training.model import Top10Selector, pairwise_margin_loss  # noqa: E402
from DanRL_retrieval.training.schema import CANDIDATE_DIM, FEATURE_VERSION, STATE_DIM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC/listwise selector on frozen retrieval top10 data.")
    parser.add_argument("--data", default=str(ROOT / "DanRL_retrieval" / "data" / "top10_bc.npz"))
    parser.add_argument("--output", default=str(ROOT / "DanRL_retrieval" / "checkpoints" / "top10_selector_bc.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--candidate-hidden-dim", type=int, default=192)
    parser.add_argument("--loss", choices=("ce", "pairwise", "ce_pairwise"), default="ce_pairwise")
    parser.add_argument("--pairwise-weight", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, npu:0, ...")
    parser.add_argument("--amp", action="store_true", help="Use CUDA/NPU automatic mixed precision.")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--compile-mode", default=None, choices=("default", "reduce-overhead", "max-autotune"), help="Optional torch.compile mode.")
    parser.add_argument("--fused-optimizer", action="store_true", help="Use fused AdamW on CUDA/NPU when supported.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Pin host tensors for faster CUDA transfer.")
    parser.add_argument("--preload-to-device", action="store_true", help="Move the full dataset to the training device and batch with device-side indices.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--eval-max-samples", type=int, default=0, help="Use at most this many fixed validation samples; 0 means full validation.")
    parser.add_argument("--skip-first-eval", action="store_true", help="Do not force validation after epoch 1.")
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without validation improvement.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation metric improvement.")
    return parser.parse_args()


def make_loader(
    arrays: tuple[torch.Tensor, ...],
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    non_blocking = device.type == "cuda"
    return tuple(x.to(device, non_blocking=non_blocking) for x in batch)


def iter_device_batches(
    arrays: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
):
    if shuffle:
        order = indices[torch.randperm(indices.numel(), device=indices.device)]
    else:
        order = indices
    for start in range(0, order.numel(), batch_size):
        batch_idx = order[start : start + batch_size]
        yield tuple(array.index_select(0, batch_idx) for array in arrays)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if device.type == "npu":
        from torch_npu.npu.amp import GradScaler

        return GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_dtype_from_arg(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_adamw(parameters, *, lr: float, weight_decay: float, fused: bool, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if fused and device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except TypeError:
            print("warning: fused AdamW not supported by this torch build; falling back to AdamW", flush=True)
    if fused and device.type == "npu":
        from torch_npu.optim import NpuFusedAdamW

        return NpuFusedAdamW(parameters, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)


@torch.no_grad()
def evaluate(model: Top10Selector, loader: DataLoader, device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top3 = 0
    mean_slot = 0.0
    for batch in loader:
        state, candidates, mask, label = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(state, candidates, mask)
        loss = F.cross_entropy(logits, label)
        pred = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
        total += label.numel()
        loss_sum += float(loss.item()) * label.numel()
        top1 += int((pred[:, 0] == label).sum().item())
        top3 += int((pred == label[:, None]).any(dim=1).sum().item())
        mean_slot += float((pred[:, 0].float() + 1.0).sum().item())
    denom = max(1, total)
    return {
        "loss": loss_sum / denom,
        "top1": top1 / denom,
        "top3": top3 / denom,
        "mean_selected_slot": mean_slot / denom,
    }


@torch.no_grad()
def evaluate_device(
    model: Top10Selector,
    arrays: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top3 = 0
    mean_slot = 0.0
    for batch in iter_device_batches(arrays, indices, batch_size, shuffle=False):
        state, candidates, mask, label = batch
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(state, candidates, mask)
        loss = F.cross_entropy(logits, label)
        pred = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
        total += label.numel()
        loss_sum += float(loss.item()) * label.numel()
        top1 += int((pred[:, 0] == label).sum().item())
        top3 += int((pred == label[:, None]).any(dim=1).sum().item())
        mean_slot += float((pred[:, 0].float() + 1.0).sum().item())
    denom = max(1, total)
    return {
        "loss": loss_sum / denom,
        "top1": top1 / denom,
        "top3": top3 / denom,
        "mean_selected_slot": mean_slot / denom,
    }


def retrieval_baseline(label: np.ndarray) -> dict[str, float]:
    denom = max(1, int(label.shape[0]))
    return {
        "top1": float((label == 0).sum() / denom),
        "top3": float((label < 3).sum() / denom),
        "mean_human_slot": float((label.astype(np.float32) + 1.0).mean()) if denom else 0.0,
    }


def main() -> None:
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
    pin_memory = bool(args.pin_memory or device.type == "cuda")
    if is_accelerator(device):
        seed_accelerator(device, args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if is_accelerator(device):
        print(
            f"device={device} accelerator_name={device_name(device)} "
            f"amp={use_amp} amp_dtype={args.amp_dtype} matmul_precision={args.matmul_precision} "
            f"npu_jit_compile={npu_jit_compile}",
            flush=True,
        )
    else:
        print(f"device={device} amp=False", flush=True)

    data = np.load(args.data, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    data_feature_version = metadata.get("feature_version")
    if str(data_feature_version) != FEATURE_VERSION:
        raise ValueError(f"dataset feature_version mismatch: {data_feature_version!r} != {FEATURE_VERSION!r}")
    state = data["state"].astype(np.float32)
    candidates = data["candidates"].astype(np.float32)
    mask = data["mask"].astype(np.float32)
    label = data["label"].astype(np.int64)
    valid = label >= 0
    state, candidates, mask, label = state[valid], candidates[valid], mask[valid], label[valid]
    if state.shape[1] != STATE_DIM or candidates.shape[2] != CANDIDATE_DIM:
        raise ValueError(
            f"feature dimension mismatch: state={state.shape[1]}/{STATE_DIM} "
            f"candidate={candidates.shape[2]}/{CANDIDATE_DIM}"
        )
    n = label.shape[0]
    if n < 2:
        raise RuntimeError("need at least two samples to train")
    order = np.random.permutation(n)
    val_n = max(1, int(n * args.val_ratio))
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    if len(train_idx) == 0:
        train_idx, val_idx = order, order
    eval_idx = val_idx
    if args.eval_max_samples > 0 and len(eval_idx) > args.eval_max_samples:
        eval_idx = eval_idx[: args.eval_max_samples]
    base_train = retrieval_baseline(label[train_idx])
    base_val = retrieval_baseline(label[val_idx])
    print(
        f"samples train={len(train_idx)} val={len(val_idx)} "
        f"retrieval_baseline_train top1={base_train['top1']:.1%} top3={base_train['top3']:.1%} "
        f"mean_human_slot={base_train['mean_human_slot']:.2f}; "
        f"val top1={base_val['top1']:.1%} top3={base_val['top3']:.1%} "
        f"mean_human_slot={base_val['mean_human_slot']:.2f}",
        flush=True,
    )

    tensors = tuple(torch.from_numpy(x) for x in (state, candidates, mask, label))
    if args.preload_to_device:
        device_tensors = tuple(t.to(device, non_blocking=True) for t in tensors)
        train_index_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        val_index_tensor = torch.as_tensor(eval_idx, dtype=torch.long, device=device)
        train_loader = None
        val_loader = None
        print(f"preload_to_device=true rows={n} approx_mb={(state.nbytes + candidates.nbytes + mask.nbytes + label.nbytes) / 1024 / 1024:.1f}", flush=True)
    else:
        device_tensors = None
        train_index_tensor = None
        val_index_tensor = None
        train_loader = make_loader(
            tuple(t[train_idx] for t in tensors),
            args.batch_size,
            True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_loader(
            tuple(t[eval_idx] for t in tensors),
            args.batch_size,
            False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    raw_model = Top10Selector(STATE_DIM, CANDIDATE_DIM, args.hidden_dim, args.candidate_hidden_dim).to(device)
    model: torch.nn.Module = raw_model
    if args.compile:
        if hasattr(torch, "compile"):
            compile_mode = None if args.compile_mode in (None, "default") else args.compile_mode
            model = torch.compile(model, mode=compile_mode)
        else:
            raise RuntimeError("torch.compile requested, but this torch version does not provide it.")
    optimizer = make_adamw(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_optimizer, device=device)
    scaler = make_grad_scaler(device, use_amp and amp_dtype == torch.float16)
    best_top1 = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    best_payload: dict[str, object] | None = None
    last_val: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        if args.preload_to_device:
            assert device_tensors is not None and train_index_tensor is not None
            train_iter = iter_device_batches(device_tensors, train_index_tensor, args.batch_size, shuffle=True)
        else:
            assert train_loader is not None
            train_iter = (move_batch(batch, device) for batch in train_loader)
        for batch_state, batch_candidates, batch_mask, batch_label in train_iter:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(batch_state, batch_candidates, batch_mask)
                ce = F.cross_entropy(logits, batch_label)
                pair = pairwise_margin_loss(logits, batch_label, batch_mask)
                if args.loss == "ce":
                    loss = ce
                elif args.loss == "pairwise":
                    loss = pair
                else:
                    loss = ce + args.pairwise_weight * pair
            if type(optimizer).__module__.startswith("torch_npu.optim"):
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += batch_label.numel()
            loss_sum += float(loss.item()) * batch_label.numel()

        should_eval = (
            (epoch == 1 and not args.skip_first_eval)
            or epoch == args.epochs
            or args.eval_every <= 1
            or epoch % args.eval_every == 0
        )
        if should_eval:
            if args.preload_to_device:
                assert device_tensors is not None and val_index_tensor is not None
                val = evaluate_device(model, device_tensors, val_index_tensor, args.batch_size, device, use_amp, amp_dtype)
            else:
                assert val_loader is not None
                val = evaluate(model, val_loader, device, use_amp, amp_dtype)
            last_val = val
        else:
            val = last_val
        train_loss = loss_sum / max(1, total)
        eval_text = "eval=run" if should_eval else "eval=skip"
        if val is None:
            val_text = "val_loss=NA top1=NA top3=NA mean_slot=NA"
        else:
            val_text = (
                f"val_loss={val['loss']:.4f} top1={val['top1']:.1%} "
                f"top3={val['top3']:.1%} mean_slot={val['mean_selected_slot']:.2f}"
            )
        print(f"epoch={epoch} {eval_text} train_loss={train_loss:.4f} {val_text}", flush=True)
        improved_top1 = should_eval and val is not None and val["top1"] > best_top1 + args.min_delta
        improved_loss = should_eval and val is not None and val["loss"] < best_loss - args.min_delta
        if improved_top1 or improved_loss:
            best_top1 = val["top1"]
            best_loss = min(best_loss, val["loss"])
            stale_epochs = 0
            best_payload = {
                "model_state_dict": cpu_state_dict(raw_model),
                "state_dim": STATE_DIM,
                "candidate_dim": CANDIDATE_DIM,
                "feature_version": FEATURE_VERSION,
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "candidate_hidden_dim": args.candidate_hidden_dim,
                },
                "args": vars(args),
                "device": str(device),
                "metadata": metadata,
                "retrieval_baseline_train": base_train,
                "retrieval_baseline_val": base_val,
                "val": val,
            }
        else:
            stale_epochs += 1 if should_eval else 0
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} stale_epochs={stale_epochs} best_top1={best_top1:.1%} best_loss={best_loss:.4f}", flush=True)
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        best_payload
        or {
            "model_state_dict": cpu_state_dict(raw_model),
            "state_dim": STATE_DIM,
            "candidate_dim": CANDIDATE_DIM,
            "feature_version": FEATURE_VERSION,
            "args": vars(args),
            "metadata": metadata,
        },
        output,
    )
    print(f"saved={output} best_top1={best_top1:.1%}")


if __name__ == "__main__":
    main()
