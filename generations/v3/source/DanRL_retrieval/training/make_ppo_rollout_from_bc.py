#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.training.model import build_selector_from_checkpoint  # noqa: E402
from DanRL_retrieval.training.ppo import compute_gae, masked_categorical  # noqa: E402
from DanRL_retrieval.training.schema import CANDIDATE_DIM, FEATURE_VERSION, STATE_DIM  # noqa: E402
from DanRL_retrieval.training.accelerator import initialize_device, seed_accelerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PPO-format smoke rollout from frozen top10 BC data.")
    parser.add_argument("--data", default=str(ROOT / "DanRL_retrieval" / "data" / "top10_bc_plm_go_ai_50k_fullsearch_v3_rank2fix.npz"))
    parser.add_argument("--checkpoint", default=str(ROOT / "DanRL_retrieval" / "checkpoints" / "top10_selector_ce_pairwise_plm_go_ai_50k_fullsearch_v3_rank2fix.pt"))
    parser.add_argument("--output", default=str(ROOT / "DanRL_retrieval" / "data" / "ppo_rollout_bc_smoke.npz"))
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reward-correct", type=float, default=1.0)
    parser.add_argument("--reward-wrong", type=float, default=-0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = initialize_device(args.device)
    if device.type in {"cuda", "npu"}:
        seed_accelerator(device, args.seed)

    data = np.load(args.data, allow_pickle=True)
    data_metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    data_feature_version = data_metadata.get("feature_version")
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

    n = min(args.samples, len(label))
    idx = np.random.choice(len(label), size=n, replace=False)
    state = state[idx]
    candidates = candidates[idx]
    mask = mask[idx]
    label = label[idx]

    payload = torch.load(args.checkpoint, map_location="cpu")
    ckpt_state_dim = int(payload.get("state_dim", -1))
    ckpt_candidate_dim = int(payload.get("candidate_dim", -1))
    if ckpt_state_dim != STATE_DIM or ckpt_candidate_dim != CANDIDATE_DIM:
        raise ValueError(f"checkpoint dim mismatch: {ckpt_state_dim}/{ckpt_candidate_dim}")
    ckpt_metadata = payload.get("metadata") or {}
    ckpt_feature_version = payload.get("feature_version") or ckpt_metadata.get("feature_version")
    if str(ckpt_feature_version) != FEATURE_VERSION:
        raise ValueError(f"checkpoint feature_version mismatch: {ckpt_feature_version!r} != {FEATURE_VERSION!r}")
    model = build_selector_from_checkpoint(payload, device=device)
    model.eval()

    with torch.no_grad():
        st = torch.from_numpy(state).to(device)
        ca = torch.from_numpy(candidates).to(device)
        ma = torch.from_numpy(mask).to(device)
        logits, value = model(st, ca, ma)
        pi = masked_categorical(logits, ma)
        # For the smoke rollout, use human action as behavior action. PPO then
        # verifies clipped-policy updates on the exact frozen top10 interface.
        action_t = torch.from_numpy(label).to(device)
        logp = pi.log_prob(action_t).detach().cpu().numpy().astype(np.float32)
        values = value.detach().cpu().numpy().astype(np.float32)

    rewards = np.full(n, args.reward_correct, dtype=np.float32)
    # Independent one-step pseudo episodes: this is a code-path smoke test, not
    # a substitute for true self-play returns.
    dones = np.ones(n, dtype=np.float32)
    advantage, returns = compute_gae(rewards, values, dones, gamma=args.gamma, lam=args.lam)

    metadata = {
        "source": str(args.data),
        "checkpoint": str(args.checkpoint),
        "samples": int(n),
        "kind": "bc_one_step_smoke",
        "note": "PPO-format smoke rollout; rewards are imitation placeholders, not self-play returns.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state=state.astype(np.float32),
        candidates=candidates.astype(np.float32),
        mask=mask.astype(np.float32),
        action=label.astype(np.int64),
        logp=logp,
        value=values,
        reward=rewards,
        done=dones,
        advantage=advantage,
        returns=returns,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"wrote={output} samples={n} device={device}")
    print(f"reward_mean={float(rewards.mean()):.3f} value_mean={float(values.mean()):.3f} adv_mean={float(advantage.mean()):.3f}")


if __name__ == "__main__":
    main()
