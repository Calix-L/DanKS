#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.training.ppo import compute_gae  # noqa: E402
from DanRL_retrieval.training.schema import (  # noqa: E402
    CANDIDATE_DIM,
    HISTORY_EVENT_DIM,
    HISTORY_EVENT_SEMANTICS,
    HISTORY_LENGTH,
    HISTORY_PROTOCOL,
    STATE_DIM,
    TOPK,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sampled interactive_server JSONL trajectories to PPO npz.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reward-json", default=None, help="Optional mapping: game_id -> reward or game_id:uid -> reward.")
    parser.add_argument("--default-reward", type=float, default=0.0)
    parser.add_argument("--reward-mode", choices=("terminal", "dense"), default="terminal")
    parser.add_argument("--skip-forced", action="store_true", default=True)
    parser.add_argument("--keep-forced", action="store_false", dest="skip_forced")
    parser.add_argument("--allow-offpolicy-temperature", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    return parser.parse_args()


def load_rewards(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in raw.items()}


def reward_for(record: dict[str, Any], rewards: dict[str, float], default: float) -> float:
    game_id = str(record.get("game_id"))
    uid = str(record.get("uid"))
    episode_id = record.get("_episode_id")
    if episode_id is not None:
        specific = f"{game_id}:{uid}:{episode_id}"
        if specific in rewards:
            return rewards[specific]
    round_id = record.get("round_id")
    if round_id is not None:
        specific = f"{game_id}:{uid}:{round_id}"
        if specific in rewards:
            return rewards[specific]
    return rewards.get(f"{game_id}:{uid}", rewards.get(game_id, default))


def assign_episode_ids(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[int]]:
    by_player: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_player[(str(row.get("game_id")), str(row.get("uid")))].append(idx)

    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for (game_id, uid), indices in by_player.items():
        indices.sort(key=lambda i: int(rows[i].get("decision_no", i)))
        episode = 0
        last_history_len: int | None = None
        last_round_id: str | None = None
        for idx in indices:
            row = rows[idx]
            history_len = int(row.get("history_len", 0) or 0)
            round_id = row.get("round_id")
            round_text = str(round_id) if round_id is not None else None
            if last_history_len is not None and history_len < last_history_len:
                episode += 1
            elif last_round_id is not None and round_text is not None and round_text != last_round_id:
                episode += 1
            row["_episode_id"] = episode
            grouped[(game_id, uid, episode)].append(idx)
            last_history_len = history_len
            if round_text is not None:
                last_round_id = round_text
    return grouped


def main() -> None:
    args = parse_args()
    rewards_map = load_rewards(args.reward_json)
    rows: list[dict[str, Any]] = []
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if args.skip_forced and record.get("forced"):
            continue
        temperature = float(record.get("sample_temperature", 1.0) or 1.0)
        if not args.allow_offpolicy_temperature and abs(temperature - 1.0) > 1.0e-6:
            raise RuntimeError(
                "trajectory contains sample_temperature != 1.0; "
                "rerun collection with --sample-temperature 1.0 or pass --allow-offpolicy-temperature"
            )
        rows.append(record)

    if not rows:
        raise RuntimeError("no usable trajectory rows")

    state = np.asarray([row["state"] for row in rows], dtype=np.float32)
    candidates = np.asarray([row["candidates"] for row in rows], dtype=np.float32)
    mask = np.asarray([row["mask"] for row in rows], dtype=np.float32)
    history_synthesized = any(
        not isinstance(row.get("history"), list) or len(row["history"]) == 0
        for row in rows
    )
    history = np.asarray(
        [
            row.get("history")
            or np.zeros((HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
            for row in rows
        ],
        dtype=np.float32,
    )
    action = np.asarray([row["action_slot"] for row in rows], dtype=np.int64)
    logp = np.asarray([row["logp"] for row in rows], dtype=np.float32)
    value = np.asarray([row["value"] for row in rows], dtype=np.float32)
    dims_ok = (
        state.ndim == 2
        and candidates.ndim == 3
        and mask.ndim == 2
        and history.ndim == 3
        and state.shape[0] == candidates.shape[0] == mask.shape[0]
        and state.shape[1] == STATE_DIM
        and candidates.shape[1:] == (TOPK, CANDIDATE_DIM)
        and mask.shape[1] == TOPK
        and history.shape[1:] == (HISTORY_LENGTH, HISTORY_EVENT_DIM)
    )
    if not dims_ok:
        raise ValueError(
            f"trajectory dim mismatch: state={state.shape}, candidates={candidates.shape}, "
            f"mask={mask.shape}, history={history.shape}, "
            f"expected state_dim={STATE_DIM}, candidates=(*,{TOPK},{CANDIDATE_DIM}), mask=(*,{TOPK})"
        )
    if (
        not np.isfinite(state).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(mask).all()
        or not np.isfinite(history).all()
    ):
        raise ValueError("trajectory contains non-finite state/candidate/mask/history features")
    if not np.isfinite(logp).all() or not np.isfinite(value).all():
        raise ValueError("trajectory contains non-finite logp/value")
    if (action < 0).any() or (action >= TOPK).any():
        raise ValueError("trajectory action_slot outside top-k range")
    valid_action_mask = mask[np.arange(len(action)), action] > 0
    if not bool(valid_action_mask.all()):
        bad = np.where(~valid_action_mask)[0][:10].tolist()
        raise ValueError(f"trajectory contains action_slot masked out at rows={bad}")

    grouped = assign_episode_ids(rows)
    final_reward = np.asarray([reward_for(row, rewards_map, args.default_reward) for row in rows], dtype=np.float32)
    reward = np.zeros_like(final_reward, dtype=np.float32) if args.reward_mode == "terminal" else final_reward.copy()
    advantage = np.zeros_like(reward, dtype=np.float32)
    returns = np.zeros_like(reward, dtype=np.float32)
    done = np.zeros_like(reward, dtype=np.float32)
    for _key, indices in grouped.items():
        indices.sort(key=lambda i: int(rows[i].get("decision_no", i)))
        local_rewards = reward[indices]
        local_values = value[indices]
        local_dones = np.zeros(len(indices), dtype=np.float32)
        local_dones[-1] = 1.0
        if args.reward_mode == "terminal":
            reward[indices[:-1]] = 0.0
            reward[indices[-1]] = final_reward[indices[-1]]
            local_rewards = reward[indices]
        local_adv, local_ret = compute_gae(local_rewards, local_values, local_dones, gamma=args.gamma, lam=args.lam)
        advantage[indices] = local_adv
        returns[indices] = local_ret
        done[indices[-1]] = 1.0

    metadata = {
        "source_jsonl": str(args.jsonl),
        "reward_json": str(args.reward_json) if args.reward_json else None,
        "default_reward": args.default_reward,
        "reward_mode": args.reward_mode,
        "skip_forced": args.skip_forced,
        "allow_offpolicy_temperature": args.allow_offpolicy_temperature,
        "rows": len(rows),
        "groups": len(grouped),
        "reward_mean": float(reward.mean()),
        "reward_nonzero": int((reward != 0).sum()),
        "history_protocol": None if history_synthesized else HISTORY_PROTOCOL,
        "history_event_semantics": (
            HISTORY_EVENT_SEMANTICS
            if not history_synthesized
            and {row.get("history_event_semantics") for row in rows}
            == {HISTORY_EVENT_SEMANTICS}
            else None
        ),
        "history_synthesized": history_synthesized,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state=state,
        candidates=candidates,
        mask=mask,
        history=history,
        action=action,
        logp=logp,
        value=value,
        reward=reward,
        done=done,
        advantage=advantage,
        returns=returns,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"wrote={output} rows={len(rows)} groups={len(grouped)} reward_mean={float(reward.mean()):.3f}")


if __name__ == "__main__":
    main()
