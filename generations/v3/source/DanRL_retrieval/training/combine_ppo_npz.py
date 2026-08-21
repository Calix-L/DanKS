#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.training.schema import (
    FEATURE_VERSION, HISTORY_EVENT_DIM, HISTORY_EVENT_SEMANTICS, HISTORY_LENGTH, HISTORY_PROTOCOL,
    TEAM_BELIEF_PROTOCOL, TOPK, FULL_LEGAL,
    normalize_candidate_contract,
)


PPO_KEYS = (
    "state",
    "candidates",
    "mask",
    "history",
    "action",
    "logp",
    "value",
    "reward",
    "done",
    "advantage",
    "returns",
)
TEAM_BELIEF_KEYS = ("team_belief_labels", "team_belief_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine PPO rollout shard npz files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compressed", action="store_true", help="Write compressed NPZ; default is uncompressed for faster training iteration handoff.")
    parser.add_argument("--max-rows", type=int, default=0, help="Keep at most this many rows after combining; 0 keeps all rows.")
    parser.add_argument("--sample", choices=("recent", "random"), default="recent", help="Row selection policy when --max-rows is smaller than the combined row count.")
    parser.add_argument("--seed", type=int, default=20260707, help="Random seed for --sample random.")
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in (*PPO_KEYS, *TEAM_BELIEF_KEYS)
    }
    metadata = []
    contracts: list[tuple[int, str]] = []
    storage_capacities: list[int] = []
    history_protocols: set[str | None] = set()
    history_event_semantics: set[str | None] = set()
    history_synthesized = False
    feature_versions: set[str | None] = set()
    team_belief_presence: list[bool] = []
    team_belief_protocols: set[str | None] = set()
    kept_inputs = []
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        required = tuple(key for key in PPO_KEYS if key != "history")
        if any(key not in data.files for key in required):
            raise ValueError(f"{path} is missing PPO rollout keys")
        rows = int(data["action"].shape[0])
        if rows <= 0:
            continue
        source_metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
        feature_versions.add(
            str(source_metadata["feature_version"])
            if source_metadata.get("feature_version")
            else None
        )
        history_present = "history" in data.files
        history_synthesized = bool(
            history_synthesized
            or source_metadata.get("history_synthesized", False)
            or not history_present
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
        belief_present = [key in data.files for key in TEAM_BELIEF_KEYS]
        if any(belief_present) and not all(belief_present):
            raise ValueError(f"{path} has an incomplete team-belief rollout contract")
        has_team_belief = all(belief_present)
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
        storage_capacity = int(source_metadata.get("storage_candidate_capacity", data["candidates"].shape[1]))
        if (storage_capacity <= 0 or data["candidates"].shape[1] != storage_capacity
                or data["mask"].shape[1] != storage_capacity):
            raise ValueError(f"{path} candidate storage metadata/shape mismatch")
        if contract[1] != FULL_LEGAL and storage_capacity != contract[0]:
            raise ValueError(f"{path} structured_topk width does not match candidate_capacity")
        contracts.append(contract)
        storage_capacities.append(storage_capacity)
        for key in PPO_KEYS:
            if key == "history" and key not in data.files:
                arrays[key].append(
                    np.zeros((rows, HISTORY_LENGTH, HISTORY_EVENT_DIM), dtype=np.float32)
                )
            else:
                arrays[key].append(data[key])
        if has_team_belief:
            for key in TEAM_BELIEF_KEYS:
                arrays[key].append(data[key])
        if "metadata" in data.files:
            metadata.append(source_metadata)
        kept_inputs.append(str(path))

    if not arrays["state"]:
        raise RuntimeError("no PPO shard data found")

    if len(set(contracts)) != 1:
        raise ValueError("PPO shards mix candidate contracts")
    if feature_versions != {FEATURE_VERSION}:
        raise ValueError(
            "PPO shard feature versions do not match the active schema: "
            f"shards={sorted(map(str, feature_versions))} current={FEATURE_VERSION!r}"
        )
    if any(team_belief_presence) and not all(team_belief_presence):
        raise ValueError("PPO shards mix team-belief supervision contracts")
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
        key: np.concatenate(values, axis=0)
        for key, values in arrays.items()
        if values
    }
    combined_rows = int(out["action"].shape[0])
    selected_rows = combined_rows
    if args.max_rows > 0 and combined_rows > args.max_rows:
        selected_rows = int(args.max_rows)
        if args.sample == "random":
            rng = np.random.default_rng(args.seed)
            indices = np.sort(rng.choice(combined_rows, size=selected_rows, replace=False))
        else:
            indices = np.arange(combined_rows - selected_rows, combined_rows)
        out = {key: value[indices] for key, value in out.items()}
    combined_meta = {
        "source": "combined_ppo_rollout",
        "inputs": kept_inputs,
        "input_count": len(kept_inputs),
        "rows": int(out["action"].shape[0]),
        "combined_rows_before_sampling": combined_rows,
        "selected_rows": selected_rows,
        "sample": args.sample,
        "max_rows": int(args.max_rows),
        "source_metadata": metadata,
        "feature_version": FEATURE_VERSION,
        "candidate_capacity": candidate_capacity,
        "storage_candidate_capacity": storage_candidate_capacity,
        "action_support": action_support,
        "history_protocol": (
            HISTORY_PROTOCOL
            if history_protocols == {HISTORY_PROTOCOL} and not history_synthesized
            else None
        ),
        "history_event_semantics": (
            HISTORY_EVENT_SEMANTICS
            if history_event_semantics == {HISTORY_EVENT_SEMANTICS}
            and not history_synthesized
            else None
        ),
        "history_synthesized": history_synthesized,
        "team_belief_protocol": (
            TEAM_BELIEF_PROTOCOL
            if team_belief_protocols == {TEAM_BELIEF_PROTOCOL}
            and team_belief_presence and all(team_belief_presence)
            else None
        ),
        "reward_mean": float(out["reward"].mean()),
        "reward_nonzero": int((out["reward"] != 0).sum()),
        "done_count": int(out["done"].sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(output, **out, metadata=json.dumps(combined_meta, ensure_ascii=False))
    print(f"wrote={output} rows={combined_meta['rows']} inputs={len(kept_inputs)} compressed={bool(args.compressed)}")


if __name__ == "__main__":
    main()
