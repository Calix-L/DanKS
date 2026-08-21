#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OpenGuanDan rollout profile JSON shards.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--slowest", type=int, default=50)
    parser.add_argument("--compact", action="store_true", help="Do not embed every shard profile in the merged JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = []
    slowest = []
    rank_values = []
    total_rows = 0
    total_elapsed = 0.0
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles.append(data)
        total_rows += int(data.get("rows", 0))
        total_elapsed += float(data.get("elapsed_sec", 0.0))
        for item in data.get("slowest") or []:
            item = dict(item)
            item["profile"] = str(path)
            slowest.append(item)
        values = data.get("rank_ms_values") or []
        if values:
            rank_values.extend(float(value) for value in values)
        else:
            # Fallback for older shard profiles. Slowest rows remain exact from
            # each shard, while percentiles are approximated from worker avgs.
            rank = data.get("rank_ms") or {}
            count = int(rank.get("count", 0))
            avg = float(rank.get("avg", 0.0))
            if count > 0:
                rank_values.extend([avg] * count)

    arr = np.asarray(rank_values, dtype=np.float64)
    slowest = sorted(slowest, key=lambda item: float(item.get("rank_ms", 0.0)), reverse=True)[: args.slowest]
    merged = {
        "source": "merged_rollout_profiles",
        "inputs": [str(path) for path in args.inputs],
        "input_count": len(profiles),
        "rows": total_rows,
        "elapsed_sec_sum": total_elapsed,
        "rank_ms": {
            "count": int(arr.size),
            "avg": float(arr.mean()) if arr.size else 0.0,
            "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
            "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "exact": bool(any((profile.get("rank_ms_values") or []) for profile in profiles)),
        },
        "slowest": slowest,
        "profiles": [] if args.compact else profiles,
    }
    if args.compact:
        merged["profile_inputs"] = [str(path) for path in args.inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output} profiles={len(profiles)} rows={total_rows}")


if __name__ == "__main__":
    main()
