#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine top10 BC shard npz files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays: dict[str, list[np.ndarray]] = {
        "state": [],
        "candidates": [],
        "mask": [],
        "label": [],
        "retrieval_rank": [],
        "sample_id": [],
        "human_kind": [],
    }
    stats = Counter()
    metadata: list[dict] = []
    seen: set[str] = set()
    duplicates = 0

    for path_text in args.inputs:
        path = Path(path_text)
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        sample_ids = data["sample_id"].astype(str)
        keep = []
        for i, sample_id in enumerate(sample_ids):
            if sample_id in seen:
                duplicates += 1
                continue
            seen.add(sample_id)
            keep.append(i)
        if not keep:
            continue
        keep_idx = np.array(keep, dtype=np.int64)
        for key in arrays:
            arrays[key].append(data[key][keep_idx])
        if "metadata" in data.files:
            meta = json.loads(str(data["metadata"]))
            metadata.append(meta)
            stats.update(meta.get("stats") or {})

    if not arrays["state"]:
        raise RuntimeError("no shard data found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined_meta = {
        "shards": [str(path) for path in args.inputs],
        "shard_count": len(args.inputs),
        "duplicates_dropped": duplicates,
        "source_metadata": metadata,
        "stats_sum": dict(stats),
    }
    np.savez_compressed(
        output,
        state=np.concatenate(arrays["state"], axis=0).astype(np.float32),
        candidates=np.concatenate(arrays["candidates"], axis=0).astype(np.float32),
        mask=np.concatenate(arrays["mask"], axis=0).astype(np.float32),
        label=np.concatenate(arrays["label"], axis=0).astype(np.int64),
        retrieval_rank=np.concatenate(arrays["retrieval_rank"], axis=0),
        sample_id=np.concatenate(arrays["sample_id"], axis=0),
        human_kind=np.concatenate(arrays["human_kind"], axis=0),
        metadata=json.dumps(combined_meta, ensure_ascii=False),
    )
    print(f"wrote={output} samples={sum(x.shape[0] for x in arrays['label'])} duplicates_dropped={duplicates}")


if __name__ == "__main__":
    main()
