"""Strict readers for frozen CardKS evaluation split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_guandan_paired_split(path: str | Path) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Return ``(pair_index, deal_seed)`` rows without silently reshaping a split."""

    split_path = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid split JSON at line {line_number}: {exc}") from exc
        if row.get("schema_version") != "cardks.ablation.split_seed.v1":
            raise ValueError(f"unexpected split schema at line {line_number}")
        if row.get("game") != "guandan":
            raise ValueError(f"split row {line_number} is not for guandan")
        if int(row.get("games_per_pair", -1)) != 2 or row.get("assignments") != ["A", "B"]:
            raise ValueError(f"split row {line_number} is not an A/B paired deal")
        rows.append(row)
    if not rows:
        raise ValueError("split manifest is empty")
    protocol_ids = {str(row.get("protocol_id")) for row in rows}
    splits = {str(row.get("split")) for row in rows}
    if len(protocol_ids) != 1 or len(splits) != 1:
        raise ValueError("split manifest mixes protocol ids or split names")
    pairs = [(int(row["pair_index"]), int(row["deal_seed"])) for row in rows]
    pair_indices = [pair_index for pair_index, _ in pairs]
    seeds = [seed for _, seed in pairs]
    if len(set(pair_indices)) != len(pairs) or len(set(seeds)) != len(pairs):
        raise ValueError("split manifest contains duplicate pair indices or deal seeds")
    if sorted(pair_indices) != list(range(len(pairs))):
        raise ValueError("split pair indices must be contiguous from zero")
    identity = {
        "path": str(split_path),
        "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "protocol_id": next(iter(protocol_ids)),
        "split": next(iter(splits)),
        "pairs": len(pairs),
    }
    return pairs, identity
