#!/usr/bin/env python3
"""Rank a deterministic set of legal actions with one DanKS generation."""

from __future__ import annotations

import argparse

import DanKS
from DanKS.retrieval import StructuralCandidateRanker, build_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DanKS.GENERATION != args.version:
        raise RuntimeError(
            f"installed {DanKS.GENERATION} package does not match --version {args.version}; "
            f"install it with: python -m pip install -e versions/{args.version}"
        )

    hand = ["S3", "H3", "C4", "D5", "S6"]
    legal_actions = [
        {"index": 0, "kind": "Single", "cards": ["S3"], "rank": "3"},
        {"index": 1, "kind": "Pair", "cards": ["S3", "H3"], "rank": "3"},
    ]
    context = build_context(
        {"my_seat": 0, "curRank": "2", "current_kind": "Lead"}
    )
    ranked = StructuralCandidateRanker(max_partitions=4).rank(
        hand,
        legal_actions,
        context,
        top_k=2,
    )
    if not ranked:
        raise RuntimeError("retrieval returned no ranked action")

    top = ranked[0]
    cards = " ".join(top.action.cards)
    print(f"DanKS {args.version.upper()} retrieval ready")
    print(f"Top action: {top.action.kind} [{cards}] score={top.score:.3f}")


if __name__ == "__main__":
    main()
