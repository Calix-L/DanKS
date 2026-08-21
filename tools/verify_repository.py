#!/usr/bin/env python3
"""Validate the DanKS public repository and all generation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from danks_repo.verify import verify_repository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = verify_repository(args.repository)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: repository contract and all generation manifests are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
