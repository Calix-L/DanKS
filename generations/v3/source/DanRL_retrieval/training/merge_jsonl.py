#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate JSONL files while skipping blank lines.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", encoding="utf-8") as out:
        for raw_path in args.inputs:
            path = Path(raw_path)
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                out.write(line.rstrip("\n") + "\n")
                rows += 1
    print(f"wrote={output} rows={rows} inputs={len(args.inputs)}")


if __name__ == "__main__":
    main()
