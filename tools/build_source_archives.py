#!/usr/bin/env python3
"""Build deterministic DanKS source archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from danks_repo.archives import build_source_archive
from danks_repo.repository import GENERATIONS, sha256_file


ARCHIVE_NAMES = {
    "v1": "DanKS-v1-source.tar.gz",
    "v2": "DanKS-v2-source.tar.gz",
    "v3": "DanKS-v3-source.tar.gz",
    "combined": "DanKS-source.tar.gz",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generation", choices=(*GENERATIONS, "all"), default="all")
    args = parser.parse_args()
    repository = args.repository.resolve()
    output_dir = (args.output_dir or repository / "dist").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [args.generation]
    if args.generation == "all":
        targets = [*GENERATIONS, "combined"]
    outputs = []
    for target in targets:
        archive_generation = "all" if target == "combined" else target
        output = build_source_archive(
            repository, archive_generation, output_dir / ARCHIVE_NAMES[target]
        )
        outputs.append(output)
        print(f"built {output.name}")
    checksums = "\n".join(
        f"{sha256_file(path)}  {path.name}" for path in sorted(outputs, key=lambda p: p.name)
    ) + "\n"
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
