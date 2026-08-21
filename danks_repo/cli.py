"""Unified command-line interface for navigating and validating DanKS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .archives import build_source_archive
from .catalog import GENERATION_CATALOG, generation_summary
from .repository import GENERATIONS, sha256_file
from .verify import verify_repository


ARCHIVE_NAMES = {
    "v1": "DanKS-v1-source.tar.gz",
    "v2": "DanKS-v2-source.tar.gz",
    "v3": "DanKS-v3-source.tar.gz",
    "all": "DanKS-source.tar.gz",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m danks_repo")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="DanKS repository root",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list the three generations")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    show_parser = subparsers.add_parser("show", help="show one generation")
    show_parser.add_argument("generation", choices=GENERATIONS)

    subparsers.add_parser("verify", help="audit repository and manifests")

    package_parser = subparsers.add_parser("package", help="build deterministic archives")
    package_parser.add_argument("--generation", choices=(*GENERATIONS, "all"), default="all")
    package_parser.add_argument("--output-dir", type=Path)
    return parser


def _list(repository: Path, *, as_json: bool) -> int:
    summaries = [generation_summary(repository, generation) for generation in GENERATIONS]
    if as_json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("Generation  Package          Files  Size")
    for summary in summaries:
        print(
            f"{summary['display_name']:<11} "
            f"{summary['package']:<16} "
            f"{summary['files']:>5}  "
            f"{summary['bytes']:>8} B"
        )
    return 0


def _show(repository: Path, generation: str) -> int:
    summary = generation_summary(repository, generation)
    metadata = GENERATION_CATALOG[generation]
    print(f"{summary['display_name']} ({generation})")
    print(f"Focus: {metadata['focus']}")
    print(f"Source: {summary['source_path']}")
    print(f"Package: {summary['package']}")
    print(f"Files: {summary['files']} ({summary['bytes']} bytes)")
    print(f"PYTHONPATH={summary['source_path']} python -c \"import {summary['package']}\"")
    print("Model weights: not included")
    print(f"Guide: generations/{generation}/README.md")
    return 0


def _verify(repository: Path) -> int:
    errors = verify_repository(repository)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: repository contract and all generation manifests are valid")
    return 0


def _package(repository: Path, generation: str, output_dir: Path | None) -> int:
    output_dir = (output_dir or repository / "dist").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [*GENERATIONS, "all"] if generation == "all" else [generation]
    outputs = []
    for item in requested:
        output = build_source_archive(repository, item, output_dir / ARCHIVE_NAMES[item])
        outputs.append(output)
        print(f"built {output}")
    checksums = "\n".join(
        f"{sha256_file(path)}  {path.name}" for path in sorted(outputs, key=lambda p: p.name)
    ) + "\n"
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve()
    if args.command == "list":
        return _list(repository, as_json=args.as_json)
    if args.command == "show":
        return _show(repository, args.generation)
    if args.command == "verify":
        return _verify(repository)
    if args.command == "package":
        return _package(repository, args.generation, args.output_dir)
    raise AssertionError(args.command)
