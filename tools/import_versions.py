#!/usr/bin/env python3
"""Import the three private upstream trees as filtered code-only snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from danks_repo.repository import copy_snapshot, write_generation_manifest


V3_INCLUDED_TESTS = frozenset(
    {
        "test_accelerator.py",
        "test_action_generator.py",
        "test_b5_trace.py",
        "test_candidate_coverage.py",
        "test_card_memory.py",
        "test_diagnostics.py",
        "test_eval_manifest.py",
        "test_persistent_ppo_transport.py",
        "test_plm_rules.py",
        "test_pressure.py",
        "test_type_suppression.py",
        "test_uniform_action_seed.py",
    }
)


def v3_excluded_relative_paths(source_root: Path) -> set[str]:
    """Return the frozen V3 boundary, including its runnable core test set."""

    exclusions = {
        "training/v13_labels.py",
        "training/v13_model.py",
        "training/v13_schema.py",
    }
    tests_root = source_root.resolve() / "tests"
    if tests_root.is_dir():
        exclusions.update(
            f"tests/{path.name}"
            for path in tests_root.glob("test_*.py")
            if path.name not in V3_INCLUDED_TESTS
        )
    return exclusions


def build_text_replacements(prefixes: list[str]) -> dict[bytes, bytes]:
    """Map caller-supplied private workspace prefixes to a public placeholder."""

    replacements: dict[bytes, bytes] = {}
    for prefix in prefixes:
        if not prefix or not Path(prefix).is_absolute() or "\x00" in prefix:
            raise ValueError("redaction prefixes must be non-empty absolute paths")
        normalized = prefix.rstrip("/")
        if not normalized:
            raise ValueError("redaction prefixes must be non-empty absolute paths")
        replacements[normalized.encode("utf-8")] = b"/workspace/danks"
    return replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--v1-source", type=Path, required=True)
    parser.add_argument("--v2-source", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-model-source", type=Path, required=True)
    parser.add_argument("--v3-ppo-source", type=Path, required=True)
    parser.add_argument("--v3-train-ppo-source", type=Path, required=True)
    parser.add_argument(
        "--redact-prefix",
        action="append",
        required=True,
        help="private absolute path prefix to replace; repeat as needed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    release_roots = {"DanKS", "KSplatform", "scripts", "requirements.txt"}
    v3_roots = {
        "__init__.py",
        "environment",
        "openguandan_adapter",
        "plm_adapter",
        "retrieval",
        "tests",
        "training",
    }
    sources = {
        "v1": (args.v1_source, "sanitized-v1-source", release_roots),
        "v2": (args.v2_source, "sanitized-v2-source", release_roots),
        "v3": (args.v3_source, "sanitized-v3-source", v3_roots),
    }
    text_replacements = build_text_replacements(args.redact_prefix)
    v3_exclusions = v3_excluded_relative_paths(args.v3_source)
    v3_overlays = {
        "training/model.py": args.v3_model_source,
        "training/ppo.py": args.v3_ppo_source,
        "training/train_ppo.py": args.v3_train_ppo_source,
    }
    for generation, (source, source_label, allowed_top_level) in sources.items():
        destination = repository / "generations" / generation / "source"
        excluded_relative_paths = None
        overlay_files = None
        if generation == "v3":
            destination = destination / "DanRL_retrieval"
            excluded_relative_paths = v3_exclusions
            overlay_files = v3_overlays
        copied = copy_snapshot(
            source,
            destination,
            repository,
            allowed_top_level=allowed_top_level,
            excluded_relative_paths=excluded_relative_paths,
            overlay_files=overlay_files,
            text_replacements=text_replacements,
        )
        write_generation_manifest(repository, generation, source_label=source_label)
        print(f"{generation}: imported {len(copied)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
