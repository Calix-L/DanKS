from pathlib import Path

import pytest

from danks_repo.repository import (
    ALLOWED_SOURCE_SUFFIXES,
    FORBIDDEN_DIRECTORY_NAMES,
    FORBIDDEN_SUFFIXES,
    GENERATIONS,
    REQUIRED_GENERATION_PATHS,
    REQUIRED_ROOT_PATHS,
    assert_safe_generation_destination,
)


def test_repository_contract_names_the_three_real_generations() -> None:
    assert tuple(GENERATIONS) == ("v1", "v2", "v3")


def test_repository_contract_requires_public_project_files() -> None:
    assert {
        "README.md",
        "ARCHITECTURE.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "LICENSE",
        "NOTICE",
        "Makefile",
        "pyproject.toml",
        ".gitignore",
        ".github/workflows/ci.yml",
        "docs/REPOSITORY_LAYOUT.md",
        "docs/REPRODUCIBILITY.md",
        "docs/MODEL_WEIGHTS.md",
        "docs/LICENSING.md",
        "docs/QUICKSTART.md",
        "docs/SOURCE_MAP.md",
        "danks_repo/__main__.py",
    } <= set(REQUIRED_ROOT_PATHS)


def test_repository_contract_requires_each_generation_snapshot() -> None:
    assert set(REQUIRED_GENERATION_PATHS) == {
        "generations/v1/source",
        "generations/v1/manifest.json",
        "generations/v1/README.md",
        "generations/v2/source",
        "generations/v2/manifest.json",
        "generations/v2/README.md",
        "generations/v3/source",
        "generations/v3/manifest.json",
        "generations/v3/README.md",
    }


def test_repository_contract_defines_source_only_boundary() -> None:
    assert {".pt", ".pth", ".onnx", ".npz", ".so", ".whl", ".zip", ".md", ".doc", ".docx", ".pdf", ".env"} <= set(
        FORBIDDEN_SUFFIXES
    )
    assert {"models", "checkpoints", "datasets", "eval_runs", ".venv", "__pycache__", "docs"} <= set(
        FORBIDDEN_DIRECTORY_NAMES
    )
    assert {".py", ".cpp", ".h", ".hpp", ".c", ".go", ".sh", ".json", ".toml", ".yaml", ".yml"} <= set(
        ALLOWED_SOURCE_SUFFIXES
    )
    assert {".md", ".doc", ".docx", ".pdf", ".txt", ".env"}.isdisjoint(
        ALLOWED_SOURCE_SUFFIXES
    )


def test_generation_destination_must_stay_below_generations(tmp_path: Path) -> None:
    root = tmp_path / "DanKS"
    safe = root / "generations" / "v3" / "source"
    assert assert_safe_generation_destination(root, safe) == safe.resolve()

    with pytest.raises(ValueError, match="generations"):
        assert_safe_generation_destination(root, root / "docs")
