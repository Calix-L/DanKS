import ast
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PLATFORM_TOKEN = "p" + "lm"
LEGACY_V3_PACKAGE = "Dan" + "RL_retrieval"
FORBIDDEN_GENERATION_DIRECTORIES = {
    "KSplatform",
    "scripts",
    "tests",
    "gdai_adapter",
    "openguandan_adapter",
    "table_runtime",
    FORBIDDEN_PLATFORM_TOKEN + "_adapter",
    FORBIDDEN_PLATFORM_TOKEN + "_eval",
}


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


def test_public_generations_are_lean_and_platform_neutral() -> None:
    generation_root = REPOSITORY_ROOT / "generations"
    source_files = [
        path
        for path in generation_root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and path.suffix.lower() in {".py", ".cpp", ".h", ".hpp", ".c", ".sh", ".md", ".txt"}
    ]

    assert not {
        part
        for path in source_files
        for part in path.relative_to(generation_root).parts
        if part in FORBIDDEN_GENERATION_DIRECTORIES
    }
    assert not [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in source_files
        if FORBIDDEN_PLATFORM_TOKEN in path.as_posix().lower()
        or FORBIDDEN_PLATFORM_TOKEN in path.read_text(encoding="utf-8").lower()
    ]

    v3_training = generation_root / "v3/source/DanKS/training"
    assert {
        "model.py",
        "ppo.py",
        "train_ppo.py",
        "featurizer.py",
        "schema.py",
        "team_belief.py",
    } <= {path.name for path in v3_training.glob("*.py")}


def test_generation_absolute_imports_resolve_inside_each_snapshot() -> None:
    missing: list[str] = []
    for generation, package in (("v1", "DanKS"), ("v2", "DanKS"), ("v3", "DanKS")):
        source_root = REPOSITORY_ROOT / "generations" / generation / "source"
        for source_path in source_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith(package + "."):
                    continue
                target = source_root.joinpath(*node.module.split("."))
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    missing.append(
                        f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno}: {node.module}"
                    )

    assert missing == []


def test_legacy_v3_package_name_is_absent() -> None:
    offenders = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "dist", "__pycache__"} for part in path.parts):
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if LEGACY_V3_PACKAGE.lower() in relative.lower():
            offenders.append(relative)
            continue
        if path.suffix.lower() in {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt"}:
            if LEGACY_V3_PACKAGE.lower() in path.read_text(encoding="utf-8").lower():
                offenders.append(relative)

    assert offenders == []


def test_shared_guandan_engine_is_packaged_once() -> None:
    from guandan.engine import Environment, Move, Moves

    assert Environment is not None
    assert Move is not None
    assert Moves is not None
    assert not list((REPOSITORY_ROOT / "generations").glob("*/source/KSplatform"))
