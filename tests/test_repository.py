from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v1", "v2", "v3")
RETIRED_PACKAGE = "Dan" + "RL_retrieval"
PLATFORM_TOKEN = "p" + "lm"
IGNORED_DIRECTORIES = {".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist"}
TEXT_SUFFIXES = {".c", ".cpp", ".h", ".hpp", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".doc", ".docx", ".npy", ".npz", ".onnx", ".pdf", ".pt", ".pth", ".so", ".whl", ".zip"}
PRIVATE_MARKERS = (
    "/home/" + "share/user/",
    "/" + "Users/",
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
)


def public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_DIRECTORIES for part in path.parts)
    ]


def test_repository_uses_compact_developer_layout() -> None:
    assert all((ROOT / "versions" / version / "DanKS").is_dir() for version in VERSIONS)
    assert all((ROOT / path).is_file() for path in ("README.md", "LICENSE", "NOTICE", "pyproject.toml"))
    assert not any((ROOT / name).exists() for name in ("generations", "danks_repo", "tools", "docs"))


def test_v3_keeps_model_and_ppo_core() -> None:
    training = ROOT / "versions/v3/DanKS/training"
    assert {
        "accelerator.py",
        "candidate_pool.py",
        "featurizer.py",
        "model.py",
        "persistent_ppo_server.py",
        "persistent_ppo_transport.py",
        "ppo.py",
        "recall_model.py",
        "recall_runtime.py",
        "schema.py",
        "team_belief.py",
        "train_ppo.py",
        "training_state.py",
        "type_suppression.py",
    } <= {path.name for path in training.glob("*.py")}


def test_version_internal_imports_resolve() -> None:
    missing: list[str] = []
    for version in VERSIONS:
        version_root = ROOT / "versions" / version
        for source in version_root.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("DanKS."):
                    continue
                target = version_root.joinpath(*node.module.split("."))
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    missing.append(f"{source.relative_to(ROOT)}:{node.lineno}: {node.module}")
    assert missing == []


def test_each_version_imports_in_isolation() -> None:
    for version in VERSIONS:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(ROOT / "versions" / version)
        result = subprocess.run(
            [sys.executable, "-c", "import DanKS; print(DanKS.__file__)"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert f"versions/{version}/DanKS/__init__.py" in result.stdout.replace("\\", "/")


def test_public_tree_contains_only_source_and_public_metadata() -> None:
    offenders: list[str] = []
    for path in public_files():
        relative = path.relative_to(ROOT).as_posix()
        lower_relative = relative.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(relative)
            continue
        if RETIRED_PACKAGE.lower() in lower_relative or PLATFORM_TOKEN in lower_relative:
            offenders.append(relative)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "NOTICE", ".gitignore"}:
            continue
        content = path.read_text(encoding="utf-8")
        lower_content = content.lower()
        if RETIRED_PACKAGE.lower() in lower_content or PLATFORM_TOKEN in lower_content:
            offenders.append(relative)
            continue
        if any(marker in content for marker in PRIVATE_MARKERS):
            offenders.append(relative)
    assert offenders == []


def test_all_python_sources_compile() -> None:
    for source in (ROOT / "versions").rglob("*.py"):
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
