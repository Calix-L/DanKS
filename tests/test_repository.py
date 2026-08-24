from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v1", "v2", "v3")
RETIRED_PACKAGE = "Dan" + "RL"
PLATFORM_TOKEN = "p" + "lm"
IGNORED_DIRECTORIES = {".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist"}
TEXT_SUFFIXES = {".c", ".cpp", ".h", ".hpp", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".doc", ".docx", ".npy", ".npz", ".onnx", ".pdf", ".pt", ".pth", ".so", ".whl", ".zip"}
PRIVATE_MARKERS = (
    "/home/" + "share/user/",
    "/" + "Users/",
    "/" + "workspace/",
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    "card" + "ks",
    "xzz" + "_",
)


def public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_DIRECTORIES for part in path.parts)
    ]


def test_repository_uses_compact_developer_layout() -> None:
    assert all((ROOT / "versions" / version / "DanKS").is_dir() for version in VERSIONS)
    assert all((ROOT / "versions" / version / "pyproject.toml").is_file() for version in VERSIONS)
    assert all((ROOT / path).is_file() for path in ("README.md", "LICENSE", "NOTICE", "pyproject.toml"))
    assert all(
        (ROOT / "examples" / name).is_file()
        for name in (
            "engine_quickstart.py",
            "retrieval_quickstart.py",
            "v3_model_smoke.py",
            "v3_ppo_smoke.py",
        )
    )
    assert not any((ROOT / name).exists() for name in ("generations", "danks_repo", "tools", "docs"))


def test_readme_visual_assets_are_versioned() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in (
        "assets/danks-v3-architecture.png",
        "assets/structure-aware-delayed-outcomes.png",
    ):
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size > 0
        assert relative in readme


def test_each_version_has_installable_distribution_metadata() -> None:
    root_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert root_metadata["project"]["name"] == "danks-engine"
    assert not any(
        classifier.startswith("License ::")
        for classifier in root_metadata["project"].get("classifiers", [])
    )
    assert any(
        dependency.startswith("tomli")
        for dependency in root_metadata["project"]["optional-dependencies"]["dev"]
    )
    for version in VERSIONS:
        metadata = tomllib.loads(
            (ROOT / "versions" / version / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert metadata["project"]["name"] == f"danks-{version}"
        assert metadata["project"]["license"] == "Apache-2.0"
        assert metadata["project"]["license-files"] == ["LICENSE", "NOTICE"]
        assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["DanKS*"]
        assert all(
            (ROOT / "versions" / version / document).is_file()
            for document in ("LICENSE", "NOTICE")
        )
        if version == "v3":
            package_data = metadata["tool"]["setuptools"]["package-data"]["DanKS"]
            assert "retrieval/native_cpp/*.cpp" in package_data


def test_notice_has_no_broken_document_reference() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "docs/" + "LICENSING.md" not in notice


def test_repository_has_compact_community_files() -> None:
    assert all(
        (ROOT / path).is_file()
        for path in (
            ".github/CONTRIBUTING.md",
            ".github/ISSUE_TEMPLATE/bug.yml",
            ".github/pull_request_template.md",
        )
    )


def test_ci_installs_generations_and_runs_v3_model() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'package-version: ["v1", "v2", "v3"]' in workflow
    assert "python -m pip install -e versions/${{ matrix.package-version }}" in workflow
    assert "python examples/retrieval_quickstart.py --version ${{ matrix.package-version }}" in workflow
    assert "python examples/v3_model_smoke.py" in workflow
    assert "python examples/v3_ppo_smoke.py" in workflow
    assert "python -m DanKS.training.train_ppo --help" in workflow


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
            [
                sys.executable,
                "-c",
                "import DanKS; print(DanKS.__file__, DanKS.GENERATION)",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert f"versions/{version}/DanKS/__init__.py" in result.stdout.replace("\\", "/")
        assert result.stdout.rstrip().endswith(version)


def test_public_uniform_action_seed_contract_is_stable() -> None:
    expected = "5062237962189776735"
    for version in ("v2", "v3"):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(ROOT / "versions" / version)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from DanKS.training.schema import derive_uniform_action_seed; "
                "print(derive_uniform_action_seed(2026, 123456, 0))",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.stdout.strip() == expected


def test_examples_do_not_rewrite_python_import_paths() -> None:
    for source in (ROOT / "examples").glob("*.py"):
        assert "sys.path" not in source.read_text(encoding="utf-8")


def test_gpu_frontier_discovers_cuda_portably() -> None:
    for version in ("v2", "v3"):
        source = ROOT / "versions" / version / "DanKS" / "retrieval" / "gpu_frontier.py"
        content = source.read_text(encoding="utf-8")
        assert "DEFAULT_CUDA_HOME" not in content
        assert 'shutil.which("nvcc")' in content
        assert "TORCH_CUDA_HOME" in content
        assert 'TORCH_CUDA_ARCH_LIST", "9.0"' not in content


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
