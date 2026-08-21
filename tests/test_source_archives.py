from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from danks_repo.archives import build_source_archive, normalized_archive_path
from danks_repo.repository import write_generation_manifest
from danks_repo.verify import verify_generation, verify_repository


def write_file(root: Path, relative: str, content: bytes = b"source\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_repository(root: Path) -> Path:
    for relative in (
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
        "danks_repo/__init__.py",
        "danks_repo/__main__.py",
        "tools/verify_repository.py",
        "tests/test_placeholder.py",
    ):
        write_file(root, relative)
    for generation in ("v1", "v2", "v3"):
        write_file(root, f"generations/{generation}/README.md")
        write_file(root, f"generations/{generation}/source/{generation}.py", generation.encode())
        write_generation_manifest(root, generation, source_label=f"{generation}-snapshot")
    return root


def archive_bytes(path: Path) -> bytes:
    return path.read_bytes()


def open_members(path: Path) -> tuple[list[tarfile.TarInfo], dict[str, bytes]]:
    with gzip.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            members = archive.getmembers()
            contents = {
                member.name: archive.extractfile(member).read()
                for member in members
                if member.isfile()
            }
    return members, contents


def test_repeated_archive_builds_are_byte_identical(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    first = build_source_archive(repository, "all", tmp_path / "first.tar.gz")
    second = build_source_archive(repository, "all", tmp_path / "second.tar.gz")
    assert archive_bytes(first) == archive_bytes(second)


def test_archive_members_are_sorted_normalized_and_rooted(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    output = build_source_archive(repository, "v1", tmp_path / "bundle.tar.gz")
    members, contents = open_members(output)
    names = [member.name for member in members]

    assert names == sorted(names)
    assert names[0] == "DanKS"
    assert all(name == "DanKS" or name.startswith("DanKS/") for name in names)
    assert all(member.uid == 0 and member.gid == 0 and member.mtime == 0 for member in members)
    assert all(member.uname == "root" and member.gname == "root" for member in members)
    assert any("generations/v1/source/v1.py" in name for name in names)
    assert not any("generations/v2/" in name for name in names)
    assert not any("generations/v3/" in name for name in names)

    bundle = json.loads(contents["DanKS/BUNDLE.json"])
    assert bundle["generations"] == ["v1"]
    checksum_lines = contents["DanKS/SHA256SUMS"].decode().splitlines()
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256(contents[f"DanKS/{relative}"]).hexdigest() == digest


def test_archive_path_normalization_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        normalized_archive_path(Path("../escape.py"))
    with pytest.raises(ValueError, match="unsafe"):
        normalized_archive_path(Path("/absolute.py"))


def test_manifest_audit_detects_modified_and_untracked_files(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    assert verify_generation(repository, "v3") == []

    write_file(repository, "generations/v3/source/v3.py", b"changed")
    write_file(repository, "generations/v3/source/untracked.py")
    errors = verify_generation(repository, "v3")
    assert any("sha256 mismatch" in error for error in errors)
    assert any("untracked source file" in error for error in errors)


def test_repository_audit_rejects_private_artifact_inside_snapshot(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    write_file(repository, "generations/v1/source/models/private.pt")
    errors = verify_repository(repository)
    assert any("forbidden" in error and "private.pt" in error for error in errors)

    with pytest.raises(ValueError, match="repository audit failed"):
        build_source_archive(repository, "all", tmp_path / "bad.tar.gz")


def test_repository_audit_rejects_private_paths_and_key_material(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    source = repository / "generations" / "v3" / "source" / "v3.py"
    source.write_text('root = "/home/share/user/example/private"\n', encoding="utf-8")
    errors = verify_generation(repository, "v3")
    assert any("private content marker" in error for error in errors)

    source.write_text('key = "-----BEGIN PRIVATE KEY-----"\n', encoding="utf-8")
    errors = verify_generation(repository, "v3")
    assert any("private content marker" in error for error in errors)

    token = "ghp_" + "abcdefghijklmnopqrstuvwxyz" + "1234567890"
    source.write_text(f'token = "{token}"\n', encoding="utf-8")
    errors = verify_generation(repository, "v3")
    assert any("private credential pattern" in error for error in errors)


def test_repository_audit_rejects_private_runtime_fingerprints(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    source = repository / "generations" / "v3" / "source" / "v3.py"
    source.write_text(
        'EXPECTED_MODEL_SHA256 = "' + "a" * 64 + '"\n', encoding="utf-8"
    )

    errors = verify_generation(repository, "v3")
    assert any("private runtime fingerprint" in error for error in errors)


def test_repository_audit_rejects_fixed_remote_service_endpoints(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    source = repository / "generations" / "v3" / "source" / "v3.py"
    source.write_text(
        'robot_url = "http://192.0.2.10:9093/callRobot"\n', encoding="utf-8"
    )

    errors = verify_generation(repository, "v3")
    assert any("private remote endpoint" in error for error in errors)


def test_repository_audit_rejects_platform_specific_source_markers(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    token = "p" + "lm"
    source = repository / "generations" / "v3" / "source" / "v3.py"
    source.write_text(f'adapter = "{token}_adapter"\n', encoding="utf-8")
    write_generation_manifest(repository, "v3", source_label="v3-snapshot")

    errors = verify_generation(repository, "v3")
    assert any("platform-specific marker" in error for error in errors)


def test_repository_audit_rejects_root_level_bypass_artifacts(tmp_path: Path) -> None:
    repository = make_repository(tmp_path / "DanKS")
    write_file(repository, "private.pt")
    write_file(repository, "docs/internal.docx")
    write_file(repository, ".DS_Store")

    errors = verify_repository(repository)
    assert any("forbidden repository artifact: private.pt" in error for error in errors)
    assert any("forbidden repository artifact: docs/internal.docx" in error for error in errors)
    assert any("forbidden repository artifact: .DS_Store" in error for error in errors)


def test_repository_audit_and_archive_ignore_editable_install_metadata(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path / "DanKS")
    write_file(repository, "danks_repository_tools.egg-info/PKG-INFO")

    assert verify_repository(repository) == []
    output = build_source_archive(repository, "all", tmp_path / "bundle.tar.gz")
    members, _ = open_members(output)
    assert not any(".egg-info/" in member.name for member in members)
