"""Validation for DanKS repository structure and generation manifests."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .repository import (
    GENERATIONS,
    REQUIRED_GENERATION_PATHS,
    REQUIRED_ROOT_PATHS,
    collect_source_files,
    sha256_file,
)


PRIVATE_CONTENT_MARKERS = (
    b"/home/share/user/",
    b"/Users/",
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
)
PRIVATE_CREDENTIAL_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
)
PRIVATE_RUNTIME_FINGERPRINT_PATTERNS = (
    re.compile(rb"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])"),
)
PRIVATE_REMOTE_ENDPOINT_PATTERNS = (
    re.compile(
        rb"https?://(?!(?:127\.0\.0\.1|0\.0\.0\.0)(?::|/))"
        rb"(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?(?:/|\b)"
    ),
)

IGNORED_LOCAL_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "dist"}
)
PUBLIC_CODE_DIRECTORIES = frozenset({"danks_repo", "tests", "tools"})


def _all_snapshot_entries(source_root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    symlinks: list[Path] = []
    if not source_root.is_dir():
        return files, symlinks
    for directory, directory_names, file_names in os.walk(source_root, followlinks=False):
        current = Path(directory)
        retained_directories = []
        for name in sorted(directory_names):
            path = current / name
            if path.is_symlink():
                symlinks.append(path.relative_to(source_root))
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(source_root)
            if path.is_symlink():
                symlinks.append(relative)
            elif path.is_file():
                files.append(relative)
    return sorted(files, key=lambda p: p.as_posix()), sorted(
        symlinks, key=lambda p: p.as_posix()
    )


def verify_generation(repository_root: Path, generation: str) -> list[str]:
    """Return all validation errors for one generation without mutating it."""

    errors: list[str] = []
    repository_root = repository_root.resolve()
    if generation not in GENERATIONS:
        return [f"unknown generation: {generation}"]
    generation_root = repository_root / "generations" / generation
    source_root = generation_root / "source"
    manifest_path = generation_root / "manifest.json"
    if not source_root.is_dir():
        errors.append(f"{generation}: missing source directory")
        return errors
    if not manifest_path.is_file():
        errors.append(f"{generation}: missing manifest.json")
        return errors

    all_files, symlinks = _all_snapshot_entries(source_root)
    allowed_files = collect_source_files(source_root)
    allowed_names = {path.as_posix() for path in allowed_files}
    for path in symlinks:
        errors.append(f"{generation}: forbidden symlink: {path.as_posix()}")
    for path in all_files:
        if path.as_posix() not in allowed_names:
            errors.append(f"{generation}: forbidden artifact: {path.as_posix()}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{generation}: invalid manifest: {exc}")
        return errors
    if manifest.get("generation") != generation:
        errors.append(f"{generation}: manifest generation mismatch")
    source_label = manifest.get("source")
    if not isinstance(source_label, str) or not source_label or "/" in source_label or "\\" in source_label:
        errors.append(f"{generation}: manifest source must be a public label, not a path")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        errors.append(f"{generation}: manifest files must be a list")
        return errors
    manifest_names = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(manifest_names) != len(entries):
        errors.append(f"{generation}: malformed manifest file entry")
        return errors
    if manifest_names != sorted(manifest_names) or len(manifest_names) != len(set(manifest_names)):
        errors.append(f"{generation}: manifest paths must be sorted and unique")

    manifest_name_set = set(manifest_names)
    for name in sorted(allowed_names - manifest_name_set):
        errors.append(f"{generation}: untracked source file: {name}")
    for name in sorted(manifest_name_set - allowed_names):
        errors.append(f"{generation}: missing or forbidden manifest file: {name}")

    for entry in entries:
        name = entry.get("path")
        if name not in allowed_names:
            continue
        path = source_root / name
        content = path.read_bytes()
        if any(marker in content for marker in PRIVATE_CONTENT_MARKERS):
            errors.append(f"{generation}: private content marker: {name}")
        if any(pattern.search(content) for pattern in PRIVATE_CREDENTIAL_PATTERNS):
            errors.append(f"{generation}: private credential pattern: {name}")
        if any(pattern.search(content) for pattern in PRIVATE_RUNTIME_FINGERPRINT_PATTERNS):
            errors.append(f"{generation}: private runtime fingerprint: {name}")
        if any(pattern.search(content) for pattern in PRIVATE_REMOTE_ENDPOINT_PATTERNS):
            errors.append(f"{generation}: private remote endpoint: {name}")
        if entry.get("size") != path.stat().st_size:
            errors.append(f"{generation}: size mismatch: {name}")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"{generation}: sha256 mismatch: {name}")
    return errors


def verify_repository(repository_root: Path) -> list[str]:
    """Return all public repository contract violations."""

    repository_root = repository_root.resolve()
    errors = []
    for relative in (*REQUIRED_ROOT_PATHS, *REQUIRED_GENERATION_PATHS):
        if not (repository_root / relative).exists():
            errors.append(f"missing required path: {relative}")
    for generation in GENERATIONS:
        errors.extend(verify_generation(repository_root, generation))

    required_root_names = set(REQUIRED_ROOT_PATHS)
    for directory, directory_names, file_names in os.walk(repository_root, followlinks=False):
        current = Path(directory)
        retained = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(repository_root)
            if path.is_symlink():
                errors.append(f"forbidden repository symlink: {relative.as_posix()}")
            elif name not in IGNORED_LOCAL_DIRECTORIES and not name.endswith(".egg-info"):
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(repository_root)
            relative_name = relative.as_posix()
            if path.is_symlink():
                errors.append(f"forbidden repository symlink: {relative_name}")
                continue
            if relative.parts[0] == "generations":
                continue
            if relative_name in required_root_names:
                continue
            if relative.parts[0] in PUBLIC_CODE_DIRECTORIES and path.suffix == ".py":
                continue
            errors.append(f"forbidden repository artifact: {relative_name}")
    return errors
