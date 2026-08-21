"""Deterministic DanKS source archive construction."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

from .repository import GENERATIONS
from .verify import verify_repository


EXCLUDED_REPOSITORY_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "dist"}
)


def normalized_archive_path(path: Path) -> str:
    """Convert a relative path to a safe POSIX archive member name."""

    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {path}")
    normalized = PurePosixPath(*path.parts).as_posix()
    if normalized.startswith("/") or normalized.startswith("../"):
        raise ValueError(f"unsafe archive path: {path}")
    return normalized


def _selected_repository_files(repository_root: Path, generation: str) -> list[Path]:
    selected: list[Path] = []
    for directory, directory_names, file_names in os.walk(repository_root, followlinks=False):
        current = Path(directory)
        relative_directory = current.relative_to(repository_root)
        retained = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(repository_root)
            if path.is_symlink():
                raise ValueError(f"repository audit failed: forbidden symlink: {relative}")
            if name in EXCLUDED_REPOSITORY_DIRECTORIES:
                continue
            if relative_directory == Path("generations") and name in GENERATIONS:
                if generation != "all" and name != generation:
                    continue
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(repository_root)
            if path.is_symlink():
                raise ValueError(f"repository audit failed: forbidden symlink: {relative}")
            selected.append(relative)
    return sorted(selected, key=lambda path: path.as_posix())


def _file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_info(name: str, *, is_directory: bool, size: int = 0, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    if is_directory:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o755 if executable else 0o644
        info.size = size
    return info


def build_source_archive(repository_root: Path, generation: str, output_path: Path) -> Path:
    """Build one deterministic per-generation or combined source archive."""

    if generation not in (*GENERATIONS, "all"):
        raise ValueError(f"unknown generation: {generation}")
    repository_root = repository_root.resolve()
    errors = verify_repository(repository_root)
    if errors:
        raise ValueError("repository audit failed:\n" + "\n".join(errors))

    selected = _selected_repository_files(repository_root, generation)
    file_contents = {
        normalized_archive_path(path): (repository_root / path).read_bytes() for path in selected
    }
    bundle = {
        "schema_version": 1,
        "generations": list(GENERATIONS) if generation == "all" else [generation],
        "layout_root": "DanKS",
    }
    file_contents["BUNDLE.json"] = (
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    checksum_lines = [
        f"{_file_digest(data)}  {name}" for name, data in sorted(file_contents.items())
    ]
    file_contents["SHA256SUMS"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    directories = {"DanKS"}
    for relative in file_contents:
        member = PurePosixPath("DanKS") / relative
        for parent in member.parents:
            if parent.as_posix() != ".":
                directories.add(parent.as_posix())
    member_names = sorted(
        directories | {f"DanKS/{relative}" for relative in file_contents}
    )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for member_name in member_names:
                    if member_name in directories:
                        archive.addfile(_tar_info(member_name, is_directory=True))
                        continue
                    relative = member_name.removeprefix("DanKS/")
                    data = file_contents[relative]
                    source_path = repository_root / relative
                    executable = source_path.exists() and bool(source_path.stat().st_mode & 0o111)
                    info = _tar_info(
                        member_name,
                        is_directory=False,
                        size=len(data),
                        executable=executable,
                    )
                    archive.addfile(info, io.BytesIO(data))
    return output_path
