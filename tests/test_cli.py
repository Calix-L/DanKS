from __future__ import annotations

import json
from pathlib import Path

import pytest

from danks_repo.catalog import generation_summary, generation_source_path
from danks_repo.cli import main


def make_catalog_repository(root: Path) -> Path:
    for generation, package_path in (
        ("v1", "DanKS"),
        ("v2", "DanKS"),
        ("v3", "DanRL_retrieval"),
    ):
        generation_root = root / "generations" / generation
        source = generation_root / "source" / package_path
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("\n", encoding="utf-8")
        (generation_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generation": generation,
                    "source": f"{generation}-source",
                    "files": [
                        {
                            "path": f"{package_path}/__init__.py",
                            "sha256": "0" * 64,
                            "size": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    return root


def test_generation_source_paths_preserve_import_namespaces(tmp_path: Path) -> None:
    repository = make_catalog_repository(tmp_path / "DanKS")
    assert generation_source_path(repository, "v1") == (
        repository / "generations/v1/source"
    )
    assert generation_source_path(repository, "v2") == (
        repository / "generations/v2/source"
    )
    assert generation_source_path(repository, "v3") == (
        repository / "generations/v3/source"
    )
    with pytest.raises(ValueError, match="unknown generation"):
        generation_source_path(repository, "legacy")


def test_generation_summary_is_manifest_backed(tmp_path: Path) -> None:
    repository = make_catalog_repository(tmp_path / "DanKS")
    summary = generation_summary(repository, "v3")
    assert summary == {
        "generation": "v3",
        "display_name": "V3",
        "package": "DanRL_retrieval",
        "files": 1,
        "bytes": 1,
        "source_path": "generations/v3/source",
    }


def test_cli_list_and_show_are_immediately_useful(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repository = make_catalog_repository(tmp_path / "DanKS")
    assert main(["--repository", str(repository), "list"]) == 0
    listed = capsys.readouterr().out
    assert "V1" in listed
    assert "V2" in listed
    assert "V3" in listed
    assert "DanRL_retrieval" in listed

    assert main(["--repository", str(repository), "show", "v3"]) == 0
    shown = capsys.readouterr().out
    assert "PYTHONPATH=generations/v3/source" in shown
    assert "import DanRL_retrieval" in shown
    assert "Model weights: not included" in shown
