"""Human-facing catalog for the three DanKS code generations."""

from __future__ import annotations

import json
from pathlib import Path

from .repository import GENERATIONS


GENERATION_CATALOG = {
    "v1": {
        "display_name": "V1",
        "package": "DanKS",
        "focus": "historical retrieval and NumPy selector code",
    },
    "v2": {
        "display_name": "V2",
        "package": "DanKS",
        "focus": "staged-training retrieval and selector code",
    },
    "v3": {
        "display_name": "V3",
        "package": "DanRL_retrieval",
        "focus": "team-belief retrieval and PPO training code",
    },
}


def generation_source_path(repository_root: Path, generation: str) -> Path:
    if generation not in GENERATIONS:
        raise ValueError(f"unknown generation: {generation}")
    return repository_root.resolve() / "generations" / generation / "source"


def generation_summary(repository_root: Path, generation: str) -> dict[str, object]:
    if generation not in GENERATIONS:
        raise ValueError(f"unknown generation: {generation}")
    repository_root = repository_root.resolve()
    manifest_path = repository_root / "generations" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    metadata = GENERATION_CATALOG[generation]
    return {
        "generation": generation,
        "display_name": metadata["display_name"],
        "package": metadata["package"],
        "files": len(files),
        "bytes": sum(int(item["size"]) for item in files),
        "source_path": f"generations/{generation}/source",
    }
