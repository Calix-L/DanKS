"""Build the public Guandan hand arranger from audited source (Go 1.23+).

Run: python scripts/build_arranger.py
Optional: --output PATH (also writes PATH.json build metadata), --go PATH.
No binary downloads, external Go modules, or private deployment are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "arranger"


def verify_source() -> dict:
    manifest = json.loads((SOURCE / "source-manifest.json").read_text(encoding="utf-8"))
    actual_names = sorted(
        p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*.go")
        if not p.name.endswith("_test.go")
    ) + ["go.mod"]
    if set(actual_names) != set(manifest["files"]):
        raise RuntimeError("arranger source manifest file set mismatch")
    for name, expected in manifest["files"].items():
        raw = (SOURCE / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise RuntimeError(f"arranger source fingerprint mismatch: {name}")
    fingerprint = hashlib.sha256("".join(
        f"{name}\0{digest}\n" for name, digest in sorted(manifest["files"].items())
    ).encode()).hexdigest()
    if fingerprint != manifest["source_fingerprint"]:
        raise RuntimeError("arranger source manifest fingerprint mismatch")
    return manifest


def refresh_manifest() -> dict:
    """Explicitly record reviewed local edits; never bypass checks implicitly.

    algorithm_source_sha identifies the historical algorithm lineage, not the
    current edited bytes. source_fingerprint/files identify the exact current
    public source, after normalizing CRLF to LF for cross-platform checkouts.
    """
    path = SOURCE / "source-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*.go")
                   if not p.name.endswith("_test.go")) + ["go.mod"]
    files = {name: hashlib.sha256((SOURCE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for name in names}
    manifest["files"] = files
    manifest["source_fingerprint"] = hashlib.sha256("".join(
        f"{name}\0{digest}\n" for name, digest in sorted(files.items())
    ).encode()).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build(output: Path | None = None, go: str = "go") -> Path:
    manifest = verify_source()
    executable = shutil.which(go)
    if executable is None:
        raise RuntimeError("Go compiler not found; install Go 1.23+ and retry python scripts/build_arranger.py")
    output = output or ROOT / "runtime" / ("hand-arranger.exe" if os.name == "nt" else "hand-arranger")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([executable, "version"], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix=".arranger-build-", dir=output.parent) as temporary:
        target = Path(temporary) / output.name
        environment = {**os.environ, "GOTOOLCHAIN": "local", "GOWORK": "off", "CGO_ENABLED": "0"}
        subprocess.run([executable, "build", "-trimpath", "-ldflags=-s -w", "-o", str(target),
                        "./cmd/hand-arranger"], cwd=SOURCE, env=environment, check=True)
        probe = subprocess.run([str(target)], input=json.dumps({
            "id": "build-probe", "game": "guandan", "mode": 4, "level": "2", "cards": [],
        }) + "\n", text=True, encoding="utf-8", capture_output=True, check=True, timeout=5)
        response = json.loads(probe.stdout)
        if response.get("error") or response.get("groups") != [] or response.get("source_sha") != manifest["algorithm_source_sha"]:
            raise RuntimeError("built arranger failed the empty Guandan identity probe")
        metadata = {
            "implementation": "public-go-splitter-ndjson",
            "go_version": version,
            "source_fingerprint": manifest["source_fingerprint"],
            "algorithm_source_sha": manifest["algorithm_source_sha"],
            "binary_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        # Only generated runtime artifacts are replaced; source files stay unchanged.
        target.replace(output)
        Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--go", default="go")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="explicitly record reviewed source edits before building; preserves historical algorithm lineage identity")
    arguments = parser.parse_args()
    try:
        if arguments.refresh_manifest:
            refresh_manifest()
        result = build(arguments.output, arguments.go)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Arranger build failed: {exc}\n")
    print(f"Built Guandan arranger: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
