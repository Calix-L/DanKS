from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_example(
    name: str,
    *arguments: str,
    python_path: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_path)
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_engine_example_runs() -> None:
    result = run_example("engine_quickstart.py", python_path=ROOT)
    assert "DanKS engine ready" in result.stdout
    assert "27 cards each" in result.stdout


def test_retrieval_example_runs_for_every_version() -> None:
    for version in ("v1", "v2", "v3"):
        result = run_example(
            "retrieval_quickstart.py",
            "--version",
            version,
            python_path=ROOT / "versions" / version,
        )
        assert f"DanKS {version.upper()} retrieval ready" in result.stdout
        assert "Top action:" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")
def test_v3_model_smoke_example_runs() -> None:
    result = run_example("v3_model_smoke.py", python_path=ROOT / "versions" / "v3")
    assert "DanKS V3 model ready" in result.stdout
    assert "logits=(1, 10)" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")
def test_v3_ppo_smoke_example_runs() -> None:
    result = run_example("v3_ppo_smoke.py", python_path=ROOT / "versions" / "v3")
    assert "DanKS V3 PPO ready" in result.stdout
    assert "iters=1" in result.stdout
