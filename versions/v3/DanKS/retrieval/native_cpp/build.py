from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


NATIVE_DIRECTORY = Path(__file__).resolve().parent


def build_invocation() -> tuple[list[str], Path]:
    return [sys.executable, "setup.py", "build_ext", "--inplace"], NATIVE_DIRECTORY


def verify_extensions() -> tuple[bool, bool]:
    importlib.invalidate_caches()
    from DanKS.retrieval import native_actor_core, native_cover

    importlib.reload(native_cover)
    importlib.reload(native_actor_core)
    return native_cover.available(), native_actor_core.available()


def main() -> None:
    command, working_directory = build_invocation()
    subprocess.run(command, cwd=working_directory, check=True)

    cover, actor = verify_extensions()
    if not cover or not actor:
        raise RuntimeError(
            "native build completed, but the V3 kernels could not be imported "
            f"(cover={cover}, actor={actor})"
        )
    print("DanKS V3 native kernels are ready (cover=True, actor=True).")


if __name__ == "__main__":
    main()
