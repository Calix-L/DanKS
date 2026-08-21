from __future__ import annotations

import os
import sys
from setuptools import Extension, setup
import pybind11


compile_args = [
    "-O3",
    "-std=c++17",
    "-flto",
    "-ffp-contract=off",
]
# A universal2 macOS interpreter compiles both arm64 and x86_64. Host-native
# ISA flags resolve to the running Apple CPU and are invalid for the other
# architecture, so leave architecture selection to distutils on macOS.
if sys.platform != "darwin":
    if os.environ.get("DANKS_NATIVE_GENERIC_ISA", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        compile_args.append("-mtune=native")
    else:
        compile_args.append("-march=native")
link_args = ["-flto"]

pgo_mode = os.environ.get("DANKS_NATIVE_PGO", "off").strip().lower()
pgo_dir = os.environ.get("DANKS_NATIVE_PGO_DIR", "").strip()
if pgo_mode not in {"off", "generate", "use"}:
    raise RuntimeError("DANKS_NATIVE_PGO must be off, generate, or use")
if pgo_mode != "off":
    if not pgo_dir:
        raise RuntimeError("DANKS_NATIVE_PGO_DIR is required when PGO is enabled")
    profile_flag = f"-fprofile-{pgo_mode}={os.path.abspath(pgo_dir)}"
    compile_args.append(profile_flag)
    link_args.append(profile_flag)
    if pgo_mode == "use":
        compile_args.extend(
            [
                "-fprofile-correction",
                "-Wno-missing-profile",
                "-Wno-coverage-mismatch",
            ]
        )


extensions = [
    Extension(
        "danks_cover",
        ["cover.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        "danks_actor_core",
        ["actor_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

selected_extension = os.environ.get("DANKS_NATIVE_EXTENSION", "").strip()
if selected_extension:
    extensions = [extension for extension in extensions if extension.name == selected_extension]
    if not extensions:
        raise RuntimeError(f"unknown DANKS_NATIVE_EXTENSION: {selected_extension}")


setup(
    name="danks-cover",
    ext_modules=extensions,
)
