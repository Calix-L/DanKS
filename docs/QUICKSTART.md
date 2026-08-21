# Quickstart

This guide gets you from a fresh source checkout to a verified generation import without model weights.

## 1. Inspect the archive

```bash
python -m danks_repo list
python -m danks_repo show v1
python -m danks_repo show v2
python -m danks_repo show v3
```

Use the lowercase identifiers `v1`, `v2`, and `v3` on the command line.

## 2. Verify integrity

```bash
python -m danks_repo verify
```

A successful audit checks required files, generation manifests, SHA256 values, forbidden artifacts, symlinks, private path markers, and common credential patterns.

## 3. Import one generation

Create a separate virtual environment for the generation you are studying. Do not mix generations in one Python process.

### V1

```bash
python -m pip install -r generations/v1/source/requirements.txt
PYTHONPATH=generations/v1/source python -c "import DanKS; print(DanKS.__file__)"
```

### V2

```bash
python -m pip install -r generations/v2/source/requirements.txt
PYTHONPATH=generations/v2/source python -c "import DanKS; print(DanKS.__file__)"
```

### V3

```bash
python -m pip install -r generations/v3/source/DanRL_retrieval/environment/requirements-training-core.txt
PYTHONPATH=generations/v3/source python -c "import DanRL_retrieval; print(DanRL_retrieval.__file__)"
```

V3 also needs a PyTorch build appropriate for your hardware. To run its
self-contained public tests, install `pytest` and then use:

```bash
PYTHONPATH=generations/v3/source python -m pytest \
  generations/v3/source/DanRL_retrieval/tests
```

These commands verify package layout only. They do not download or provide private weights.

## 4. Build source bundles

```bash
python -m danks_repo package --generation v3
python -m danks_repo package --generation all
sha256sum -c dist/SHA256SUMS
```

## Troubleshooting

- `ModuleNotFoundError`: run from the repository root and use the exact generation-specific `PYTHONPATH` above.
- Manifest mismatch: restore the modified source file or regenerate the snapshot through the maintainer importer.
- Missing checkpoint or data: expected; those assets are outside the public repository.
- Dependency conflict: create a new virtual environment and install only one generation's requirements.
