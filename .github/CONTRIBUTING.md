# Contributing to DanKS

Thank you for helping improve DanKS. Keep changes focused, reproducible, and confined to the generation whose contract they modify.

## Development setup

Install the shared engine and checks from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

For AI changes, install exactly one generation in a separate environment:

```bash
python -m pip install -e versions/v3
```

Replace `v3` with the version being changed. Do not mix generations in one environment or silently change another generation's feature/checkpoint contract.

## Pull requests

- Add or update a failing test before changing behavior.
- Run `python -m pytest -q` and the relevant example.
- Keep generated files, checkpoints, datasets, and evaluation output out of Git.
- Never include credentials, private filesystem paths, proprietary platform code, or private operational documents.
- Explain compatibility impact when changing features, model schemas, checkpoints, or native kernels.

Small, reviewable pull requests are preferred.
