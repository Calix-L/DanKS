# Architecture

DanKS is organized around a small shared rules layer, isolated model generations, and reproducible source packaging.

## Repository layer

The root `danks_repo` package implements manifest generation, validation, and deterministic archives. It is independent of gameplay and model code. The `tools/` directory contains only thin verification and packaging entry points.

## Shared game layer

`guandan.engine` contains the Python 108-card game environment, move generator, action types, table lifecycle, tribute flow, and settlement logic. It is packaged once and can be imported independently of V1, V2, or V3.

## Generation layer

Each directory below `generations/` has exactly three public elements:

```text
<generation>/
├── README.md
├── manifest.json
└── source/
```

`source/` retains the generation's original package layout. `manifest.json` records every included file by relative path, byte size, and SHA256. Cross-generation imports are intentionally unsupported.

## Private runtime boundary

The code snapshots originated in systems that also use models, data, and operational configuration. Those assets are outside this repository. Their absence is a product boundary, not a packaging omission, and repository validation prevents common private artifact types from entering a snapshot or archive.

## Design principles

1. **No implicit compatibility.** A checkpoint or runtime from one generation must not be assumed compatible with another.
2. **Core source only.** Generation trees contain retrieval, feature, model, and training code—not copied platform services, evaluation harnesses, or operational wrappers.
3. **Content-addressed provenance.** Generation manifests make changes visible and reviewable.
4. **Deterministic distribution.** Archive ordering and metadata are normalized.
5. **Fail closed.** Unknown generations, path escapes, symlinks, and forbidden artifacts cause validation failure.
