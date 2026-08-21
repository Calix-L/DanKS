# Architecture

DanKS is organized around isolation and provenance rather than a merged runtime.

## Repository layer

The root `danks_repo` package implements source selection, manifest generation, validation, and deterministic archives. It is independent of gameplay and model code. The `tools/` commands are thin entry points over this package.

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
2. **Code-only snapshots.** Import policy admits explicitly allowlisted code roots and public-safe source configuration, not upstream documentation or runtime artifacts. Private machine-path prefixes are normalized during import.
3. **Content-addressed provenance.** Generation manifests make changes visible and reviewable.
4. **Deterministic distribution.** Archive ordering and metadata are normalized.
5. **Fail closed.** Unknown generations, path escapes, symlinks, and forbidden artifacts cause validation failure.
