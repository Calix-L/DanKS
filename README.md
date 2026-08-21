# DanKS

DanKS is a code-only archive of three generations of Guandan AI research. The public repository names them simply **V1**, **V2**, and **V3**. Each generation keeps its original package namespace and compatibility boundary, while a small repository CLI makes the archive easy to navigate, validate, and package.

> [!IMPORTANT]
> Model weights, datasets, internal documents, evaluation results, and private deployment material are not distributed.

## Start here

From the repository root, no installation is needed for navigation:

```bash
python -m danks_repo list
python -m danks_repo show v3
python -m danks_repo verify
```

The first command shows the three generations, package names, file counts, and sizes. The second prints the exact `PYTHONPATH` import command for a generation. The third verifies every manifest, source hash, privacy boundary, and repository path.

For a guided walkthrough, read [Quickstart](docs/QUICKSTART.md). To locate important code immediately, use the [Source map](docs/SOURCE_MAP.md).

## Choose a generation

| Generation | Python package | Start with | Scope |
| --- | --- | --- | --- |
| `V1` | `DanKS` | [`generations/v1/README.md`](generations/v1/README.md) | Retrieval and NumPy selector foundations |
| `V2` | `DanKS` | [`generations/v2/README.md`](generations/v2/README.md) | Staged-training retrieval and selector stack |
| `V3` | `DanRL_retrieval` | [`generations/v3/README.md`](generations/v3/README.md) | Team-belief retrieval and PPO training stack |

Do not put two generations on the same `PYTHONPATH`. The two `DanKS` packages are intentionally isolated snapshots, not interchangeable implementations.

## Repository layout

```text
DanKS/
├── generations/
│   ├── v1/
│   │   ├── README.md
│   │   ├── manifest.json
│   │   └── source/DanKS/
│   ├── v2/
│   │   ├── README.md
│   │   ├── manifest.json
│   │   └── source/DanKS/
│   └── v3/
│       ├── README.md
│       ├── manifest.json
│       └── source/DanRL_retrieval/
├── danks_repo/          # list/show/verify/package implementation
├── docs/                # public usage and repository documentation
├── tests/               # repository-boundary tests
└── tools/               # maintainer wrappers and snapshot importer
```

## Common commands

```bash
make help
make overview
make show GENERATION=v3
make test
make audit
make package
```

`make package` writes four deterministic archives to `dist/`: one per generation and one combined bundle. Rebuilding unchanged content produces byte-identical files.

## What works without private assets

- Browse and analyze every included source file.
- Import the generation package after installing that generation's public dependencies.
- Run repository validation and deterministic packaging.
- Run source tests that do not require private models, data, services, or hardware.

Pretrained inference, original evaluation reproduction, and full training reproduction require private assets and are not promised by this code-only release.

## Public boundary

The importer uses explicit source-root allowlists, excludes upstream documents and generated artifacts, and normalizes approved machine-specific workspace prefixes to `/workspace/danks`. Generation manifests record every included file by path, size, and SHA256.

## License

Original DanKS repository material is licensed under the [Apache License 2.0](LICENSE). Third-party material retains its applicable notices; see [Licensing](docs/LICENSING.md).
