# DanKS

DanKS is a compact, code-only repository for three generations of GuanDan AI research. The public generations are named **V1**, **V2**, and **V3**. Each keeps its original model namespace and compatibility boundary, while shared game rules and repository tooling live only once at the root.

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
| `V3` | `DanKS` | [`generations/v3/README.md`](generations/v3/README.md) | Team-belief retrieval and PPO training stack |

Do not put multiple generations on the same `PYTHONPATH`. All three expose the `DanKS` namespace but remain intentionally isolated, non-interchangeable implementations.

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
│       └── source/DanKS/
├── guandan/             # shared 108-card rules engine
├── danks_repo/          # list/show/verify/package implementation
├── docs/                # public usage and repository documentation
├── tests/               # repository-boundary tests
└── tools/               # small verification and packaging wrappers
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
- Run the shared Python GuanDan rules engine without model weights.
- Import the generation package after installing that generation's public dependencies.
- Run repository validation and deterministic packaging.
- Study and run V3's model, feature, team-belief, and PPO training implementation with your own data.

Pretrained inference, original evaluation reproduction, and full training reproduction require private assets and are not promised by this code-only release.

## Public boundary

Generation manifests record every included file by path, size, and SHA256. The repository audit rejects weights, datasets, generated binaries, private paths, credentials, fixed remote endpoints, and platform-specific evaluation code.

## License

Original DanKS repository material is licensed under the [Apache License 2.0](LICENSE). Third-party material retains its applicable notices; see [Licensing](docs/LICENSING.md).
