# Repository Layout

The repository separates public project infrastructure from immutable generation snapshots.

| Path | Role |
| --- | --- |
| `generations/v1` | V1 code snapshot and manifest |
| `generations/v2` | V2 code snapshot and manifest |
| `generations/v3` | V3 code snapshot and manifest |
| `danks_repo` | Import, audit, and archive implementation |
| `tools` | Command-line interfaces |
| `tests` | Repository tooling and boundary tests |
| `docs` | Newly written public repository documentation |
| `dist` | Generated archives; ignored by Git |

Generation manifests are authoritative for snapshot contents. A file below `source/` that is absent from its manifest fails the audit, as does a manifest entry whose size or SHA256 no longer matches.

The root project version describes packaging tools only. It must not be used as an alias for V1, V2, or V3.

V3 is nested below `source/DanRL_retrieval` so setting `PYTHONPATH=generations/v3/source` preserves its actual import namespace. V1 and V2 retain their original `source/DanKS` package roots.
