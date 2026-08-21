# Repository Layout

The repository separates shared game rules, public project infrastructure, and isolated generation snapshots.

| Path | Role |
| --- | --- |
| `generations/v1` | V1 code snapshot and manifest |
| `generations/v2` | V2 code snapshot and manifest |
| `generations/v3` | V3 code snapshot and manifest |
| `guandan` | Shared Python rules engine and table environment |
| `danks_repo` | Catalog, audit, and archive implementation |
| `tools` | Thin verification and packaging wrappers |
| `tests` | Repository tooling and boundary tests |
| `docs` | Newly written public repository documentation |
| `dist` | Generated archives; ignored by Git |

Generation manifests are authoritative for snapshot contents. A file below `source/` that is absent from its manifest fails the audit, as does a manifest entry whose size or SHA256 no longer matches.

The root project version describes packaging tools only. It must not be used as an alias for V1, V2, or V3.

Every generation is nested below `source/DanKS`. Select exactly one generation with its `PYTHONPATH`; the shared namespace does not imply checkpoint or runtime compatibility across generations.

Operational services, model-serving adapters, evaluation harnesses, copied per-generation tests, and data-preparation utilities are intentionally outside the public generation trees. V3 retains the full PPO learner and its runtime dependencies.
