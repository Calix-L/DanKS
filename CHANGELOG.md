# Changelog

All notable public repository changes are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning for repository tooling, independently of the three model-code generations.

## [Unreleased]

### Added

- Code-only snapshots for V1, V2, and V3.
- Generation manifests with SHA256 and size metadata.
- Deterministic per-generation and combined source archives.
- Source-boundary auditing and CI configuration.
- Apache-2.0 licensing and public contributor documentation.
- Unified `python -m danks_repo` navigation, verification, and packaging commands.
- Generation-specific Quickstart and source map.
- Importable `DanRL_retrieval` namespace for the V3 snapshot.
- Shared importable `guandan.engine` rules package.
- Complete V3 PPO learner, persistent transport, training-state, model, feature, and team-belief source.

### Changed

- Reduced each generation to its retrieval and model/training core.
- Consolidated duplicated game rules into one shared package.

### Removed

- Copied platform services, per-generation tests, evaluation harnesses, deployment wrappers, and non-core data conversion utilities.
