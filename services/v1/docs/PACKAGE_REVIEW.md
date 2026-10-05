# Local source-package review

The original generation-one browser files and ten-module import closure match
the captured old service byte for byte. Captured backend/worker sources were
adapted only to portable settings, public rules and the generic external AI
contract; generation-two recovery behavior is not backported.

## Prior developer verification

Before this test-free public repack, the developer-side candidate was verified
on macOS Apple Silicon / Python 3.12: 45 Python checks, one browser
module-closure smoke check and page-script syntax. Complete seeded games test
108-card conservation; an HTTP AI mock tests actor-visible requests, and invalid
actions and AI failure never silently substitute another policy.

In that prior developer verification, the full source-built Go hand arranger
passed its Go tests. On 23 fixtures,
including all 13 levels, level-10 aliases, duplicate cards, wildcards, jokers,
empty hands and a priority-conflict hand, its adapter matched the original
implementation on 92 mode layouts and 287 variant selections, with no differences.
These are regression fixtures, not exhaustive proof of every possible hand.

## Test-free public package

This repack is derived only from the previously reviewed source ZIP. It removes
developer test source (Python/browser test directories and Go test files) and
updates the release-facing Markdown; runtime source, artwork, source identity,
licenses and NOTICE remain unchanged. The counts above describe prior developer
verification, not tests bundled with or newly executed by this package.

The external packaging helper checks safe paths and the exact approved file set,
compares every retained non-Markdown file byte for byte with the prior ZIP, and
regenerates `MANIFEST.sha256` for every shipped file except itself. The adjacent
ZIP SHA-256 is supplied separately. It checks the resulting ZIP's entries and
manifest against the source folder. No runtime smoke-test claim is made by the
packaging step; use [DEVELOPMENT](DEVELOPMENT.md) for manual acceptance checks.

Test source, runtime executables, caches, private capture files, deployments,
weights and player logs are not shipped. Developer tests and the previous ZIP
are retained outside this public package. No upload or production-service change
was made.

The owner confirmed source/artwork ownership. Third-party licenses and company
trademark distinctions remain in NOTICE. See VERSION.md for preserved historical
session behavior and AI-interface adaptations.
