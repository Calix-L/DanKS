# Package review

This directory is a test-free public repack of the previously reviewed,
allowlisted source ZIP, not a filtered Git export.
Only the single-player frontend and its import closure are included. Models,
AI implementation, production configuration, topology, credentials, logs,
deployment scripts and Git history are excluded.

The external HTTP connector sends a bounded actor-visible schema; legal action
indices and request identity are checked. An unavailable configured AI is an
error, not a switch to the example bot.

## Prior developer verification

Before this test-free repack, developer-side verification covered archive
content/boundary scanning, browser module regression
tests, room API authentication/static assets/WebSocket, illegal-action rejection
and complete seeded rules games with 108-card conservation. These are candidate
smoke/regression checks, not an exhaustive proof of all tournament rules or a
benchmark of private AI behavior.

That prior verification on macOS Apple Silicon / Python 3.12 recorded 56 Python
checks and 146 browser
module checks. Three complete seeded example-bot games conserve all 108 cards;
one complete mocked HTTP-AI game validates actor-visible requests and normalized
pass history at every turn. The configured external production AI itself is not
part of those tests. The earlier candidate ZIP was also re-tested after fresh
extraction. These counts are historical developer evidence, not newly executed
checks or test source supplied in this test-free package.

In that prior verification, the source-built Go hand arranger passed its Go
tests. Across 23 fixtures (all
13 levels, level-10 aliases, duplicates, wildcards, jokers, empty hands and a
priority-conflict hand), the adapter matched the original implementation on
92 mode layouts and 287 variant selections with no differences. These fixtures
are regression evidence, not exhaustive proof of every possible hand.

The maintainer confirmed ownership of the supplied source and card/avatar artwork.
Full source-built hand arrangement is included; it is independent of private AI.
Third-party licenses and company trademark notices are retained. The production
service is untouched.

## Test-free public package

Only developer test source (Python/browser test directories and Go test files)
and the release-facing Markdown differ from the prior ZIP. Runtime source,
artwork, arranger fingerprints, licenses and NOTICE remain unchanged. Developer
tests and previous archives are retained separately and are not bundled here.

The external packaging helper checks safe paths and the exact approved file set,
compares every retained non-Markdown file byte for byte with the prior ZIP, and
regenerates `MANIFEST.sha256` for every shipped file except itself. The adjacent
ZIP hash is supplied separately. It checks the resulting ZIP entries and
manifest against the source folder. Packaging does not itself establish a new
runtime regression result; follow [DEVELOPMENT](DEVELOPMENT.md) for manual checks.
No test source, virtual environment, installed dependency wheel, runtime binary,
generated audit record, cache, credentials or model file is included in the archive.

## Revision r1

AI request failures pause before referee commit and expose an authenticated,
versioned retry without changing the observation or legal decision. Revoked
WebSockets close and their queued private views are removed; slow clients have
bounded sends. Browser auth/missing-room failures stop reconnecting and return to
entry. Late ready/arrangement/retry responses cannot restore a departed room.
The public client uses the solo API schema and no longer exposes a nonexistent
seat-mutation route. Health and emitted audit logs share one schema identifier.

Ordinary Guandan ranks use A above 2 in full hand arrangement and the
example bot. Prior developer single-card rule checks confirmed A beats 2, equal ranks cannot beat,
and level cards and jokers are promoted. Level 10/T/t aliases agree. These checks
do not change the legal rule that 2 is promoted when the current level itself is 2.
