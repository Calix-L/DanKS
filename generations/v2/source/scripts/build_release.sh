#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
NAME="$(basename "$ROOT")"
ARCHIVE="${ARCHIVE:-$PARENT/$NAME.tar.gz}"

find "$ROOT" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

(
  cd "$ROOT"
  find . \
    -path './.venv' -prune -o \
    -path './runs' -prune -o \
    -type f ! -name MANIFEST.sha256 \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum >MANIFEST.sha256
)

tar \
  --exclude="$NAME/.venv" \
  --exclude="$NAME/runs" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "$ARCHIVE" \
  -C "$PARENT" "$NAME"

(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" >"$(basename "$ARCHIVE").sha256"
)
echo "release_ready archive=$ARCHIVE"
cat "$ARCHIVE.sha256"
