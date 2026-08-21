#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

for command in curl perl sha256sum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required system command missing: $command" >&2
    exit 2
  }
done

[[ -x "$PYTHON" ]] || {
  echo "environment missing; run: bash scripts/create_env.sh" >&2
  exit 2
}

"$PYTHON" "$ROOT/scripts/preflight.py"
