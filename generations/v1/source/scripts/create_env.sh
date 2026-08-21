#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"

command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || {
  echo "Python 3.11 is required; command not found: $PYTHON_BOOTSTRAP" >&2
  exit 2
}

version="$("$PYTHON_BOOTSTRAP" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$version" != "3.11" ]]; then
  echo "Python 3.11 is required for the bundled native extensions; found $version" >&2
  exit 2
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$ROOT/.venv"
fi

"$ROOT/.venv/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --no-index \
  --find-links "$ROOT/wheels" \
  -r "$ROOT/requirements.txt"

echo "environment_ready python=$ROOT/.venv/bin/python"
