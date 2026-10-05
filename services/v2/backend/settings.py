from __future__ import annotations

import os
from typing import Any


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool) -> bool:
    """Read a strict boolean environment variable.

    Deployment mistakes should fail during application construction instead of
    silently enabling debug or internal-detail endpoints.
    """

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of "
        "1/0, true/false, yes/no, or on/off"
    )


def normalize_base_path(value: str | None) -> str:
    """Return an ASGI root path such as ``/arena`` or ``""``.

    Nginx removes this prefix before proxying to FastAPI.  Restricting it to
    ordinary path segments prevents a malformed environment variable from producing
    ambiguous URLs.
    """

    raw = str(value or "").strip()
    if not raw or raw == "/":
        return ""
    if any(marker in raw for marker in ("?", "#", "\\", "://")):
        raise ValueError("BATTLE_BASE_PATH must be a URL path, not a URL")
    segments = [segment for segment in raw.split("/") if segment]
    if not segments or any(segment in {".", ".."} for segment in segments):
        raise ValueError("BATTLE_BASE_PATH contains an invalid path segment")
    return "/" + "/".join(segments)


def redact_internal_details(value: Any) -> Any:
    """Remove host paths and executable details from a public API payload."""

    if isinstance(value, list):
        return [redact_internal_details(item) for item in value]
    if not isinstance(value, dict):
        return value

    public: dict[str, Any] = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if (
            normalized in {
                "path",
                "checkpoint_path",
                "command",
                "config_path",
                "interpreter",
                "source_manifest",
            }
            or normalized.endswith("_path")
            or normalized.endswith("_root")
        ):
            continue
        public[key] = redact_internal_details(item)
    return public
