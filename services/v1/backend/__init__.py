"""Battle platform backend.

The package intentionally keeps the game engines outside of the serving
environment.  Workers communicate with :mod:`backend.orchestrator` over a
small NDJSON protocol.
"""

from .models import MatchManifest, ReplayEvent

__all__ = ["MatchManifest", "ReplayEvent"]
