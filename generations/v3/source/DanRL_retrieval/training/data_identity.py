"""Stable replay grouping helpers used by the human BC dataset builder."""

from __future__ import annotations

import re


_SOURCE_PREFIX = re.compile(r"^s\d+:")


def canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("sample id must be nonempty")
    return _SOURCE_PREFIX.sub("", text)


def replay_group_from_sample_id(value: object) -> str:
    """Remove only decision-local suffixes while preserving replay identity."""

    text = canonical_sample_id(value)
    parts = text.split(":")
    if len(parts) >= 5 and parts[0].startswith("line"):
        # Raw replay ids are line:document:replay:event:uid.
        return ":".join(parts[:3])
    if len(parts) >= 3:
        # Adapter sample ids end in decision/seat-local components.
        return ":".join(parts[:-2])
    return text
