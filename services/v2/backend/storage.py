from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Iterable

from .models import (
    Game,
    MatchManifest,
    MatchStatus,
    ReplayEvent,
    ReplayEventType,
    RoundStatus,
    utc_now,
)


class MatchNotFoundError(FileNotFoundError):
    pass


class CorruptMatchError(RuntimeError):
    pass


class MatchStore:
    """Durable, append-only replay storage.

    Manifests use atomic replacement, while events are fsynced JSONL records.
    A lock protects event-id allocation and concurrent API/worker access inside
    one serving process.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._next_ids: dict[str, int] = {}

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_writable(self) -> bool:
        try:
            self.ensure()
            probe = self.root / f".write-probe-{uuid.uuid4().hex}"
            probe.write_bytes(b"ok")
            probe.unlink()
            return True
        except OSError:
            return False

    def match_dir(self, match_id: str) -> Path:
        self._validate_id(match_id)
        return self.root / match_id

    @staticmethod
    def _validate_id(match_id: str) -> None:
        if (
            not match_id
            or match_id in {".", ".."}
            or "/" in match_id
            or "\\" in match_id
            or "\x00" in match_id
        ):
            raise ValueError("invalid match id")

    def create(self, manifest: MatchManifest) -> Path:
        with self._lock:
            self.ensure()
            directory = self.match_dir(manifest.match_id)
            try:
                directory.mkdir(mode=0o750)
            except FileExistsError as exc:
                raise FileExistsError(f"match already exists: {manifest.match_id}") from exc
            self._atomic_manifest_write(directory / "manifest.json", manifest)
            (directory / "events.jsonl").touch(mode=0o640)
            (directory / "worker.log").touch(mode=0o640)
            self._next_ids[manifest.match_id] = 1
            return directory

    def save(self, manifest: MatchManifest) -> None:
        with self._lock:
            path = self.match_dir(manifest.match_id) / "manifest.json"
            if not path.is_file():
                raise MatchNotFoundError(manifest.match_id)
            self._atomic_manifest_write(path, manifest)

    @staticmethod
    def _atomic_manifest_write(path: Path, manifest: MatchManifest) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = manifest.model_dump_json(indent=2)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def get(self, match_id: str) -> MatchManifest:
        path = self.match_dir(match_id) / "manifest.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise MatchNotFoundError(match_id) from exc
        try:
            return MatchManifest.model_validate_json(raw)
        except Exception as exc:
            raise CorruptMatchError(f"invalid manifest for {match_id}: {exc}") from exc

    def count(self) -> int:
        if not self.root.exists():
            return 0
        return sum(
            1
            for item in self.root.iterdir()
            if item.is_dir() and (item / "manifest.json").is_file()
        )

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        game: Game | None = None,
        status: MatchStatus | None = None,
    ) -> tuple[list[MatchManifest], int]:
        if not self.root.exists():
            return [], 0
        manifests: list[MatchManifest] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not (directory / "manifest.json").is_file():
                continue
            try:
                manifest = self.get(directory.name)
            except (CorruptMatchError, MatchNotFoundError, ValueError):
                continue
            if game is not None and manifest.game is not game:
                continue
            if status is not None and manifest.status is not status:
                continue
            manifests.append(manifest)
        manifests.sort(key=lambda item: (item.created_at, item.match_id), reverse=True)
        total = len(manifests)
        return manifests[offset : offset + limit], total

    def append_event(self, event: ReplayEvent) -> ReplayEvent:
        with self._lock:
            path = self.match_dir(event.match_id) / "events.jsonl"
            if not path.is_file():
                raise MatchNotFoundError(event.match_id)
            next_id = self._next_ids.get(event.match_id)
            if next_id is None:
                existing = self.read_events(event.match_id)
                next_id = (existing[-1].event_id + 1) if existing else 1
            event = event.model_copy(update={"event_id": next_id})
            payload = event.model_dump_json()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_ids[event.match_id] = next_id + 1
            return event

    def read_events(
        self,
        match_id: str,
        *,
        round_index: int | None = None,
        committed_only: bool = False,
        after_event_id: int = 0,
    ) -> list[ReplayEvent]:
        path = self.match_dir(match_id) / "events.jsonl"
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            if not (self.match_dir(match_id) / "manifest.json").is_file():
                raise MatchNotFoundError(match_id) from exc
            return []
        events: list[ReplayEvent] = []
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = ReplayEvent.model_validate_json(line)
                except Exception as exc:
                    raise CorruptMatchError(
                        f"invalid event in {match_id} at line {line_number}: {exc}"
                    ) from exc
                if event.event_id <= after_event_id:
                    continue
                if round_index is not None and event.round_index != round_index:
                    continue
                if (
                    committed_only
                    and event.event_type is not ReplayEventType.ACTION_COMMITTED
                ):
                    continue
                events.append(event)
        return events

    def latest_event_id(self, match_id: str) -> int:
        events = self.read_events(match_id)
        return events[-1].event_id if events else 0

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        if not self.root.exists():
            return recovered
        for directory in self.root.iterdir():
            if not directory.is_dir() or not (directory / "manifest.json").is_file():
                continue
            try:
                manifest = self.get(directory.name)
            except (CorruptMatchError, MatchNotFoundError, ValueError):
                continue
            if manifest.status not in {MatchStatus.RUNNING, MatchStatus.QUEUED}:
                continue
            now = utc_now()
            manifest.status = MatchStatus.INTERRUPTED
            manifest.ended_at = now
            manifest.error = "service restarted before the worker completed"
            for summary in manifest.rounds:
                if summary.status in {RoundStatus.RUNNING, RoundStatus.QUEUED}:
                    summary.status = RoundStatus.INTERRUPTED
                    summary.ended_at = now
            self.save(manifest)
            recovered.append(manifest.match_id)
        return recovered

    def append_worker_log(self, match_id: str, lines: Iterable[str]) -> None:
        path = self.match_dir(match_id) / "worker.log"
        if not path.parent.is_dir():
            raise MatchNotFoundError(match_id)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line.rstrip("\n"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
