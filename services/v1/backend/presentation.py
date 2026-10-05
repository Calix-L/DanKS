from __future__ import annotations

import copy
import hashlib
import json
import os
import math
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from .models import Game, MatchManifest, ReplayEvent, ReplayEventType


PRESENTATION_VERSION = "battle-platform.presentation.v2"
NORMALIZER_VERSION = "battle-platform.play-area.v2.2"
ORIGINAL_SOURCE_SHA256 = {
    "hand.go": "b91ce83760ef3c60d076e79b6afc6a1a5ed50349ee8c17b97ccc86ef27b6e36b",
    "pattern.go": "2a712692b3e2ab984bba927cb1e2674adb1767fdfba75bf9d257b8c01376aa21",
    "splitter.go": "778438601e6811775a70268292a51c3f5925123ee2682ee95dcd005020f2c325",
    "splitter1.go": "3c8a0d25aa84b3ffc8206b1a72dd08f2bbd6daf16207530eab9d541a58bc6f6a",
    "splitter2.go": "7f751f1ba0b70af30e42e3f2d3b2f0200e0ef9a3bde0e692f56a634fa9506c4b",
    "splitter3.go": "92e64551f258689830c167bec7c23b973df52f40acc0581a7fa16856588c4983",
    "splitter4.go": "3996c711b22352d7554f78786b4a08a282e80d5fc1c96273dd950d24a45f3c68",
}
ARRANGER_SOURCE_SHA = ";".join(
    f"{Path(name).stem}:{digest}"
    for name, digest in ORIGINAL_SOURCE_SHA256.items()
)
SUPPORTED_MODES = (1, 2, 3, 4)
DEFAULT_ARRANGER_RESPONSE_TIMEOUT_SECONDS = 1.0

class ArrangerError(RuntimeError, ValueError):
    pass


GUANDAN_LEVELS = frozenset(
    ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
)


def _canonicalize_guandan_level(raw: Any) -> str:
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise ArrangerError(
            f"invalid guandan level {raw!r}; expected one of "
            "2-9,T,J,Q,K,A (10 is accepted as T)"
        )
    level = str(raw).strip().upper()
    if level == "10":
        level = "T"
    if level not in GUANDAN_LEVELS:
        raise ArrangerError(
            f"invalid guandan level {raw!r}; expected one of "
            "2-9,T,J,Q,K,A (10 is accepted as T)"
        )
    return level


def _binary_name() -> str:
    return "hand-arranger.exe" if os.name == "nt" else "hand-arranger"


def _normalize_action(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        nested = raw.get("action")
        cards = raw.get("cards")
        if not isinstance(cards, list) and isinstance(nested, list):
            cards = nested[2] if len(nested) > 2 and isinstance(nested[2], list) else []
        action_type = raw.get("type") or raw.get("kind")
        rank = raw.get("rank")
        if isinstance(nested, list):
            action_type = action_type or (nested[0] if nested else None)
            rank = rank or (nested[1] if len(nested) > 1 else None)
        passed = bool(raw.get("pass")) or action_type == "PASS"
        empty = (
            not cards
            and (
                action_type is None
                or str(action_type).strip().upper() in {"", "NONE", "NULL"}
            )
        )
        return {
            "type": str(action_type or ("PASS" if passed else "None")),
            "rank": rank,
            "cards": [str(card) for card in (cards or [])],
            "pass": passed,
            "label": (
                ""
                if empty
                else str(raw.get("label") or ("不出" if passed else action_type or ""))
            ),
        }
    if isinstance(raw, list):
        action_type = raw[0] if raw else "PASS"
        rank = raw[1] if len(raw) > 1 else None
        cards = raw[2] if len(raw) > 2 and isinstance(raw[2], list) else []
        passed = action_type == "PASS"
        empty = (
            not cards
            and (
                action_type is None
                or str(action_type).strip().upper() in {"", "NONE", "NULL"}
            )
        )
        return {
            "type": str(action_type or "None"),
            "rank": rank,
            "cards": [str(card) for card in cards],
            "pass": passed,
            "label": "" if empty else "不出" if passed else str(action_type),
        }
    raise ArrangerError(f"unsupported play_area payload: {type(raw).__name__}")


class HandArranger:
    """Full Guandan splitter and priority-preserving partition variants.

    Each uncached exact request runs one bounded child process; this works on
    Windows, macOS and Linux without selecting on subprocess pipes.
    """

    def __init__(
        self,
        project_root: Path | None,
        *,
        response_timeout: float | None = None,
    ) -> None:
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        self.project_root = root.parent if root.name == "web" else root
        configured = os.getenv("BATTLE_ARRANGER_PATH")
        self.path = (
            Path(configured).expanduser().resolve()
            if configured
            else (self.project_root / "runtime" / _binary_name()).resolve()
        )
        self._lock = threading.RLock()
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._variant_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._binary_sha: str | None = None
        if response_timeout is None:
            configured_timeout = os.getenv(
                "BATTLE_ARRANGER_RESPONSE_TIMEOUT_SECONDS",
                str(DEFAULT_ARRANGER_RESPONSE_TIMEOUT_SECONDS),
            )
            try:
                response_timeout = float(configured_timeout)
            except ValueError as exc:
                raise ValueError(
                    "BATTLE_ARRANGER_RESPONSE_TIMEOUT_SECONDS must be a number"
                ) from exc
        if not math.isfinite(response_timeout) or response_timeout <= 0:
            raise ValueError("arranger response timeout must be positive")
        self.response_timeout = response_timeout

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "version": PRESENTATION_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "implementation": "public-go-splitter-ndjson",
            "source_sha": ARRANGER_SOURCE_SHA,
            "original_source_sha256": dict(ORIGINAL_SOURCE_SHA256),
            "source_fingerprint": self._source_fingerprint(),
            "binary_sha256": self.binary_sha256(),
            "guandan_mode": 1,
            "supported_modes": list(SUPPORTED_MODES),
        }

    def binary_sha256(self) -> str | None:
        if self._binary_sha is not None:
            return self._binary_sha
        if not self.path.is_file():
            return None
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self._binary_sha = digest.hexdigest()
        return self._binary_sha

    def _source_fingerprint(self) -> str | None:
        manifest = self.project_root / "arranger" / "source-manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return data["source_fingerprint"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _require_binary(self) -> None:
        if not self.path.is_file() or (os.name != "nt" and not os.access(self.path, os.X_OK)):
            raise ArrangerError(
                f"arranger binary is unavailable: {self.path}; "
                "build it with python scripts/build_arranger.py "
                "(Go 1.23+ required), or set BATTLE_ARRANGER_PATH"
            )

    def health(self) -> dict[str, Any]:
        try:
            self._require_binary()
            # Builds carry public source/binary metadata, not private deployment pins.
            metadata_path = Path(str(self.path) + ".json")
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ArrangerError(f"invalid arranger build metadata: {exc}") from exc
                if not isinstance(metadata, dict) or metadata.get("binary_sha256") != self.binary_sha256():
                    raise ArrangerError("arranger build metadata binary SHA mismatch")
                fingerprint = self._source_fingerprint()
                if fingerprint is not None and metadata.get("source_fingerprint") != fingerprint:
                    raise ArrangerError("arranger build metadata source fingerprint mismatch")
            probe = self.arrange(Game.GUANDAN, [], level="2", mode=4)
            if probe["groups"] != []:
                raise ArrangerError("arranger empty-hand probe returned non-empty groups")
            return {"ready": True, "executable": True, "path": str(self.path), **self.identity}
        except (OSError, ArrangerError, ValueError) as exc:
            return {
                "ready": False,
                "executable": self.path.is_file() and (os.name == "nt" or os.access(self.path, os.X_OK)),
                "path": str(self.path), **self.identity, "error": str(exc),
            }

    def close(self) -> None:
        """No persistent child remains between requests."""
        return None

    def _arrange_exact(
        self,
        game: Game,
        cards: Iterable[str],
        *,
        level: str | int | None,
        mode: int,
    ) -> dict[str, Any]:
        if game is not Game.GUANDAN:
            raise ArrangerError("expected Guandan")
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in SUPPORTED_MODES:
            raise ArrangerError("arranger mode must be an integer in 1..4")
        level = _canonicalize_guandan_level(level)
        normalized_cards = tuple(sorted(str(card) for card in cards))
        cache_key = (self.binary_sha256(), game.value, level, mode, normalized_cards)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)
            self._require_binary()
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id, "game": game.value, "level": level,
                "mode": mode, "cards": list(normalized_cards),
            }
            try:
                completed = subprocess.run(
                    [str(self.path)], input=json.dumps(payload, separators=(",", ":")) + "\n",
                    capture_output=True, text=True, encoding="utf-8",
                    timeout=self.response_timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ArrangerError(
                    f"arranger response timed out after {self.response_timeout:g} seconds"
                ) from exc
            except OSError as exc:
                raise ArrangerError(f"arranger process failed: {exc}") from exc
            if completed.returncode:
                raise ArrangerError(
                    f"arranger exited with status {completed.returncode}: {completed.stderr[:4096].strip()}"
                )
            try:
                response = json.loads(completed.stdout)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ArrangerError(f"arranger returned invalid JSON: {exc}") from exc
            if not isinstance(response, dict):
                raise ArrangerError("arranger response must be an object")
            if response.get("id") != request_id:
                raise ArrangerError("arranger response id does not match request")
            if response.get("error"):
                raise ArrangerError(str(response["error"]))
            if response.get("version") != PRESENTATION_VERSION:
                raise ArrangerError("arranger presentation version mismatch")
            if response.get("source_sha") != ARRANGER_SOURCE_SHA:
                raise ArrangerError("arranger source identity mismatch")
            if response.get("mode") != mode or isinstance(response.get("mode"), bool):
                raise ArrangerError("arranger response mode does not match request")
            groups = response.get("groups")
            if not isinstance(groups, list):
                raise ArrangerError("arranger groups must be a list")
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("cards"), list):
                    raise ArrangerError("arranger group must contain a card list")
                if not group["cards"] or any(not isinstance(c, str) for c in group["cards"]):
                    raise ArrangerError("arranger group cards must be non-empty strings")
                for field, minimum, maximum in (("pattern", 1, 8), ("minor", 0, 3), ("value", 1, 15)):
                    value = group.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                        raise ArrangerError(f"arranger group {field} is invalid")
            flattened = [card for group in groups for card in group["cards"]]
            if sorted(flattened) != list(normalized_cards):
                raise ArrangerError("arranger output violates card multiset conservation")
            result = {
                "version": PRESENTATION_VERSION, "mode": mode,
                "source_sha": ARRANGER_SOURCE_SHA, "groups": groups,
            }
            self._cache[cache_key] = copy.deepcopy(result)
            return result

    @staticmethod
    def _partition_signature(layout: dict[str, Any]) -> tuple[str, ...]:
        """Identify a partition without treating display order as semantics."""

        signatures = []
        for group in layout.get("groups", []):
            signatures.append(
                json.dumps(
                    {
                        "pattern": group.get("pattern"),
                        "minor": group.get("minor"),
                        "value": group.get("value"),
                        "cards": sorted(str(card) for card in group.get("cards", [])),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return tuple(sorted(signatures))

    @staticmethod
    def _ordered_signature(layout: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            json.dumps(group, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            for group in layout.get("groups", [])
        )

    @staticmethod
    def _layout_metrics(layout: dict[str, Any]) -> tuple[int, int, int, int]:
        groups = layout.get("groups", [])
        bombs = sum(group.get("pattern") == 8 for group in groups)
        flushes = sum(
            group.get("pattern") == 8 and group.get("minor") == 2
            for group in groups
        )
        singles = sum(
            group.get("pattern") == 1 and len(group.get("cards", [])) == 1
            for group in groups
        )
        return bombs, flushes, singles, len(groups)

    @classmethod
    def _keeps_primary_objective(
        cls,
        mode: int,
        base: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        base_bombs, base_flushes, base_singles, _ = cls._layout_metrics(base)
        bombs, flushes, singles, _ = cls._layout_metrics(candidate)
        if mode == 1:
            return bombs == base_bombs
        if mode == 2:
            return flushes == base_flushes
        if mode == 3:
            return singles == base_singles
        return cls._partition_signature(candidate) == cls._partition_signature(base)

    @classmethod
    def _candidate_sort_key(
        cls,
        mode: int,
        layout: dict[str, Any],
    ) -> tuple[Any, ...]:
        bombs, flushes, singles, hand_count = cls._layout_metrics(layout)
        if mode == 1:
            score = (-bombs, -flushes, hand_count, singles)
        elif mode == 2:
            score = (-flushes, -bombs, hand_count, singles)
        elif mode == 3:
            score = (singles, -bombs, -flushes, hand_count)
        else:
            score = (hand_count, singles, -bombs, -flushes)
        return (*score, cls._partition_signature(layout))

    def _guandan_variants(
        self,
        cards: tuple[str, ...],
        *,
        level: str | int | None,
        mode: int,
    ) -> list[dict[str, Any]]:
        cache_key = (
            self.binary_sha256(),
            Game.GUANDAN.value,
            str(level),
            mode,
            cards,
            "partition-variants-v2",
        )
        cached = self._variant_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        base = self._arrange_exact(Game.GUANDAN, cards, level=level, mode=mode)
        exact_layouts = [base]
        for source_mode in SUPPORTED_MODES:
            if source_mode == mode:
                continue
            exact_layouts.append(
                self._arrange_exact(
                    Game.GUANDAN,
                    cards,
                    level=level,
                    mode=source_mode,
                )
            )

        partitions: list[dict[str, Any]] = []
        seen_partitions: set[tuple[str, ...]] = set()
        for layout in exact_layouts:
            if not self._keeps_primary_objective(mode, base, layout):
                continue
            signature = self._partition_signature(layout)
            if signature in seen_partitions:
                continue
            seen_partitions.add(signature)
            partitions.append(layout)
        partitions[1:] = sorted(
            partitions[1:],
            key=lambda layout: self._candidate_sort_key(mode, layout),
        )

        # A repeated click is a new candidate only when the physical cards are
        # assigned to genuinely different melds. Merely reversing or otherwise
        # reordering the same groups must never be exposed as another layout.
        # Every exact candidate has already passed through the same Go Sort,
        # therefore all variants retain one consistent display direction.
        variants = [copy.deepcopy(layout) for layout in partitions[:8]] or [base]
        self._variant_cache[cache_key] = copy.deepcopy(variants)
        return variants

    def arrange(
        self,
        game: Game,
        cards: Iterable[str],
        *,
        level: str | int | None,
        mode: int,
        variant: int = 0,
    ) -> dict[str, Any]:
        if game is not Game.GUANDAN:
            raise ArrangerError("expected Guandan")
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in SUPPORTED_MODES:
            raise ArrangerError("arranger mode must be an integer in 1..4")
        level = _canonicalize_guandan_level(level)
        if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
            raise ArrangerError("arranger variant must be a non-negative integer")
        normalized_cards = tuple(sorted(str(card) for card in cards))
        with self._lock:
            if game is Game.GUANDAN and normalized_cards:
                variants = self._guandan_variants(
                    normalized_cards,
                    level=level,
                    mode=mode,
                )
            else:
                variants = [
                    self._arrange_exact(game, normalized_cards, level=level, mode=mode)
                ]
            selected_index = variant % len(variants)
            selected = copy.deepcopy(variants[selected_index])
            selected["mode"] = mode
            selected["variant_index"] = selected_index
            selected["variant_count"] = len(variants)
            return selected

    def enrich_state(
        self,
        raw_state: dict[str, Any] | None,
        game: Game,
        *,
        reconstructed_plays: list[dict[str, Any] | None] | None = None,
    ) -> dict[str, Any] | None:
        if raw_state is None:
            return None
        state = copy.deepcopy(raw_state)
        players = state.get("players")
        if not isinstance(players, list):
            return state
        if state.get("game") not in {game.value, game}:
            # Protocol tests and terminal diagnostics may carry a generic
            # ``players`` field without claiming to be an observer snapshot.
            return state
        if game is not Game.GUANDAN:
            raise ArrangerError("expected Guandan")
        expected = 4
        if len(players) != expected:
            raise ArrangerError(
                f"{game.value} observer state has {len(players)} players, expected {expected}"
            )
        raw_metrics = state.get("metrics")
        if raw_metrics is None:
            metrics: dict[str, Any] = {}
        elif isinstance(raw_metrics, dict):
            metrics = raw_metrics
        else:
            raise ArrangerError("observer state metrics must be an object")
        if game is Game.GUANDAN:
            level: Any = metrics.get("level")
            if level is None:
                level = state.get("curRank")
            if level is None:
                level = state.get("cur_rank")
            if level is None:
                raise ArrangerError("guandan observer state is missing the current level")
            level = _canonicalize_guandan_level(level)
            metrics["level"] = level
            state["metrics"] = metrics
        else:
            level = metrics.get("level")
        mode = 1
        for fallback_position, player in enumerate(players):
            if not isinstance(player, dict):
                raise ArrangerError("observer player must be an object")
            position = player.get("position", fallback_position)
            if isinstance(position, bool) or not isinstance(position, int):
                raise ArrangerError("observer player position must be an integer")
            hand = player.get("hand")
            if not isinstance(hand, list):
                raise ArrangerError(
                    f"observer state is missing the full hand for seat {position}"
                )
            player["hand_layout"] = self.arrange(
                game,
                [str(card) for card in hand],
                level=level,
                mode=mode,
            )
            play = player.get("play_area")
            if play is None and reconstructed_plays is not None:
                play = reconstructed_plays[position]
            player["play_area"] = _normalize_action(play)
        state["presentation_identity"] = {
            **self.identity,
            "reconstructed_plays": reconstructed_plays is not None,
        }
        return state

    def enrich_event(
        self,
        event: ReplayEvent,
        game: Game,
        *,
        before_plays: list[dict[str, Any] | None] | None = None,
        after_plays: list[dict[str, Any] | None] | None = None,
    ) -> ReplayEvent:
        return event.model_copy(
            update={
                "before_state": self.enrich_state(
                    event.before_state,
                    game,
                    reconstructed_plays=before_plays,
                ),
                "after_state": self.enrich_state(
                    event.after_state,
                    game,
                    reconstructed_plays=after_plays,
                ),
            },
            deep=True,
        )


def materialize_presentation(
    arranger: HandArranger,
    manifest: MatchManifest,
    events: list[ReplayEvent],
    match_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Build an atomic, deterministic presentation cache without rewriting events."""

    target = match_dir / "presentation.v2.jsonl"
    last_event_id = events[-1].event_id if events else 0
    cached = _load_presentation_cache(
        target,
        manifest,
        source_event_id=last_event_id,
        arranger_sha256=arranger.binary_sha256(),
    )
    if cached is not None:
        return cached

    rounds: dict[int, dict[str, Any]] = {}
    play_areas: dict[int, list[dict[str, Any] | None]] = {}
    reconstructed = False
    presented_events: list[ReplayEvent] = []
    for event in events:
        round_index = event.round_index
        current = play_areas.setdefault(
            round_index,
            [None] * 4,
        )
        before = copy.deepcopy(current)
        after = copy.deepcopy(current)
        if event.event_type is ReplayEventType.ROUND_STARTED:
            current[:] = [None] * len(current)
            before = copy.deepcopy(current)
            after = copy.deepcopy(current)
        if event.event_type is ReplayEventType.ACTION_COMMITTED:
            actor = event.actor_pos
            if actor is None:
                raise ArrangerError("committed replay event is missing actor_pos")
            after[actor] = _normalize_action(event.chosen_action)
            current[:] = copy.deepcopy(after)
        states_have_plays = True
        for state in (event.before_state, event.after_state):
            if state is None:
                continue
            players = state.get("players")
            if isinstance(players, list) and any(
                not isinstance(player, dict) or "play_area" not in player
                for player in players
            ):
                states_have_plays = False
        reconstructed = reconstructed or not states_have_plays
        presented = arranger.enrich_event(
            event,
            manifest.game,
            before_plays=None if states_have_plays else before,
            after_plays=None if states_have_plays else after,
        )
        presented_events.append(presented)
        bucket = rounds.setdefault(
            round_index,
            {"initial_state": None, "items": []},
        )
        if event.event_type is ReplayEventType.ROUND_STARTED:
            bucket["initial_state"] = (
                presented.before_state or presented.after_state
            )
        elif event.event_type is ReplayEventType.ACTION_COMMITTED:
            bucket["items"].append(presented)

    lines = [
        {
            "kind": "meta",
            "presentation_version": PRESENTATION_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "arranger_sha256": arranger.binary_sha256(),
            "source_sha": ARRANGER_SOURCE_SHA,
            "source_event_id": last_event_id,
            "presentation_reconstructed": reconstructed,
        }
    ]
    for round_index in sorted(rounds):
        initial = rounds[round_index]["initial_state"]
        if initial is not None:
            lines.append(
                {
                    "kind": "initial_state",
                    "round_index": round_index,
                    "state": initial,
                }
            )
        for item in rounds[round_index]["items"]:
            lines.append(
                {
                    "kind": "action_committed",
                    "round_index": round_index,
                    "event": item.model_dump(mode="json"),
                }
            )
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    for bucket in rounds.values():
        bucket["presentation_reconstructed"] = reconstructed
    return rounds


def _load_presentation_cache(
    target: Path,
    manifest: MatchManifest,
    *,
    source_event_id: int,
    arranger_sha256: str | None,
) -> dict[int, dict[str, Any]] | None:
    """Return a complete matching derivative cache, or regenerate it safely."""

    try:
        raw_lines = target.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    if not raw_lines:
        return None
    try:
        meta = json.loads(raw_lines[0])
        if (
            not isinstance(meta, dict)
            or meta.get("kind") != "meta"
            or meta.get("presentation_version") != PRESENTATION_VERSION
            or meta.get("normalizer_version") != NORMALIZER_VERSION
            or meta.get("arranger_sha256") != arranger_sha256
            or meta.get("source_sha") != ARRANGER_SOURCE_SHA
            or meta.get("source_event_id") != source_event_id
        ):
            return None
        reconstructed = bool(meta.get("presentation_reconstructed"))
        rounds: dict[int, dict[str, Any]] = {}
        for raw_line in raw_lines[1:]:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                return None
            round_index = row.get("round_index")
            if (
                isinstance(round_index, bool)
                or not isinstance(round_index, int)
                or round_index < 0
            ):
                return None
            bucket = rounds.setdefault(
                round_index,
                {
                    "initial_state": None,
                    "items": [],
                    "presentation_reconstructed": reconstructed,
                },
            )
            if row.get("kind") == "initial_state":
                state = row.get("state")
                if not isinstance(state, dict) or bucket["initial_state"] is not None:
                    return None
                bucket["initial_state"] = state
                continue
            if row.get("kind") == "action_committed":
                event = ReplayEvent.model_validate(row.get("event"))
                if (
                    event.match_id != manifest.match_id
                    or event.round_index != round_index
                    or event.event_type is not ReplayEventType.ACTION_COMMITTED
                ):
                    return None
                bucket["items"].append(event)
                continue
            return None
        return rounds
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
