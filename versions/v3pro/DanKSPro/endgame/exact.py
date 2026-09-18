"""Complete reversible partnership minimax with bounded transposition reuse.

This low-level solver sees a complete synthetic deal. The public runtime
installs only allocations derived from actor hand, played cards and counts.
"""

from __future__ import annotations

import copy

import json

import random

import time

from collections import deque

from contextlib import contextmanager

from dataclasses import dataclass, field

from typing import Any, Iterable


def wire_action(action: Any) -> Any:
    return action.to_json() if hasattr(action, "to_json") else action


def action_cards(action: Any) -> list[Any]:
    payload = wire_action(action)
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        return list(payload[2] or [])
    if isinstance(payload, dict):
        return list(payload.get("cards") or [])
    return list(getattr(action, "cards", ()) or ())


def action_kind(action: Any) -> str:
    payload = wire_action(action)
    if isinstance(payload, (list, tuple)) and payload:
        return str(payload[0]).upper()
    if isinstance(payload, dict):
        return str(payload.get("kind") or payload.get("type") or "").upper()
    return str(getattr(action, "type", getattr(action, "kind", ""))).upper()


def action_key(action: Any) -> str:
    return json.dumps(
        wire_action(action), ensure_ascii=False, sort_keys=True, default=str
    )


def order_action_indices(actions: Iterable[Any], hand_size: int) -> list[int]:
    """Domain ordering only; it never removes a legal action."""

    rows: list[tuple[int, int, int, int]] = []
    for index, action in enumerate(actions):
        cards = action_cards(action)
        finishes = int(bool(cards) and len(cards) == hand_size)
        is_pass = int(action_kind(action) == "PASS" or not cards)
        rows.append((index, finishes, len(cards), is_pass))
    rows.sort(key=lambda row: (-row[1], -row[2], row[3], row[0]))
    return [row[0] for row in rows]


def protected_root_indices(
    actions: list[Any],
    v3pro_order: Iterable[int],
    chosen_index: int,
    hand_size: int,
    cap: int,
) -> list[int]:
    """Return a cap plus tactical escape hatches; the result may exceed cap."""

    if cap < 0:
        raise ValueError("cap must be non-negative")
    valid = set(range(len(actions)))
    ranking = [int(index) for index in v3pro_order if int(index) in valid]
    kept = set(ranking[:cap])
    if chosen_index in valid:
        kept.add(int(chosen_index))

    nonpass_nonfinish: list[int] = []
    for index, action in enumerate(actions):
        cards = action_cards(action)
        if action_kind(action) == "PASS" or not cards:
            kept.add(index)
        elif len(cards) == hand_size:
            kept.add(index)
        else:
            nonpass_nonfinish.append(index)

    # Preserve the engine's weakest ordinary move.  Several observed oracle
    # corrections were precisely PASS or a low single, so policy score alone
    # is not a safe hard-pruning criterion.
    if nonpass_nonfinish:
        kept.add(min(nonpass_nonfinish))
    return [index for index in range(len(actions)) if index in kept]


@dataclass(frozen=True)
class TTEntry:
    value: float
    flag: str  # EXACT, LOWER, or UPPER


@dataclass
class ExactTranspositionTable:
    entries: dict[tuple[Any, ...], TTEntry] = field(default_factory=dict)
    lookups: int = 0
    hits: int = 0
    stores: int = 0
    max_entries: int | None = None

    def __post_init__(self) -> None:
        if self.max_entries is not None:
            if (
                isinstance(self.max_entries, bool)
                or int(self.max_entries) != self.max_entries
                or self.max_entries <= 0
            ):
                raise ValueError("max_entries must be a positive integer or None")
            self.max_entries = int(self.max_entries)

    def _touch(self, key: tuple[Any, ...], entry: TTEntry) -> None:
        self.entries.pop(key, None)
        self.entries[key] = entry

    def lookup(self, key: tuple[Any, ...]) -> float | None:
        """Small exact-only interface used by unit tests and diagnostics."""

        self.lookups += 1
        entry = self.entries.get(key)
        if entry is None or entry.flag != "EXACT":
            return None
        self.hits += 1
        self._touch(key, entry)
        return entry.value

    def probe(self, key: tuple[Any, ...]) -> TTEntry | None:
        self.lookups += 1
        entry = self.entries.get(key)
        if entry is not None:
            self.hits += 1
            self._touch(key, entry)
        return entry

    def store(self, key: tuple[Any, ...], value: float, flag: str = "EXACT") -> None:
        previous = self.entries.get(key)
        if previous is None or flag == "EXACT" or previous.flag != "EXACT":
            entry = TTEntry(float(value), flag)
            self._touch(key, entry)
            self.stores += 1
            if self.max_entries is not None:
                while len(self.entries) > self.max_entries:
                    oldest = next(iter(self.entries))
                    del self.entries[oldest]


@dataclass
class SearchStats:
    max_nodes: int
    nodes: int = 0
    expanded_actions: int = 0
    cutoffs: int = 0


class NodeBudgetExceeded(RuntimeError):
    pass


def state_key(env: Any, root_team: int) -> tuple[Any, ...]:
    greater = env.state.greater_action
    current = env.state.current_action
    return (
        int(root_team),
        _restoration_semantic(getattr(env, "rank", None)),
        _restoration_semantic(getattr(env, "rank_order", None)),
        _restoration_semantic(getattr(env, "phase", None)),
        _restoration_semantic(getattr(env.state, "number_order", None)),
        _restoration_semantic(getattr(env.state, "deck_shuffle_times", None)),
        tuple(tuple(str(card) for card in player.hand_cards) for player in env.players),
        int(env.state.current_pos),
        int(env.state.greater_pos),
        bool(env.action_first),
        action_key(greater) if greater is not None else None,
        action_key(current) if current is not None else None,
        tuple(int(value) for value in env.settlement.order),
    )


def _restoration_semantic(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_restoration_semantic(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _restoration_semantic(item)) for key, item in value.items()
            )
        )
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return ("json", _restoration_semantic(to_json()))
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(
                value,
                "__qualname__",
                getattr(value, "__name__", type(value).__qualname__),
            ),
        )
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                sorted(
                    (str(key), _restoration_semantic(item))
                    for key, item in fields.items()
                    if not str(key).startswith("_")
                )
            ),
        )
    return (type(value).__module__, type(value).__qualname__)


def restoration_key(env: Any, root_team: int) -> tuple[Any, ...]:
    """Fingerprint rule state plus engine metadata that loop_back may omit."""
    env_fields = tuple(
        (name, _restoration_semantic(getattr(env, name, None)))
        for name in (
            "results",
            "loop",
            "belongs_to",
            "rank_inc",
            "rank_belongs",
            "action_first",
            "shuffle_times",
            "trace",
        )
    )
    player_fields = tuple(
        _restoration_semantic(getattr(player, "__dict__", {})) for player in env.players
    )
    state_fields = _restoration_semantic(getattr(env.state, "__dict__", {}))
    settlement_fields = _restoration_semantic(getattr(env.settlement, "__dict__", {}))
    legal = tuple(
        action_key(env.legal_moves[int(index)])
        for index in getattr(env.legal_moves, "valid_range", ())
    )
    return (
        state_key(env, root_team),
        env_fields,
        player_fields,
        state_fields,
        settlement_fields,
        legal,
    )


def search_semantic_key(env: Any, root_team: int) -> tuple[Any, ...]:
    """Compact fingerprint of every field that can affect later play search."""

    loop = getattr(env, "loop", None)
    loop_identity = getattr(loop, "__func__", loop)
    player_metadata = tuple(
        (
            int(getattr(player, "victory", 0)),
            int(getattr(player, "hearts_num", 0)),
            int(getattr(player, "rank", 0)),
            int(getattr(player, "stuck_times", 0)),
        )
        for player in env.players
    )
    settlement = _restoration_semantic(
        getattr(getattr(env, "settlement", None), "__dict__", {})
    )
    legal = tuple(
        action_key(env.legal_moves[int(index)])
        for index in getattr(env.legal_moves, "valid_range", ())
    )
    return (
        state_key(env, root_team),
        player_metadata,
        settlement,
        legal,
        len(getattr(env, "trace", ())),
        loop_identity,
        _restoration_semantic(getattr(env, "results", None)),
        _restoration_semantic(getattr(env, "belongs_to", None)),
        _restoration_semantic(getattr(env, "rank_inc", None)),
        _restoration_semantic(getattr(env, "rank_belongs", None)),
        _restoration_semantic(getattr(env, "shuffle_times", None)),
    )


@dataclass
class EngineSearchSnapshot:
    """Restorable metadata boundary around one reversible engine transition."""

    player_dicts: tuple[dict[str, Any], ...]
    player_hand_objects: tuple[Any, ...]
    state_object: Any
    state_dict: dict[str, Any]
    settlement_object: Any
    settlement_dict: dict[str, Any]
    trace_object: Any
    trace_values: Any
    legal_moves_object: Any
    legal_moves_dict: dict[str, Any]
    env_fields: dict[str, tuple[bool, Any]]

    @classmethod
    def capture(cls, env: Any) -> "EngineSearchSnapshot":
        names = (
            "results",
            "loop",
            "belongs_to",
            "rank_inc",
            "rank_belongs",
            "action_first",
            "shuffle_times",
        )
        env_fields: dict[str, tuple[bool, Any]] = {}
        for name in names:
            exists = hasattr(env, name)
            value = getattr(env, name, None)
            env_fields[name] = (
                exists,
                value if callable(value) else copy.deepcopy(value),
            )
        return cls(
            player_dicts=tuple(
                copy.deepcopy(getattr(player, "__dict__", {})) for player in env.players
            ),
            player_hand_objects=tuple(
                getattr(player, "hand_cards", None) for player in env.players
            ),
            state_object=env.state,
            state_dict=copy.deepcopy(getattr(env.state, "__dict__", {})),
            settlement_object=env.settlement,
            settlement_dict=copy.deepcopy(getattr(env.settlement, "__dict__", {})),
            trace_object=getattr(env, "trace", None),
            trace_values=copy.deepcopy(getattr(env, "trace", None)),
            legal_moves_object=getattr(env, "legal_moves", None),
            legal_moves_dict=copy.deepcopy(
                getattr(getattr(env, "legal_moves", None), "__dict__", {})
            ),
            env_fields=env_fields,
        )

    def restore(self, env: Any) -> None:
        for player, fields, hand_object in zip(
            env.players,
            self.player_dicts,
            self.player_hand_objects,
            strict=True,
        ):
            restored_fields = copy.deepcopy(fields)
            hand_values = restored_fields.pop("hand_cards", None)
            player.__dict__.clear()
            player.__dict__.update(restored_fields)
            if hand_object is not None and hand_values is not None:
                hand_object[:] = hand_values
                player.hand_cards = hand_object
            elif hand_values is not None:
                player.hand_cards = hand_values
        self.state_object.__dict__.clear()
        self.state_object.__dict__.update(copy.deepcopy(self.state_dict))
        env.state = self.state_object
        self.settlement_object.__dict__.clear()
        self.settlement_object.__dict__.update(copy.deepcopy(self.settlement_dict))
        env.settlement = self.settlement_object
        if isinstance(self.trace_object, list) and isinstance(self.trace_values, list):
            self.trace_object[:] = copy.deepcopy(self.trace_values)
            env.trace = self.trace_object
        elif self.trace_object is not None:
            env.trace = copy.deepcopy(self.trace_values)
        if self.legal_moves_object is not None:
            legal_fields = getattr(self.legal_moves_object, "__dict__", None)
            if isinstance(legal_fields, dict):
                legal_fields.clear()
                legal_fields.update(copy.deepcopy(self.legal_moves_dict))
            env.legal_moves = self.legal_moves_object
        for name, (existed, value) in self.env_fields.items():
            if existed:
                setattr(env, name, value if callable(value) else copy.deepcopy(value))
            elif hasattr(env, name):
                delattr(env, name)


@dataclass
class EngineSearchEdgeSnapshot:
    """Cheap per-edge rollback for metadata omitted by the engine undo frame."""

    player_dicts: tuple[dict[str, Any], ...]
    player_hand_objects: tuple[Any, ...]
    player_hand_values: tuple[tuple[Any, ...], ...]
    state_object: Any
    settlement_object: Any
    settlement_dict: dict[str, Any]
    trace_object: Any
    trace_values: tuple[Any, ...]
    legal_moves_object: Any
    env_fields: dict[str, tuple[bool, Any]]

    @classmethod
    def capture(cls, env: Any) -> "EngineSearchEdgeSnapshot":
        names = (
            "results",
            "loop",
            "belongs_to",
            "rank_inc",
            "rank_belongs",
            "action_first",
            "shuffle_times",
        )
        env_fields: dict[str, tuple[bool, Any]] = {}
        for name in names:
            exists = hasattr(env, name)
            value = getattr(env, name, None)
            env_fields[name] = (
                exists,
                value if callable(value) else copy.deepcopy(value),
            )
        return cls(
            player_dicts=tuple(
                copy.deepcopy(
                    {
                        name: value
                        for name, value in getattr(player, "__dict__", {}).items()
                        if name != "hand_cards"
                    }
                )
                for player in env.players
            ),
            player_hand_objects=tuple(
                getattr(player, "hand_cards", None) for player in env.players
            ),
            player_hand_values=tuple(
                tuple(getattr(player, "hand_cards", ())) for player in env.players
            ),
            state_object=env.state,
            settlement_object=env.settlement,
            settlement_dict=copy.deepcopy(getattr(env.settlement, "__dict__", {})),
            trace_object=getattr(env, "trace", None),
            trace_values=tuple(getattr(env, "trace", ())),
            legal_moves_object=getattr(env, "legal_moves", None),
            env_fields=env_fields,
        )

    def restore_metadata(self, env: Any) -> None:
        for player, fields, hand_object, hand_values in zip(
            env.players,
            self.player_dicts,
            self.player_hand_objects,
            self.player_hand_values,
            strict=True,
        ):
            player.__dict__.clear()
            player.__dict__.update(copy.deepcopy(fields))
            if hand_object is not None:
                hand_object[:] = hand_values
                player.hand_cards = hand_object
        env.state = self.state_object
        self.settlement_object.__dict__.clear()
        self.settlement_object.__dict__.update(copy.deepcopy(self.settlement_dict))
        env.settlement = self.settlement_object
        if isinstance(self.trace_object, list):
            self.trace_object[:] = self.trace_values
            env.trace = self.trace_object
        if self.legal_moves_object is not None:
            env.legal_moves = self.legal_moves_object
        for name, (existed, value) in self.env_fields.items():
            if existed:
                setattr(env, name, value if callable(value) else copy.deepcopy(value))
            elif hasattr(env, name):
                delattr(env, name)


def terminal_order(messages: list[Any]) -> list[int] | None:
    for message in messages:
        body = message.body
        if body.get("type") == "notify" and body.get("stage") == "episodeOver":
            return [int(value) for value in body.get("order", [])]
    return None


def unique_action_indices(env: Any, domain_order: bool) -> list[int]:
    seen: set[str] = set()
    indices: list[int] = []
    for raw_index in env.legal_moves.valid_range:
        index = int(raw_index)
        key = action_key(env.legal_moves[index])
        if key not in seen:
            seen.add(key)
            indices.append(index)
    if not domain_order:
        return indices
    hand_size = len(env.players[int(env.state.current_pos)].hand_cards)
    action_subset = [env.legal_moves[index] for index in indices]
    return [
        indices[offset] for offset in order_action_indices(action_subset, hand_size)
    ]


@contextmanager
def reversible_transition(
    env: Any, index: int, before: tuple[Any, ...], root_team: int, label: str
):
    """Execute exactly one engine transition and restore it on every exit path."""
    snapshot = EngineSearchSnapshot.capture(env)
    restoration_before = restoration_key(env, root_team)
    entered = False
    try:
        messages = list(env.loop({"actIndex": int(index)}))
        entered = True
    except BaseException as exc:
        if state_key(env, root_team) == before:
            snapshot.restore(env)
            if restoration_key(env, root_team) != restoration_before:
                raise RuntimeError(
                    f"{label} failed before a restorable transition"
                ) from exc
            raise
        try:
            env.loop_back()
        except BaseException as restore_exc:
            snapshot.restore(env)
            raise RuntimeError(
                f"{label} transition failed and restoration failed"
            ) from restore_exc
        snapshot.restore(env)
        core_restored = state_key(env, root_team) == before
        if not core_restored or restoration_key(env, root_team) != restoration_before:
            raise RuntimeError(
                f"{label} transition failed and loop_back did not restore state"
            ) from exc
        raise
    try:
        yield messages
    finally:
        if entered:
            try:
                env.loop_back()
            except BaseException as restore_exc:
                snapshot.restore(env)
                raise RuntimeError(f"{label} restoration failed") from restore_exc
            snapshot.restore(env)
            core_restored = state_key(env, root_team) == before
            if (
                not core_restored
                or restoration_key(env, root_team) != restoration_before
            ):
                raise RuntimeError(f"{label} loop_back did not restore search state")


@contextmanager
def transactional_reversible_transition(
    env: Any,
    index: int,
    before: tuple[Any, ...],
    root_team: int,
    label: str,
):
    """Undo one edge cheaply inside a full root-level safety transaction."""

    del before
    edge_snapshot = EngineSearchEdgeSnapshot.capture(env)
    semantic_before = search_semantic_key(env, root_team)
    entered = False
    try:
        messages = list(env.loop({"actIndex": int(index)}))
        entered = True
        yield messages
    finally:
        if entered:
            try:
                env.loop_back()
            finally:
                edge_snapshot.restore_metadata(env)
            if search_semantic_key(env, root_team) != semantic_before:
                raise RuntimeError(
                    f"{label} loop_back did not restore search semantics"
                )
        else:
            edge_snapshot.restore_metadata(env)


def minimax(
    env: Any,
    root_team: int,
    alpha: float,
    beta: float,
    stats: SearchStats,
    table: ExactTranspositionTable,
    reward_by_order: Any,
    domain_order: bool,
    transition_context: Any = reversible_transition,
) -> float:
    stats.nodes += 1
    if stats.nodes > stats.max_nodes:
        raise NodeBudgetExceeded

    key = state_key(env, root_team)
    alpha_original, beta_original = alpha, beta
    cached = table.probe(key)
    if cached is not None:
        if cached.flag == "EXACT":
            return cached.value
        if cached.flag == "LOWER":
            alpha = max(alpha, cached.value)
        elif cached.flag == "UPPER":
            beta = min(beta, cached.value)
        if alpha >= beta:
            return cached.value

    maximizing = int(env.state.current_pos) % 2 == root_team
    best = -4.0 if maximizing else 4.0
    before = key
    for index in unique_action_indices(env, domain_order):
        stats.expanded_actions += 1
        with transition_context(env, index, before, root_team, "minimax") as messages:
            order = terminal_order(messages)
            if order is not None:
                value = float(reward_by_order(order, root_team))
            else:
                value = minimax(
                    env,
                    root_team,
                    alpha,
                    beta,
                    stats,
                    table,
                    reward_by_order,
                    domain_order,
                    transition_context,
                )

        if maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if beta <= alpha:
            stats.cutoffs += 1
            break

    if best <= alpha_original:
        flag = "UPPER"
    elif best >= beta_original:
        flag = "LOWER"
    else:
        flag = "EXACT"
    table.store(key, best, flag)
    return best


def _solve_root_actions(
    env: Any,
    root_indices: Iterable[int],
    max_nodes: int,
    table: ExactTranspositionTable,
    reward_by_order: Any,
    domain_order: bool,
    transition_context: Any,
) -> dict[str, Any]:
    root_team = int(env.state.current_pos) % 2
    before = state_key(env, root_team)
    stats = SearchStats(max_nodes=max_nodes)
    start_hits = table.hits
    start_lookups = table.lookups
    started = time.perf_counter()
    values: dict[int, float] = {}
    try:
        for raw_index in root_indices:
            index = int(raw_index)
            with transition_context(env, index, before, root_team, "root") as messages:
                order = terminal_order(messages)
                if order is not None:
                    value = float(reward_by_order(order, root_team))
                else:
                    value = minimax(
                        env,
                        root_team,
                        -4.0,
                        4.0,
                        stats,
                        table,
                        reward_by_order,
                        domain_order,
                        transition_context,
                    )
            values[index] = value
    except NodeBudgetExceeded:
        return {
            "status": "node_budget_exceeded",
            "nodes": stats.nodes,
            "expanded_actions": stats.expanded_actions,
            "cutoffs": stats.cutoffs,
            "cache_lookups": table.lookups - start_lookups,
            "cache_hits": table.hits - start_hits,
            "cache_entries": len(table.entries),
            "elapsed_sec": time.perf_counter() - started,
        }
    return {
        "status": "ok",
        "values": values,
        "nodes": stats.nodes,
        "expanded_actions": stats.expanded_actions,
        "cutoffs": stats.cutoffs,
        "cache_lookups": table.lookups - start_lookups,
        "cache_hits": table.hits - start_hits,
        "cache_entries": len(table.entries),
        "elapsed_sec": time.perf_counter() - started,
    }


def solve_root_actions(
    env: Any,
    root_indices: Iterable[int],
    max_nodes: int,
    table: ExactTranspositionTable,
    reward_by_order: Any,
    domain_order: bool,
) -> dict[str, Any]:
    return _solve_root_actions(
        env,
        root_indices,
        max_nodes,
        table,
        reward_by_order,
        domain_order,
        reversible_transition,
    )


def solve_root_actions_transactional(
    env: Any,
    root_indices: Iterable[int],
    max_nodes: int,
    table: ExactTranspositionTable,
    reward_by_order: Any,
    domain_order: bool,
) -> dict[str, Any]:
    """Search with one full root snapshot and compact per-edge restoration."""

    root_team = int(env.state.current_pos) % 2
    snapshot = EngineSearchSnapshot.capture(env)
    restoration_before = restoration_key(env, root_team)
    try:
        result = _solve_root_actions(
            env,
            root_indices,
            max_nodes,
            table,
            reward_by_order,
            domain_order,
            transactional_reversible_transition,
        )
    finally:
        snapshot.restore(env)
        if restoration_key(env, root_team) != restoration_before:
            raise RuntimeError(
                "root transaction did not restore the complete engine state"
            )
    result["full_snapshots"] = 1
    return result
