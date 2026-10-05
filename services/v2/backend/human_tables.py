from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import hmac
import inspect
import json
import os
import secrets
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Protocol
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .human_table_models import (
    EngineFullState,
    HumanTableActionRequest,
    HumanTableArrangeRequest,
    HumanTableCreateRequest,
    HumanTableJoinRequest,
    HumanTableReadyRequest,
    HumanTableRetryRequest,
    HumanTableSeatRequest,
    HumanTableSession,
    SeatKind,
    ViewerRole,
)
from .models import (
    ActorIdentity,
    Game,
    MatchManifest,
    MatchStatus,
    ModelIdentity,
    Pairing,
    PlayerIdentity,
    ReplayEvent,
    ReplayEventType,
    RoundStatus,
    RoundSummary,
    RunMode,
    utc_now,
)
from .storage import MatchStore
from .public_trick_history import PublicTrickHistory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCHEMA = "danks_human_table_audit_v1"
TURN_SECONDS = 20
ARRANGER_TIMEOUT_SECONDS = 1.5
VIEW_KEYS = frozenset(
    {
        "table_no",
        "game",
        "human_slots",
        "round_no",
        "level",
        "guandan_levels",
        "phase",
        "failure",
        "version",
        "viewer",
        "seats",
        "spectator_count",
        "own_hand",
        "spectator_hands",
        "arrange_mode",
        "own_hand_layout",
        "decision",
        "current_trick",
        "trick_history",
        "turn_started_at_ms",
        "turn_seconds",
        "bot_policy",
        "rule_profile",
        "doudizhu_state",
        "result",
        "series",
        "finish_order",
        "replay_history",
        "latest_replay_match_id",
        "server_time_ms",
    }
)
PUBLIC_BOT_POLICY_KEYS = ("id", "ready")
PUBLIC_GUANDAN_RULE_PROFILES = frozenset(
    {"standard_v1", "arena_client_pdf_v1", "sports_bureau"}
)
GUANDAN_LEVEL_SEQUENCE = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
_GUANDAN_PLAY_CARD_CODES = frozenset(
    f"{suit}{rank}" for suit in "SHCD" for rank in "23456789TJQKA"
) | {"SB", "HR"}
_GUANDAN_PLAY_TYPES = frozenset(
    {
        "Single", "Pair", "Trips", "ThreeWithTwo", "Straight",
        "ThreePair", "TwoTrips", "StraightFlush", "Bomb", "FourKings",
    }
)
_GUANDAN_PLAY_RANKS = frozenset("23456789TJQKABR") | {"10"}


def _normalize_guandan_level(value: Any) -> str:
    text = str(value or "2").strip().upper()
    if text == "T":
        text = "10"
    return text if text in GUANDAN_LEVEL_SEQUENCE else "2"


def _advance_guandan_level(level: Any, delta: int) -> str:
    """Advance without skipping or wrapping past A."""

    current = _normalize_guandan_level(level)
    index = GUANDAN_LEVEL_SEQUENCE.index(current)
    return GUANDAN_LEVEL_SEQUENCE[
        min(index + max(0, delta), len(GUANDAN_LEVEL_SEQUENCE) - 1)
    ]


def _is_complete_guandan_order(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and not any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in value
        )
        and set(value) == set(range(4))
    )


_DOUDIZHU_POSITIONS = range(3)
_DOUDIZHU_CARD_CODES = frozenset(
    [
        f"{suit}{rank}"
        for suit in ("S", "H", "C", "D")
        for rank in (
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K",
            "A",
            "2",
        )
    ]
    + ["SB", "HR"]
)


def _is_protocol_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _doudizhu_position(value: Any) -> int | None:
    return value if _is_protocol_int(value) and value in _DOUDIZHU_POSITIONS else None


def _doudizhu_rank_counts(cards: list[str]) -> Counter[str] | None:
    if any(card not in _DOUDIZHU_CARD_CODES for card in cards):
        return None
    # Card labels are exactly two characters; B/R are distinct joker ranks.
    return Counter(card[1] for card in cards)


def _replay_rank_matches(rank: Any, expected: Any) -> bool:
    """Compare setup-action ranks without bool/int protocol ambiguity."""

    if isinstance(expected, bool):
        if isinstance(rank, bool):
            return rank is expected
        if _is_protocol_int(rank):
            return rank in {0, 1} and bool(rank) is expected
        normalized = str(rank).strip().casefold()
        accepted = {"1", "true", "yes"} if expected else {"0", "false", "no"}
        return normalized in accepted
    if _is_protocol_int(expected):
        if isinstance(rank, bool):
            return False
        try:
            return int(str(rank).strip()) == expected
        except ValueError:
            return False
    return str(rank).strip().casefold() == str(expected).strip().casefold()


def _validate_finished_doudizhu_result(state: EngineFullState) -> None:
    """Reject contradictory settlement data before it reaches any browser."""

    result = state.result
    winner = _doudizhu_position(result.get("winner"))
    landlord = _doudizhu_position(result.get("landlord"))
    winning_side = result.get("winningSide")
    scores = result.get("handScores")
    if winner is None or landlord is None:
        raise EngineProtocolError(
            "finished DouDizhu state requires winner and landlord in 0..2"
        )
    if winning_side not in {"landlord", "defenders"}:
        raise EngineProtocolError(
            "finished DouDizhu state requires winningSide landlord or defenders"
        )
    expected_side = "landlord" if winner == landlord else "defenders"
    if winning_side != expected_side:
        raise EngineProtocolError(
            "finished DouDizhu winningSide contradicts winner and landlord"
        )
    if (
        not isinstance(scores, list)
        or len(scores) != 3
        or any(not _is_protocol_int(score) for score in scores)
        or sum(scores) != 0
    ):
        raise EngineProtocolError(
            "finished DouDizhu handScores must be three integers with zero sum"
        )
    if scores[winner] <= 0:
        raise EngineProtocolError(
            "finished DouDizhu winner must have a positive hand score"
        )
    for position, score in enumerate(scores):
        won = position == landlord if winning_side == "landlord" else position != landlord
        if (won and score <= 0) or (not won and score >= 0):
            raise EngineProtocolError(
                "finished DouDizhu handScores contradict the winning side"
            )

    metric_landlord = _doudizhu_position(state.metrics.get("landlord_position"))
    if metric_landlord is None or metric_landlord != landlord:
        raise EngineProtocolError(
            "finished DouDizhu result landlord contradicts engine metrics"
        )
    empty_hands = [
        player.position for player in state.players if player.hand_count == 0
    ]
    if empty_hands != [winner]:
        raise EngineProtocolError(
            "finished DouDizhu winner must be the only player with an empty hand"
        )
    for player in state.players:
        expected_role = "landlord" if player.position == landlord else "farmer"
        if player.role != expected_role:
            raise EngineProtocolError(
                f"finished DouDizhu role mismatch at position {player.position}"
            )


class HumanTableError(Exception):
    """An error that can be mapped directly to a JSON HTTP response."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.status_code = status
        self.code = code
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "code": self.code, "detail": self.detail}


class TableNotFoundError(HumanTableError):
    def __init__(self, table_no: str) -> None:
        super().__init__(404, "table_not_found", f"table {table_no!r} was not found")


class InvalidTokenError(HumanTableError):
    def __init__(self) -> None:
        super().__init__(401, "invalid_token", "the table token is invalid")


class TableConflictError(HumanTableError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(409, code, detail)


class TablePermissionError(HumanTableError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(403, code, detail)


class TableRequestError(HumanTableError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(422, code, detail)


class EngineFailureError(HumanTableError):
    def __init__(self, detail: str) -> None:
        super().__init__(502, "engine_failure", detail)


class EngineProtocolError(RuntimeError):
    pass


class EngineRpcError(EngineProtocolError):
    """A worker rejection with its structured fault code intact."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class EngineClient(Protocol):
    async def start(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def act(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def bot_act(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class HandArrangerClient(Protocol):
    def arrange(
        self,
        game: Game,
        cards: list[str],
        *,
        level: str | int | None,
        mode: int,
        variant: int = 0,
    ) -> dict[str, Any]: ...


EngineFactory = Callable[[Game, str], EngineClient]


def _parse_worker_command(raw: str, variable: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{variable} must be a JSON array") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{variable} must be a non-empty JSON array of strings")
    return list(value)


def human_worker_command(
    game: Game | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve the persistent rules-worker command without invoking a shell."""

    normalized_game = Game(game)
    values = os.environ if environ is None else environ
    variable = f"BATTLE_HUMAN_{normalized_game.value.upper()}_WORKER_COMMAND"
    configured = values.get(variable)
    if configured:
        return _parse_worker_command(configured, variable)
    mode = values.get("BATTLE_HUMAN_ENGINE_MODE") or values.get(
        "BATTLE_WORKER_MODE", "real"
    )
    if mode not in {"real", "mock"}:
        raise ValueError("BATTLE_HUMAN_ENGINE_MODE must be 'real' or 'mock'")
    return [
        sys.executable,
        "-m",
        "workers.table_worker",
        "--game",
        normalized_game.value,
        "--mode",
        mode,
    ]


class SubprocessEngineClient:
    """One persistent NDJSON subprocess for one table.

    Stdout is a strict request/response stream containing one JSON object per
    line.  Stderr is continuously consumed in a separate task so a noisy model
    runtime can never fill the pipe and deadlock a game.
    """

    def __init__(
        self,
        game: Game | str,
        table_no: str,
        *,
        command: list[str] | None = None,
        cwd: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        close_timeout: float = 3.0,
    ) -> None:
        self.game = Game(game)
        self.table_no = table_no
        self.command = list(
            command
            if command is not None
            else human_worker_command(self.game, environ=environ)
        )
        if not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("worker command must be a non-empty list of strings")
        self.cwd = Path(cwd or PROJECT_ROOT).resolve()
        self.environ = dict(environ) if environ is not None else None
        self.close_timeout = close_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=64)
        self._spawn_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._request_sequence = 0
        self._closed = False

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail)[-8192:]

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._closed:
            raise EngineProtocolError("engine client is closed")
        process = self._process
        if process is not None:
            if process.returncode is not None:
                raise EngineProtocolError(
                    f"rules worker exited with code {process.returncode}: "
                    f"{self.stderr_tail}"
                )
            return process
        async with self._spawn_lock:
            process = self._process
            if process is not None:
                return process
            child_env = None
            if self.environ is not None:
                child_env = dict(os.environ)
                child_env.update(self.environ)
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.command,
                    cwd=str(self.cwd),
                    env=child_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024 * 1024,
                )
            except (OSError, ValueError) as exc:
                raise EngineProtocolError(f"could not start rules worker: {exc}") from exc
            self._process = process
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(process),
                name=f"human-engine-stderr-{self.table_no}",
            )
            return process

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        try:
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return
                self._stderr_tail.append(chunk.decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # draining must never break request handling
            self._stderr_tail.append(f"[stderr drain failed: {exc}]\n")

    async def _exchange(self, operation: str, payload: dict[str, Any]) -> Any:
        async with self._request_lock:
            process = await self._ensure_process()
            assert process.stdin is not None
            assert process.stdout is not None
            self._request_sequence += 1
            request_id = f"{self.table_no}:{self._request_sequence}"
            request = {
                "request_id": request_id,
                "type": operation,
                "payload": payload,
            }
            try:
                encoded = json.dumps(
                    request,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8") + b"\n"
            except (TypeError, ValueError) as exc:
                raise EngineProtocolError(f"engine request is not JSON-safe: {exc}") from exc
            try:
                process.stdin.write(encoded)
                await process.stdin.drain()
                raw = await process.stdout.readline()
            except (BrokenPipeError, ConnectionError, OSError) as exc:
                raise EngineProtocolError(
                    f"rules worker connection failed: {exc}; {self.stderr_tail}"
                ) from exc
            if not raw:
                return_code = await process.wait()
                raise EngineProtocolError(
                    f"rules worker exited with code {return_code} without a response: "
                    f"{self.stderr_tail}"
                )
            try:
                response = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EngineProtocolError(
                    "rules worker stdout must contain exactly one JSON object per line"
                ) from exc
            if not isinstance(response, dict):
                raise EngineProtocolError("rules worker response must be a JSON object")
            if response.get("request_id") != request_id:
                raise EngineProtocolError("rules worker response request_id did not match")
            if not isinstance(response.get("ok"), bool):
                raise EngineProtocolError("rules worker response must contain boolean ok")
            if not response["ok"]:
                error = response.get("error")
                if isinstance(error, dict):
                    code = error.get("code", "engine_error")
                    message = error.get("message", "unknown error")
                    detail = f"{code}: {message}"
                else:
                    detail = str(error or "unknown error")
                if isinstance(error, dict) and isinstance(code, str):
                    raise EngineRpcError(code, f"rules worker rejected {operation}: {detail}")
                raise EngineProtocolError(f"rules worker rejected {operation}: {detail}")
            if "state" not in response:
                raise EngineProtocolError("rules worker response is missing state")
            return response["state"]

    async def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = await self._exchange(operation, payload)
        if not isinstance(state, dict):
            raise EngineProtocolError("rules worker response state must be an object")
        return state

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("start", payload)

    async def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("act", payload)

    async def bot_act(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("bot_act", payload)

    async def close(self) -> None:
        if self._closed:
            return
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(EngineProtocolError):
                await self._exchange("close", {})
            if process.stdin is not None:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=self.close_timeout)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.close_timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        self._closed = True
        stderr_task = self._stderr_task
        if stderr_task is not None:
            if not stderr_task.done():
                stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        self._stderr_task = None
        self._process = None


def subprocess_engine_factory(
    *,
    cwd: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> EngineFactory:
    def create(game: Game, table_no: str) -> SubprocessEngineClient:
        return SubprocessEngineClient(
            game,
            table_no,
            cwd=cwd,
            environ=environ,
        )

    return create


@dataclass
class _Seat:
    position: int
    kind: SeatKind
    name: str | None = None
    viewer_id: str | None = None
    ready: bool = False


@dataclass
class _Viewer:
    participant_id: str
    user_id: str
    nickname: str
    token_hash: str
    role: ViewerRole
    arrange_mode: int
    arrange_variant: int = 0
    layout_key: tuple[str, str | None, int, int, tuple[str, ...]] | None = field(
        default=None,
        repr=False,
    )
    raw_layout: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _ActionRecord:
    decision_id: str
    action_index: int
    selected_cards: tuple[str, ...] | None


@dataclass
class _Room:
    table_no: str
    game: Game
    human_slots: int
    seed: int
    engine: EngineClient
    seats: list[_Seat]
    round_no: int = 1
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    viewers: dict[str, _Viewer] = field(default_factory=dict, repr=False)
    token_index: dict[str, str] = field(default_factory=dict, repr=False)
    subscribers: dict[asyncio.Queue[dict[str, Any]], str] = field(
        default_factory=dict, repr=False
    )
    processed_actions: dict[tuple[str, str], _ActionRecord] = field(
        default_factory=dict, repr=False
    )
    next_first_player: int | None = None
    version: int = 1
    phase: str = "waiting"
    failure: dict[str, Any] | None = None
    started: bool = False
    engine_state: EngineFullState | None = field(default=None, repr=False)
    turn_started_at_ms: int | None = None
    bot_task: asyncio.Task[None] | None = field(default=None, repr=False)
    pending_engine: EngineClient | None = field(default=None, repr=False)
    visible_play_areas: list[Any] = field(default_factory=list, repr=False)
    guandan_response_target: dict[str, Any] | None = field(default=None, repr=False)
    public_trick_history: PublicTrickHistory = field(default_factory=PublicTrickHistory, repr=False)
    public_trick_history_round: tuple[int, int] | None = field(default=None, repr=False)
    guandan_team_levels: dict[str, str] = field(
        default_factory=lambda: {"even": "2", "odd": "2"}
    )
    series_no: int = 1
    guandan_current_level: str = "2"
    guandan_active_team: str | None = None
    guandan_a_challenger: str | None = None
    guandan_a_attempt_count: dict[str, int] = field(
        default_factory=lambda: {"even": 0, "odd": 0}
    )
    guandan_head_finish_count_by_seat: list[int] = field(
        default_factory=lambda: [0, 0, 0, 0]
    )
    guandan_team_victory_count: dict[str, int] = field(
        default_factory=lambda: {"even": 0, "odd": 0}
    )
    guandan_rounds_played: int = 0
    guandan_match_winner_team: str | None = None
    guandan_match_finished: bool = False
    guandan_final_result: dict[str, Any] | None = None
    guandan_last_settlement: dict[str, Any] | None = None
    settled_round_no: int = 0
    guandan_finish_order: list[int] = field(default_factory=list)
    replay_match_id: str | None = None
    replay_turn_index: int = 0
    replay_history: list[dict[str, Any]] = field(default_factory=list)
    replay_failed: bool = False
    audit_path: Path | None = field(default=None, repr=False)
    audit_sequence: int = 0
    closed: bool = False


class RoomManager:
    """In-memory human rooms with viewer-specific state projection."""

    def __init__(
        self,
        engine_factory: EngineFactory | None = None,
        *,
        arranger: HandArrangerClient | None = None,
        arranger_timeout_seconds: float = ARRANGER_TIMEOUT_SECONDS,
        bot_delay_ms: int = 800,
        subscriber_queue_size: int = 32,
        audit_root: Path | str | None = None,
        audit_enabled: bool = True,
        replay_store: MatchStore | None = None,
        audit_partition_by_user: bool = False,
    ) -> None:
        if bot_delay_ms < 0:
            raise ValueError("bot_delay_ms must be non-negative")
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        if arranger_timeout_seconds <= 0:
            raise ValueError("arranger_timeout_seconds must be positive")
        self.engine_factory = engine_factory or subprocess_engine_factory()
        self.arranger = arranger
        self.arranger_timeout_seconds = arranger_timeout_seconds
        self.bot_delay_ms = bot_delay_ms
        self.subscriber_queue_size = subscriber_queue_size
        self.replay_store = replay_store
        self.audit_partition_by_user = audit_partition_by_user
        configured_audit_root = (
            audit_root or os.environ.get("BATTLE_HUMAN_LOG_ROOT")
            if audit_enabled
            else None
        )
        self.audit_root = (
            None
            if configured_audit_root is None
            else Path(configured_audit_root).expanduser().resolve()
        )
        if self.audit_root is not None:
            self.audit_root.mkdir(parents=True, exist_ok=True)
        self._rooms: dict[str, _Room] = {}
        self._manager_lock = asyncio.Lock()
        self._engine_close_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_participant_id() -> str:
        return f"participant_{secrets.token_hex(12)}"

    @staticmethod
    def _default_arrange_mode(game: Game) -> int:
        return 1 if game is Game.GUANDAN else 4

    def _create_engine(self, game: Game, table_no: str) -> EngineClient:
        try:
            engine = self.engine_factory(game, table_no)
        except Exception as exc:
            raise EngineFailureError(f"could not create rules engine: {exc}") from exc
        if inspect.isawaitable(engine):
            raise TypeError("engine_factory must return an EngineClient synchronously")
        return engine

    def _new_token(self, room: _Room) -> tuple[str, str]:
        while True:
            token = secrets.token_urlsafe(32)
            digest = self._hash_token(token)
            if digest not in room.token_index:
                return token, digest

    def _new_table_no_locked(self) -> str:
        for _ in range(1000):
            table_no = str(100000 + secrets.randbelow(900000))
            if table_no not in self._rooms:
                return table_no
        raise EngineFailureError("could not allocate a unique table number")

    @staticmethod
    def _coerce(model: type[Any], value: Any) -> Any:
        if isinstance(value, model):
            return value
        return model.model_validate(value)

    async def create(
        self, request: HumanTableCreateRequest | Mapping[str, Any]
    ) -> HumanTableSession:
        request = self._coerce(HumanTableCreateRequest, request)
        async with self._manager_lock:
            if self._closed:
                raise TableConflictError("manager_closed", "room manager is closed")
            table_no = self._new_table_no_locked()
            engine = self._create_engine(request.game, table_no)
            seat_count = 4 if request.game is Game.GUANDAN else 3
            seats = [
                _Seat(position=position, kind=SeatKind.OPEN)
                if position < request.human_slots
                else _Seat(
                    position=position,
                    kind=SeatKind.BOT,
                    name=f"Bot {position + 1}",
                    ready=True,
                )
                for position in range(seat_count)
            ]
            room = _Room(
                table_no=table_no,
                game=request.game,
                human_slots=request.human_slots,
                seed=secrets.randbits(63),
                engine=engine,
                seats=seats,
                audit_path=(
                    None
                    if self.audit_root is None
                    else self.audit_root / table_no / "events.jsonl"
                ),
            )
            token, digest = self._new_token(room)
            participant_id = self._new_participant_id()
            viewer = _Viewer(
                participant_id=participant_id,
                user_id=request.user_id or participant_id,
                nickname=request.nickname,
                token_hash=digest,
                role=ViewerRole.PLAYER,
                arrange_mode=self._default_arrange_mode(request.game),
            )
            room.viewers[participant_id] = viewer
            room.token_index[digest] = participant_id
            self._fill_open_seat(room.seats[0], viewer)
            if self.audit_root is not None and self.audit_partition_by_user:
                timezone_name = os.environ.get("BATTLE_LOG_TIMEZONE", "Asia/Shanghai")
                try:
                    local_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
                except Exception:
                    local_date = datetime.now().date().isoformat()
                room.audit_path = (
                    self.audit_root
                    / local_date
                    / viewer.user_id
                    / table_no
                    / "events.jsonl"
                )
            self._rooms[table_no] = room
            self._audit_locked(
                room,
                "table_created",
                {
                    "creator": participant_id,
                    "user_id": viewer.user_id,
                    "nickname": request.nickname,
                    "game": request.game.value,
                    "human_slots": request.human_slots,
                    "seed": room.seed,
                    "seats": [self._audit_seat(seat) for seat in room.seats],
                },
            )
        return self._session(room, viewer, token)

    async def join(
        self,
        table_no: str,
        request: HumanTableJoinRequest | Mapping[str, Any],
    ) -> HumanTableSession:
        request = self._coerce(HumanTableJoinRequest, request)
        room = self._room(table_no)
        async with room.lock:
            if room.closed:
                raise TableNotFoundError(str(table_no))
            if request.resume_token is not None:
                viewer = self._authenticate_locked(room, request.resume_token)
                self._audit_locked(
                    room,
                    "participant_resumed",
                    {"participant_id": viewer.participant_id, "role": viewer.role.value},
                )
                return self._session(room, viewer, request.resume_token)
            token, digest = self._new_token(room)
            participant_id = self._new_participant_id()
            open_seat = None
            if not room.started and room.phase == "waiting":
                open_seat = next(
                    (seat for seat in room.seats if seat.kind is SeatKind.OPEN), None
                )
            role = ViewerRole.PLAYER if open_seat is not None else ViewerRole.SPECTATOR
            viewer = _Viewer(
                participant_id=participant_id,
                user_id=request.user_id or participant_id,
                nickname=request.nickname,
                token_hash=digest,
                role=role,
                arrange_mode=self._default_arrange_mode(room.game),
            )
            room.viewers[participant_id] = viewer
            room.token_index[digest] = participant_id
            if open_seat is not None:
                self._fill_open_seat(open_seat, viewer)
                room.version += 1
            self._audit_locked(
                room,
                "participant_joined",
                {
                    "participant_id": participant_id,
                    "nickname": request.nickname,
                    "role": role.value,
                    "seat": open_seat.position if open_seat is not None else None,
                },
            )
            self._broadcast_locked(room)
            return self._session(room, viewer, token)

    async def resume_solo(
        self,
        table_no: str,
        request: HumanTableJoinRequest | Mapping[str, Any],
    ) -> HumanTableSession:
        """Resume the sole human seat using its stable browser user id."""

        request = self._coerce(HumanTableJoinRequest, request)
        room = self._room(table_no)
        async with room.lock:
            if request.resume_token is not None:
                viewer = self._authenticate_locked(room, request.resume_token)
                return self._session(room, viewer, request.resume_token)
            if request.user_id is None:
                raise InvalidTokenError()
            viewer = next(
                (
                    item
                    for item in room.viewers.values()
                    if item.role is ViewerRole.PLAYER
                    and hmac.compare_digest(item.user_id, request.user_id)
                ),
                None,
            )
            if viewer is None:
                raise TablePermissionError(
                    "user_mismatch",
                    "this table belongs to a different browser user id",
                )
            token, digest = self._new_token(room)
            room.token_index.pop(viewer.token_hash, None)
            viewer.token_hash = digest
            room.token_index[digest] = viewer.participant_id
            self._revoke_viewer_subscriptions_locked(room, viewer)
            self._audit_locked(
                room,
                "participant_resumed",
                {
                    "participant_id": viewer.participant_id,
                    "user_id": viewer.user_id,
                    "role": viewer.role.value,
                    "method": "stable_user_id",
                },
            )
            return self._session(room, viewer, token)

    async def get_view(self, table_no: str, token: str) -> dict[str, Any]:
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            await self._warm_private_layout_locked(room, viewer)
            return self._project_locked(room, viewer)

    # ``view`` is a convenient route-level spelling.
    view = get_view

    async def retry(
        self,
        table_no: str,
        token: str,
        request: HumanTableRetryRequest | Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._coerce(HumanTableRetryRequest, request)
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            self._expect_version(room, request.expected_version)
            if self._seat_for_viewer(room, viewer) is None:
                raise TablePermissionError("not_player", "spectators cannot retry")
            state = room.engine_state
            if (
                room.phase != "paused"
                or room.failure != {"code": "ai_unavailable", "retryable": True}
                or state is None
                or state.decision is None
                or room.seats[state.decision.actor].kind is not SeatKind.BOT
            ):
                raise TableConflictError("not_retryable", "the table has no retryable AI selection failure")
            room.phase = state.phase
            room.failure = None
            room.version += 1
            self._ensure_bot_task_locked(room)
            self._broadcast_locked(room)
            return self._project_locked(room, viewer)

    async def arrange(
        self,
        table_no: str,
        token: str,
        request: HumanTableArrangeRequest | Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._coerce(HumanTableArrangeRequest, request)
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            if self._seat_for_viewer(room, viewer) is None:
                raise TablePermissionError(
                    "not_player", "spectators do not have a private hand to arrange"
                )
            if room.game is Game.DOUDIZHU and request.mode != 4:
                raise TableRequestError(
                    "invalid_arrange_mode",
                    "doudizhu supports only the same-rank layout mode 4",
                )
            if viewer.arrange_mode == request.mode:
                viewer.arrange_variant += 1
            else:
                viewer.arrange_mode = request.mode
                viewer.arrange_variant = 0
            await self._warm_private_layout_locked(room, viewer)
            view = self._project_locked(room, viewer)
            self._broadcast_viewer_locked(room, viewer, view=view)
            return view

    async def seat(
        self,
        table_no: str,
        token: str,
        request: HumanTableSeatRequest | Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._coerce(HumanTableSeatRequest, request)
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            self._expect_version(room, request.expected_version)
            if room.started or room.phase != "waiting":
                raise TableConflictError(
                    "table_already_started", "seats can only change while waiting"
                )
            source = self._seat_for_viewer(room, viewer)
            if source is None:
                raise TablePermissionError("not_player", "spectators cannot change seats")
            if source.ready:
                raise TableConflictError(
                    "player_ready", "unready before changing seats"
                )
            if request.position >= len(room.seats):
                raise TableRequestError("invalid_seat", "seat is outside this table")
            target = room.seats[request.position]
            if target.position == source.position:
                return self._project_locked(room, viewer)
            if target.kind is SeatKind.HUMAN:
                raise TableConflictError("seat_occupied", "target seat has a human player")
            self._swap_seat_occupants(source, target)
            room.version += 1
            self._audit_locked(
                room,
                "seat_changed",
                {
                    "participant_id": viewer.participant_id,
                    "from": source.position,
                    "to": target.position,
                    "seats": [self._audit_seat(seat) for seat in room.seats],
                },
            )
            self._broadcast_locked(room)
            return self._project_locked(room, viewer)

    async def ready(
        self,
        table_no: str,
        token: str,
        request: HumanTableReadyRequest | Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._coerce(HumanTableReadyRequest, request)
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            self._expect_version(room, request.expected_version)
            if room.phase not in {"waiting", "finished", "game_over"}:
                raise TableConflictError(
                    "table_already_started",
                    "players can ready only before a round or after the whole series finishes",
                )
            is_rematch = room.phase in {"finished", "game_over"}
            is_new_series = room.phase == "game_over"
            seat = self._seat_for_viewer(room, viewer)
            if seat is None:
                raise TablePermissionError("not_player", "spectators cannot ready")
            if seat.ready == request.ready:
                return self._project_locked(room, viewer)
            seat.ready = request.ready
            room.version += 1
            self._audit_locked(
                room,
                "ready_changed",
                {
                    "participant_id": viewer.participant_id,
                    "seat": seat.position,
                    "ready": request.ready,
                },
            )
            should_start = all(
                seat.kind is not SeatKind.OPEN and seat.ready for seat in room.seats
            )
            if not should_start:
                self._broadcast_locked(room)
                return self._project_locked(room, viewer)

            # Mark the transition before awaiting the engine.  The per-room
            # lock plus this flag guarantees exactly one start invocation.
            room.started = True
            room.phase = "starting"
            self._broadcast_locked(room)
            engine = room.engine
            candidate_engine: EngineClient | None = None
            next_seed = room.seed
            try:
                if is_rematch:
                    candidate_engine = self._create_engine(room.game, room.table_no)
                    room.pending_engine = candidate_engine
                    engine = candidate_engine
                    next_seed = secrets.randbits(63)
                raw_state = await engine.start(
                    self._start_payload(
                        room,
                        seed=next_seed,
                        new_series=is_new_series,
                    )
                )
                state = self._validate_engine_state(room, raw_state, initial=True)
                if room.closed or self._closed:
                    raise TableNotFoundError(room.table_no)
            except asyncio.CancelledError:
                if candidate_engine is not None:
                    await self._retire_pending_engine(room, candidate_engine)
                raise
            except Exception as exc:
                if candidate_engine is not None:
                    await self._retire_pending_engine(room, candidate_engine)
                if room.closed or self._closed:
                    raise TableNotFoundError(room.table_no) from exc
                room.phase = "error"
                room.failure = {"code": "engine_failure", "retryable": False}
                room.turn_started_at_ms = None
                room.version += 1
                self._audit_locked(
                    room,
                    "engine_start_failed",
                    {"round_no": room.round_no, "error": repr(exc)},
                )
                self._broadcast_locked(room)
                if isinstance(exc, HumanTableError):
                    raise
                raise EngineFailureError("rules engine failed to start") from exc
            previous_engine: EngineClient | None = None
            if is_rematch:
                previous_engine = room.engine
                room.engine = engine
                room.pending_engine = None
                room.seed = next_seed
                if is_new_series:
                    self._reset_guandan_series_locked(room)
                else:
                    room.round_no += 1
                room.processed_actions.clear()
                room.next_first_player = None
                room.visible_play_areas = [None] * len(room.seats)
                room.guandan_response_target = None
                room.public_trick_history = PublicTrickHistory()
                room.guandan_finish_order = []
                room.replay_match_id = None
                room.replay_turn_index = 0
                room.replay_failed = False
            self._install_engine_state_locked(room, state)
            self._start_replay_locked(room, state)
            # Projection is always memory-only.  Publish a safe flat layout
            # immediately, then replace it with a warmed Go layout below.
            self._broadcast_locked(room)
            self._ensure_bot_task_locked(room)
            close_task = (
                self._schedule_engine_close(previous_engine)
                if previous_engine is not None
                else None
            )
            await self._warm_all_private_layouts_locked(room)
            if not room.closed:
                self._broadcast_locked(room)
            view = self._project_locked(room, viewer)
            if close_task is not None:
                await asyncio.shield(close_task)
            return view

    async def action(
        self,
        table_no: str,
        token: str,
        request: HumanTableActionRequest | Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._coerce(HumanTableActionRequest, request)
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            seat = self._seat_for_viewer(room, viewer)
            if seat is None:
                raise TablePermissionError("not_player", "spectators cannot play actions")
            if room.phase in {"error", "paused"}:
                raise TableConflictError(
                    "table_paused" if room.phase == "paused" else "table_error",
                    "the table is stopped after an engine or AI selection failure",
                )
            action_key = (viewer.participant_id, request.client_action_id)
            selected_cards_key = (
                None
                if request.selected_cards is None
                else tuple(sorted(request.selected_cards))
            )
            if request.selected_cards is not None and room.game is not Game.DOUDIZHU:
                raise TableRequestError(
                    "invalid_selected_cards",
                    "selected_cards is supported only for DouDizhu play actions",
                )
            previous = room.processed_actions.get(action_key)
            if previous is not None:
                if (
                    previous.decision_id != request.decision_id
                    or previous.action_index != request.action_index
                    or previous.selected_cards != selected_cards_key
                ):
                    raise TableConflictError(
                        "client_action_id_reused",
                        "client_action_id was already used for a different action",
                    )
                return self._project_locked(room, viewer)
            self._expect_version(room, request.expected_version)
            state = room.engine_state
            if state is None or not room.started:
                raise TableConflictError("game_not_started", "the game has not started")
            decision = state.decision
            if decision is None:
                raise TableConflictError("no_active_decision", "there is no active decision")
            if decision.id != request.decision_id:
                raise TableConflictError(
                    "stale_decision", "the decision is no longer active"
                )
            if decision.actor != seat.position:
                raise TableConflictError("not_your_turn", "it is not this player's turn")
            selected = next(
                (
                    action
                    for action in decision.actions
                    if action.action_index == request.action_index
                ),
                None,
            )
            if selected is None:
                raise TableRequestError(
                    "illegal_action", "action_index is not legal for this decision"
                )
            if request.selected_cards is not None:
                if (
                    decision.phase.casefold() != "play"
                    or selected.type.casefold() == "pass"
                ):
                    raise TableRequestError(
                        "invalid_selected_cards",
                        "selected_cards requires a non-PASS DouDizhu play action",
                    )
                selected_rank_counts = _doudizhu_rank_counts(
                    request.selected_cards
                )
                canonical_rank_counts = _doudizhu_rank_counts(selected.cards)
                player = next(
                    item for item in state.players if item.position == seat.position
                )
                if (
                    selected_rank_counts is None
                    or canonical_rank_counts is None
                    or selected_rank_counts != canonical_rank_counts
                    or bool(
                        Counter(request.selected_cards) - Counter(player.hand)
                    )
                ):
                    raise TableRequestError(
                        "invalid_selected_cards",
                        "selected_cards must be an equivalent subset of the current hand",
                    )
            payload = {
                "table_no": room.table_no,
                "game": room.game.value,
                "seat": seat.position,
                "decision_id": decision.id,
                "action_index": request.action_index,
                "client_action_id": request.client_action_id,
                "expected_state_version": state.state_version,
                "action": selected.model_dump(mode="json"),
            }
            if request.selected_cards is not None:
                payload["selected_cards"] = list(request.selected_cards)
            inference_started = time.perf_counter()
            try:
                raw_state = await room.engine.act(payload)
                next_state = self._validate_engine_state(room, raw_state, initial=False)
            except Exception as exc:
                # The worker may already have consumed the action before its
                # reply was lost or failed validation. Continuing from the
                # controller's old snapshot would make the states diverge.
                room.phase = "error"
                room.failure = {"code": "engine_failure", "retryable": False}
                room.turn_started_at_ms = None
                room.version += 1
                self._audit_locked(
                    room,
                    "human_action_failed",
                    {
                        "participant_id": viewer.participant_id,
                        "seat": seat.position,
                        "decision_id": request.decision_id,
                        "action_index": request.action_index,
                        "error": repr(exc),
                    },
                )
                self._broadcast_locked(room)
                if isinstance(exc, HumanTableError):
                    raise
                raise EngineFailureError("rules engine rejected action") from exc
            room.processed_actions[action_key] = _ActionRecord(
                decision_id=request.decision_id,
                action_index=request.action_index,
                selected_cards=selected_cards_key,
            )
            committed_audit_error: OSError | None = None
            try:
                self._audit_locked(
                    room,
                    "human_action_committed",
                    {
                        "participant_id": viewer.participant_id,
                        "seat": seat.position,
                        "decision_id": request.decision_id,
                        "action_index": request.action_index,
                        "action": selected.model_dump(mode="json"),
                        "selected_cards": request.selected_cards,
                    },
                )
            except OSError as exc:
                committed_audit_error = exc
            replay_action = selected.model_dump(mode="json")
            if request.selected_cards is not None:
                replay_action["cards"] = list(request.selected_cards)
            self._record_replay_action_locked(
                room,
                state,
                next_state,
                replay_action,
                inference_ms=(time.perf_counter() - inference_started) * 1000,
            )
            self._install_engine_state_locked(
                room,
                next_state,
                prior_audit_error=committed_audit_error,
            )
            self._broadcast_locked(room)
            self._ensure_bot_task_locked(room)
            await self._warm_all_private_layouts_locked(room)
            if not room.closed:
                self._broadcast_locked(room)
            return self._project_locked(room, viewer)

    async def subscribe(
        self,
        table_no: str,
        token: str,
    ) -> asyncio.Queue[dict[str, Any]]:
        room = self._room(table_no)
        async with room.lock:
            viewer = self._authenticate_locked(room, token)
            await self._warm_private_layout_locked(room, viewer)
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
                maxsize=self.subscriber_queue_size
            )
            room.subscribers[queue] = viewer.participant_id
            queue.put_nowait(self._project_locked(room, viewer))
            return queue

    async def unsubscribe(
        self,
        table_no: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        room = self._room(table_no)
        async with room.lock:
            room.subscribers.pop(queue, None)

    @contextlib.asynccontextmanager
    async def subscription_guard(
        self, table_no: str, token: str, queue: asyncio.Queue[dict[str, Any]],
    ) -> AsyncIterator[bool]:
        """Serialize private socket sends with token rotation, including dequeued states."""

        room = self._room(table_no)
        async with room.lock:
            try:
                viewer = self._authenticate_locked(room, token)
            except InvalidTokenError:
                yield False
                return
            yield room.subscribers.get(queue) == viewer.participant_id

    @staticmethod
    def _revoke_viewer_subscriptions_locked(room: _Room, viewer: _Viewer) -> None:
        for queue, participant_id in tuple(room.subscribers.items()):
            if participant_id != viewer.participant_id:
                continue
            room.subscribers.pop(queue, None)
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait({"type": "revoked"})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._manager_lock:
            rooms = list(self._rooms.values())
            self._rooms.clear()
        tasks: list[asyncio.Task[None]] = []
        engines: list[EngineClient] = []
        seen_engines: set[int] = set()
        for room in rooms:
            room.closed = True
            self._audit_locked(room, "table_service_stopped", {})
            for engine in (room.engine, room.pending_engine):
                if engine is not None and id(engine) not in seen_engines:
                    seen_engines.add(id(engine))
                    engines.append(engine)
            room.pending_engine = None
            task = room.bot_task
            room.bot_task = None
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        close_tasks = [self._schedule_engine_close(engine) for engine in engines]
        close_tasks.extend(
            task
            for task in tuple(self._engine_close_tasks)
            if task not in close_tasks
        )
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

    stop = close

    async def _close_engine(self, engine: EngineClient) -> None:
        with contextlib.suppress(Exception):
            await engine.close()

    def _schedule_engine_close(self, engine: EngineClient) -> asyncio.Task[None]:
        task = asyncio.create_task(self._close_engine(engine))
        self._engine_close_tasks.add(task)
        task.add_done_callback(self._engine_close_tasks.discard)
        return task

    async def _retire_pending_engine(
        self,
        room: _Room,
        engine: EngineClient,
    ) -> None:
        # Manager.close() atomically clears pending_engine when it assumes
        # ownership of the close.  Avoid a second concurrent close in that case.
        if room.pending_engine is not engine:
            return
        room.pending_engine = None
        await asyncio.shield(self._schedule_engine_close(engine))

    def _room(self, table_no: str) -> _Room:
        room = self._rooms.get(str(table_no))
        if room is None or room.closed:
            raise TableNotFoundError(str(table_no))
        return room

    @staticmethod
    def _audit_seat(seat: _Seat) -> dict[str, Any]:
        return {
            "position": seat.position,
            "kind": seat.kind.value,
            "name": seat.name,
            "participant_id": seat.viewer_id,
            "ready": seat.ready,
        }

    def _audit_locked(
        self,
        room: _Room,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Append and fsync one replay-grade human-table audit event."""

        path = room.audit_path
        if path is None:
            return
        room.audit_sequence += 1
        event = {
            "schema": AUDIT_SCHEMA,
            "sequence": room.audit_sequence,
            "timestamp_ms": int(time.time() * 1000),
            "event_type": event_type,
            "table_no": room.table_no,
            "game": room.game.value,
            "round_no": room.round_no,
            "room_version": room.version,
            "payload": copy.deepcopy(dict(payload)),
        }
        encoded = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fill_open_seat(seat: _Seat, viewer: _Viewer) -> None:
        if seat.kind is not SeatKind.OPEN:
            raise RuntimeError("cannot fill a non-open seat")
        seat.kind = SeatKind.HUMAN
        seat.name = viewer.nickname
        seat.viewer_id = viewer.participant_id
        seat.ready = False

    @staticmethod
    def _swap_seat_occupants(source: _Seat, target: _Seat) -> None:
        source_values = (source.kind, source.name, source.viewer_id, source.ready)
        target_values = (target.kind, target.name, target.viewer_id, target.ready)
        source.kind, source.name, source.viewer_id, source.ready = target_values
        target.kind, target.name, target.viewer_id, target.ready = source_values

    @staticmethod
    def _seat_for_viewer(room: _Room, viewer: _Viewer) -> _Seat | None:
        if viewer.role is not ViewerRole.PLAYER:
            return None
        return next(
            (seat for seat in room.seats if seat.viewer_id == viewer.participant_id),
            None,
        )

    def _authenticate_locked(self, room: _Room, token: str) -> _Viewer:
        if room.closed:
            raise TableNotFoundError(room.table_no)
        if not isinstance(token, str) or not token:
            raise InvalidTokenError()
        digest = self._hash_token(token)
        participant_id = room.token_index.get(digest)
        if participant_id is None:
            # Keep the comparison workload independent of which real token was
            # supplied.  Only SHA-256 digests are retained in memory.
            for stored in room.token_index:
                hmac.compare_digest(stored, digest)
            raise InvalidTokenError()
        viewer = room.viewers.get(participant_id)
        if viewer is None or not hmac.compare_digest(viewer.token_hash, digest):
            raise InvalidTokenError()
        return viewer

    @staticmethod
    def _expect_version(room: _Room, expected: int) -> None:
        if expected != room.version:
            raise TableConflictError(
                "stale_version",
                f"expected version {expected}, current version is {room.version}",
            )

    def _session(
        self,
        room: _Room,
        viewer: _Viewer,
        token: str,
    ) -> HumanTableSession:
        seat = self._seat_for_viewer(room, viewer)
        return HumanTableSession(
            table_no=room.table_no,
            token=token,
            participant_id=viewer.participant_id,
            role=viewer.role,
            seat=seat.position if seat is not None else None,
            view=self._project_locked(room, viewer),
        )

    @staticmethod
    def _start_payload(
        room: _Room,
        *,
        seed: int | None = None,
        new_series: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "table_no": room.table_no,
            "game": room.game.value,
            "human_slots": room.human_slots,
            "seed": room.seed if seed is None else seed,
            "seats": [
                {
                    "position": seat.position,
                    "name": seat.name,
                    "kind": seat.kind.value,
                }
                for seat in room.seats
            ],
        }
        if (
            room.game is Game.GUANDAN
            and room.next_first_player is not None
            and not new_series
        ):
            payload["first_player"] = room.next_first_player
        if room.game is Game.GUANDAN:
            payload["level"] = "2" if new_series else room.guandan_current_level
            payload["team_levels"] = (
                {"even": "2", "odd": "2"}
                if new_series
                else copy.deepcopy(room.guandan_team_levels)
            )
        return payload

    @staticmethod
    def _reset_guandan_series_locked(room: _Room) -> None:
        """Reset series-only state after every human confirms ``再来一场``."""

        if room.game is not Game.GUANDAN:
            room.round_no += 1
            return
        room.series_no += 1
        room.round_no = 1
        room.guandan_team_levels = {"even": "2", "odd": "2"}
        room.guandan_current_level = "2"
        room.guandan_active_team = None
        room.guandan_a_challenger = None
        room.guandan_a_attempt_count = {"even": 0, "odd": 0}
        room.guandan_head_finish_count_by_seat = [0, 0, 0, 0]
        room.guandan_team_victory_count = {"even": 0, "odd": 0}
        room.guandan_rounds_played = 0
        room.guandan_match_winner_team = None
        room.guandan_match_finished = False
        room.guandan_final_result = None
        room.guandan_last_settlement = None
        room.settled_round_no = 0

    @staticmethod
    def _guandan_finish_order_from_state(state: EngineFullState) -> list[int]:
        """Return the valid finish positions explicitly supplied by the referee."""

        if state.game is not Game.GUANDAN:
            return []
        terminal = state.result.get("order")
        if _is_complete_guandan_order(terminal):
            return list(terminal)
        raw = state.metrics.get("finish_order")
        if not isinstance(raw, list):
            return []
        order: list[int] = []
        for position in raw:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position not in range(4)
                or position in order
            ):
                return []
            order.append(position)
        return order

    def _update_guandan_finish_order_locked(
        self,
        room: _Room,
        previous_state: EngineFullState | None,
        state: EngineFullState,
    ) -> None:
        """Track live finish order from authoritative engine transitions.

        The GuanDan engine exposes the full order only at episodeOver.  Before that
        point, a referee hand transition from non-empty to empty is the
        authoritative finish event.  The terminal order replaces this partial
        history so settlement remains exactly aligned with GuanDan.
        """

        if room.game is not Game.GUANDAN:
            return
        authoritative = self._guandan_finish_order_from_state(state)
        if _is_complete_guandan_order(authoritative):
            room.guandan_finish_order = authoritative
            return
        for position in authoritative:
            if position not in room.guandan_finish_order:
                room.guandan_finish_order.append(position)
        if previous_state is None:
            return
        before = {player.position: player.hand_count for player in previous_state.players}
        for player in state.players:
            if (
                before.get(player.position, 0) > 0
                and player.hand_count == 0
                and player.position not in room.guandan_finish_order
            ):
                room.guandan_finish_order.append(player.position)

    @staticmethod
    def _guandan_team_for_seat(position: int) -> str:
        return "even" if position % 2 == 0 else "odd"

    @staticmethod
    def _guandan_team_positions(team: str) -> list[int]:
        return [0, 2] if team == "even" else [1, 3]

    @staticmethod
    def _current_series_replay_ids(room: _Room) -> list[str]:
        replay_ids = [
            str(item["match_id"])
            for item in room.replay_history
            if item.get("series_no", 1) == room.series_no and item.get("match_id")
        ]
        if room.replay_match_id and room.replay_match_id not in replay_ids:
            replay_ids.append(room.replay_match_id)
        return replay_ids

    def _settle_guandan_round_locked(
        self,
        room: _Room,
        order: list[int],
    ) -> None:
        """Apply the Pailemen independent-level and explicit pass-A rules."""

        if room.settled_round_no == room.round_no:
            return
        if not _is_complete_guandan_order(order):
            raise EngineProtocolError("finished GuanDan state requires a complete order")

        first_position = order[0]
        winning_team = self._guandan_team_for_seat(first_position)
        losing_team = "odd" if winning_team == "even" else "even"
        partner_position = (first_position + 2) % 4
        partner_finish = order.index(partner_position) + 1
        delta = {2: 3, 3: 2, 4: 1}[partner_finish]
        levels_before = copy.deepcopy(room.guandan_team_levels)
        previous_level = levels_before[winning_team]

        challenge_team = (
            room.guandan_a_challenger
            if room.guandan_current_level == "A"
            and room.guandan_a_challenger in {"even", "odd"}
            else None
        )
        if challenge_team is not None:
            room.guandan_a_attempt_count[challenge_team] += 1

        next_level = _advance_guandan_level(previous_level, delta)
        room.guandan_team_levels[winning_team] = next_level
        room.guandan_current_level = next_level
        room.guandan_active_team = winning_team
        room.guandan_rounds_played += 1
        room.guandan_head_finish_count_by_seat[first_position] += 1
        room.guandan_team_victory_count[winning_team] += 1

        passed_a = (
            challenge_team == winning_team
            and previous_level == "A"
            and partner_finish in {2, 3}
        )
        if passed_a:
            room.guandan_match_finished = True
            room.guandan_match_winner_team = winning_team
            room.guandan_a_challenger = winning_team
        else:
            # Reaching A only registers the challenger for the *next* hand.
            # The same rule also rotates the challenger when both teams are A.
            room.guandan_a_challenger = winning_team if next_level == "A" else None

        outcome = (
            "double_up"
            if partner_finish == 2
            else "first_third"
            if partner_finish == 3
            else "first_last"
        )
        label = {2: "双上", 3: "一三游", 4: "一四游"}[partner_finish]
        room.guandan_last_settlement = {
            "schema_version": "guandan_round_settlement_v1",
            "series_no": room.series_no,
            "round_no": room.round_no,
            "outcome": outcome,
            "summary": f"{label}：头游方升 {delta} 级",
            "winning_team": winning_team,
            "losing_team": losing_team,
            "winning_positions": self._guandan_team_positions(winning_team),
            "finish_order": list(order),
            "partner_position": partner_position,
            "partner_finish": partner_finish,
            "upgrade_levels": delta,
            "previous_level": previous_level,
            "next_level": next_level,
            "team_levels_before": levels_before,
            "team_levels": copy.deepcopy(room.guandan_team_levels),
            "next_current_level": room.guandan_current_level,
            "a_challenge_team": challenge_team,
            "a_challenge_attempted": challenge_team is not None,
            "a_challenge_succeeded": passed_a,
            "passed_a": passed_a,
        }
        if passed_a:
            room.guandan_final_result = {
                "schema_version": "guandan_series_result_v1",
                "series_no": room.series_no,
                "winner_team": winning_team,
                "loser_team": losing_team,
                "team_levels": copy.deepcopy(room.guandan_team_levels),
                "winning_team_level": "A",
                "losing_team_level": room.guandan_team_levels[losing_team],
                "head_finish_count_by_seat": list(
                    room.guandan_head_finish_count_by_seat
                ),
                "team_victory_count": copy.deepcopy(
                    room.guandan_team_victory_count
                ),
                "a_attempt_count": copy.deepcopy(room.guandan_a_attempt_count),
                "winning_a_attempt_no": room.guandan_a_attempt_count[winning_team],
                "last_finish_order": list(order),
                "rounds_played": room.guandan_rounds_played,
                "replay_ids": self._current_series_replay_ids(room),
            }
        room.settled_round_no = room.round_no

    @staticmethod
    def _replay_legal_actions_hash(state: EngineFullState) -> str:
        actions = (
            []
            if state.decision is None
            else [action.model_dump(mode="json") for action in state.decision.actions]
        )
        encoded = json.dumps(
            actions,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _infer_replay_action(
        before: EngineFullState,
        after: EngineFullState,
        actor: int,
    ) -> Any:
        """Recover the committed bot action from authoritative state deltas."""

        decision = before.decision
        before_player = next(item for item in before.players if item.position == actor)
        after_player = next(item for item in after.players if item.position == actor)
        removed = Counter(before_player.hand) - Counter(after_player.hand)
        if decision is not None:
            matches = []
            if removed:
                matches = [
                    action
                    for action in decision.actions
                    if Counter(action.cards) == removed
                ]
            if len(matches) == 1:
                return matches[0].model_dump(mode="json")
            phase = decision.phase.casefold()
            missing = object()
            expected: Any = missing
            expected_type = phase
            if phase == "bid":
                bids = after.metrics.get("bids")
                if isinstance(bids, Mapping):
                    expected = bids.get(str(actor), bids.get(actor, missing))
                if expected is missing and after.metrics.get("highest_bidder") == actor:
                    expected = after.metrics.get("highest_bid", missing)
            elif phase == "double":
                doubles = after.metrics.get("doubles")
                if isinstance(doubles, Mapping):
                    expected = doubles.get(str(actor), doubles.get(actor, missing))
            elif phase == "redouble":
                expected = after.metrics.get("redouble", missing)
            if expected is not missing:
                setup_matches = [
                    action
                    for action in decision.actions
                    if action.type.casefold() == expected_type
                    and _replay_rank_matches(action.rank, expected)
                ]
                if len(setup_matches) == 1:
                    return setup_matches[0].model_dump(mode="json")
            passes = [
                action
                for action in decision.actions
                if action.type.casefold() == "pass"
            ]
            if len(passes) == 1:
                return passes[0].model_dump(mode="json")
            if len(decision.actions) == 1:
                return decision.actions[0].model_dump(mode="json")
        if after_player.play_area is not None:
            return copy.deepcopy(after_player.play_area)
        return {"type": "PASS", "rank": "", "cards": [], "label": "不出"}

    @staticmethod
    def _replay_player_identity(
        room: _Room,
        state: EngineFullState,
        position: int,
    ) -> PlayerIdentity:
        player = next(item for item in state.players if item.position == position)
        policy = state.metrics.get("bot_policy")
        policy_id = "human"
        if room.seats[position].kind is SeatKind.BOT:
            policy_id = (
                str(policy.get("id"))
                if isinstance(policy, Mapping) and policy.get("id")
                else "bot"
            )
        return PlayerIdentity(
            position=position,
            name=player.name,
            policy_id=policy_id,
            team=player.team,
            role=player.role,
        )

    def _disable_replay_locked(
        self,
        room: _Room,
        operation: str,
        exc: BaseException,
    ) -> None:
        room.replay_failed = True
        try:
            self._audit_locked(
                room,
                "replay_write_failed",
                {"operation": operation, "error": repr(exc)},
            )
        except OSError:
            # Replay and audit persistence are observability layers.  A shared
            # disk failure must never stop installation of an engine state
            # after that engine has already consumed the committed action.
            pass

    def _start_replay_locked(self, room: _Room, state: EngineFullState) -> None:
        store = self.replay_store
        if store is None or room.replay_failed:
            return
        match_id = (
            f"human-{room.game.value}-{room.table_no}-s{room.series_no}-r{room.round_no}-"
            f"{uuid.uuid4().hex[:12]}"
        )
        now = utc_now()
        players = [
            self._replay_player_identity(room, state, position)
            for position in range(len(room.seats))
        ]
        bot_policy = state.metrics.get("bot_policy")
        models: list[ModelIdentity] = []
        if isinstance(bot_policy, dict):
            policy_id = bot_policy.get("id")
            if isinstance(policy_id, str) and policy_id and policy_id != "rule_bot":
                version = bot_policy.get("release_version")
                checkpoint_sha = bot_policy.get("sha256")
                models.append(
                    ModelIdentity(
                        policy_id=policy_id,
                        display_name="DanKS",
                        version=(
                            version
                            if isinstance(version, str) and version
                            else policy_id
                        ),
                        kind="model",
                        expected_sha256=(
                            checkpoint_sha
                            if isinstance(checkpoint_sha, str)
                            else None
                        ),
                        actual_sha256=(
                            checkpoint_sha
                            if isinstance(checkpoint_sha, str)
                            else None
                        ),
                        verified=isinstance(checkpoint_sha, str),
                        source_identity={
                            "retrieval_profile": bot_policy.get("retrieval_profile"),
                            "onnx_sha256": bot_policy.get("onnx_sha256"),
                        },
                    )
                )
        manifest = MatchManifest(
            match_id=match_id,
            game=room.game,
            run_mode=RunMode.SINGLE,
            pairing=Pairing.SINGLE,
            seed=room.seed,
            status=MatchStatus.RUNNING,
            created_at=now,
            started_at=now,
            players=players,
            models=models,
            source_identity={
                "replay_protocol": "1.0",
                "backend": "human-table",
                "table_no": room.table_no,
                "series_no": room.series_no,
                "round_no": room.round_no,
                "audit_schema": AUDIT_SCHEMA,
            },
            presentation_identity=(
                copy.deepcopy(getattr(self.arranger, "identity", {}))
                if self.arranger is not None
                else {}
            ),
            rules={
                "profile": state.metrics.get("rule_profile", ""),
                "strict_action_list_index": True,
                "fallback_allowed": False,
            },
            request={
                "source": "human_table",
                "table_no": room.table_no,
                "series_no": room.series_no,
                "round_no": room.round_no,
                "human_slots": room.human_slots,
            },
            rounds=[
                RoundSummary(
                    round_index=0,
                    status=RoundStatus.RUNNING,
                    started_at=now,
                )
            ],
        )
        try:
            store.create(manifest)
            store.append_event(
                ReplayEvent(
                    event_id=1,
                    match_id=match_id,
                    event_type=ReplayEventType.MATCH_STARTED,
                    phase="setup",
                    metadata={
                        "table_no": room.table_no,
                        "human_table_series_no": room.series_no,
                        "human_table_round_no": room.round_no,
                    },
                )
            )
            store.append_event(
                ReplayEvent(
                    event_id=1,
                    match_id=match_id,
                    event_type=ReplayEventType.ROUND_STARTED,
                    phase=state.phase,
                    before_state=state.model_dump(mode="json"),
                    metadata={
                        "human_table_series_no": room.series_no,
                        "human_table_round_no": room.round_no,
                    },
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._disable_replay_locked(room, "start", exc)
            return
        room.replay_match_id = match_id
        room.replay_turn_index = 0

    def _record_replay_action_locked(
        self,
        room: _Room,
        before: EngineFullState,
        after: EngineFullState,
        chosen_action: Any,
        *,
        inference_ms: float,
    ) -> None:
        store = self.replay_store
        match_id = room.replay_match_id
        decision = before.decision
        if (
            store is None
            or match_id is None
            or room.replay_failed
            or decision is None
        ):
            return
        identity = self._replay_player_identity(room, before, decision.actor)
        metadata: dict[str, Any] = {
            "decision_id": decision.id,
            "source": "human_table",
        }
        bot_policy = after.metrics.get("bot_policy")
        if (
            room.seats[decision.actor].kind is SeatKind.BOT
            and isinstance(bot_policy, dict)
        ):
            for key in ("trace_ref", "trace_hash"):
                value = bot_policy.get(key)
                if isinstance(value, str) and value:
                    metadata[key] = value
        try:
            store.append_event(
                ReplayEvent(
                    event_id=1,
                    match_id=match_id,
                    round_index=0,
                    turn_index=room.replay_turn_index,
                    event_type=ReplayEventType.DECISION_STARTED,
                    phase=decision.phase,
                    actor_pos=decision.actor,
                    actor=ActorIdentity(**identity.model_dump()),
                    before_state=before.model_dump(mode="json"),
                    legal_action_count=decision.action_count,
                    legal_actions_hash=self._replay_legal_actions_hash(before),
                    metadata={
                        "decision_id": decision.id,
                        "source": "human_table",
                    },
                )
            )
            store.append_event(
                ReplayEvent(
                    event_id=1,
                    match_id=match_id,
                    round_index=0,
                    turn_index=room.replay_turn_index,
                    event_type=ReplayEventType.ACTION_COMMITTED,
                    phase=decision.phase,
                    actor_pos=decision.actor,
                    actor=ActorIdentity(**identity.model_dump()),
                    before_state=before.model_dump(mode="json"),
                    chosen_action=copy.deepcopy(chosen_action),
                    after_state=after.model_dump(mode="json"),
                    inference_ms=max(0.0, inference_ms),
                    legal_action_count=decision.action_count,
                    legal_actions_hash=self._replay_legal_actions_hash(before),
                    metadata=metadata,
                )
            )
            manifest = store.get(match_id)
            manifest.rounds[0].turn_count = room.replay_turn_index + 1
            store.save(manifest)
            room.replay_turn_index += 1
        except (OSError, ValueError, RuntimeError) as exc:
            self._disable_replay_locked(room, "action", exc)

    def _finalize_replay_locked(self, room: _Room, state: EngineFullState) -> None:
        store = self.replay_store
        match_id = room.replay_match_id
        if (
            store is None
            or match_id is None
            or room.replay_failed
            or any(item.get("match_id") == match_id for item in room.replay_history)
        ):
            return
        result = self._public_result(room, state)
        if room.game is Game.GUANDAN and room.guandan_last_settlement is not None:
            result.update(copy.deepcopy(room.guandan_last_settlement))
            result["team_levels"] = copy.deepcopy(room.guandan_team_levels)
            result["series_state"] = {
                "series_no": room.series_no,
                "rounds_played": room.guandan_rounds_played,
                "head_finish_count_by_seat": list(
                    room.guandan_head_finish_count_by_seat
                ),
                "team_victory_count": copy.deepcopy(
                    room.guandan_team_victory_count
                ),
                "a_attempt_count": copy.deepcopy(room.guandan_a_attempt_count),
                "match_finished": room.guandan_match_finished,
                "match_winner_team": room.guandan_match_winner_team,
            }
            if room.guandan_final_result is not None:
                result["final_result"] = copy.deepcopy(room.guandan_final_result)
        now = utc_now()
        terminal_state = state.model_dump(mode="json")
        terminal_state["result"] = copy.deepcopy(result)
        terminal_state.setdefault("metrics", {})["platform_series"] = copy.deepcopy(
            result.get("series_state", {})
        )
        try:
            for event_type, phase in (
                (ReplayEventType.ROUND_FINISHED, "settle"),
                (ReplayEventType.MATCH_FINISHED, "finished"),
            ):
                store.append_event(
                    ReplayEvent(
                        event_id=1,
                        match_id=match_id,
                        round_index=0,
                        turn_index=room.replay_turn_index,
                        event_type=event_type,
                        phase=phase,
                        after_state=terminal_state,
                        metadata={"result": copy.deepcopy(result)},
                    )
                )
            manifest = store.get(match_id)
            manifest.status = MatchStatus.COMPLETED
            manifest.ended_at = now
            manifest.result = copy.deepcopy(result)
            manifest.rounds[0].status = RoundStatus.COMPLETED
            manifest.rounds[0].ended_at = now
            manifest.rounds[0].turn_count = room.replay_turn_index
            manifest.rounds[0].result = copy.deepcopy(result)
            store.save(manifest)
        except (OSError, ValueError, RuntimeError) as exc:
            self._disable_replay_locked(room, "finalize", exc)
            return
        room.replay_history.append(
            {
                "series_no": room.series_no,
                "round_no": room.round_no,
                "match_id": match_id,
            }
        )

    def _validate_engine_state(
        self,
        room: _Room,
        raw_state: Any,
        *,
        initial: bool,
    ) -> EngineFullState:
        try:
            state = EngineFullState.model_validate(raw_state)
        except ValidationError as exc:
            raise EngineProtocolError(f"invalid full engine state: {exc}") from exc
        if state.game is not room.game:
            raise EngineProtocolError("rules engine returned the wrong game")
        if not initial and room.engine_state is not None:
            if state.state_version <= room.engine_state.state_version:
                raise EngineProtocolError("engine state_version did not advance")
        for player in state.players:
            seat = room.seats[player.position]
            if player.kind is not seat.kind or player.name != seat.name:
                raise EngineProtocolError(
                    f"engine identity mismatch at position {player.position}"
                )
        finished = state.status == "finished" or state.phase == "finished"
        if (
            finished
            and state.game is Game.GUANDAN
            and not _is_complete_guandan_order(state.result.get("order"))
        ):
            raise EngineProtocolError(
                "finished GuanDan state requires order to be a permutation of 0..3"
            )
        if finished and state.game is Game.DOUDIZHU:
            _validate_finished_doudizhu_result(state)
        return state

    @staticmethod
    def _update_guandan_response_target_locked(
        room: _Room, state: EngineFullState
    ) -> None:
        """Remember only the worker's committed public action, never legal choices.

        Worker ``table`` describes the latest action, including PASS. The actual
        response target therefore needs to survive PASS, but a free-lead decision
        clears it even if the worker retains an old last-action table.
        """
        decision = state.decision
        if (
            room.game is not Game.GUANDAN
            or state.status != "active"
            or state.phase != "play"
            or decision is None
            or decision.phase != "play"
            or not decision.can_pass
        ):
            room.guandan_response_target = None
            return
        table = state.table
        seat = table.get("owner_position")
        kind = table.get("action_type")
        cards = table.get("cards")
        if (
            _is_protocol_int(seat)
            and seat in range(4)
            and isinstance(kind, str)
            and kind.casefold() == "pass"
            and cards == []
        ):
            return
        # Fail closed for missing/legacy metadata instead of treating an old
        # visible play, a removed-card delta or an ambiguous legal action as fact.
        room.guandan_response_target = None
        if (
            not _is_protocol_int(seat)
            or seat not in range(4)
            or not isinstance(kind, str)
            or kind not in _GUANDAN_PLAY_TYPES
            or not isinstance(cards, list)
            or not 1 <= len(cards) <= 27
            or any(
                not isinstance(card, str) or card not in _GUANDAN_PLAY_CARD_CODES
                for card in cards
            )
        ):
            return
        rank = table.get("action_rank")
        if rank is None:
            player = next(
                (item for item in state.players if item.position == seat), None
            )
            play = None if player is None else player.play_area
            if (
                isinstance(play, dict)
                and play.get("type") == kind
                and isinstance(play.get("cards"), list)
                and all(isinstance(card, str) for card in play["cards"])
                and Counter(play["cards"]) == Counter(cards)
            ):
                rank = play.get("rank")
        # Rank belongs to the committed interpretation (especially wild hearts).
        # Never recover it from the physical cards or private action list.
        if not isinstance(rank, str) or rank not in _GUANDAN_PLAY_RANKS:
            return
        room.guandan_response_target = {
            "seat": seat,
            "type": kind,
            "rank": rank,
            "label": kind,
            "cards": list(cards),
        }

    @staticmethod
    def _public_current_trick(
        room: _Room, state: EngineFullState | None
    ) -> dict[str, Any]:
        if (
            room.game is not Game.GUANDAN
            or room.phase != "play"
            or state is None
            or state.status != "active"
            or state.decision is None
            or state.decision.phase != "play"
        ):
            return {"mode": "unknown", "target": None}
        if not state.decision.can_pass:
            return {"mode": "lead", "target": None}
        target = room.guandan_response_target
        return {
            "mode": "follow" if target is not None else "unknown",
            "target": copy.deepcopy(target),
        }

    def _install_engine_state_locked(
        self,
        room: _Room,
        state: EngineFullState,
        *,
        prior_audit_error: OSError | None = None,
    ) -> None:
        self._update_guandan_response_target_locked(room, state)
        previous_state = room.engine_state
        history_round = (room.series_no, room.round_no)
        if room.public_trick_history_round != history_round:
            room.public_trick_history = PublicTrickHistory()
            room.public_trick_history_round = history_round
        room.public_trick_history.observe(state)
        previous_decision_id = (
            previous_state.decision.id
            if previous_state is not None and previous_state.decision is not None
            else None
        )
        next_decision_id = state.decision.id if state.decision is not None else None
        previous_actor = (
            previous_state.decision.actor
            if previous_state is not None and previous_state.decision is not None
            else None
        )
        if len(room.visible_play_areas) != len(room.seats):
            room.visible_play_areas = [None] * len(room.seats)
        if previous_actor is not None:
            next_player = next(
                (player for player in state.players if player.position == previous_actor),
                None,
            )
            if next_player is not None:
                room.visible_play_areas[previous_actor] = copy.deepcopy(
                    next_player.play_area
                )
        # Keep every committed play/PASS visible for the rest of the table
        # cycle.  Only clear a seat's previous decision when a new decision
        # reaches that same seat again.  This mirrors a physical table: the
        # other three seats retain their latest decisions until their own next
        # turns instead of the whole table being erased at a trick boundary.
        if state.decision is not None and next_decision_id != previous_decision_id:
            room.visible_play_areas[state.decision.actor] = None
        self._update_guandan_finish_order_locked(room, previous_state, state)
        room.engine_state = state
        finished = state.status == "finished" or state.phase == "finished"
        room.phase = "finished" if finished else state.phase
        room.failure = None
        room.version += 1
        if finished:
            for seat in room.seats:
                seat.ready = seat.kind is SeatKind.BOT
            room.turn_started_at_ms = None
            if room.game is Game.GUANDAN:
                order = state.result.get("order")
                first = order[0] if isinstance(order, list) and order else None
                room.next_first_player = (
                    first
                    if isinstance(first, int)
                    and not isinstance(first, bool)
                    and first in range(4)
                    else None
                )
                if (
                    room.settled_round_no != room.round_no
                    and _is_complete_guandan_order(order)
                ):
                    assert isinstance(order, list)
                    self._settle_guandan_round_locked(room, order)
                    if room.guandan_match_finished:
                        room.phase = "game_over"
            self._finalize_replay_locked(room, state)
        elif next_decision_id is None:
            room.turn_started_at_ms = None
        elif next_decision_id != previous_decision_id:
            room.turn_started_at_ms = int(time.time() * 1000)
        audit_error = prior_audit_error
        try:
            self._audit_locked(
                room,
                "engine_state",
                {
                    "state": state.model_dump(mode="json"),
                    "guandan_team_levels": copy.deepcopy(room.guandan_team_levels),
                    "guandan_current_level": room.guandan_current_level,
                    "guandan_active_team": room.guandan_active_team,
                    "guandan_a_challenger": room.guandan_a_challenger,
                    "guandan_a_attempt_count": copy.deepcopy(
                        room.guandan_a_attempt_count
                    ),
                    "guandan_head_finish_count_by_seat": list(
                        room.guandan_head_finish_count_by_seat
                    ),
                    "guandan_team_victory_count": copy.deepcopy(
                        room.guandan_team_victory_count
                    ),
                    "guandan_rounds_played": room.guandan_rounds_played,
                    "guandan_match_finished": room.guandan_match_finished,
                    "guandan_final_result": copy.deepcopy(room.guandan_final_result),
                    "guandan_last_settlement": copy.deepcopy(
                        room.guandan_last_settlement
                    ),
                },
            )
        except OSError as exc:
            if audit_error is None:
                audit_error = exc
        if audit_error is not None and not finished:
            # The authoritative engine state is installed before stopping the
            # room, so a durable-log failure can never leave controller and
            # engine on opposite sides of the committed action.
            room.phase = "error"
            room.failure = {"code": "engine_failure", "retryable": False}
            room.turn_started_at_ms = None
            room.version += 1

    def _ensure_bot_task_locked(self, room: _Room) -> None:
        state = room.engine_state
        if (
            state is None
            or state.decision is None
            or room.closed
            or room.phase in {"error", "paused"}
        ):
            return
        actor = state.decision.actor
        if room.seats[actor].kind is not SeatKind.BOT:
            return
        if room.bot_task is not None and not room.bot_task.done():
            return
        room.bot_task = asyncio.create_task(
            self._run_bot_turns(room),
            name=f"human-table-bots-{room.table_no}",
        )

    async def _run_bot_turns(self, room: _Room) -> None:
        current_task = asyncio.current_task()
        try:
            while not room.closed:
                async with room.lock:
                    state = room.engine_state
                    if (
                        state is None
                        or state.decision is None
                        or room.phase in {"error", "paused"}
                    ):
                        return
                    decision = state.decision
                    if room.seats[decision.actor].kind is not SeatKind.BOT:
                        return
                    decision_id = decision.id
                    engine_version = state.state_version
                if self.bot_delay_ms:
                    await asyncio.sleep(self.bot_delay_ms / 1000)
                else:
                    await asyncio.sleep(0)
                async with room.lock:
                    state = room.engine_state
                    if (
                        room.closed
                        or state is None
                        or state.decision is None
                        or room.phase in {"error", "paused"}
                        or state.decision.id != decision_id
                        or state.state_version != engine_version
                        or room.seats[state.decision.actor].kind is not SeatKind.BOT
                    ):
                        continue
                    payload = {
                        "table_no": room.table_no,
                        "game": room.game.value,
                        "seat": state.decision.actor,
                        "decision_id": decision_id,
                        "expected_state_version": engine_version,
                    }
                    inference_started = time.perf_counter()
                    try:
                        raw_state = await room.engine.bot_act(payload)
                        next_state = self._validate_engine_state(
                            room, raw_state, initial=False
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        retryable = getattr(exc, "code", None) == "ai_unavailable"
                        room.phase = "paused" if retryable else "error"
                        room.failure = {
                            "code": "ai_unavailable" if retryable else "engine_failure",
                            "retryable": retryable,
                        }
                        room.turn_started_at_ms = None
                        room.version += 1
                        self._audit_locked(
                            room,
                            "bot_action_failed",
                            {
                                "seat": state.decision.actor,
                                "decision_id": decision_id,
                                "error": repr(exc),
                            },
                        )
                        self._broadcast_locked(room)
                        return
                    committed_audit_error: OSError | None = None
                    try:
                        self._audit_locked(
                            room,
                            "bot_action_committed",
                            {
                                "seat": state.decision.actor,
                                "decision_id": decision_id,
                                "bot_policy": copy.deepcopy(
                                    state.metrics.get("bot_policy")
                                ),
                            },
                        )
                    except OSError as exc:
                        committed_audit_error = exc
                    self._record_replay_action_locked(
                        room,
                        state,
                        next_state,
                        self._infer_replay_action(
                            state,
                            next_state,
                            state.decision.actor,
                        ),
                        inference_ms=(time.perf_counter() - inference_started) * 1000,
                    )
                    self._install_engine_state_locked(
                        room,
                        next_state,
                        prior_audit_error=committed_audit_error,
                    )
                    self._broadcast_locked(room)
                    await self._warm_all_private_layouts_locked(room)
                    if not room.closed:
                        self._broadcast_locked(room)
        finally:
            if room.bot_task is current_task:
                room.bot_task = None

    def _broadcast_locked(self, room: _Room) -> None:
        for queue, participant_id in tuple(room.subscribers.items()):
            viewer = room.viewers.get(participant_id)
            if viewer is None:
                room.subscribers.pop(queue, None)
                continue
            view = self._project_locked(room, viewer)
            self._enqueue_view(queue, view)

    def _broadcast_viewer_locked(
        self,
        room: _Room,
        viewer: _Viewer,
        *,
        view: dict[str, Any] | None = None,
    ) -> None:
        projected = view if view is not None else self._project_locked(room, viewer)
        for queue, participant_id in tuple(room.subscribers.items()):
            if participant_id == viewer.participant_id:
                self._enqueue_view(queue, projected)

    @staticmethod
    def _enqueue_view(
        queue: asyncio.Queue[dict[str, Any]],
        view: dict[str, Any],
    ) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(view)

    @staticmethod
    def _public_level(
        room: _Room,
        state: EngineFullState | None,
    ) -> str | None:
        if room.game is not Game.GUANDAN or state is None:
            return None
        return room.guandan_current_level

    def _public_guandan_levels(
        self,
        room: _Room,
        viewer: _Viewer,
    ) -> dict[str, Any] | None:
        if room.game is not Game.GUANDAN:
            return None
        viewer_seat = self._seat_for_viewer(room, viewer)
        our_team = (
            "odd"
            if viewer_seat is not None and viewer_seat.position % 2 == 1
            else "even"
        )
        opponent_team = "even" if our_team == "odd" else "odd"
        active_side = (
            "our"
            if room.guandan_active_team == our_team
            else "opponent"
            if room.guandan_active_team == opponent_team
            else ""
        )
        return {
            "our_level": room.guandan_team_levels[our_team],
            "opponent_level": room.guandan_team_levels[opponent_team],
            "current_level": room.guandan_current_level,
            "active_side": active_side,
            "a_challenger_side": (
                "our"
                if room.guandan_a_challenger == our_team
                else "opponent"
                if room.guandan_a_challenger == opponent_team
                else ""
            ),
        }

    def _public_guandan_series(
        self,
        room: _Room,
        viewer: _Viewer,
    ) -> dict[str, Any] | None:
        if room.game is not Game.GUANDAN:
            return None
        viewer_seat = self._seat_for_viewer(room, viewer)
        viewer_team = (
            self._guandan_team_for_seat(viewer_seat.position)
            if viewer_seat is not None
            else None
        )
        return {
            "schema_version": "guandan_series_state_v1",
            "series_no": room.series_no,
            "round_no": room.round_no,
            "team_levels": copy.deepcopy(room.guandan_team_levels),
            "current_level": room.guandan_current_level,
            "active_team": room.guandan_active_team,
            "a_challenger": room.guandan_a_challenger,
            "a_attempt_count": copy.deepcopy(room.guandan_a_attempt_count),
            "head_finish_count_by_seat": list(
                room.guandan_head_finish_count_by_seat
            ),
            "team_victory_count": copy.deepcopy(room.guandan_team_victory_count),
            "rounds_played": room.guandan_rounds_played,
            "match_winner_team": room.guandan_match_winner_team,
            "match_finished": room.guandan_match_finished,
            "viewer_team": viewer_team,
            "final_result": copy.deepcopy(room.guandan_final_result),
        }

    @staticmethod
    def _public_bot_policy(
        state: EngineFullState | None,
    ) -> dict[str, Any] | None:
        if state is None:
            return None
        raw = state.metrics.get("bot_policy")
        if not isinstance(raw, Mapping):
            return None
        return {
            key: copy.deepcopy(raw.get(key))
            for key in PUBLIC_BOT_POLICY_KEYS
        }

    @staticmethod
    def _public_rule_profile(
        state: EngineFullState | None,
    ) -> str | None:
        if state is None or state.game is not Game.GUANDAN:
            return None
        value = state.metrics.get("rule_profile")
        return value if value in PUBLIC_GUANDAN_RULE_PROFILES else None

    @staticmethod
    def _public_doudizhu_state(
        state: EngineFullState | None,
    ) -> dict[str, Any] | None:
        """Build a type-checked allow-list from otherwise private metrics."""

        if state is None or state.game is not Game.DOUDIZHU:
            return None
        metrics = state.metrics

        landlord = _doudizhu_position(metrics.get("landlord_position"))
        highest_bid_raw = metrics.get("highest_bid")
        highest_bid = (
            highest_bid_raw
            if _is_protocol_int(highest_bid_raw) and highest_bid_raw in range(4)
            else 0
        )
        highest_bidder = _doudizhu_position(metrics.get("highest_bidder"))

        bids: dict[str, int] = {}
        raw_bids = metrics.get("bids")
        if isinstance(raw_bids, Mapping):
            for position in _DOUDIZHU_POSITIONS:
                value = raw_bids.get(str(position), raw_bids.get(position))
                if _is_protocol_int(value) and value in range(4):
                    bids[str(position)] = value

        pending_double: list[int] = []
        raw_pending = metrics.get("pending_double")
        if isinstance(raw_pending, list):
            pending_double = sorted(
                {
                    position
                    for position in raw_pending
                    if _doudizhu_position(position) is not None
                }
            )

        doubles: dict[str, bool] = {}
        raw_doubles = metrics.get("doubles")
        if isinstance(raw_doubles, Mapping):
            for position in _DOUDIZHU_POSITIONS:
                value = raw_doubles.get(str(position), raw_doubles.get(position))
                if isinstance(value, bool):
                    doubles[str(position)] = value

        redouble_raw = metrics.get("redouble")
        redouble = redouble_raw if isinstance(redouble_raw, bool) else False
        base_score_raw = metrics.get("base_score")
        base_score = (
            base_score_raw
            if _is_protocol_int(base_score_raw) and base_score_raw in range(4)
            else 0
        )

        bottom_cards: list[str] = []
        raw_bottom = metrics.get("bottom_cards")
        bottom_visible = state.phase.casefold() in {"play", "finished", "gameover"}
        if (
            bottom_visible
            and isinstance(raw_bottom, list)
            and len(raw_bottom) <= 3
            and all(
                isinstance(card, str) and card in _DOUDIZHU_CARD_CODES
                for card in raw_bottom
            )
            and len(set(raw_bottom)) == len(raw_bottom)
        ):
            bottom_cards = list(raw_bottom)

        multiplier_raw = metrics.get("multiplier")
        multiplier = (
            multiplier_raw
            if _is_protocol_int(multiplier_raw)
            and 1 <= multiplier_raw <= 2**31 - 1
            else 1
        )
        defender_multipliers: dict[str, int] = {}
        raw_defender_multipliers = metrics.get("defender_multipliers")
        if landlord is not None and isinstance(raw_defender_multipliers, Mapping):
            for position in _DOUDIZHU_POSITIONS:
                if position == landlord:
                    continue
                value = raw_defender_multipliers.get(
                    str(position), raw_defender_multipliers.get(position)
                )
                if (
                    _is_protocol_int(value)
                    and 1 <= value <= 2**31 - 1
                ):
                    defender_multipliers[str(position)] = value
        bombs_raw = metrics.get("bombs")
        bombs = (
            bombs_raw
            if _is_protocol_int(bombs_raw) and bombs_raw in range(14)
            else 0
        )
        rockets_raw = metrics.get("rockets")
        rockets = (
            rockets_raw
            if _is_protocol_int(rockets_raw) and rockets_raw in range(2)
            else 0
        )
        return {
            "landlord_position": landlord,
            "highest_bid": highest_bid,
            "highest_bidder": highest_bidder,
            "bids": bids,
            "pending_double": pending_double,
            "doubles": doubles,
            "redouble": redouble,
            "base_score": base_score,
            "bottom_cards": bottom_cards,
            "multiplier": multiplier,
            "defender_multipliers": defender_multipliers,
            "bombs": bombs,
            "rockets": rockets,
        }

    @staticmethod
    def _public_doudizhu_rest_cards(value: Any) -> list[list[Any]] | None:
        if not isinstance(value, list) or len(value) > 3:
            return None
        rows: list[list[Any]] = []
        positions: set[int] = set()
        cards_seen: set[str] = set()
        for row in value:
            if not isinstance(row, list) or len(row) != 2:
                return None
            position = _doudizhu_position(row[0])
            cards = row[1]
            if (
                position is None
                or position in positions
                or not isinstance(cards, list)
                or not cards
                or len(cards) > 20
                or not all(
                    isinstance(card, str) and card in _DOUDIZHU_CARD_CODES
                    for card in cards
                )
                or len(set(cards)) != len(cards)
                or cards_seen.intersection(cards)
            ):
                return None
            positions.add(position)
            cards_seen.update(cards)
            rows.append([position, list(cards)])
        return sorted(rows, key=lambda row: row[0])

    @classmethod
    def _public_doudizhu_result(
        cls,
        state: EngineFullState,
    ) -> dict[str, Any]:
        source = state.result
        if not source:
            return {}
        result: dict[str, Any] = {}

        winner = _doudizhu_position(source.get("winner"))
        if winner is not None:
            result["winner"] = winner
        winning_side = source.get("winningSide")
        if winning_side in {"landlord", "defenders"}:
            result["winningSide"] = winning_side
        landlord = _doudizhu_position(source.get("landlord"))
        if landlord is not None:
            result["landlord"] = landlord

        base_score = source.get("baseScore")
        if _is_protocol_int(base_score) and base_score in range(1, 4):
            result["baseScore"] = base_score
        bombs = source.get("bombs")
        if _is_protocol_int(bombs) and bombs in range(14):
            result["bombs"] = bombs
        rockets = source.get("rockets")
        if _is_protocol_int(rockets) and rockets in range(2):
            result["rockets"] = rockets
        for key in ("spring", "reverseSpring", "redouble"):
            value = source.get(key)
            if isinstance(value, bool):
                result[key] = value

        raw_doubles = source.get("doubles")
        doubles: dict[str, bool] = {}
        if landlord is not None and isinstance(raw_doubles, Mapping):
            for position in _DOUDIZHU_POSITIONS:
                if position == landlord:
                    continue
                value = raw_doubles.get(str(position), raw_doubles.get(position))
                if isinstance(value, bool):
                    doubles[str(position)] = value
        if doubles:
            result["doubles"] = doubles

        hand_scores = source.get("handScores")
        if (
            isinstance(hand_scores, list)
            and len(hand_scores) == 3
            and all(_is_protocol_int(score) for score in hand_scores)
            and sum(hand_scores) == 0
        ):
            result["handScores"] = list(hand_scores)
        total_scores = source.get("totalScores")
        if (
            isinstance(total_scores, list)
            and len(total_scores) == 3
            and all(_is_protocol_int(score) for score in total_scores)
            and sum(total_scores) == 0
        ):
            result["totalScores"] = list(total_scores)

        if "restCards" in source:
            rest_cards = cls._public_doudizhu_rest_cards(source.get("restCards"))
            if rest_cards is not None:
                result["restCards"] = rest_cards
        return result

    @classmethod
    def _public_result(
        cls,
        room: _Room,
        state: EngineFullState | None,
    ) -> dict[str, Any]:
        result = copy.deepcopy(state.result) if state is not None else {}
        if room.game is Game.DOUDIZHU:
            return {} if state is None else cls._public_doudizhu_result(state)
        if room.game is not Game.GUANDAN:
            return result
        order = result.get("order")
        if not _is_complete_guandan_order(order):
            return result
        assert isinstance(order, list)
        first = order[0]
        winning_parity = first % 2
        teammate_finish = order.index((first + 2) % 4) + 1
        if teammate_finish == 2:
            outcome, delta, label = "double_up", 3, "双上"
        elif teammate_finish == 3:
            outcome, delta, label = "first_third", 2, "一三游"
        else:
            outcome, delta, label = "first_last", 1, "一四游"
        winning_positions = [
            position for position in range(4) if position % 2 == winning_parity
        ]
        winners = "1、3" if winning_parity == 0 else "2、4"
        result.update(
            {
                "outcome": outcome,
                "winning_positions": winning_positions,
                "summary": f"{label}：{winners} 号位队升 {delta} 级",
            }
        )
        # ``scores`` from old replays remain readable, but new GuanDan rounds
        # never manufacture per-seat +/- points. GuanDan settles only team levels.
        result.pop("scores", None)
        return result

    @staticmethod
    def _flat_hand_layout(
        cards: list[dict[str, str]],
        mode: int,
        *,
        fallback: bool,
    ) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        if cards:
            groups.append(
                {
                    "pattern": 0,
                    "minor": 0,
                    "value": None,
                    "cards": copy.deepcopy(cards),
                }
            )
        return {
            "version": None,
            "mode": mode,
            "source_sha": None,
            "fallback": fallback,
            "variant_index": 0,
            "variant_count": 1,
            "groups": groups,
        }

    @staticmethod
    def _layout_key(
        room: _Room,
        viewer: _Viewer,
        codes: list[str],
        level: str | None,
    ) -> tuple[str, str | None, int, int, tuple[str, ...]]:
        return (
            room.game.value,
            level,
            viewer.arrange_mode,
            viewer.arrange_variant,
            tuple(sorted(codes)),
        )

    def _private_hand_codes_locked(
        self,
        room: _Room,
        viewer: _Viewer,
    ) -> list[str]:
        seat = self._seat_for_viewer(room, viewer)
        state = room.engine_state
        if seat is None or state is None:
            return []
        player = next(
            (item for item in state.players if item.position == seat.position),
            None,
        )
        return [] if player is None else list(player.hand)

    @staticmethod
    def _sanitize_raw_layout(
        arranged: Any,
        codes: list[str],
        *,
        mode: int,
    ) -> dict[str, Any]:
        if not isinstance(arranged, dict) or not isinstance(
            arranged.get("groups"), list
        ):
            raise ValueError("arranger returned an invalid layout")
        response_mode = arranged.get("mode")
        if isinstance(response_mode, bool) or response_mode != mode:
            raise ValueError("arranger returned the wrong layout mode")
        groups: list[dict[str, Any]] = []
        flattened: list[str] = []
        for raw_group in arranged["groups"]:
            if not isinstance(raw_group, dict) or not isinstance(
                raw_group.get("cards"), list
            ):
                raise ValueError("arranger returned an invalid group")
            group_codes = [str(code) for code in raw_group["cards"]]
            flattened.extend(group_codes)
            groups.append(
                {
                    "pattern": raw_group.get("pattern"),
                    "minor": raw_group.get("minor"),
                    "value": copy.deepcopy(raw_group.get("value")),
                    "cards": group_codes,
                }
            )
        if sorted(flattened) != sorted(codes):
            raise ValueError("arranger changed the private hand multiset")
        return {
            "version": arranged.get("version"),
            "mode": mode,
            "source_sha": arranged.get("source_sha"),
            "fallback": False,
            "variant_index": max(0, int(arranged.get("variant_index", 0))),
            "variant_count": max(1, int(arranged.get("variant_count", 1))),
            "groups": groups,
        }

    async def _warm_private_layout_locked(
        self,
        room: _Room,
        viewer: _Viewer,
    ) -> None:
        if self._seat_for_viewer(room, viewer) is None:
            return
        codes = self._private_hand_codes_locked(room, viewer)
        level = self._public_level(room, room.engine_state)
        hand_identity = (room.game.value, level, viewer.arrange_mode, tuple(sorted(codes)))
        if viewer.layout_key is not None:
            previous_identity = (
                viewer.layout_key[0],
                viewer.layout_key[1],
                viewer.layout_key[2],
                viewer.layout_key[4],
            )
            if previous_identity != hand_identity:
                viewer.arrange_variant = 0
        key = self._layout_key(room, viewer, codes, level)
        if viewer.layout_key == key:
            return
        if not codes:
            raw_layout = {
                "version": None,
                "mode": viewer.arrange_mode,
                "source_sha": None,
                "fallback": False,
                "variant_index": 0,
                "variant_count": 1,
                "groups": [],
            }
        elif self.arranger is None:
            raw_layout = {
                "version": None,
                "mode": viewer.arrange_mode,
                "source_sha": None,
                "fallback": True,
                "variant_index": 0,
                "variant_count": 1,
                "groups": [],
            }
        else:
            try:
                arrange_kwargs = {
                    "level": level,
                    "mode": viewer.arrange_mode,
                }
                if "variant" in inspect.signature(self.arranger.arrange).parameters:
                    arrange_kwargs["variant"] = viewer.arrange_variant
                arranged = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.arranger.arrange,
                        room.game,
                        codes,
                        **arrange_kwargs,
                    ),
                    timeout=self.arranger_timeout_seconds,
                )
                raw_layout = self._sanitize_raw_layout(
                    arranged,
                    codes,
                    mode=viewer.arrange_mode,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Presentation failure is cached for this exact hand/mode so a
                # broken helper cannot create an unbounded stream of threads.
                raw_layout = {
                    "version": None,
                    "mode": viewer.arrange_mode,
                    "source_sha": None,
                    "fallback": True,
                    "variant_index": 0,
                    "variant_count": 1,
                    "groups": [],
                }
        if room.closed:
            return
        viewer.layout_key = key
        viewer.raw_layout = raw_layout

    async def _warm_all_private_layouts_locked(self, room: _Room) -> None:
        for viewer in room.viewers.values():
            if self._seat_for_viewer(room, viewer) is not None:
                await self._warm_private_layout_locked(room, viewer)

    def _project_private_layout_locked(
        self,
        room: _Room,
        viewer: _Viewer,
        cards: list[dict[str, str]],
        *,
        level: str | None,
    ) -> dict[str, Any]:
        if not cards:
            return self._flat_hand_layout(cards, viewer.arrange_mode, fallback=False)
        key = self._layout_key(
            room,
            viewer,
            [card["code"] for card in cards],
            level,
        )
        raw_layout = viewer.raw_layout if viewer.layout_key == key else None
        if raw_layout is None or raw_layout.get("fallback") is True:
            return self._flat_hand_layout(cards, viewer.arrange_mode, fallback=True)
        try:
            by_code: dict[str, deque[dict[str, str]]] = {}
            for card in cards:
                by_code.setdefault(card["code"], deque()).append(card)
            groups: list[dict[str, Any]] = []
            consumed = 0
            for raw_group in raw_layout["groups"]:
                mapped_cards: list[dict[str, str]] = []
                for code in raw_group["cards"]:
                    pool = by_code.get(code)
                    if pool is None or not pool:
                        raise ValueError("cached layout changed the hand multiset")
                    mapped_cards.append(copy.deepcopy(pool.popleft()))
                    consumed += 1
                groups.append(
                    {
                        "pattern": raw_group.get("pattern"),
                        "minor": raw_group.get("minor"),
                        "value": copy.deepcopy(raw_group.get("value")),
                        "cards": mapped_cards,
                    }
                )
            if consumed != len(cards) or any(pool for pool in by_code.values()):
                raise ValueError("cached layout did not conserve every card")
            return {
                "version": raw_layout.get("version"),
                "mode": viewer.arrange_mode,
                "source_sha": raw_layout.get("source_sha"),
                "fallback": False,
                "variant_index": raw_layout.get("variant_index", 0),
                "variant_count": raw_layout.get("variant_count", 1),
                "groups": groups,
            }
        except (KeyError, TypeError, ValueError):
            return self._flat_hand_layout(cards, viewer.arrange_mode, fallback=True)

    def _project_locked(self, room: _Room, viewer: _Viewer) -> dict[str, Any]:
        state = room.engine_state
        players = (
            {player.position: player for player in state.players}
            if state is not None
            else {}
        )
        viewer_seat = self._seat_for_viewer(room, viewer)
        seats: list[dict[str, Any]] = []
        for seat in room.seats:
            public_seat: dict[str, Any] = {
                "position": seat.position,
                "display_name": seat.name,
                "kind": seat.kind.value,
                "ready": seat.ready,
            }
            player = players.get(seat.position)
            if player is not None:
                public_seat.update(
                    {
                        "hand_count": player.hand_count,
                        "play_area": copy.deepcopy(
                            room.visible_play_areas[seat.position]
                            if len(room.visible_play_areas) == len(room.seats)
                            else player.play_area
                        ),
                        "team": player.team,
                        "role": player.role,
                    }
                )
            else:
                public_seat.update({"hand_count": 0, "play_area": None})
            seats.append(public_seat)

        decision: dict[str, Any] | None = None
        if room.phase != "error" and state is not None and state.decision is not None:
            engine_decision = state.decision
            decision = {
                "id": engine_decision.id,
                "actor": engine_decision.actor,
                "phase": engine_decision.phase,
            }
            if (
                viewer_seat is not None
                and viewer_seat.position == engine_decision.actor
                and viewer_seat.kind is SeatKind.HUMAN
            ):
                decision["can_pass"] = engine_decision.can_pass
                decision["action_count"] = engine_decision.action_count
                decision["actions"] = [
                    {
                        "action_index": action.action_index,
                        "type": action.type,
                        "rank": copy.deepcopy(action.rank),
                        "cards": list(action.cards),
                        "label": action.label,
                    }
                    for action in engine_decision.actions
                ]

        result = self._public_result(room, state)
        if (
            room.game is Game.GUANDAN
            and result
            and room.guandan_last_settlement is not None
        ):
            result.update(copy.deepcopy(room.guandan_last_settlement))
            result["team_levels"] = copy.deepcopy(room.guandan_team_levels)
            result["series_state"] = {
                "series_no": room.series_no,
                "rounds_played": room.guandan_rounds_played,
                "head_finish_count_by_seat": list(
                    room.guandan_head_finish_count_by_seat
                ),
                "team_victory_count": copy.deepcopy(
                    room.guandan_team_victory_count
                ),
                "a_attempt_count": copy.deepcopy(room.guandan_a_attempt_count),
                "match_finished": room.guandan_match_finished,
                "match_winner_team": room.guandan_match_winner_team,
            }
            if room.guandan_final_result is not None:
                result["final_result"] = copy.deepcopy(room.guandan_final_result)
        level = self._public_level(room, state)
        projection: dict[str, Any] = {
            "table_no": room.table_no,
            "game": room.game.value,
            "human_slots": room.human_slots,
            "round_no": room.round_no,
            "level": level,
            "guandan_levels": self._public_guandan_levels(room, viewer),
            "series": self._public_guandan_series(room, viewer),
            "phase": room.phase,
            "failure": copy.deepcopy(room.failure),
            "version": room.version,
            "viewer": {
                "participant_id": viewer.participant_id,
                "nickname": viewer.nickname,
                "role": viewer.role.value,
                "seat": viewer_seat.position if viewer_seat is not None else None,
            },
            "seats": seats,
            "spectator_count": sum(
                item.role is ViewerRole.SPECTATOR for item in room.viewers.values()
            ),
            "decision": decision,
            "current_trick": self._public_current_trick(room, state),
            "trick_history": (
                room.public_trick_history.snapshot()
                if room.game is Game.GUANDAN
                else None
            ),
            "turn_started_at_ms": room.turn_started_at_ms,
            "turn_seconds": TURN_SECONDS,
            "bot_policy": self._public_bot_policy(state),
            "rule_profile": self._public_rule_profile(state),
            "doudizhu_state": self._public_doudizhu_state(state),
            "result": result,
            "finish_order": (
                list(room.guandan_finish_order)
                if room.game is Game.GUANDAN
                else []
            ),
            "replay_history": copy.deepcopy(room.replay_history),
            "latest_replay_match_id": (
                str(room.replay_history[-1]["match_id"])
                if room.replay_history
                else ""
            ),
            "server_time_ms": int(time.time() * 1000),
        }
        reveal_all_hands = viewer.role is ViewerRole.SPECTATOR or room.phase.casefold() in {
            "finished",
            "completed",
            "game_over",
            "gameover",
        }
        if reveal_all_hands:
            projection["spectator_hands"] = [
                [
                    {
                        "id": f"s{seat}-v{state.state_version if state else 0}-c{index}",
                        "code": code,
                    }
                    for index, code in enumerate(
                        (players.get(seat).hand if players.get(seat) is not None else [])
                    )
                ]
                for seat in range(len(room.seats))
            ]
        if viewer_seat is not None:
            player = players.get(viewer_seat.position)
            hand = player.hand if player is not None else []
            own_hand = [
                {
                    "id": f"p{viewer_seat.position}-v{state.state_version if state else 0}-c{index}",
                    "code": code,
                }
                for index, code in enumerate(hand)
            ]
            projection["own_hand"] = own_hand
            projection["arrange_mode"] = viewer.arrange_mode
            projection["own_hand_layout"] = self._project_private_layout_locked(
                room,
                viewer,
                own_hand,
                level=level,
            )
        if not set(projection).issubset(VIEW_KEYS):  # defensive tripwire
            raise RuntimeError("human-table projection contains a non-whitelisted key")
        return projection


HumanRoomManager = RoomManager
RoomError = HumanTableError


__all__ = [
    "EngineClient",
    "EngineFactory",
    "EngineFailureError",
    "EngineProtocolError",
    "HandArrangerClient",
    "HumanRoomManager",
    "HumanTableError",
    "InvalidTokenError",
    "RoomError",
    "RoomManager",
    "SubprocessEngineClient",
    "TURN_SECONDS",
    "TableConflictError",
    "TableNotFoundError",
    "TablePermissionError",
    "TableRequestError",
    "VIEW_KEYS",
    "human_worker_command",
    "subprocess_engine_factory",
]
