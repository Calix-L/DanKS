from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Game(str, Enum):
    GUANDAN = "guandan"
    DOUDIZHU = "doudizhu"


class RunMode(str, Enum):
    SINGLE = "single"
    FULL_MATCH = "full_match"


class Pairing(str, Enum):
    SINGLE = "single"
    SWAPPED = "swapped"


class MatchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RoundStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ReplayEventType(str, Enum):
    MATCH_STARTED = "match_started"
    ROUND_STARTED = "round_started"
    DECISION_STARTED = "decision_started"
    ACTION_COMMITTED = "action_committed"
    ROUND_FINISHED = "round_finished"
    MATCH_FINISHED = "match_finished"
    WORKER_ERROR = "worker_error"


class PlayerIdentity(StrictModel):
    position: int = Field(ge=0, le=3)
    name: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=128)
    team: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default=None, max_length=64)


class ActorIdentity(StrictModel):
    position: int = Field(ge=0, le=3)
    name: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=128)
    team: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default=None, max_length=64)


class ModelIdentity(StrictModel):
    policy_id: str
    display_name: str
    version: str
    kind: str
    checkpoint_path: str | None = None
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    verified: bool = False
    source_identity: dict[str, Any] = Field(default_factory=dict)


class RoundSummary(StrictModel):
    round_index: int = Field(ge=0)
    status: RoundStatus = RoundStatus.QUEUED
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turn_count: int = Field(default=0, ge=0)
    result: dict[str, Any] = Field(default_factory=dict)


class MatchManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    match_id: str = Field(min_length=1, max_length=128)
    pair_id: str | None = Field(default=None, max_length=128)
    pair_index: int | None = Field(default=None, ge=0, le=1)
    game: Game
    run_mode: RunMode
    pairing: Pairing
    seed: int
    status: MatchStatus = MatchStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    players: list[PlayerIdentity]
    models: list[ModelIdentity]
    source_identity: dict[str, Any] = Field(default_factory=dict)
    presentation_identity: dict[str, Any] = Field(default_factory=dict)
    rules: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    rounds: list[RoundSummary] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_seats(self) -> "MatchManifest":
        expected = 4 if self.game is Game.GUANDAN else 3
        if len(self.players) != expected:
            raise ValueError(f"{self.game.value} requires exactly {expected} players")
        positions = sorted(player.position for player in self.players)
        if positions != list(range(expected)):
            raise ValueError("player positions must be unique and contiguous from zero")
        if self.pairing is Pairing.SWAPPED:
            if self.pair_id is None or self.pair_index is None:
                raise ValueError("swapped matches require pair_id and pair_index")
        return self


class TopCandidate(StrictModel):
    action_index: int | None = Field(default=None, ge=0)
    action: Any = None
    score: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayEvent(StrictModel):
    schema_version: str = SCHEMA_VERSION
    event_id: int = Field(ge=1)
    match_id: str = Field(min_length=1, max_length=128)
    round_index: int = Field(default=0, ge=0)
    turn_index: int = Field(default=0, ge=0)
    event_type: ReplayEventType
    phase: str = Field(default="play", max_length=64)
    actor_pos: int | None = Field(default=None, ge=0, le=3)
    actor: ActorIdentity | None = None
    before_state: dict[str, Any] | None = None
    chosen_action: Any = None
    after_state: dict[str, Any] | None = None
    inference_ms: float | None = Field(default=None, ge=0)
    legal_action_count: int | None = Field(default=None, ge=0)
    legal_actions_hash: str | None = Field(default=None, max_length=128)
    top_candidates: list[TopCandidate] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "ReplayEvent":
        decision_events = {
            ReplayEventType.DECISION_STARTED,
            ReplayEventType.ACTION_COMMITTED,
        }
        if self.event_type in decision_events:
            if self.actor is None and self.actor_pos is None:
                raise ValueError("decision events require an explicit actor position")
            if self.actor is not None:
                if self.actor_pos is None:
                    self.actor_pos = self.actor.position
                elif self.actor.position != self.actor_pos:
                    raise ValueError("actor_pos must match actor.position")
            if self.before_state is None:
                raise ValueError("decision events require before_state")
        if self.event_type is ReplayEventType.ACTION_COMMITTED:
            if self.chosen_action is None:
                raise ValueError("action_committed requires chosen_action")
            if self.after_state is None:
                raise ValueError("action_committed requires after_state")
            if self.inference_ms is None:
                raise ValueError("action_committed requires inference_ms")
            if self.legal_action_count is None or self.legal_actions_hash is None:
                raise ValueError(
                    "action_committed requires legal_action_count and legal_actions_hash"
                )
        return self


class MatchCreateRequest(StrictModel):
    game: Game
    run_mode: RunMode = RunMode.SINGLE
    pairing: Pairing = Pairing.SINGLE
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    assignment: dict[str, str] = Field(default_factory=dict)
    decision_delay_ms: int = Field(default=3000, ge=0, le=30_000)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        pairing = normalized.get("pairing")
        if pairing in {"swap", "side_swap", "paired"}:
            normalized["pairing"] = Pairing.SWAPPED.value
        run_mode = normalized.get("run_mode")
        if run_mode in {"single_round", "single_hand", "baseline_play"}:
            normalized["run_mode"] = RunMode.SINGLE.value
        if run_mode in {"full", "complete"}:
            normalized["run_mode"] = RunMode.FULL_MATCH.value
        return normalized


class MatchCreateResponse(StrictModel):
    job_id: str
    match_ids: list[str]
    pair_id: str | None = None
    status: str = "queued"


class MatchListResponse(StrictModel):
    items: list[MatchManifest]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)


class RoundEventsResponse(StrictModel):
    match_id: str
    round_index: int = Field(ge=0)
    initial_state: dict[str, Any] | None = None
    presentation_version: str
    presentation_reconstructed: bool = False
    items: list[ReplayEvent]
    count: int = Field(ge=0)
