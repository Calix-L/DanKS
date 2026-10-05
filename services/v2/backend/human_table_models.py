from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Game


class StrictModel(BaseModel):
    """Base model for the public human-table protocol.

    Rejecting unknown fields is intentional: these objects cross a trust
    boundary between a browser, the serving process, and a rules worker.
    """

    model_config = ConfigDict(extra="forbid")


class SeatKind(str, Enum):
    OPEN = "open"
    HUMAN = "human"
    BOT = "bot"


class ViewerRole(str, Enum):
    PLAYER = "player"
    SPECTATOR = "spectator"


class HumanTableCreateRequest(StrictModel):
    game: Game
    human_slots: int
    nickname: str = Field(min_length=1, max_length=64)
    user_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("nickname must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_human_slots(self) -> "HumanTableCreateRequest":
        # A Dou Dizhu table has three seats, so it supports the same complete
        # 1/2/3-human progression as the four-seat GuanDan room (whose fourth
        # human seat remains intentionally unavailable).
        maximum = 3
        if not 1 <= self.human_slots <= maximum:
            raise ValueError(
                f"{self.game.value} human_slots must be between 1 and {maximum}"
            )
        return self


class HumanTableJoinRequest(StrictModel):
    nickname: str = Field(min_length=1, max_length=64)
    resume_token: str | None = Field(default=None, min_length=16, max_length=512)
    user_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("nickname must not be blank")
        return normalized


class HumanTableSeatRequest(StrictModel):
    position: int = Field(ge=0, le=3)
    expected_version: int = Field(ge=0)


class HumanTableReadyRequest(StrictModel):
    ready: bool
    expected_version: int = Field(ge=0)


class HumanTableRetryRequest(StrictModel):
    expected_version: int = Field(ge=0)


class HumanTableArrangeRequest(StrictModel):
    mode: int = Field(ge=1, le=4)


class HumanTableActionRequest(StrictModel):
    action_index: int = Field(ge=0)
    expected_version: int = Field(ge=0)
    decision_id: str = Field(min_length=1, max_length=256)
    client_action_id: str = Field(min_length=1, max_length=256)
    selected_cards: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )

    @field_validator("selected_cards", mode="before")
    @classmethod
    def validate_selected_cards(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("selected_cards must be a list of strings")
        if any(
            not isinstance(card, str)
            or not card
            or len(card) > 8
            or card != card.strip()
            for card in value
        ):
            raise ValueError(
                "selected_cards entries must be nonblank strings of at most 8 characters"
            )
        return value


class HumanTableSession(StrictModel):
    table_no: str
    token: str
    participant_id: str
    role: ViewerRole
    seat: int | None = Field(default=None, ge=0, le=3)
    view: dict[str, Any]


class EngineAction(StrictModel):
    action_index: int = Field(ge=0)
    type: str = Field(min_length=1, max_length=128)
    rank: Any = None
    cards: list[str] = Field(default_factory=list)
    label: str = Field(default="", max_length=512)


class EngineDecision(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    actor: int = Field(ge=0, le=3)
    phase: str = Field(min_length=1, max_length=128)
    actions: list[EngineAction]
    can_pass: bool
    action_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_actions(self) -> "EngineDecision":
        indexes = [item.action_index for item in self.actions]
        if len(indexes) != len(set(indexes)):
            raise ValueError("decision action_index values must be unique")
        if self.action_count != len(self.actions):
            raise ValueError("decision action_count must match actions length")
        return self


class EnginePlayer(StrictModel):
    position: int = Field(ge=0, le=3)
    name: str = Field(min_length=1, max_length=128)
    kind: SeatKind
    team: str | None = Field(default=None, max_length=64)
    role: str | None = Field(default=None, max_length=64)
    hand: list[str]
    hand_count: int = Field(ge=0)
    play_area: Any = None

    @model_validator(mode="after")
    def validate_hand_count(self) -> "EnginePlayer":
        if self.hand_count != len(self.hand):
            raise ValueError("player hand_count must match hand length")
        return self


class EngineFullState(StrictModel):
    """Private, unprojected rules-engine state.

    Instances of this model must never be returned directly to a browser.  The
    room manager below builds a new allow-listed projection for every viewer.
    """

    game: Game
    phase: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=128)
    state_version: int = Field(ge=0)
    players: list[EnginePlayer]
    decision: EngineDecision | None
    table: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_game_shape(self) -> "EngineFullState":
        expected = 4 if self.game is Game.GUANDAN else 3
        if len(self.players) != expected:
            raise ValueError(f"{self.game.value} requires {expected} engine players")
        positions = sorted(player.position for player in self.players)
        if positions != list(range(expected)):
            raise ValueError("engine player positions must be unique and contiguous")
        if self.decision is not None and self.decision.actor >= expected:
            raise ValueError("decision actor is outside the table")
        return self


# Compact aliases are useful in route modules while the prefixed names avoid
# collisions with the existing replay-oriented request models.
CreateTableRequest = HumanTableCreateRequest
JoinTableRequest = HumanTableJoinRequest
SeatRequest = HumanTableSeatRequest
ReadyRequest = HumanTableReadyRequest
ActionRequest = HumanTableActionRequest
ArrangeRequest = HumanTableArrangeRequest
TableSession = HumanTableSession


__all__ = [
    "ActionRequest",
    "ArrangeRequest",
    "CreateTableRequest",
    "EngineAction",
    "EngineDecision",
    "EngineFullState",
    "EnginePlayer",
    "HumanTableActionRequest",
    "HumanTableArrangeRequest",
    "HumanTableCreateRequest",
    "HumanTableJoinRequest",
    "HumanTableReadyRequest",
    "HumanTableSeatRequest",
    "HumanTableSession",
    "JoinTableRequest",
    "ReadyRequest",
    "SeatKind",
    "SeatRequest",
    "TableSession",
    "ViewerRole",
]
