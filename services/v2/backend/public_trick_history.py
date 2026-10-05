"""Bounded Guandan history from referee-confirmed public action metadata."""

from __future__ import annotations

import copy
from typing import Any

from .human_table_models import EngineFullState
from .models import Game


MAX_TRICK_EVENTS = 128
_CARDS = frozenset(f"{suit}{rank}" for suit in "SHCD" for rank in "23456789TJQKA") | {"SB", "HR"}
_TYPES = frozenset({"Single", "Pair", "Trips", "ThreeWithTwo", "Straight", "ThreePair", "TwoTrips", "StraightFlush", "Bomb", "FourKings"})
_RANKS = frozenset("23456789TJQKABR") | {"10"}


def public_play_event(seat: int, play: Any) -> dict[str, Any] | None:
    """Whitelist a committed play; never recover facts from cards or actions."""
    if type(seat) is not int or seat not in range(4):
        return None
    if isinstance(play, list) and len(play) >= 3:
        kind, rank, cards = play[:3]
    elif isinstance(play, dict):
        kind, rank, cards = play.get("type"), play.get("rank"), play.get("cards")
        if play.get("pass") is True and kind != "PASS":
            return None
    else:
        return None
    if not isinstance(kind, str) or not isinstance(cards, list):
        return None
    if kind.casefold() == "pass":
        if cards or (rank is not None and rank not in ("", "PASS", "pass")):
            return None
        return {"seat": seat, "cards": [], "pass": True, "type": "PASS", "rank": "PASS"}
    if (
        kind not in _TYPES or not isinstance(rank, str) or rank not in _RANKS
        or not 1 <= len(cards) <= 27
        or any(not isinstance(card, str) or card not in _CARDS for card in cards)
    ):
        return None
    return {"seat": seat, "cards": list(cards), "pass": False, "type": kind, "rank": rank}


class PublicTrickHistory:
    """Only the current and last completed trick survive in room memory.

    ``complete`` means evidence completeness, not trick lifecycle. A service
    attached mid-trick, missing public play or capped list remains incomplete.
    The high-water marker ignores stale snapshots, including after room state
    itself has been replaced by one. No full engine/private state is retained.
    """

    def __init__(self) -> None:
        self.current: dict[str, Any] | None = None
        self.previous: dict[str, Any] | None = None
        self._version: int | None = None
        self._decision: tuple[str, int, str] | None = None

    def observe(self, state: EngineFullState) -> None:
        if state.game is not Game.GUANDAN:
            return
        decision = state.decision
        next_decision = None if decision is None else (decision.id, decision.actor, decision.phase)
        play_phase = state.status == "active" and state.phase == "play" and decision is not None and decision.phase == "play"
        terminal = state.status == "finished" or state.phase == "finished"
        if self._version is None:
            self._version, self._decision = state.state_version, next_decision
            if play_phase:
                self.current = {"number": 1, "complete": not decision.can_pass, "events": []}
            return
        if state.state_version <= self._version:
            return
        previous_decision = self._decision
        self._version = state.state_version
        # A metadata-only version change is not a committed action or boundary.
        if not terminal and previous_decision is not None and next_decision is not None and previous_decision[0] == next_decision[0]:
            return
        self._decision = None if terminal else next_decision
        if previous_decision is None or previous_decision[2] != "play" or not (play_phase or terminal):
            if play_phase and self.current is None:
                self.current = {"number": 1, "complete": not decision.can_pass, "events": []}
            return
        if self.current is None:
            self.current = {"number": 1, "complete": False, "events": []}
        actor = previous_decision[1]
        player = next((player for player in state.players if player.position == actor), None)
        # The referee may clear/mutate player play areas while handing back
        # the lead. The worker's copied latest-action table survives that reset.
        # Its owner must match the previous decision; never infer a missing PASS.
        table = state.table
        owner = table.get("owner_position")
        event = None
        if type(owner) is int and owner == actor:
            event = public_play_event(actor, {"type": table.get("action_type"),
                "rank": table.get("action_rank"), "cards": table.get("cards")})
        if event is None:
            event = public_play_event(actor, None if player is None else player.play_area)
        if event is None:
            self.current["complete"] = False
        else:
            self.current["events"].append(event)
            if len(self.current["events"]) > MAX_TRICK_EVENTS:
                self.current["events"] = self.current["events"][-MAX_TRICK_EVENTS:]
                self.current["complete"] = False
        if terminal or (play_phase and not decision.can_pass):
            self.previous = self.current
            self.current = None if terminal else {"number": self.previous["number"] + 1, "complete": True, "events": []}

    def snapshot(self) -> dict[str, Any]:
        return {"available": True, "current": copy.deepcopy(self.current), "previous": copy.deepcopy(self.previous)}
