#!/usr/bin/env python3
"""Generation-one GuanDan rules worker and public AI bridge.

The process owns one table and speaks request/response NDJSON on stdin/stdout.
It deliberately contains no clock: a table advances only after an explicit
``act`` or ``bot_act`` request.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, TextIO



STATE_KEYS = {
    "game",
    "phase",
    "status",
    "state_version",
    "players",
    "decision",
    "table",
    "metrics",
    "result",
}

_RANK_VALUE = {
    rank: value
    for value, rank in enumerate(
        ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "B", "R")
    )
}
_SUIT_VALUE = {"S": 0, "H": 1, "C": 2, "D": 3}
GUANDAN_LEVELS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
GUANDAN_SPORTS_RULE_PROFILE = "sports_bureau"
GUANDAN_ARENA_RULE_PROFILE = "arena_client_pdf_v1"
GUANDAN_STANDARD_RULE_PROFILE = "standard_v1"
GUANDAN_RULE_PROFILES = frozenset(
    {
        GUANDAN_SPORTS_RULE_PROFILE,
        GUANDAN_ARENA_RULE_PROFILE,
        GUANDAN_STANDARD_RULE_PROFILE,
    }
)


class GuandanBotPolicy(Protocol):
    def reset_table_history(self) -> None: ...

    def consume_messages(self, messages: Iterable[Any]) -> int: ...

    def choose_action_from_environment(
        self, environment: Any, *, seat: int | None = None
    ) -> int: ...

    def metadata(self) -> Mapping[str, Any]: ...


def _rule_bot_policy_metadata():
    return {"id": "example_rule_bot", "ready": True}


def _filtered_bot_policy_metadata(policy):
    if policy is None:
        return _rule_bot_policy_metadata()
    raw = policy.metadata()
    return {"id": str(raw["id"]), "ready": bool(raw["ready"])}


def _load_public_policy(*, game, mode):
    from .http_policy import HTTPPolicy
    endpoint = os.getenv("DANKS_AI_ENDPOINT", "").strip()
    return HTTPPolicy(endpoint) if endpoint and game == "guandan" and mode == "real" else None


def _guandan_rule_profile(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    profile = values.get(
        "BATTLE_GUANDAN_RULE_PROFILE", GUANDAN_STANDARD_RULE_PROFILE
    ).strip()
    if profile not in GUANDAN_RULE_PROFILES:
        _fault(
            "engine_unavailable",
            f"unsupported GuanDan rule profile {profile!r}",
        )
    return profile


def verify_arena_guandan_rule_contract(rules_module: Any) -> None:
    """Fail closed when GuanDan rules no longer matches the supplied PDF.

    These probes cover the three ambiguous wildcard examples that previously
    diverged from the human-table contract: arbitrary straight substitution,
    bomb-first interpretation, steel-plate priority on lead, and greatest-rank
    selection within one physical card multiset.
    """

    first_actions = getattr(rules_module, "first_actions", None)
    second_actions = getattr(rules_module, "second_actions", None)
    can_beat = getattr(rules_module, "can_beat", None)
    if not all(callable(value) for value in (first_actions, second_actions, can_beat)):
        raise RuntimeError(
            "GuanDan rules does not expose first_actions, second_actions and can_beat"
        )

    cases = (
        (
            ["HJ", "SQ", "SJ", "ST", "S9"],
            "J",
            "StraightFlush",
            "K",
        ),
        (
            ["S3", "H3", "S4", "H4", "HT", "HT"],
            "T",
            "TwoTrips",
            "4",
        ),
        (
            ["S3", "H3", "S4", "H4", "HT"],
            "T",
            "ThreeWithTwo",
            "4",
        ),
    )
    for hand, level, expected_type, expected_rank in cases:
        actions = first_actions(hand, hand.count(f"H{level}"), level)
        matches = [
            action
            for action in actions
            if isinstance(action, list)
            and len(action) >= 3
            and isinstance(action[2], list)
            and sorted(str(card) for card in action[2]) == sorted(hand)
        ]
        if len(matches) != 1 or matches[0][0:2] != [expected_type, expected_rank]:
            raise RuntimeError(
                "GuanDan rules failed the Arena PDF wildcard contract for "
                f"{hand!r}: {matches!r}"
            )

    lead_actions = first_actions(["S3", "H4"], 0, "J")
    if any(action[0] == "PASS" for action in lead_actions):
        raise RuntimeError("GuanDan rules offered PASS on a free lead")

    wildcard_hand = ["HJ", "SQ", "SJ", "ST", "S9"]
    follow_actions = second_actions(
        wildcard_hand,
        1,
        "J",
        ["Straight", "Q", ["S8", "S9", "ST", "SJ", "SQ"]],
    )
    exact_follow = [
        action
        for action in follow_actions
        if isinstance(action, list)
        and len(action) >= 3
        and isinstance(action[2], list)
        and sorted(str(card) for card in action[2]) == sorted(wildcard_hand)
    ]
    if len(exact_follow) != 1 or exact_follow[0][0:2] != ["StraightFlush", "K"]:
        raise RuntimeError(
            "GuanDan rules failed bomb-first canonicalization while following: "
            f"{exact_follow!r}"
        )

    bomb_four = ["Bomb", "3", ["S3", "H3", "C3", "D3"]]
    bomb_five = ["Bomb", "3", ["S3", "H3", "C3", "D3", "S3"]]
    bomb_six = ["Bomb", "3", ["S3", "H3", "C3", "D3", "S3", "H3"]]
    straight_flush = ["StraightFlush", "K", ["S9", "ST", "SJ", "SQ", "SK"]]
    four_kings = ["FourKings", "R", ["SB", "SB", "HR", "HR"]]
    hierarchy = (bomb_four, bomb_five, straight_flush, bomb_six, four_kings)
    if any(
        not can_beat(higher, lower, "J") or can_beat(lower, higher, "J")
        for lower, higher in zip(hierarchy, hierarchy[1:])
    ):
        raise RuntimeError("GuanDan rules failed the Arena bomb hierarchy contract")


class RpcFault(RuntimeError):
    """A recoverable request error returned to the controller."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fault(code: str, message: str) -> "Any":
    raise RpcFault(code, message)


@contextlib.contextmanager
def _quiet_engine() -> Iterable[None]:
    """Keep third-party/research runtime prints off the RPC stream."""

    with contextlib.redirect_stdout(sys.stderr):
        yield


def _raw_action(value: Any) -> list[Any] | dict[str, Any] | str | None:
    if value is None:
        return None
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return to_json()
    return value


def _action_view(raw: Any, action_index: int) -> dict[str, Any]:
    raw = _raw_action(raw)
    if isinstance(raw, dict):
        action_type = raw.get("type", raw.get("kind", "action"))
        rank = raw.get("rank")
        cards_value = raw.get("cards", raw.get("card_list", []))
    elif isinstance(raw, (list, tuple)):
        action_type = raw[0] if raw else "PASS"
        rank = raw[1] if len(raw) > 1 else None
        cards_value = raw[2] if len(raw) > 2 else []
    else:
        action_type = "PASS" if str(raw).upper() == "PASS" else "action"
        rank = None
        cards_value = []

    kind = str(action_type or "action")
    rank_value = None if rank is None else str(rank)
    cards = (
        [str(card) for card in cards_value]
        if isinstance(cards_value, (list, tuple))
        else []
    )
    label = "PASS" if kind.upper() == "PASS" else kind
    return {
        "action_index": action_index,
        "type": kind,
        "rank": rank_value,
        "cards": cards,
        "label": label,
    }


def _play_area(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    action = _action_view(value, -1)
    action.pop("action_index")
    return action


def _empty_table() -> dict[str, Any]:
    return {
        "cards": [],
        "action_label": "",
        "action_type": "PASS",
        "owner_position": None,
    }


def _table_from_action(action: dict[str, Any], seat: int) -> dict[str, Any]:
    return {
        "cards": list(action["cards"]),
        "action_label": action["label"],
        "action_type": action["type"],
        "owner_position": seat,
    }


def _card_key(card: str) -> tuple[int, int, str]:
    label = str(card).upper()
    if label in {"SB", "BJ", "B"}:
        return _RANK_VALUE["B"], 9, label
    if label in {"HR", "RJ", "R"}:
        return _RANK_VALUE["R"], 9, label
    rank = label[-1:] or "?"
    suit = label[:1]
    return _RANK_VALUE.get(rank, 99), _SUIT_VALUE.get(suit, 9), label




def _action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    cards = action["cards"]
    first = _card_key(cards[0]) if cards else (99, 9, "")
    return first, len(cards), str(action["rank"] or ""), action["action_index"]


def choose_rule_action(
    actions: list[dict[str, Any]], *, phase: str, is_lead: bool
) -> int:
    """Choose a deterministic, deliberately small rule action."""

    if not actions:
        _fault("no_legal_action", "the current decision has no legal action")
    if phase in {"tribute", "back"}:
        return actions[0]["action_index"]

    non_pass = [action for action in actions if action["type"].upper() != "PASS"]
    if is_lead:
        singles = [
            action for action in non_pass if action["type"].casefold() == "single"
        ]
        candidates = singles or non_pass
    else:
        candidates = non_pass
    if candidates:
        return min(candidates, key=_action_key)["action_index"]

    passes = [action for action in actions if action["type"].upper() == "PASS"]
    if passes:
        return passes[0]["action_index"]
    _fault("no_legal_action", "the rule bot could not select a legal action")




def _message_body(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        body = message.get("body", message)
    else:
        body = getattr(message, "body", None)
    return body if isinstance(body, dict) else None


def normalize_guandan_finish_order(
    raw_order: Any,
    raw_rest_cards: Any,
    *,
    tie_rng: random.Random,
) -> list[int]:
    """Return the PDF-defined complete finish order for one terminal hand.

    GuanDan rules ends a hand as soon as both partners have gone out.  Its wire
    payload then appends players who still hold cards to ``order`` in seat
    order.  ``restCards`` is what distinguishes those synthetic suffix entries
    from players who actually emptied their hands.
    """

    if (
        not isinstance(raw_order, list)
        or len(raw_order) != 4
        or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in raw_order
        )
        or set(raw_order) != set(range(4))
    ):
        raise ValueError(
            "GuanDan rules episodeOver.order must be a permutation of 0..3"
        )
    if not isinstance(raw_rest_cards, list):
        raise ValueError("GuanDan rules episodeOver.restCards must be a list")

    remaining_counts: dict[int, int] = {}
    for row in raw_rest_cards:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(
                "GuanDan rules episodeOver.restCards rows must be [position, cards]"
            )
        position, cards = row
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in range(4)
            or position in remaining_counts
        ):
            raise ValueError(
                "GuanDan rules episodeOver.restCards positions must be unique 0..3"
            )
        if not isinstance(cards, (list, tuple)) or not cards:
            raise ValueError(
                "GuanDan rules episodeOver.restCards must contain nonempty hands"
            )
        remaining_counts[position] = len(cards)

    if len(remaining_counts) > 2:
        raise ValueError(
            "GuanDan rules episodeOver cannot leave more than two players unfinished"
        )

    explicit_finishers = [
        position for position in raw_order if position not in remaining_counts
    ]
    remaining_positions = sorted(remaining_counts)
    if len(remaining_positions) == 2:
        if (
            len(explicit_finishers) != 2
            or explicit_finishers[0] % 2 != explicit_finishers[1] % 2
        ):
            raise ValueError(
                "GuanDan rules episodeOver unfinished players contradict team finish"
            )
        first, second = remaining_positions
        first_count = remaining_counts[first]
        second_count = remaining_counts[second]
        if first_count == second_count:
            if tie_rng.randrange(2):
                remaining_positions.reverse()
        else:
            remaining_positions.sort(key=remaining_counts.__getitem__)
    elif len(remaining_positions) == 1 and len(explicit_finishers) != 3:
        raise ValueError(
            "GuanDan rules episodeOver has an invalid explicit finish count"
        )

    order = [*explicit_finishers, *remaining_positions]
    if len(order) != 4 or set(order) != set(range(4)):
        raise ValueError("normalized GuanDan finish order is not a permutation of 0..3")
    return order


def _valid_indices(environment: Any) -> list[int]:
    result: list[int] = []
    action_count = len(environment.legal_moves.action_list)
    for value in environment.legal_moves.valid_range:
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < action_count:
            result.append(value)
    return result


class BaseAdapter:
    def __init__(self, game: str, seed: int, seats: list[dict[str, Any]]) -> None:
        self.game = game
        self.seed = seed
        self.seats = copy.deepcopy(seats)
        self.status = "active"
        self.result: dict[str, Any] | None = None
        self.table = _empty_table()
        self.turn_count = 0

    @property
    def phase(self) -> str:
        raise NotImplementedError

    @property
    def actor(self) -> int | None:
        raise NotImplementedError

    @property
    def is_lead(self) -> bool:
        raise NotImplementedError

    def legal_actions(self) -> list[tuple[int, Any]]:
        raise NotImplementedError

    def apply(self, seat: int, action_index: int) -> None:
        raise NotImplementedError

    def players_state(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    def _record_action(self, seat: int, action_index: int, raw: Any) -> None:
        self.table = _table_from_action(_action_view(raw, action_index), seat)
        self.turn_count += 1
        policy = getattr(self, "bot_policy", None)
        if policy is not None:
            policy.record_action(seat, raw)


class MockAdapter(BaseAdapter):
    """Tiny standard-library game used for local UI and protocol testing."""

    def __init__(
        self,
        game: str,
        seed: int,
        seats: list[dict[str, Any]],
        *,
        first_player: int | None = None,
        level: str | None = None,
    ) -> None:
        super().__init__(game, seed, seats)
        self.seat_count = 4 if game == "guandan" else 3
        deck = [
            suit + rank
            for suit in "SHCD"
            for rank in "3456789TJQKA2"
        ] + ["SB", "HR"]
        rng = random.Random(seed)
        rng.shuffle(deck)
        self.hands = [
            sorted(deck[position * 3 : position * 3 + 3], key=_card_key)
            for position in range(self.seat_count)
        ]
        self.play_areas: list[Any] = [None] * self.seat_count
        self.current_actor = (
            first_player
            if game == "guandan" and first_player is not None
            else seed % self.seat_count
        )
        self.level = level if game == "guandan" and level else (
            random.Random(seed).choice(GUANDAN_LEVELS)
            if game == "guandan"
            else None
        )
        self.leader: int | None = None
        self.target_rank: int | None = None
        self.pass_count = 0

    @property
    def phase(self) -> str:
        return "finished" if self.status == "finished" else "play"

    @property
    def actor(self) -> int | None:
        return self.current_actor if self.status == "active" else None

    @property
    def is_lead(self) -> bool:
        return self.target_rank is None

    def legal_actions(self) -> list[tuple[int, Any]]:
        if self.status != "active":
            return []
        choices: list[list[Any]] = []
        seen: set[str] = set()
        for card in self.hands[self.current_actor]:
            if card in seen:
                continue
            seen.add(card)
            if self.target_rank is None or _card_key(card)[0] > self.target_rank:
                choices.append(["Single", card[-1], [card]])
        if self.target_rank is not None:
            choices.append(["PASS", "PASS", []])
        return list(enumerate(choices))

    def apply(self, seat: int, action_index: int) -> None:
        legal = dict(self.legal_actions())
        if action_index not in legal:
            _fault("illegal_action", f"action_index {action_index} is not legal")
        raw = legal[action_index]
        action = _action_view(raw, action_index)
        self.play_areas[seat] = copy.deepcopy(raw)
        self._record_action(seat, action_index, raw)

        if action["type"].upper() == "PASS":
            self.pass_count += 1
            if self.pass_count >= self.seat_count - 1:
                self.current_actor = int(self.leader)
                self.target_rank = None
                self.pass_count = 0
            else:
                self.current_actor = (seat + 1) % self.seat_count
            return

        card = action["cards"][0]
        self.hands[seat].remove(card)
        self.leader = seat
        self.target_rank = _card_key(card)[0]
        self.pass_count = 0
        if not self.hands[seat]:
            self.status = "finished"
            rest_cards = [
                [position, list(hand)]
                for position, hand in enumerate(self.hands)
                if hand
            ]
            if self.game == "guandan":
                remainder = sorted(
                    (position for position in range(self.seat_count) if position != seat),
                    key=lambda position: (len(self.hands[position]), position),
                )
                self.result = {
                    "order": [seat, *remainder],
                    "restCards": rest_cards,
                }
            else:
                landlord = 0
                landlord_won = seat == landlord
                hand_scores = (
                    [2, -1, -1]
                    if landlord_won
                    else [-2, 1, 1]
                )
                self.result = {
                    "winner": seat,
                    "winningSide": (
                        "landlord" if landlord_won else "defenders"
                    ),
                    "landlord": landlord,
                    "baseScore": 1,
                    "bombs": 0,
                    "rockets": 0,
                    "spring": False,
                    "reverseSpring": False,
                    "doubles": {"1": False, "2": False},
                    "redouble": False,
                    "handScores": hand_scores,
                    "totalScores": list(hand_scores),
                    "restCards": rest_cards,
                }
            return
        self.current_actor = (seat + 1) % self.seat_count

    def players_state(self) -> list[dict[str, Any]]:
        players: list[dict[str, Any]] = []
        for seat in self.seats:
            position = seat["position"]
            player = {
                "position": position,
                "name": seat["name"],
                "kind": seat["kind"],
                "hand": list(self.hands[position]),
                "hand_count": len(self.hands[position]),
                "play_area": _play_area(self.play_areas[position]),
            }
            if self.game == "guandan":
                player["team"] = "even" if position % 2 == 0 else "odd"
            else:
                player["role"] = "landlord" if position == 0 else "farmer"
            players.append(player)
        return players

    def metrics(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "engine": "mock",
            "turn_count": self.turn_count,
            "bot_policy": _rule_bot_policy_metadata(),
        }
        if self.game == "guandan":
            values.update(
                {
                    "level": self.level,
                    "trick_index": self.turn_count,
                    "finish_order": (
                        [] if self.result is None else list(self.result["order"])
                    ),
                }
            )
        else:
            values.update(
                {
                    "landlord_position": 0,
                    "highest_bid": 1,
                    "highest_bidder": 0,
                    "bids": {"0": 1},
                    "pending_double": [],
                    "doubles": {"1": False, "2": False},
                    "redouble": False,
                    "base_score": 1,
                    "bottom_cards": [],
                    "multiplier": 1,
                    "defender_multipliers": {"1": 1, "2": 1},
                    "bombs": 0,
                    "rockets": 0,
                }
            )
        return values


class RealGuandanAdapter(BaseAdapter):
    def __init__(
        self,
        seed: int,
        seats: list[dict[str, Any]],
        *,
        first_player: int | None = None,
        level_index: int | None = None,
        bot_policy: GuandanBotPolicy | None = None,
    ) -> None:
        super().__init__("guandan", seed, seats)
        self.bot_policy = bot_policy
        if self.bot_policy is not None:
            self.bot_policy.reset_table_history()
        self._finish_tiebreak_rng = random.Random(seed ^ 0x44414E)
        self.rule_profile = _guandan_rule_profile()
        from rules.engine import environment as module
        if self.rule_profile == GUANDAN_ARENA_RULE_PROFILE:
            from rules.engine import python_rules
            verify_arena_guandan_rule_contract(python_rules)
        level_index = (
            random.Random(seed).randrange(2, 15)
            if level_index is None
            else level_index
        )
        random.seed(seed)
        with _quiet_engine():
            self.environment = module.Environment(
                first_player=seed % 4 if first_player is None else first_player
            )
            for seat in seats:
                self.environment.add_player(seat["name"], seat["position"])
            self.environment.set_rank(level_index, -1)
            messages = list(self.environment.start())
            # GuanDan rules's start() rebuilds the trick rank order with 2 as
            # trump even when ``set_rank`` selected another level beforehand.
            # Keep every comparison surface aligned with the rank used for
            # dealing, wild hearts and legal-action generation.  Tribute is
            # disabled for human tables, but leaving this table stale would
            # still make the environment internally contradictory.
            self.environment.state.update_order("2", self.environment.rank)
        self._consume(messages)

    @property
    def phase(self) -> str:
        if self.status == "finished":
            return "finished"
        name = str(getattr(self.environment.loop, "__name__", "play"))
        return name if name in {"play", "tribute", "back"} else "play"

    @property
    def actor(self) -> int | None:
        if self.status != "active":
            return None
        value = getattr(self.environment.state, "current_pos", None)
        return value if isinstance(value, int) and value in range(4) else None

    @property
    def is_lead(self) -> bool:
        return bool(getattr(self.environment, "action_first", False))

    def legal_actions(self) -> list[tuple[int, Any]]:
        if self.status != "active":
            return []
        return [
            (index, self.environment.legal_moves.action_list[index])
            for index in _valid_indices(self.environment)
        ]

    def apply(self, seat: int, action_index: int) -> None:
        legal = dict(self.legal_actions())
        if action_index not in legal:
            _fault("illegal_action", f"action_index {action_index} is not legal")
        if not self.environment.validate(seat, {"actIndex": action_index}):
            _fault("illegal_action", "GuanDan rules rejected the selected action")
        raw = copy.deepcopy(legal[action_index])
        committed_phase = self.phase
        with _quiet_engine():
            messages = list(self.environment.loop({"actIndex": action_index}))
        self._record_action(seat, action_index, raw)
        if self.bot_policy is not None and committed_phase == "play":
            self.bot_policy.consume_messages(messages)
        self._consume(messages)

    def _consume(self, messages: Iterable[Any]) -> None:
        if self.status == "finished":
            return
        for message in messages:
            body = _message_body(message)
            if body is None or body.get("stage") != "episodeOver":
                continue
            rest_cards = copy.deepcopy(body.get("restCards"))
            self.result = {
                "order": normalize_guandan_finish_order(
                    body.get("order", []),
                    rest_cards,
                    tie_rng=self._finish_tiebreak_rng,
                ),
                "restCards": rest_cards,
            }
            self.status = "finished"
            return

    def players_state(self) -> list[dict[str, Any]]:
        players: list[dict[str, Any]] = []
        for seat, engine_player in zip(self.seats, self.environment.players):
            hand = [str(card) for card in engine_player.hand_cards]
            players.append(
                {
                    "position": seat["position"],
                    "name": seat["name"],
                    "kind": seat["kind"],
                    "hand": hand,
                    "hand_count": len(hand),
                    "play_area": _play_area(engine_player.play_area),
                    "team": "even" if seat["position"] % 2 == 0 else "odd",
                }
            )
        return players

    def metrics(self) -> dict[str, Any]:
        return {
            "engine": "GuanDan rules",
            "rule_profile": self.rule_profile,
            "level": str(self.environment.rank),
            "trick_index": self.turn_count,
            "finish_order": (
                [] if self.result is None else copy.deepcopy(self.result["order"])
            ),
            "bot_policy": _filtered_bot_policy_metadata(self.bot_policy),
        }




def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fault("invalid_request", f"payload.{key} must be an integer")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        _fault("invalid_request", f"payload.{key} must be a nonempty string")
    return value




def _optional_guandan_level(payload: dict[str, Any]) -> tuple[str | None, int | None]:
    if "level" not in payload:
        return None, None
    value = payload.get("level")
    if isinstance(value, bool):
        _fault("invalid_request", "payload.level must be a GuanDan level")
    text = str(value).strip().upper()
    if text == "10":
        text = "T"
    if text not in GUANDAN_LEVELS:
        _fault("invalid_request", "payload.level must be 2 through A")
    level_index = 10 if text == "T" else 11 + "JQKA".index(text) if text in "JQKA" else int(text)
    return text, level_index


def validate_seats(game: str, raw_seats: Any) -> list[dict[str, Any]]:
    seat_count = 4 if game == "guandan" else 3
    if not isinstance(raw_seats, list) or len(raw_seats) != seat_count:
        _fault(
            "invalid_roster",
            f"{game} requires exactly {seat_count} seats",
        )
    seats: list[dict[str, Any]] = []
    for raw in raw_seats:
        if not isinstance(raw, dict):
            _fault("invalid_roster", "every seat must be an object")
        position = raw.get("position")
        name = raw.get("name")
        kind = raw.get("kind")
        if isinstance(position, bool) or not isinstance(position, int):
            _fault("invalid_roster", "seat.position must be an integer")
        if not isinstance(name, str) or not name.strip():
            _fault("invalid_roster", "seat.name must be a nonempty string")
        if not isinstance(kind, str) or not kind.strip():
            _fault("invalid_roster", "seat.kind must be a nonempty string")
        normalized_name = name.strip()
        normalized_kind = kind.strip().lower()
        if len(normalized_name) > 128:
            _fault("invalid_roster", "seat.name must be at most 128 characters")
        if normalized_kind not in {"human", "bot"}:
            _fault("invalid_roster", "seat.kind must be human or bot")
        seats.append(
            {
                "position": position,
                "name": normalized_name,
                "kind": normalized_kind,
            }
        )
    seats.sort(key=lambda seat: seat["position"])
    if [seat["position"] for seat in seats] != list(range(seat_count)):
        _fault(
            "invalid_roster",
            f"seat positions must be exactly 0 through {seat_count - 1}",
        )
    return seats


class GameSession:
    """One state-versioned table hosted by an NDJSON process."""

    def __init__(
        self,
        game: str,
        mode: str,
    ) -> None:
        self.game = game
        self.mode = mode
        if game != "guandan":
            raise ValueError("only guandan is supported")
        self.adapter: BaseAdapter | None = None
        self.state_version = 0
        self.closed = False

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.adapter is not None:
            _fault("already_started", "this worker already owns a table")
        seed = _required_integer(payload, "seed")
        if seed < 0 or seed > 2**63 - 1:
            _fault("invalid_request", "payload.seed is outside the supported range")
        seats = validate_seats(self.game, payload.get("seats"))
        level, level_index = (
            _optional_guandan_level(payload)
            if self.game == "guandan"
            else (None, None)
        )
        first_player = payload.get("first_player")
        if first_player is not None:
            if self.game != "guandan":
                _fault(
                    "invalid_request",
                    "payload.first_player is supported only for guandan",
                )
            if (
                isinstance(first_player, bool)
                or not isinstance(first_player, int)
                or first_player not in range(4)
            ):
                _fault(
                    "invalid_request",
                    "payload.first_player must be an integer from 0 through 3",
                )
        if self.mode == "mock":
            adapter: BaseAdapter = MockAdapter(
                self.game,
                seed,
                seats,
                first_player=first_player,
                level=level,
            )
        elif self.game == "guandan":
            bot_policy = _load_public_policy(
                game=self.game,
                mode=self.mode,
            )
            adapter = RealGuandanAdapter(
                seed,
                seats,
                first_player=first_player,
                level_index=level_index,
                bot_policy=bot_policy,
            )
        else:
            raise ValueError("only guandan is supported")
        self.adapter = adapter
        self.state_version = 1
        return self.snapshot()

    def _turn(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        if self.adapter is None:
            _fault("not_started", "start the table before submitting an action")
        if self.adapter.status != "active":
            _fault("game_finished", "the table has already finished")
        expected_version = _required_integer(payload, "expected_state_version")
        if expected_version != self.state_version:
            _fault(
                "stale_state",
                f"expected state_version {self.state_version}, got {expected_version}",
            )
        decision = self._decision()
        if decision is None:
            _fault("game_finished", "the table has no pending decision")
        decision_id = _required_string(payload, "decision_id")
        if decision_id != decision["id"]:
            _fault("stale_decision", "decision_id does not match the current turn")
        seat = _required_integer(payload, "seat")
        if seat != decision["actor"]:
            _fault(
                "wrong_actor",
                f"seat {decision['actor']} must act, not seat {seat}",
            )
        return decision, seat

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision, seat = self._turn(payload)
        action_index = _required_integer(payload, "action_index")
        legal_actions = {
            action["action_index"]: action for action in decision["actions"]
        }
        legal_indices = set(legal_actions)
        if action_index not in legal_indices:
            _fault(
                "illegal_action",
                f"action_index {action_index} is not in the current decision",
            )
        assert self.adapter is not None
        if "selected_cards" in payload:
            _fault("illegal_selected_cards", "generation one uses the canonical action index only")
        self.adapter.apply(seat, action_index)
        self.state_version += 1
        return self.snapshot()

    def bot_act(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision, seat = self._turn(payload)
        assert self.adapter is not None
        if (
            self.game == "guandan"
            and self.mode == "real"
            and isinstance(self.adapter, RealGuandanAdapter)
            and self.adapter.bot_policy is not None
            and decision["phase"] == "play"
        ):
            action_index = self.adapter.bot_policy.choose_action_from_environment(
                self.adapter.environment,
                seat=seat,
            )
        else:
            chooser = choose_rule_action
            action_index = chooser(
                decision["actions"],
                phase=decision["phase"],
                is_lead=self.adapter.is_lead,
            )
        legal_indices = {
            action["action_index"] for action in decision["actions"]
        }
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index not in legal_indices
        ):
            _fault(
                "policy_error",
                f"bot selected action_index {action_index!r} outside the current decision",
            )
        self.adapter.apply(seat, action_index)
        self.state_version += 1
        return self.snapshot()

    def close(self) -> dict[str, Any] | None:
        self.closed = True
        return None if self.adapter is None else self.snapshot()

    def _decision(self) -> dict[str, Any] | None:
        adapter = self.adapter
        if adapter is None or adapter.status != "active" or self.closed:
            return None
        actor = adapter.actor
        if actor is None:
            _fault("engine_error", "active engine has no current actor")
        phase = adapter.phase
        actions = [
            _action_view(raw, index) for index, raw in adapter.legal_actions()
        ]
        decision_id = f"v{self.state_version}:p{actor}:{phase}"
        return {
            "id": decision_id,
            "actor": actor,
            "phase": phase,
            "actions": actions,
            "can_pass": any(
                action["type"].upper() == "PASS" for action in actions
            ),
            "action_count": len(actions),
        }

    def snapshot(self) -> dict[str, Any]:
        if self.adapter is None:
            _fault("not_started", "the table has not started")
        status = "closed" if self.closed else self.adapter.status
        phase = "closed" if self.closed else self.adapter.phase
        state = {
            "game": self.game,
            "phase": phase,
            "status": status,
            "state_version": self.state_version,
            "players": self.adapter.players_state(),
            "decision": self._decision(),
            "table": copy.deepcopy(self.adapter.table),
            "metrics": self.adapter.metrics(),
            "result": (
                {}
                if self.adapter.result is None
                else copy.deepcopy(self.adapter.result)
            ),
        }
        if set(state) != STATE_KEYS:
            raise AssertionError("internal state schema drift")
        return state

    @staticmethod
    def operation(request: dict[str, Any]) -> str:
        value = request.get(
            "type",
            request.get("op", request.get("method", request.get("command"))),
        )
        if not isinstance(value, str) or not value:
            _fault("invalid_request", "request type is required")
        return value

    @staticmethod
    def payload(request: dict[str, Any]) -> dict[str, Any]:
        if "payload" not in request:
            return {
                key: value
                for key, value in request.items()
                if key not in {"request_id", "type", "op", "method", "command"}
            }
        payload = request["payload"]
        if not isinstance(payload, dict):
            _fault("invalid_request", "request.payload must be an object")
        return payload

    def dispatch(self, request: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        operation = self.operation(request)
        payload = self.payload(request)
        if operation == "start":
            return self.start(payload), False
        if operation == "act":
            return self.act(payload), False
        if operation == "bot_act":
            return self.bot_act(payload), False
        if operation == "close":
            return self.close(), True
        _fault("unknown_request", f"unsupported request type {operation!r}")


class NdjsonServer:
    def __init__(
        self,
        session: GameSession,
        *,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
    ) -> None:
        self.session = session
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

    def emit(self, value: dict[str, Any]) -> None:
        line = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.stdout.write(line + "\n")
        self.stdout.flush()

    @staticmethod
    def _request_id(request: Any) -> str | int:
        if not isinstance(request, dict):
            _fault("invalid_request", "each NDJSON line must contain an object")
        value = request.get("request_id")
        if (
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or (isinstance(value, str) and not value)
        ):
            _fault("invalid_request", "request_id must be a string or integer")
        return value

    def run(self) -> int:
        for raw_line in self.stdin:
            if not raw_line.strip():
                continue
            request_id: str | int | None = None
            try:
                try:
                    request = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    _fault("invalid_json", f"invalid JSON: {exc.msg}")
                request_id = self._request_id(request)
                state, should_close = self.session.dispatch(request)
                self.emit({"request_id": request_id, "ok": True, "state": state})
                if should_close:
                    return 0
            except RpcFault as exc:
                self.emit(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": {"code": exc.code, "message": exc.message},
                    }
                )
            except Exception as exc:  # keep the RPC process diagnosable and alive
                traceback.print_exc(file=self.stderr)
                self.emit(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": {
                            "code": "internal_error",
                            "message": str(exc) or type(exc).__name__,
                        },
                    }
                )
        return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("guandan",), required=True)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    session = GameSession(args.game, args.mode)
    return NdjsonServer(session).run()


if __name__ == "__main__":
    raise SystemExit(main())
