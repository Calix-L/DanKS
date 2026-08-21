from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .cards import ALL_CARDS, CARD_INDEX, card_rank, normalize_cards
from .plm_rules import normalize_kind, normalize_rank


SEAT_COUNT = 4
PLAY_STAT_FIELDS = (
    "play_actions",
    "played_cards_count",
    "pass_actions",
    "straight_actions",
    "triple_plus_actions",
    "bomb_actions",
    "normal_bomb_actions",
    "bomb_4_actions",
    "bomb_5_actions",
    "bomb_6_actions",
    "bomb_7_actions",
    "bomb_8_actions",
    "bomb_9_actions",
    "bomb_10_actions",
    "straight_flush_actions",
    "four_kings_actions",
    "black_jokers",
    "red_jokers",
)
PLAY_STAT_SCALES = (
    27.0,
    27.0,
    40.0,
    5.0,
    5.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    4.0,
    1.0,
    2.0,
    2.0,
)
# PPO consumes only irreducible public summaries.  The complete statistics
# above remain available to diagnostics and response-evidence construction.
NETWORK_PLAY_STAT_FIELDS = (
    "play_actions",
    "pass_actions",
    "straight_actions",
    "triple_plus_actions",
    "bomb_4_actions",
    "bomb_5_actions",
    "bomb_6_actions",
    "bomb_7_actions",
    "bomb_8_actions",
    "bomb_9_actions",
    "bomb_10_actions",
    "straight_flush_actions",
    "four_kings_actions",
)
NETWORK_PLAY_STAT_SCALES = (
    27.0,
    40.0,
    5.0,
    5.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    7.0,
    4.0,
    1.0,
)
BOMB_KINDS = {"Bomb", "StraightFlush", "FourKings"}
IGNORED_ACTIONS = {"finish", "skip", "episodeover", "gameresult"}
RELATIVE_SEAT_NAMES = ("self", "left_opponent", "teammate", "right_opponent")


class CardMemoryIntegrityError(ValueError):
    """Raised when public card memory violates the two-deck contract."""


@dataclass(frozen=True)
class PlayedAction:
    seat: int
    kind: str
    rank: str | None
    cards: tuple[str, ...]


@dataclass(frozen=True)
class ResponseEvent:
    """One public opportunity where a seat passed or beat the current winner."""

    action_index: int
    seat: int
    passed: bool
    target_seat: int
    target_kind: str
    target_rank: str | None
    target_cards: tuple[str, ...]
    target_was_teammate: bool
    response_kind: str | None = None
    response_rank: str | None = None
    response_cards: tuple[str, ...] = ()

    @property
    def target_size(self) -> int:
        return len(self.target_cards)

    @property
    def response_size(self) -> int:
        return len(self.response_cards)


@dataclass(frozen=True)
class SeatPlayStats:
    seat: int
    played_cards: tuple[str, ...] = ()
    actions: tuple[PlayedAction, ...] = ()
    response_events: tuple[ResponseEvent, ...] = ()
    play_actions: int = 0
    played_cards_count: int = 0
    pass_actions: int = 0
    straight_actions: int = 0
    triple_plus_actions: int = 0
    bomb_actions: int = 0
    normal_bomb_actions: int = 0
    bomb_4_actions: int = 0
    bomb_5_actions: int = 0
    bomb_6_actions: int = 0
    bomb_7_actions: int = 0
    bomb_8_actions: int = 0
    bomb_9_actions: int = 0
    bomb_10_actions: int = 0
    straight_flush_actions: int = 0
    four_kings_actions: int = 0
    black_jokers: int = 0
    red_jokers: int = 0

    def stat_values(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, field)) for field in PLAY_STAT_FIELDS)

    def exact_counts(self) -> tuple[int, ...]:
        counts = Counter(self.played_cards)
        return tuple(int(counts.get(card, 0)) for card in ALL_CARDS)


@dataclass(frozen=True)
class CardMemory:
    seats: tuple[SeatPlayStats, SeatPlayStats, SeatPlayStats, SeatPlayStats]
    actions: tuple[PlayedAction, ...]
    remaining_exact: tuple[int, ...]
    complete: bool
    valid: bool = True
    errors: tuple[str, ...] = ()

    def remaining_detail(self) -> dict[str, int]:
        return {
            card: int(self.remaining_exact[index])
            for index, card in enumerate(ALL_CARDS)
        }

    def remaining_by_rank(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index, card in enumerate(ALL_CARDS):
            rank = card_rank(card)
            counts[rank] = counts.get(rank, 0) + int(self.remaining_exact[index])
        return counts

    def to_dict(self) -> dict[str, Any]:
        seat_rows: list[dict[str, Any]] = []
        for seat in self.seats:
            exact_counts = Counter(seat.played_cards)
            kind_counts = Counter(action.kind for action in seat.actions)
            seat_rows.append(
                {
                    "seat": seat.seat,
                    "relative_name": RELATIVE_SEAT_NAMES[seat.seat],
                    "played_cards": list(seat.played_cards),
                    "exact_card_counts": {
                        card: int(exact_counts[card])
                        for card in ALL_CARDS
                        if exact_counts[card] > 0
                    },
                    "kind_counts": dict(kind_counts),
                    "stats": {
                        field: int(getattr(seat, field))
                        for field in PLAY_STAT_FIELDS
                    },
                    "actions": [
                        {
                            "kind": action.kind,
                            "rank": action.rank,
                            "cards": list(action.cards),
                        }
                        for action in seat.actions
                    ],
                    "response_events": [
                        {
                            "action_index": event.action_index,
                            "passed": event.passed,
                            "target_seat": event.target_seat,
                            "target_kind": event.target_kind,
                            "target_rank": event.target_rank,
                            "target_cards": list(event.target_cards),
                            "target_was_teammate": event.target_was_teammate,
                            "response_kind": event.response_kind,
                            "response_rank": event.response_rank,
                            "response_cards": list(event.response_cards),
                        }
                        for event in seat.response_events
                    ],
                }
            )
        return {
            "complete": self.complete,
            "valid": self.valid,
            "errors": list(self.errors),
            "remaining_cards_detail": self.remaining_detail(),
            "seats": seat_rows,
        }


def _normalize_card_sequence(cards: Any, *, label: str) -> list[str]:
    if cards is None:
        return []
    if not isinstance(cards, (list, tuple)):
        raise CardMemoryIntegrityError(f"{label} must be a list or tuple")
    try:
        return normalize_cards(cards)
    except (TypeError, ValueError) as exc:
        raise CardMemoryIntegrityError(f"invalid card in {label}: {exc}") from exc


def _known_cards(state: dict[str, Any]) -> list[str]:
    known = state.get("known_hand_cards") or {}
    if not isinstance(known, dict):
        raise CardMemoryIntegrityError("known_hand_cards must be a dict")
    out: list[str] = []
    for seat, cards in known.items():
        out.extend(_normalize_card_sequence(cards, label=f"known_hand_cards[{seat!r}]"))
    return out


def _history_actions(
    state: dict[str, Any],
) -> tuple[list[PlayedAction], list[str], tuple[SeatPlayStats, SeatPlayStats, SeatPlayStats, SeatPlayStats]]:
    history = state.get("history") or []
    if not isinstance(history, (list, tuple)):
        raise CardMemoryIntegrityError("history must be a list or tuple")
    try:
        observer = int(state.get("history_my_seat", state.get("my_seat", 0)) or 0) % SEAT_COUNT
    except (TypeError, ValueError) as exc:
        raise CardMemoryIntegrityError("history_my_seat must be an integer seat") from exc

    action_rows: list[PlayedAction] = []
    history_cards: list[str] = []
    seat_cards: list[list[str]] = [[] for _ in range(SEAT_COUNT)]
    seat_actions: list[list[PlayedAction]] = [[] for _ in range(SEAT_COUNT)]
    seat_response_events: list[list[ResponseEvent]] = [[] for _ in range(SEAT_COUNT)]
    stats = [{field: 0 for field in PLAY_STAT_FIELDS} for _ in range(SEAT_COUNT)]
    current_target: PlayedAction | None = None

    for index, raw in enumerate(history):
        if not isinstance(raw, dict):
            continue
        wire_action = str(raw.get("action") or "").strip().lower()
        if wire_action in IGNORED_ACTIONS or bool(raw.get("finished", False)):
            continue
        raw_pos = raw.get("pos", raw.get("seat", -1))
        try:
            absolute_seat = int(raw_pos)
        except (TypeError, ValueError):
            continue
        if absolute_seat < 0 or absolute_seat >= SEAT_COUNT:
            continue
        seat = (absolute_seat - observer) % SEAT_COUNT
        cards = _normalize_card_sequence(raw.get("cards") or [], label=f"history[{index}].cards")
        raw_kind = raw.get("kind") or raw.get("action_kind")
        if not raw_kind:
            raw_kind = "PASS" if not cards else "Unknown"
        kind = normalize_kind(str(raw_kind))
        if kind in {"Pass", "pass"}:
            kind = "PASS"
        if not cards and wire_action == "pass":
            kind = "PASS"
        rank = normalize_rank(raw.get("rank"))
        action = PlayedAction(seat=seat, kind=kind, rank=rank, cards=tuple(cards))
        action_rows.append(action)
        seat_actions[seat].append(action)

        if kind == "PASS" or not cards:
            if current_target is not None and seat != current_target.seat:
                seat_response_events[seat].append(
                    ResponseEvent(
                        action_index=index,
                        seat=seat,
                        passed=True,
                        target_seat=current_target.seat,
                        target_kind=current_target.kind,
                        target_rank=current_target.rank,
                        target_cards=current_target.cards,
                        target_was_teammate=(seat % 2 == current_target.seat % 2),
                    )
                )
            stats[seat]["pass_actions"] += 1
            continue
        if current_target is not None and seat != current_target.seat:
            seat_response_events[seat].append(
                ResponseEvent(
                    action_index=index,
                    seat=seat,
                    passed=False,
                    target_seat=current_target.seat,
                    target_kind=current_target.kind,
                    target_rank=current_target.rank,
                    target_cards=current_target.cards,
                    target_was_teammate=(seat % 2 == current_target.seat % 2),
                    response_kind=action.kind,
                    response_rank=action.rank,
                    response_cards=action.cards,
                )
            )
        current_target = action
        stats[seat]["play_actions"] += 1
        stats[seat]["played_cards_count"] += len(cards)
        if kind == "Straight":
            stats[seat]["straight_actions"] += 1
        if kind == "TriplePlus":
            stats[seat]["triple_plus_actions"] += 1
        if kind in BOMB_KINDS:
            stats[seat]["bomb_actions"] += 1
        if kind == "Bomb":
            stats[seat]["normal_bomb_actions"] += 1
            if 4 <= len(cards) <= 10:
                stats[seat][f"bomb_{len(cards)}_actions"] += 1
        elif kind == "StraightFlush":
            stats[seat]["straight_flush_actions"] += 1
        elif kind == "FourKings":
            stats[seat]["four_kings_actions"] += 1
        stats[seat]["black_jokers"] += cards.count("BJ")
        stats[seat]["red_jokers"] += cards.count("RJ")
        history_cards.extend(cards)
        seat_cards[seat].extend(cards)

    seats = tuple(
        SeatPlayStats(
            seat=seat,
            played_cards=tuple(seat_cards[seat]),
            actions=tuple(seat_actions[seat]),
            response_events=tuple(seat_response_events[seat]),
            **stats[seat],
        )
        for seat in range(SEAT_COUNT)
    )
    return action_rows, history_cards, seats  # type: ignore[return-value]


def build_card_memory(state: dict[str, Any]) -> CardMemory:
    """Build one immutable, observer-relative snapshot of public card memory."""

    if not isinstance(state, dict):
        raise TypeError("state must be a dict")
    known_cards = _known_cards(state)
    actions, history_cards, seats = _history_actions(state)
    explicit_played = _normalize_card_sequence(
        state.get("played_cards") or [], label="played_cards"
    )
    complete = bool(state.get("history_is_complete", False))
    if complete and Counter(explicit_played) != Counter(history_cards):
        raise CardMemoryIntegrityError(
            "complete history cards do not match explicit played_cards"
        )
    selected_played = explicit_played if explicit_played else history_cards
    used = Counter(known_cards + selected_played)
    for card, count in sorted(used.items()):
        if count > 2:
            raise CardMemoryIntegrityError(
                f"card {card} appears {count} times, exceeds two-deck maximum 2"
            )
    remaining_exact = tuple(2 - int(used.get(card, 0)) for card in ALL_CARDS)
    return CardMemory(
        seats=seats,
        actions=tuple(actions),
        remaining_exact=remaining_exact,
        complete=complete,
    )
