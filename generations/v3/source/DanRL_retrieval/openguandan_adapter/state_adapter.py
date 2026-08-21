from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from DanRL_retrieval.retrieval.cards import CARD_INDEX, card_sort_key
from DanRL_retrieval.retrieval.models import ActionCandidate
from DanRL_retrieval.retrieval.plm_rules import (
    PLM_KIND,
    RANK_TO_VALUE,
    normalize_kind,
    normalize_rank,
)


_CANONICAL_KINDS = PLM_KIND | {"Lead", "PASS"}


def ogd_card_to_retrieval(card: Any) -> str:
    if isinstance(card, str):
        if card in CARD_INDEX:
            return card
        if card in {"SB", "B"}:
            return "BJ"
        if card in {"HR", "R"}:
            return "RJ"
    text = str(card).strip().upper().replace("10", "T")
    if text in {"SB", "B"}:
        return "BJ"
    if text in {"HR", "R"}:
        return "RJ"
    return text


def retrieval_card_to_ogd(card: str) -> str:
    if card == "BJ":
        return "SB"
    if card == "RJ":
        return "HR"
    return card


def ogd_cards_to_retrieval(cards: Any) -> list[str]:
    if not cards:
        return []
    if isinstance(cards, str):
        if cards.upper() == "PASS":
            return []
        return [ogd_card_to_retrieval(cards)]
    return [ogd_card_to_retrieval(card) for card in cards]


def ogd_rank_to_retrieval(rank: Any) -> str | None:
    if rank is None:
        return None
    if isinstance(rank, str) and rank in RANK_TO_VALUE:
        return rank
    return normalize_rank(str(rank))


def ogd_kind_to_retrieval(kind: Any) -> str:
    if isinstance(kind, str) and kind in _CANONICAL_KINDS:
        return kind
    return normalize_kind(str(kind or "PASS"))


def clean_action_entry(entry: list[Any] | tuple[Any, ...], index: int) -> dict[str, Any]:
    kind = ogd_kind_to_retrieval(entry[0] if len(entry) > 0 else "PASS")
    cards = ogd_cards_to_retrieval(entry[2] if len(entry) > 2 else [])
    rank = ogd_rank_to_retrieval(entry[1] if len(entry) > 1 else None)
    return {
        "index": index,
        "kind": kind,
        "rank": "PASS" if kind == "PASS" else rank,
        "cards": cards,
    }


def clean_action_list(action_list: list[Any]) -> list[dict[str, Any]]:
    return [clean_action_entry(action, idx) for idx, action in enumerate(action_list or [])]


def clean_action_candidates(action_list: list[Any]) -> list[ActionCandidate]:
    """Convert the official wire list directly to normalized ranker actions."""

    actions: list[ActionCandidate] = []
    for index, entry in enumerate(action_list or []):
        kind = ogd_kind_to_retrieval(entry[0] if len(entry) > 0 else "PASS")
        raw_cards = entry[2] if len(entry) > 2 else []
        if not raw_cards or kind == "PASS":
            cards = ()
        elif isinstance(raw_cards, str):
            cards = (ogd_card_to_retrieval(raw_cards),)
        else:
            cards = tuple(ogd_card_to_retrieval(card) for card in raw_cards)
        rank = ogd_rank_to_retrieval(entry[1] if len(entry) > 1 else None)
        actions.append(
            ActionCandidate(
                index=index,
                kind=kind,
                rank="PASS" if kind == "PASS" else rank,
                cards=cards,
            )
        )
    return actions


def _greater_action(body: dict[str, Any]) -> tuple[str, str | None, list[str]] | None:
    greater = body.get("greaterAction")
    if not greater or greater == [None, None, None]:
        return None
    kind = ogd_kind_to_retrieval(greater[0] if len(greater) > 0 else None)
    if kind == "PASS":
        return None
    cards = ogd_cards_to_retrieval(greater[2] if len(greater) > 2 else [])
    if not cards:
        return None
    return kind, ogd_rank_to_retrieval(greater[1] if len(greater) > 1 else None), cards


@dataclass
class OpenGuanDanTableTracker:
    """Small public-state tracker for retrieval self-play.

    The OpenGuanDan act body already contains the current hand, legal actions,
    public hand counts, and greater action.  This tracker only keeps played
    cards/history so retrieval can estimate remaining controls.
    """

    played_cards: list[str] = field(default_factory=list)
    play_history: list[dict[str, Any]] = field(default_factory=list)
    episode_index: int = 0

    def reset_episode(self) -> None:
        self.played_cards.clear()
        self.play_history.clear()
        self.episode_index += 1

    def observe_play(self, body: dict[str, Any]) -> None:
        action = body.get("curAction") or []
        cards = ogd_cards_to_retrieval(action[2] if len(action) > 2 else [])
        pos = int(body.get("curPos", -1))
        if cards:
            self.played_cards.extend(cards)
        self.play_history.append(
            {
                "pos": pos,
                "kind": ogd_kind_to_retrieval(action[0] if len(action) > 0 else "PASS"),
                "rank": ogd_rank_to_retrieval(action[1] if len(action) > 1 else None),
                "cards": cards,
                "finished": False,
            }
        )


def context_from_act_body(body: dict[str, Any], seat: int, tracker: OpenGuanDanTableTracker) -> tuple[list[str], dict[str, Any]]:
    hand = sorted(ogd_cards_to_retrieval(body.get("handCards") or []), key=card_sort_key)
    greater = _greater_action(body)
    public = body.get("publicInfo") or []
    counts_abs = [0, 0, 0, 0]
    for idx, item in enumerate(public[:4]):
        try:
            counts_abs[idx] = int(item.get("rest", 0))
        except (AttributeError, TypeError, ValueError):
            counts_abs[idx] = 0
    if 0 <= seat < 4:
        counts_abs[seat] = len(hand)
    public_counts = [counts_abs[(seat + offset) % 4] for offset in range(4)]
    if greater is None:
        current_kind = "Lead"
        current_rank = None
        current_size = 0
    else:
        current_kind, current_rank, cards = greater
        current_size = len(cards)
    greater_pos = body.get("greaterPos")
    last_player = None
    try:
        greater_pos_int = int(greater_pos)
        if greater_pos_int >= 0:
            last_player = (greater_pos_int - seat) % 4
    except (TypeError, ValueError):
        last_player = None
    ctx = {
        "curRank": ogd_rank_to_retrieval(body.get("curRank")) or "2",
        "my_seat": 0,
        "public_counts": public_counts,
        "current_kind": current_kind,
        "current_rank": current_rank,
        "current_size": current_size,
        "last_player": last_player,
        "known_hand_cards": {"0": hand},
        # Retrieval consumes played cards as a multiset. build_context copies
        # this list to an immutable tuple before the tracker can change again.
        "played_cards": tracker.played_cards,
        "history": tracker.play_history,
        "history_my_seat": seat,
        "history_is_complete": True,
    }
    return hand, ctx


def reward_by_order(order: list[int], seat: int) -> float:
    """Team reward shaped by finish order, matching the DanRL pattern scale."""

    team = {seat, (seat + 2) % 4}
    pattern = "".join("1" if int(pos) in team else "0" for pos in order)
    rewards = {"1100": 3.0, "1010": 2.0, "1001": 1.0, "0110": -1.0, "0101": -2.0, "0011": -3.0}
    return rewards.get(pattern, 0.0)


def choose_tribute_action(body: dict[str, Any]) -> int:
    rank_card = "H" + str(body.get("curRank", "2"))
    for idx, action in enumerate(body.get("actionList") or []):
        cards = action[2] if len(action) > 2 else []
        if rank_card in cards:
            return idx
    return 0


def choose_back_action(body: dict[str, Any]) -> int:
    """Return a legal tribute-back action.

    First version is conservative: prefer the first legal return action.  Back
    phases are rare and not part of PPO top10 training yet.
    """

    return 0


def action_signature(action: dict[str, Any]) -> tuple[str, str | None, tuple[str, ...]]:
    counts = Counter(action.get("cards") or [])
    cards: list[str] = []
    for card in sorted(counts, key=card_sort_key):
        cards.extend([card] * counts[card])
    return str(action.get("kind")), action.get("rank"), tuple(cards)
