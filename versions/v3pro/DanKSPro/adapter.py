"""Shared-engine public observations to V3 features; no private hand access."""
from __future__ import annotations

from DanKS.retrieval.cards import normalize_cards
from DanKS.retrieval.ranker import normalize_action


def policy_cards(cards):
    """Translate shared-engine joker labels at the policy boundary."""
    return normalize_cards([{"SB": "BJ", "HR": "RJ"}.get(str(card), str(card)) for card in cards])


def wire_actions(actions):
    """Convert engine wire moves to candidates with positional action IDs."""
    result = []
    for index, action in enumerate(actions):
        action = action.to_json() if hasattr(action, "to_json") else action
        if not isinstance(action, (tuple, list)) or len(action) != 3:
            raise ValueError("expected engine [kind, rank, cards] action")
        kind, rank, cards = action
        if str(kind).upper() == "PASS":
            cards = ()
        rank = {"B": "BJ", "R": "RJ"}.get(rank, rank)
        result.append(normalize_action(dict(index=index, kind=kind, rank=rank,
                                           cards=policy_cards(cards or ())), index))
    return result


def public_context(observation, history, actions):
    """Counts become relative; history and last-player positions stay absolute."""
    seat = int(observation["seat"])
    counts = list(observation["public_counts"])
    if seat not in range(4) or len(counts) != 4:
        raise ValueError("expected four public seat counts")
    events = []
    for item in history:
        if "curAction" in item:
            move = wire_actions([item["curAction"]])[0]
            events.append(dict(pos=int(item["curPos"]), kind=move.kind,
                               rank=move.rank, cards=list(move.cards)))
        elif "pos" in item and "cards" in item:
            events.append(dict(item, cards=policy_cards(item["cards"])))
        else:
            raise ValueError("history must contain public play events")
    target = observation.get("greater_action")
    # Absence of PASS in the engine legal set identifies regained lead even
    # when the last winning move remains in the public trick metadata.
    following = any(action.kind == "PASS" for action in actions) and target
    move = wire_actions([target])[0] if following else None
    context = dict(my_seat=seat, history_my_seat=seat, curRank=observation["rank"],
                   public_counts=[int(counts[(seat+i) % 4]) for i in range(4)],
                   current_kind=move.kind if move else "Lead",
                   current_rank=move.rank if move else None,
                   current_size=move.size if move else 0,
                   last_player=int(observation["greater_pos"]) if move else None,
                   played_cards=[card for event in events for card in event["cards"]],
                   history=events)
    return context, events
