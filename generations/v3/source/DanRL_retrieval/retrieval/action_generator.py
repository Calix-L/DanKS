from __future__ import annotations

from .context import RetrievalContext
from .models import ActionCandidate, CardGroup
from .partitioner import FullSearchPartitioner
from .plm_rules import can_follow_action, group_to_action, normalize_kind


class PLMActionGenerator:
    """Generate legal action candidates with PLM-style follow rules.

    This is a Python-side rule adapter inspired by `plm_guandan/common`.
    It enumerates legal groups from the full-search group universe, then
    filters them with PLM comparison rules. It does not use PLM's fixed splitter
    to decide one arrangement.
    """

    def __init__(self, partitioner: FullSearchPartitioner | None = None) -> None:
        self.partitioner = partitioner or FullSearchPartitioner()

    def generate(self, hand_cards: list[str], ctx: RetrievalContext, include_pass: bool = True) -> list[ActionCandidate]:
        groups = self.partitioner._generate_groups(hand_cards, ctx)
        actions: list[ActionCandidate] = []
        if include_pass and ctx.current_kind != "Lead":
            actions.append(ActionCandidate(index=0, kind="PASS", cards=(), rank="PASS"))
        target = self._target_action(ctx)
        seen = set()
        for group in groups:
            action = group_to_action(group, len(actions))
            key = (action.kind, action.rank, action.cards)
            if key in seen:
                continue
            if target is not None and not can_follow_action(action, target, ctx.cur_rank):
                continue
            seen.add(key)
            actions.append(action)
        for idx, action in enumerate(actions):
            actions[idx] = ActionCandidate(index=idx, kind=action.kind, cards=action.cards, rank=action.rank)
        return actions

    def _target_action(self, ctx: RetrievalContext) -> ActionCandidate | None:
        kind = normalize_kind(ctx.current_kind)
        if kind == "Lead":
            return None
        size = ctx.current_size
        if size <= 0:
            size = {
                "Single": 1,
                "Pair": 2,
                "Triple": 3,
                "Straight": 5,
                "TriplePlus": 5,
                "StraightPair": 6,
                "StraightTriple": 6,
                "Bomb": 4,
                "StraightFlush": 5,
                "FourKings": 4,
            }.get(kind, 0)
        cards = tuple(f"__{i}" for i in range(size))
        return ActionCandidate(index=-1, kind=kind, cards=cards, rank=ctx.current_rank)
