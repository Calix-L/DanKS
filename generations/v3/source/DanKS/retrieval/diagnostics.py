from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .cards import normalize_cards
from .context import RetrievalContext
from .models import ActionCandidate, CardGroup, Partition
from .partitioner import FullSearchPartitioner
from .scoring import group_value


HIGH_VALUE_KINDS = {
    "FourKings",
    "StraightFlush",
    "Bomb",
    "StraightTriple",
    "StraightPair",
    "TriplePlus",
    "Straight",
    "Triple",
    "Pair",
}


@dataclass(frozen=True)
class StructureOpportunity:
    group: CardGroup
    value: float
    cards: tuple[str, ...]


@dataclass(frozen=True)
class ActionStructureDiagnostic:
    action: ActionCandidate
    complete_groups: tuple[CardGroup, ...]
    broken_groups: tuple[CardGroup, ...]
    touched_groups: tuple[CardGroup, ...]
    touched_cards: dict[str, tuple[CardGroup, ...]]


def _card_counter(cards: tuple[str, ...]) -> Counter[str]:
    return Counter(cards)


def _overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum(min(count, right.get(card, 0)) for card, count in left.items())


def _same_cards(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return _card_counter(left) == _card_counter(right)


def _same_action_identity(action: ActionCandidate, group: CardGroup) -> bool:
    if action.kind != group.kind or not _same_cards(action.cards, group.cards):
        return False
    if action.rank is None or group.rank is None:
        return True
    return str(action.rank) == str(group.rank)


def structure_opportunities(
    hand_cards: list[str],
    ctx: RetrievalContext,
    partitioner: FullSearchPartitioner | None = None,
    min_kind: set[str] | None = None,
) -> list[StructureOpportunity]:
    """Return all generated high-value groups, independent of one partition.

    This is intentionally not an exact-cover result. It answers: "which
    structural identities can each physical card participate in?"
    """

    partitioner = partitioner or FullSearchPartitioner()
    kinds = min_kind or HIGH_VALUE_KINDS
    groups = partitioner._generate_groups(normalize_cards(hand_cards), ctx)
    out = [
        StructureOpportunity(group=group, value=group_value(group), cards=group.cards)
        for group in groups
        if group.kind in kinds
    ]
    out.sort(key=lambda item: (-item.value, item.group.kind, item.group.rank or "", item.group.cards))
    return out


def card_opportunity_map(opportunities: list[StructureOpportunity]) -> dict[str, tuple[CardGroup, ...]]:
    by_card: dict[str, list[CardGroup]] = defaultdict(list)
    for opportunity in opportunities:
        for card in set(opportunity.cards):
            by_card[card].append(opportunity.group)
    return {
        card: tuple(sorted(groups, key=lambda group: (-group_value(group), group.kind, group.rank or "", group.cards)))
        for card, groups in by_card.items()
    }


def action_structure_diagnostic(
    action: ActionCandidate,
    opportunities: list[StructureOpportunity],
) -> ActionStructureDiagnostic:
    if action.kind == "PASS" or not action.cards:
        return ActionStructureDiagnostic(action, (), (), (), {})
    action_counter = _card_counter(action.cards)
    touched: list[CardGroup] = []
    complete: list[CardGroup] = []
    broken: list[CardGroup] = []
    touched_cards: dict[str, list[CardGroup]] = defaultdict(list)
    for opportunity in opportunities:
        group = opportunity.group
        group_counter = _card_counter(group.cards)
        overlap = _overlap_count(action_counter, group_counter)
        if overlap <= 0:
            continue
        touched.append(group)
        for card in set(action.cards):
            if group_counter.get(card, 0):
                touched_cards[card].append(group)
        if _same_action_identity(action, group):
            complete.append(group)
        elif group.kind in HIGH_VALUE_KINDS:
            broken.append(group)

    key = lambda group: (-group_value(group), group.kind, group.rank or "", group.cards)
    return ActionStructureDiagnostic(
        action=action,
        complete_groups=tuple(sorted(complete, key=key)),
        broken_groups=tuple(sorted(broken, key=key)),
        touched_groups=tuple(sorted(touched, key=key)),
        touched_cards={
            card: tuple(sorted(groups, key=key))
            for card, groups in sorted(touched_cards.items())
        },
    )


def format_group(group: CardGroup) -> str:
    wild = group.meta.get("wild_as")
    suffix = f" wild={','.join(wild)}" if wild else ""
    return f"{group.kind}:{group.rank}:{','.join(group.cards)}{suffix}"


def format_partition(partition: Partition, max_groups: int = 12) -> str:
    groups = list(partition.groups)
    shown = groups[:max_groups]
    text = " | ".join(format_group(group) for group in shown)
    if len(groups) > max_groups:
        text += f" | ...(+{len(groups) - max_groups})"
    return text or "EMPTY"


def format_action_diagnostic(diagnostic: ActionStructureDiagnostic, max_groups: int = 4) -> str:
    complete = ", ".join(format_group(group) for group in diagnostic.complete_groups[:max_groups]) or "-"
    broken = ", ".join(format_group(group) for group in diagnostic.broken_groups[:max_groups]) or "-"
    touched = ", ".join(format_group(group) for group in diagnostic.touched_groups[:max_groups]) or "-"
    return f"complete=[{complete}] broken=[{broken}] touched=[{touched}]"
