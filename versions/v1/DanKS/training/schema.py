from __future__ import annotations

from DanKS.retrieval.cards import ALL_CARDS, RANKS


TOPK = 10
DEFAULT_CANDIDATE_CAPACITY = TOPK
STRUCTURED_TOPK = "structured_topk"
FULL_LEGAL = "full_legal"
ACTION_SUPPORTS = (STRUCTURED_TOPK, FULL_LEGAL)

ACTION_KINDS = (
    "PASS",
    "Single",
    "Pair",
    "Triple",
    "TriplePlus",
    "Straight",
    "StraightPair",
    "StraightTriple",
    "Bomb",
    "StraightFlush",
    "FourKings",
    "Unknown",
)

GROUP_KINDS = (
    "Single",
    "Pair",
    "Triple",
    "TriplePlus",
    "Straight",
    "StraightPair",
    "StraightTriple",
    "Bomb",
    "StraightFlush",
    "FourKings",
)

CARD_DIM = len(ALL_CARDS)
RANK_DIM = len(RANKS)
ACTION_KIND_DIM = len(ACTION_KINDS)
GROUP_KIND_DIM = len(GROUP_KINDS)

# state = hand card counts + current level rank + current trick kind/rank +
# current trick size + public hand counts + relative trick holder + lead flag +
# remaining-by-rank counts.
STATE_DIM = (
    CARD_DIM
    + RANK_DIM
    + ACTION_KIND_DIM
    + RANK_DIM
    + 1
    + 4
    + 5
    + 1
    + RANK_DIM
)

# candidate = action card counts + kind + rank + scalar action facts +
# retrieval score components + after-hand/partition summaries.
CANDIDATE_SCALAR_DIM = 26
CANDIDATE_DIM = CARD_DIM + ACTION_KIND_DIM + RANK_DIM + CANDIDATE_SCALAR_DIM + GROUP_KIND_DIM

FEATURE_VERSION = "top10_selector_v3_rank2fix"


def normalize_action_support(value: object, *, missing: str = STRUCTURED_TOPK) -> str:
    selected = missing if value is None else str(value)
    if selected not in ACTION_SUPPORTS:
        raise ValueError(f"action_support must be one of {ACTION_SUPPORTS}")
    return selected


def normalize_candidate_contract(
    candidate_capacity: object, action_support: object,
) -> tuple[int, str]:
    support = normalize_action_support(action_support)
    if isinstance(candidate_capacity, bool) or not isinstance(candidate_capacity, int):
        raise ValueError("candidate_capacity must be an integer")
    if support == FULL_LEGAL:
        if candidate_capacity != 0:
            raise ValueError("full_legal requires candidate_capacity=0 (dynamic all-legal width)")
    elif candidate_capacity <= 0:
        raise ValueError("structured_topk requires a positive candidate_capacity")
    return candidate_capacity, support
