from __future__ import annotations

import hashlib

from DanKS.retrieval.cards import ALL_CARDS, RANKS


TOPK = 10
MAX_HAND_CARDS = 27
MAX_ACTION_CARDS = 10
DEFAULT_CANDIDATE_CAPACITY = TOPK
STRUCTURED_TOPK = "structured_topk"
FULL_LEGAL = "full_legal"
ACTION_SUPPORTS = (STRUCTURED_TOPK, FULL_LEGAL)
SELECTOR_INFERENCE_K_LEVELS = (1, 3, 5, TOPK)
HISTORY_LENGTH = 15

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
# cards + relative seat + pass/finish + action/remaining size + team/valid
HISTORY_EVENT_DIM = CARD_DIM + 4 + 2 + 2 + 2
HISTORY_PROTOCOL = "public_action_history_v1_15x64"
# History contains one canonical event for each real public play/pass action.
# Engine bookkeeping such as per-seat broadcasts, finish and skipped-seat
# notifications is deliberately excluded.
HISTORY_EVENT_SEMANTICS = "actual_public_actions_deduplicated_v1"

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


def derive_uniform_action_seed(action_seed: int, deal_seed: int, side: int) -> int:
    """Derive a stable per-game RNG seed for uniform action selection.

    ``deal_seed`` is intentionally shared by every action-seed replicate so the
    cards remain paired.  ``action_seed`` must therefore participate in a
    separate derivation; otherwise the nominal 2026/2027/2028 replicates replay
    the same action trajectory.  Including ``side`` keeps the two partnership
    assignments independent while remaining exactly reproducible.
    """

    values = {
        "action_seed": action_seed,
        "deal_seed": deal_seed,
        "side": side,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if action_seed < 0 or deal_seed < 0:
        raise ValueError("action_seed and deal_seed must be nonnegative")
    if side not in {0, 1}:
        raise ValueError("side must be 0 or 1")
    material = f"danks-uniform-action-v1:{action_seed}:{deal_seed}:{side}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


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


def normalize_selector_inference_k(
    value: object,
    *,
    candidate_capacity: int,
    action_support: str,
    selection_mode: str,
    deterministic: bool,
) -> int | None:
    """Validate the evaluation-only prefix mask for one frozen Top-10 selector.

    This is deliberately separate from ``candidate_capacity``. The checkpoint,
    retrieval call, candidate order, tensor shape and candidate features all
    remain Top-10. The only changed model input is the valid-candidate mask:
    ``effective_mask = original_mask AND slot < K``.
    It is an inference diagnostic, not a substitute for independently trained
    K>1 policies in the paper's candidate-scale experiment.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("selector_inference_k must be an integer")
    if value not in SELECTOR_INFERENCE_K_LEVELS:
        raise ValueError(
            f"selector_inference_k must be one of {SELECTOR_INFERENCE_K_LEVELS}"
        )
    if candidate_capacity != TOPK or action_support != STRUCTURED_TOPK:
        raise ValueError(
            "selector_inference_k requires a structured_topk K10 checkpoint contract"
        )
    if selection_mode != "selector":
        raise ValueError("selector_inference_k requires selection_mode=selector")
    if not deterministic:
        raise ValueError("selector_inference_k is deterministic evaluation only")
    return value
