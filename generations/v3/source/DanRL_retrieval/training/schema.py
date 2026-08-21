from __future__ import annotations

import hashlib

from DanRL_retrieval.retrieval.card_memory import NETWORK_PLAY_STAT_FIELDS
from DanRL_retrieval.retrieval.cards import ALL_CARDS, NORMAL_RANKS, RANKS


TOPK = 10
MAX_HAND_CARDS = 27
MAX_ACTION_CARDS = 10
DEFAULT_CANDIDATE_CAPACITY = TOPK
STRUCTURED_TOPK = "structured_topk"
FULL_LEGAL = "full_legal"
ACTION_SUPPORTS = (STRUCTURED_TOPK, FULL_LEGAL)
SELECTOR_INFERENCE_K_LEVELS = (1, 3, 5, TOPK)
HISTORY_LENGTH = 15
CARD_MEMORY_SEATS = 4
CARD_MEMORY_STAT_FIELD_COUNT = len(NETWORK_PLAY_STAT_FIELDS)

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
)

# A table target is never PASS. Unknown is the explicit no-target/lead state.
TRICK_KINDS = ACTION_KINDS[1:] + ("Unknown",)
LEVEL_RANKS = NORMAL_RANKS

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
LEVEL_RANK_DIM = len(LEVEL_RANKS)
ACTION_KIND_DIM = len(ACTION_KINDS)
TRICK_KIND_DIM = len(TRICK_KINDS)
GROUP_KIND_DIM = len(GROUP_KINDS)
LAST_PLAYER_DIM = 4
# cards + relative seat + pass + remaining size + valid.  Finish was never
# emitted by the rollout tracker; action size and team are exact functions of
# the card/seat fields and are deliberately omitted in the compact schema.
HISTORY_EVENT_DIM = CARD_DIM + 4 + 1 + 1 + 1
HISTORY_PROTOCOL = "public_action_history_v2_15x61"
# History contains one canonical event for each real public play/pass action.
# Engine bookkeeping such as per-seat broadcasts, finish and skipped-seat
# notifications is deliberately excluded.
HISTORY_EVENT_SEMANTICS = "actual_public_actions_compact_v2"

# state = hand card counts + current level rank + current trick kind/rank +
# current trick size + public hand counts + relative trick holder +
# remaining-by-rank counts + high-resolution card memory.
CARD_MEMORY_REMAINING_DIM = CARD_DIM
CARD_MEMORY_PLAYED_EXACT_DIM = CARD_MEMORY_SEATS * CARD_DIM
CARD_MEMORY_STAT_DIM = CARD_MEMORY_SEATS * CARD_MEMORY_STAT_FIELD_COUNT
CARD_MEMORY_DIM = (
    CARD_MEMORY_REMAINING_DIM
    + CARD_MEMORY_PLAYED_EXACT_DIM
    + CARD_MEMORY_STAT_DIM
)
PRESSURE_STATE_FIELDS = (
    "opponent_control_actions",
    "team_failed_responses",
    "current_kind_pressure",
    "opponent_cards_since_team_control",
    "last_team_retake_was_bomb",
)
PRESSURE_STATE_DIM = len(PRESSURE_STATE_FIELDS)
STATE_DIM = (
    CARD_DIM
    + LEVEL_RANK_DIM
    + TRICK_KIND_DIM
    + RANK_DIM
    + 1
    + 4
    + LAST_PLAYER_DIM
    + RANK_DIM
    + CARD_MEMORY_DIM
    + PRESSURE_STATE_DIM
)

# candidate = action card counts + kind + rank + selected retrieval facts +
# response evidence + a set-level bomb-response fact + partition summaries.
CANDIDATE_BASE_SCALAR_FIELDS = (
    "slot_rank",
    "retrieval_score",
    "card_value",
    "retake_score",
    "current_control",
    "lead_action",
    "spend_penalty",
    "break_group",
    "low_break_preference",
    "escape_risk",
    "pass_pressure",
    "teammate_overcall",
    "my_min_steps",
    "opponent_short_pressure",
    "my_retake_count",
    "partner_follow_help",
    "must_block",
    "dynamic_strength",
)
CANDIDATE_RESPONSE_SCALAR_FIELDS = (
    "left_response_risk",
    "teammate_response_risk",
    "right_response_risk",
    "left_pass_evidence",
    "teammate_pass_evidence",
    "right_pass_evidence",
    "left_positive_response_evidence",
    "teammate_positive_response_evidence",
    "right_positive_response_evidence",
    "left_spent_evidence",
    "teammate_spent_evidence",
    "right_spent_evidence",
)
CANDIDATE_TACTICAL_SCALAR_FIELDS = (
    "only_bomb_response",
)
# V9+ removes deterministic/coarse candidate facts. Team special-pattern
# yielding and bomb interruption are reconstructed inside the network.
CANDIDATE_BASE_SCALAR_DIM = len(CANDIDATE_BASE_SCALAR_FIELDS)
CANDIDATE_SCALAR_OFFSET = CARD_DIM + ACTION_KIND_DIM + RANK_DIM
CANDIDATE_RESPONSE_SCALAR_OFFSET = (
    CANDIDATE_SCALAR_OFFSET + CANDIDATE_BASE_SCALAR_DIM
)
CANDIDATE_TACTICAL_SCALAR_OFFSET = (
    CANDIDATE_RESPONSE_SCALAR_OFFSET
    + len(CANDIDATE_RESPONSE_SCALAR_FIELDS)
)
CANDIDATE_SCALAR_DIM = (
    CANDIDATE_BASE_SCALAR_DIM
    + len(CANDIDATE_RESPONSE_SCALAR_FIELDS)
    + len(CANDIDATE_TACTICAL_SCALAR_FIELDS)
)
CANDIDATE_DIM = CARD_DIM + ACTION_KIND_DIM + RANK_DIM + CANDIDATE_SCALAR_DIM + GROUP_KIND_DIM

FEATURE_VERSION = "top10_selector_v12_calibrated_tactics1"

TEAM_BELIEF_RELATIVE_SEATS = (1, 2, 3)
TEAM_BELIEF_TARGET_NAMES = (
    "non_bomb_response",
    "bomb_response",
    "finishing_response",
    "low_cost_response",
    "bomb_preserving_response",
    "control_preserving_response",
    "five_card_response",
)
TEAM_BELIEF_SEAT_COUNT = len(TEAM_BELIEF_RELATIVE_SEATS)
TEAM_BELIEF_TARGET_DIM = len(TEAM_BELIEF_TARGET_NAMES)
TEAM_BELIEF_POS_WEIGHTS = (
    1.0,
    4.0,
    12.0,
    1.5,
    1.0,
    1.0,
    6.0,
)
TEAM_TACTICAL_INTERACTION_DIM = 4
TEAM_BELIEF_PROTOCOL = "candidate_hidden_response_quality_v3"
STATE_PUBLIC_COUNTS_OFFSET = CARD_DIM + LEVEL_RANK_DIM + TRICK_KIND_DIM + RANK_DIM + 1
STATE_TRICK_KIND_OFFSET = CARD_DIM + LEVEL_RANK_DIM
STATE_LAST_PLAYER_OFFSET = STATE_PUBLIC_COUNTS_OFFSET + 4
STATE_CARD_MEMORY_OFFSET = STATE_PUBLIC_COUNTS_OFFSET + 4 + LAST_PLAYER_DIM + RANK_DIM
STATE_PRESSURE_OFFSET = STATE_CARD_MEMORY_OFFSET + CARD_MEMORY_DIM
CARD_MEMORY_PLAYED_EXACT_OFFSET = CARD_MEMORY_REMAINING_DIM
CARD_MEMORY_STAT_OFFSET = CARD_MEMORY_REMAINING_DIM + CARD_MEMORY_PLAYED_EXACT_DIM
TEAM_BELIEF_PUBLIC_SEAT_DIM = CARD_DIM + CARD_MEMORY_STAT_FIELD_COUNT + 3


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
    material = f"cardks-uniform-action-v1:{action_seed}:{deal_seed}:{side}".encode("ascii")
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
