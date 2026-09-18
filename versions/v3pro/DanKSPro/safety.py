"""Inference-only, same-strength wildcard/bomb asset gate; no hand-count feed mask."""
from collections import Counter
from dataclasses import asdict
import math
import numpy as np
from DanKS.retrieval.cards import normalize_card, card_rank, heart_level_card
from DanKS.retrieval.rules import is_bomb_kind, normalize_kind
from . import _guarded as legacy

CONTRACT = "same_strength_asset_pareto_mask_v1"

def asset_cost(hand, action, ctx):
    wild = heart_level_card(ctx.cur_rank)
    cards = [normalize_card(c) for c in action.cards]
    ranks = Counter(card_rank(normalize_card(c)) for c in hand if normalize_card(c) != wild)
    used = Counter(card_rank(c) for c in cards if c != wild)
    broken = sum(1 for r, n in ranks.items() if n >= 4 and 0 < used[r] < n)
    if is_bomb_kind(action.kind):
        broken = 0
    return (sum(c == wild for c in cards), broken)

def same_task(a, b):
    return (normalize_kind(a.kind), a.rank, a.size) == (normalize_kind(b.kind), b.rank, b.size)

def asset_safe_gate(hand, ctx, rows, mask):
    original = np.asarray(mask, dtype=np.float32).copy()
    trace = {"contract": CONTRACT, "status": "no_dominated_asset_action", "masked": [], "witnesses": []}
    active = [i for i in range(len(rows)) if original[i] > 0]
    if any(rows[i].action.size >= len(hand) for i in active):
        trace["status"] = "immediate_finish_preserve_mask"
        return original, trace
    try:
        counts = ctx.public_counts
        if len(counts) != 4 or int(counts[0]) != len(hand) or any(int(x) < 0 for x in counts):
            raise ValueError("incomplete public counts")
        # Short opponents are a veto, never a reason to blanket-mask singles/pairs.
        if any(0 < int(counts[i]) <= 2 for i in (1, 3)):
            trace["status"] = "opponent_short_preserve_mask"
            return original, trace
        costs = {i: asset_cost(hand, rows[i].action, ctx) for i in active}
        metrics = {}
        for i in active:
            d = rows[i].details
            required = ("my_retake_count", "break_group_penalty", "spend_penalty")
            if any(k not in d for k in required):
                raise ValueError("incomplete asset metrics")
            vetoes = ("must_block", "opponent_short_pressure", "partner_follow_help")
            if not all(math.isfinite(float(d.get(k, 0.))) for k in vetoes):
                raise ValueError("nonfinite safety veto")
            m = legacy._metrics(rows[i], ctx)
            if not all(math.isfinite(float(v)) for v in asdict(m).values()):
                raise ValueError("nonfinite metrics")
            metrics[i] = m
        edges = {}
        for i in active:
            a = rows[i]
            if costs[i] == (0, 0) or is_bomb_kind(a.action.kind):
                continue
            for j in active:
                if i == j or not same_task(a.action, rows[j].action):
                    continue
                if not (all(x <= y for x, y in zip(costs[j], costs[i])) and costs[j] != costs[i]):
                    continue
                if not legacy._dominance_blockers(a, rows[j], ctx, hand_size=len(hand), opportunity=None):
                    edges.setdefault(i, []).append(j)
        # Keep explicit surviving witnesses, rather than masking their witnesses too.
        safe = set(active) - set(edges)
        masked = [i for i in edges if any(j in safe for j in edges[i])]
        result = original.copy()
        for i in masked:
            result[i] = 0
            j = next(j for j in edges[i] if j in safe)
            trace["witnesses"].append({"masked_index": int(rows[i].action.index), "safe_index": int(rows[j].action.index), "cost_before": costs[i], "cost_after": costs[j]})
        if not np.any(result > 0):
            trace["status"] = "no_safe_candidate_preserve_mask"
            return original, trace
        trace.update(status="applied" if masked else trace["status"], masked=[int(rows[i].action.index) for i in masked])
        return result, trace
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError) as exc:
        trace.update(status="incomplete_preserve_mask", detail=str(exc))
        return original, trace
