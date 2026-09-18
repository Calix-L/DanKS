"""Compose the unchanged V3 network with conservative inference-only modules.

Action IDs are caller IDs, never candidate slots. Retrieval context counts are
relative (self, next, partner, previous); history positions are absolute seats.
The policy owns its models; instances should not be used concurrently.
"""
from __future__ import annotations

import hashlib
import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import DanKS
from DanKS.retrieval.context import RetrievalContext, build_context
from DanKS.retrieval.card_memory import build_card_memory
from DanKS.retrieval.ranker import StructuralCandidateRanker, normalize_action
from DanKS.retrieval.scoring import BREAK_PENALTY_PROFILES, ScoreWeights
from DanKS.training.featurizer import featurize_topk, history_features
from DanKS.training.model import build_selector_from_checkpoint
from DanKS.training.schema import TOPK

from .rules import decide, lower_pair_legal_actions
from .safety import asset_cost, asset_safe_gate
from .adapter import public_context, wire_actions, policy_cards


def _decision_context(ctx):
    return (ctx.my_seat, ctx.cur_rank, tuple(ctx.public_counts), ctx.current_kind,
            ctx.current_rank, ctx.current_size, ctx.last_player)


class ProPolicy:
    """Frozen V3 plus asset gate, equivalent substitutions and optional experts.

    Use ``act`` for ordinary inference and ``refine_endgame`` for a reconstructed
    public search state. With no expert checkpoints, ordinary inference works
    but learned endgame proposals are explicitly unavailable.
    """

    def __init__(self, model, *, ranker=None, specialist=None, extended_specialist=None,
                 device="cpu", enabled=True):
        if DanKS.GENERATION != "v3":
            raise RuntimeError("V3Pro requires the danks-v3 package, not V1/V2")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.specialist = None if specialist is None else specialist.to(self.device).eval()
        self.extended_specialist = (None if extended_specialist is None
                                    else extended_specialist.to(self.device).eval())
        self.enabled = bool(enabled)
        self.ranker = ranker if ranker is not None else StructuralCandidateRanker(
            max_partitions=8, exact_best_cache_size=0,
            break_profile=BREAK_PENALTY_PROFILES["asset-tier-v2"],
            weights=ScoreWeights(break_group=46.))
        self._endgame = None

    @classmethod
    def from_checkpoints(cls, checkpoint, *, specialist=None, extended_specialist=None,
                         device="cpu", trusted=False, **kwargs):
        """Load caller-owned weights; pickle loading requires explicit trust.

        Each expert's parent SHA must match the supplied main checkpoint. No
        checkpoint paths, downloads, or private weights are built into this package.
        """
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = torch.load(path, map_location="cpu", weights_only=not trusted)
        model = build_selector_from_checkpoint(payload, device=device)
        experts = []
        for name in (specialist, extended_specialist):
            if name is None:
                experts.append(None)
                continue
            value = torch.load(name, map_location="cpu", weights_only=not trusted)
            parent = value.get("endgame_exact_distillation", {}).get("input_checkpoint_sha256")
            if parent != digest:
                raise ValueError("endgame expert parent checkpoint identity mismatch")
            experts.append(build_selector_from_checkpoint(value, device=device))
        return cls(model, specialist=experts[0], extended_specialist=experts[1],
                   device=device, **kwargs)

    @torch.inference_mode()
    def _logits(self, model, record):
        inputs = [torch.as_tensor(record[key], dtype=torch.float32, device=self.device)[None]
                  for key in ("state", "candidates", "mask", "history")]
        output = model(*inputs)
        values = (output[0] if isinstance(output, tuple) else output)[0].detach().cpu().numpy()
        if values.shape != (TOPK,) or not np.all(np.isfinite(values[record["mask"] > 0])):
            raise ValueError("invalid model logits on active candidates")
        return values

    def act(self, hand, context, legal_actions, *, history=None, enhanced=None):
        """Return ``(caller_action_id, record)``; input legal actions are authoritative.

        ``enhanced=False`` is the unmodified-policy continuation entry. The same
        model, ranker, features and candidate order are used without postprocessors.
        """
        enabled = self.enabled if enhanced is None else bool(enhanced)
        if not hand or not legal_actions:
            raise ValueError("nonempty hand and legal actions required")
        events = copy.deepcopy(history if history is not None else (
            context.get("history", ()) if isinstance(context, dict) else ()))
        if isinstance(context, RetrievalContext):
            ctx = context
            memory = build_card_memory(dict(my_seat=ctx.my_seat, history=events,
                                            played_cards=list(ctx.played_cards)))
            if ctx.card_memory is None:
                ctx = replace(ctx, card_memory=memory,
                              remaining_detail=memory.remaining_detail(),
                              remaining_by_rank=memory.remaining_by_rank())
            elif any(
                    getattr(ctx.card_memory, key) != getattr(memory, key)
                    for key in ("actions", "seats", "remaining_exact")):
                raise ValueError("prebuilt context does not match supplied public history")
        else:
            # One history source must drive retrieval, state and history features.
            ctx = build_context(dict(context, history=events))
        if ctx.known_hand_cards:
            raise ValueError("private opponent hands are not accepted")
        actions = [normalize_action(action, i) for i, action in enumerate(legal_actions)]
        ids = [int(action.index) for action in actions]
        if len(ids) != len(set(ids)):
            raise ValueError("legal action indices must be unique")
        rows = self.ranker.rank(list(hand), actions, ctx, top_k=TOPK)
        if not rows or any(row.action.index not in ids for row in rows):
            raise ValueError("retrieval must retain legal candidate IDs")
        state, candidates, original_mask = featurize_topk(list(hand), ctx, rows)
        mask, gate = (asset_safe_gate(hand, ctx, rows, original_mask) if enabled else
                      (original_mask, dict(status="disabled", masked=[])))
        record = dict(state=state, candidates=candidates, mask=mask,
                      history=history_features(events, my_seat=ctx.my_seat),
                      public_history=tuple(events), actor_hand=tuple(hand),
                      decision_context=_decision_context(ctx),
                      ordered_candidate_indices=[int(row.action.index) for row in rows],
                      legal_actions=actions, training_eligible=False)
        logits = self._logits(self.model, record)
        slot = int(np.argmax(np.where(mask > 0, logits, -np.inf)))
        ppo = int(rows[slot].action.index)
        record.update(experience_logits=logits, action_slot=slot)
        final = ppo
        trace = dict(safe_gate=gate, ppo_action=ppo, rules_action=ppo,
                     final_action=ppo, endgame=dict(status="not_requested"))
        if enabled:
            allowed = [row for i, row in enumerate(rows) if mask[i] > 0]
            selected_slot = next(i for i, row in enumerate(allowed) if row.action.index == ppo)
            try:
                lower = lower_pair_legal_actions(rows[slot].action, actions, ctx)
                cost = asset_cost(hand, rows[slot].action, ctx)
                lower = [a for a in lower if a.index not in gate["masked"] and
                         all(x <= y for x, y in zip(asset_cost(hand, a, ctx), cost))]
                extra = self.ranker.rank(list(hand), lower, ctx, top_k=None) if lower else ()
                choice = decide(hand, ctx, allowed, selected_slot, equivalent_rows=extra)
                if choice.final_action_index not in ids or choice.final_action_index in gate["masked"]:
                    raise ValueError("rule selected an illegal or blocked action")
                final = int(choice.final_action_index)
                trace["same_type"] = choice.trace
            except (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError) as exc:
                trace["rule_error"] = str(exc)
        trace.update(rules_action=final, final_action=final)
        record["trace"] = trace
        return final, record

    def propose_endgame(self, record, *, total_remaining, baseline_index):
        """Rank specialist proposals only; this never authorizes an action change."""
        if not self.enabled or total_remaining > 16:
            return ()
        expert = self.specialist if total_remaining <= 10 else self.extended_specialist
        if expert is None:
            return ()
        logits = self._logits(expert, record)
        parent = record["experience_logits"]
        slot = int(record["action_slot"])
        floor, cap = (-1., 3) if total_remaining <= 10 else (-20., 5)
        scores = []
        blocked = set(record["trace"]["safe_gate"]["masked"])
        for i, index in enumerate(record["ordered_candidate_indices"]):
            if record["mask"][i] <= 0 or index == baseline_index or index in blocked:
                continue
            score = float(logits[i] - logits[slot] - (parent[slot] - parent[i]))
            if np.isfinite(score) and score >= floor:
                scores.append((score, float(logits[i]), -i, int(index)))
        ordered = [item[-1] for item in sorted(scores, reverse=True)]
        # Preserve the original confidence-gated expert argmax as the first
        # proposal, before filling from the joint expert/parent score ranking.
        valid = np.asarray(record["mask"]) > 0
        expert_slot = int(np.argmax(np.where(valid, logits, -np.inf)))
        index = int(record["ordered_candidate_indices"][expert_slot])
        confidence = float(logits[expert_slot] - logits[slot] - (parent[slot] - parent[expert_slot]))
        if index != baseline_index and index not in blocked and confidence >= 1.8511199951171875:
            ordered = [index] + [value for value in ordered if value != index]
        return tuple(ordered[:cap])

    def reference_action(self, observation, legal_wire_actions, history):
        """Unenhanced deterministic V3 callback for hypothetical continuations."""
        actions = wire_actions(legal_wire_actions)
        context, events = public_context(observation, history, actions)
        return self.act(policy_cards(observation["hand"]), context, actions,
                        history=events, enhanced=False)[0]

    def refine_endgame(self, env, *, actor_seat, actor_hand, played_cards,
                       public_counts, record):
        """Refine a previous ``act`` result using a reconstructed public state.

        Counts here are ABSOLUTE seats, unlike Retrieval's relative counts.
        The caller must reconstruct the complete public root and supply all
        played cards. Never provide opponent hands to the policy. Candidate IDs
        are mapped to engine positions only after matching the entire legal set.
        This API is node-budgeted, not a hard wall-clock deadline service.
        """
        baseline = int(record["trace"]["rules_action"])
        total = sum(public_counts)
        expert = self.specialist if total <= 10 else self.extended_specialist
        if not self.enabled or total > 16 or expert is None:
            reason = ("disabled" if not self.enabled else "above_threshold"
                      if total > 16 else "expert_unavailable")
            record["trace"]["endgame"] = dict(reason=reason, applied=False)
            return baseline
        from .endgame.runtime import EndgameRefiner, FrozenPolicyContinuation, public_observation
        try:
            if Counter(policy_cards(actor_hand)) != Counter(policy_cards(record["actor_hand"])):
                raise ValueError("record hand differs from search hand")
            engine_actions = wire_actions(list(env.legal_moves.action_list))
            observation = public_observation(env)
            if observation["seat"] != actor_seat or list(public_counts) != observation["public_counts"]:
                raise ValueError("search seat/counts differ from reconstructed root")
            current, events = public_context(observation, record["public_history"], engine_actions)
            if _decision_context(build_context(current)) != record["decision_context"]:
                raise ValueError("record decision context differs from reconstructed root")
            history_cards = [card for event in events for card in event["cards"]]
            if Counter(history_cards) != Counter(policy_cards(played_cards)):
                raise ValueError("record history differs from supplied played cards")
            supplied = list(record["legal_actions"])
            def signature(action):
                return action.kind, action.rank, tuple(sorted(action.cards))
            engine_groups = {}
            for action in engine_actions:
                engine_groups.setdefault(signature(action), []).append(action.index)
            if Counter(map(signature, engine_actions)) != Counter(map(signature, supplied)):
                raise ValueError("record legal actions differ from reconstructed root")
            mapping = {action.index: engine_groups[signature(action)].pop(0) for action in supplied}
            reverse = {position: index for index, position in mapping.items()}
            proposals = self.propose_endgame(record, total_remaining=total, baseline_index=baseline)
            blocked = {mapping[index] for index in record["trace"]["safe_gate"]["masked"]}
            if self._endgame is None:
                self._endgame = EndgameRefiner()
            selected, trace = self._endgame.refine(
                env, actor_seat, policy_cards(actor_hand), policy_cards(played_cards),
                list(public_counts), mapping[baseline], tuple(mapping[index] for index in proposals),
                FrozenPolicyContinuation(self.reference_action, public_history=record["public_history"]),
                blocked_indices=blocked)
            if selected not in reverse or selected in blocked:
                raise ValueError("search returned illegal or Safe Gate blocked action")
            final = reverse[selected]
            trace["engine_position_to_action_id"] = reverse
            record["trace"].update(final_action=final, endgame=trace)
            return final
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError) as exc:
            record["trace"]["endgame"] = dict(reason="adapter_error", applied=False, error=str(exc))
            return baseline
