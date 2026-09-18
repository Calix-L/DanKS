"""Bounded search coordinator and concrete shared-engine policy continuation.

Every root index is an actual ``env.legal_moves`` position. Callers with their
own candidate IDs must map them before calling. Search is exhaustive over the
publicly consistent allocations, but is not a strategy-fusion-free solver.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import asdict
import sys

from . import exact, exhaustive, information, recovery, tie, verifier


def reward_by_order(order, root_team):
    """Partnership grade: first/second +3, first/third +2, first/fourth +1."""
    if len(order) != 4 or set(order) != set(range(4)) or root_team not in (0, 1):
        raise ValueError("a complete four-seat finish order is required")
    winner = order[0] % 2
    partner_position = order.index((order[0] + 2) % 4)
    grade = 4 - partner_position
    return grade if winner == root_team else -grade


@contextmanager
def preserved_random_state():
    """Keep process RNG streams unchanged, without importing optional runtimes."""
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def public_observation(env):
    """Expose only the current actor's hand and public trick information."""
    seat = int(env.state.current_pos)
    return dict(
        seat=seat,
        hand=[str(c) for c in env.players[seat].hand_cards],
        public_counts=[len(p.hand_cards) for p in env.players],
        rank=str(env.rank),
        current_pos=seat,
        greater_pos=int(env.state.greater_pos),
        greater_action=copy.deepcopy(exact.wire_action(env.state.greater_action)),
        current_action=copy.deepcopy(exact.wire_action(env.state.current_action)),
        finish_order=list(env.settlement.order),
        action_first=bool(env.action_first),
    )


class FrozenPolicyContinuation:
    """Concrete rollout adapter for a pure deterministic public policy.

    ``policy(observation, legal_wire_actions, public_history)`` returns a list
    position. Observation contains the current seat's hand, never other hands.
    History is a branch-local tuple of the caller's root events followed by raw
    shared-engine public ``notify/play`` bodies, exactly once per transition.
    The default policy plays the first legal move; it is a usable deterministic
    reference for examples, not a claim of learned-policy strength. Production
    callers supply their frozen reference policy. Stateful policies must be
    copied before constructing this adapter or implement a pure callback.
    """

    def __init__(self, policy=None, *, public_history=()):
        self.policy = policy
        self.public_history = tuple(copy.deepcopy(public_history))

    def __call__(
        self, env, root_indices, max_nodes, table, reward_by_order, domain_order
    ):
        del table, domain_order
        if type(max_nodes) is not int or max_nodes <= 0:
            raise ValueError("positive continuation budget required")
        roots = tuple(root_indices)
        if not roots or any(
            type(i) is not int or i not in env.legal_moves.valid_range for i in roots
        ):
            raise ValueError("continuation root must be legal engine position")
        nodes = decisions = 0
        values = {}
        root_team = int(env.state.current_pos) % 2
        with preserved_random_state():
            for root in roots:
                branch = copy.deepcopy(env)
                history = list(copy.deepcopy(self.public_history))
                index = root
                while True:
                    if nodes >= max_nodes:
                        return dict(
                            status="node_budget_exceeded",
                            nodes=nodes,
                            policy_decisions=decisions,
                        )
                    nodes += 1
                    messages = list(branch.loop({"actIndex": index}))
                    for message in messages:
                        if (
                            message.seat == 0
                            and message.body.get("type") == "notify"
                            and message.body.get("stage") == "play"
                        ):
                            history.append(copy.deepcopy(message.body))
                    order = exact.terminal_order(messages)
                    if order is not None:
                        values[root] = float(reward_by_order(order, root_team))
                        break
                    legal_indices = list(branch.legal_moves.valid_range)
                    legal = [
                        copy.deepcopy(exact.wire_action(branch.legal_moves[i]))
                        for i in legal_indices
                    ]
                    if not legal:
                        raise ValueError("continuation has no legal action")
                    position = 0
                    if len(legal) > 1 and self.policy is not None:
                        decisions += 1
                        position = self.policy(
                            public_observation(branch),
                            legal,
                            tuple(copy.deepcopy(history)),
                        )
                    if type(position) is not int or not 0 <= position < len(legal):
                        raise ValueError(
                            "continuation policy returned illegal position"
                        )
                    index = legal_indices[position]
        return dict(
            status="ok",
            values=values,
            nodes=nodes,
            policy_decisions=decisions,
            expanded_actions=0,
            cutoffs=0,
            cache_lookups=0,
            cache_hits=0,
            cache_entries=0,
        )


def standard_config():
    """Return a fresh copy of the certified five-zero-margin search contract."""
    config = dict(
        verify_blueprint_tie_fallback=True,
        verify_complete_small_proposals=True,
        extended_full_legal_proposals=True,
        exact_verify_specialist=True,
        verify_blueprint_guard=True,
        verify_root_transaction=True,
        extended_max_total_remaining=16,
        extended_max_allocations=128,
        verify_max_allocations=10000,
        verify_risk_mode="strict",
        verify_candidate_mode="joint",
        verify_blueprint_order="all_before_joint",
        verify_risk_cvar_alpha=1 / 3,
        verify_max_total_nodes=500000,
        verify_max_nodes_per_allocation=250000,
        verify_retry_max_total_nodes=2000000,
    )
    config.update({"verify_" + key: 0.0 for key in tie.THRESHOLDS})
    return config


def exact_retry_budget(*, route, status, config):
    """The larger retry is reserved for an incomplete 13–16 outer pool."""
    if route == "extended_outer" and status in recovery.INCOMPLETE:
        return config["verify_retry_max_total_nodes"]
    return None


class EndgameRefiner:
    """Public worlds, frozen-policy prefilter, exact verification, isolated repair.

    The table is reusable across decisions; completed/bounded minimax entries
    encode all hidden hands and root partnership. Isolated recovery only mutates
    private table copies. Every incomplete or invalid proof preserves baseline.
    Calling this method implies the required expert is available: an empty
    expert ranking still expands the complete legal pool under this contract.
    The caller must abstain before invoking refinement when its expert is absent.
    """

    def __init__(self):
        self.table = exact.ExactTranspositionTable(max_entries=1000000)

    def refine(
        self,
        env,
        actor_seat,
        actor_hand,
        played_cards,
        public_counts,
        baseline_index,
        proposed_indices,
        blueprint_solver,
        blocked_indices=(),
    ):
        trace = dict(
            applied=False, selected_index=baseline_index, reason="not_eligible"
        )
        legal = tuple(int(i) for i in env.legal_moves.valid_range)
        if type(baseline_index) is not int or baseline_index not in legal:
            raise ValueError("baseline must identify a legal engine action")
        if actor_seat != int(env.state.current_pos):
            trace["reason"] = "invalid_actor"
            return baseline_index, trace
        blocked = set(blocked_indices)
        proposals = tuple(
            dict.fromkeys(
                i
                for i in proposed_indices
                if type(i) is int
                and i in legal
                and i != baseline_index
                and i not in blocked
            )
        )
        proposals += tuple(
            i
            for i in legal
            if i != baseline_index and i not in proposals and i not in blocked
        )
        if not proposals:
            trace["reason"] = "no_unblocked_alternative"
            return baseline_index, trace
        config = standard_config()
        try:
            if any(type(c) is not int or c < 0 for c in public_counts):
                raise ValueError("public counts must be nonnegative integers")
            total = sum(public_counts)
            if not 0 < total <= 16:
                return baseline_index, trace
            worlds = information.count_hidden_hand_allocations(
                actor_seat, actor_hand, played_cards, public_counts
            )
        except (TypeError, ValueError, OverflowError):
            trace["reason"] = "invalid_public_information"
            return baseline_index, trace
        route = (
            "legacy" if total <= 10 else "extended" if total <= 12 else "extended_outer"
        )
        trace.update(route=route, total_allocations=worlds)
        if worlds > (10000 if route == "legacy" else 128):
            trace["reason"] = "allocation_limit_exceeded"
            return baseline_index, trace
        trace["proposed_indices"] = proposals
        snapshot = exact.EngineSearchSnapshot.capture(env)
        original_undo = env.allow_step_back
        calls = []

        def search(indices, solver, budget, table):
            return exhaustive.exhaustive_determinization_action(
                env=env,
                benchmark=exact,
                information_module=information,
                reward_by_order=reward_by_order,
                actor_seat=actor_seat,
                actor_hand=list(actor_hand),
                played_cards=list(played_cards),
                public_counts=list(public_counts),
                chosen_index=baseline_index,
                v3pro_order=[baseline_index, *indices],
                max_allocations=10000,
                max_nodes_per_allocation=250000,
                max_total_nodes=budget,
                root_candidate_indices=[baseline_index, *indices],
                transposition_table=table,
                root_action_solver=solver,
            )

        def verify_blueprint(index, rank):
            del rank
            return search((index,), blueprint_solver, 500000, self.table)

        def verify_exact(indices, *, max_total_nodes=500000, table_override=None):
            return search(
                indices,
                exact.solve_root_actions_transactional,
                max_total_nodes,
                self.table if table_override is None else table_override,
            )

        def verify_joint(indices):
            result = verify_exact(indices)
            calls.append(("initial", result))
            retry = exact_retry_budget(route=route, status=result.status, config=config)
            if retry is not None:
                result = verify_exact(indices, max_total_nodes=retry)
                calls.append(("retry", result))
            return result

        try:
            with preserved_random_state():
                env.allow_step_back = True
                if route == "legacy":
                    decision = verifier.verify_candidate_pool_blueprint_first(
                        baseline_index=baseline_index,
                        proposed_indices=proposals,
                        verify_blueprint=verify_blueprint,
                        verify_minimax=lambda index, rank: verify_exact((index,)),
                    )
                    trace["decision"] = asdict(decision)
                else:
                    decision = verifier.verify_candidate_pool_blueprint_then_joint(
                        baseline_index=baseline_index,
                        proposed_indices=proposals,
                        verify_blueprint=verify_blueprint,
                        verify_joint=verify_joint,
                    )
                    decision, proof = tie.apply_blueprint_tie_fallback(
                        decision,
                        baseline_index=baseline_index,
                        proposed_indices=proposals,
                        route="extended_outer",
                        config=config,
                    )
                    trace.update(
                        decision=asdict(decision),
                        tie_proof=proof,
                        exact_calls=[
                            (stage, asdict(result)) for stage, result in calls
                        ],
                    )
                    recovered = recovery.recover(
                        decision=decision,
                        baseline_index=baseline_index,
                        proposed_indices=proposals,
                        route=route,
                        config=config,
                        table=self.table,
                        helper=tie,
                        verifier=verifier,
                        enabled=True,
                        original_calls=calls,
                        retry_budget=exact_retry_budget,
                        verify_exact=verify_exact,
                    )
                    trace["recovery"] = recovered
                    if recovered["applied"]:
                        selected = recovered["selected_index"]
                        if selected not in blocked:
                            trace.update(
                                applied=True,
                                selected_index=selected,
                                reason="isolated_recovered",
                            )
                            return selected, trace
                if decision.applied and decision.selected_index not in blocked:
                    trace.update(
                        applied=True,
                        selected_index=decision.selected_index,
                        reason=decision.reason,
                    )
                    return decision.selected_index, trace
                trace["reason"] = decision.reason
        except (
            ArithmeticError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            trace.update(reason="verification_error", error=type(exc).__name__)
        finally:
            snapshot.restore(env)
            env.allow_step_back = original_undo
        return baseline_index, trace
