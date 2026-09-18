"""Portable public-information search and proof contracts."""

import importlib.util
from collections import Counter
import random

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("DanKSPro") is None,
    reason="optional V3Pro package is not on PYTHONPATH",
)


def modules():
    from DanKSPro.endgame import exact, information, runtime

    return exact, information, runtime


def small_table(actor_hand=("S3", "S5")):
    from guandan.engine import Environment
    from guandan.engine.types import Phase

    env = Environment(allow_step_back=True, first_player=0)
    for seat in range(4):
        env.add_player(str(seat), seat)
    env.start()
    for player, hand in zip(env.players, (actor_hand, ("S4",), ("S6",), ("S7",))):
        player.hand_cards = []
        player.hearts_num = 0
        for card in hand:
            player.add_card(card, env.rank)
    env.state.reset()
    env.settlement.clear()
    env.first_action(0)
    env.action_first = True
    env.act([], 0, Phase.PLAY)
    env.state.current_pos = 0
    env.loop = env.play
    env.trace.clear()
    return env


def public_input(env):
    _, information, _ = modules()
    remaining = Counter(str(c) for p in env.players for c in p.hand_cards)
    played = list((Counter(information.canonical_double_deck()) - remaining).elements())
    return (
        [str(c) for c in env.players[0].hand_cards],
        played,
        [len(p.hand_cards) for p in env.players],
    )


def test_complete_real_engine_search_and_restore():
    exact, _, runtime = modules()
    env = small_table()
    before = exact.restoration_key(env, 0)
    result = exact.solve_root_actions_transactional(
        env,
        (0, 1),
        10000,
        exact.ExactTranspositionTable(max_entries=1000000),
        runtime.reward_by_order,
        True,
    )
    assert result["status"] == "ok"
    assert set(result["values"]) == {0, 1}
    assert all(value in (-3, -2, -1, 1, 2, 3) for value in result["values"].values())
    assert exact.restoration_key(env, 0) == before


def test_public_enumeration_is_complete_and_capped():
    _, information, _ = modules()
    env = small_table()
    hand, played, counts = public_input(env)
    worlds = information.enumerate_hidden_hands_exhaustive(
        0, hand, played, counts, max_allocations=128
    )
    assert worlds.total_allocations == 6
    assert sum(worlds.allocation_weights) == worlds.total_physical_allocations == 6
    with pytest.raises(information.HiddenAllocationLimitExceeded):
        information.enumerate_hidden_hands_exhaustive(
            0, hand, played, counts, max_allocations=1
        )
    with pytest.raises(ValueError):
        information.enumerate_hidden_hands_exhaustive(
            0, hand, [], counts, max_allocations=128
        )


def test_refiner_empty_seed_expands_when_expert_available_and_blocked_stays_out():
    exact, _, runtime = modules()
    env = small_table()
    hand, played, counts = public_input(env)
    before = exact.restoration_key(env, 0)
    rng = random.getstate()
    refiner = runtime.EndgameRefiner()
    chosen, trace = refiner.refine(
        env, 0, hand, played, counts, 0, (), runtime.FrozenPolicyContinuation()
    )
    assert trace["proposed_indices"] == (1,)
    chosen, trace = refiner.refine(
        env,
        0,
        hand,
        played,
        counts,
        0,
        (1,),
        runtime.FrozenPolicyContinuation(),
        blocked_indices=(1,),
    )
    assert chosen == 0 and trace["reason"] == "no_unblocked_alternative"
    assert exact.restoration_key(env, 0) == before
    assert random.getstate() == rng


def test_refiner_missing_history_and_policy_failure_restore():
    exact, _, runtime = modules()
    env = small_table()
    hand, played, counts = public_input(env)
    before = exact.restoration_key(env, 0)
    chosen, trace = runtime.EndgameRefiner().refine(
        env, 0, hand, [], counts, 0, (1,), runtime.FrozenPolicyContinuation()
    )
    assert chosen == 0 and trace["reason"] == "invalid_public_information"

    def fail(*args):
        random.random()
        raise ValueError("policy failure")

    chosen, trace = runtime.EndgameRefiner().refine(
        env, 0, hand, played, counts, 0, (1,), runtime.FrozenPolicyContinuation(fail)
    )
    assert chosen == 0 and trace["reason"] == "verification_error"
    assert exact.restoration_key(env, 0) == before


def test_frozen_continuation_has_real_public_history_and_restores():
    exact, _, runtime = modules()
    env = small_table()
    before = exact.restoration_key(env, 0)
    seen = []

    def policy(observation, legal, history):
        assert "hand" in observation and "public_counts" in observation
        assert "players" not in observation
        seen.append(history)
        return 0

    solver = runtime.FrozenPolicyContinuation(policy, public_history=({"root": True},))
    result = solver(env, (0,), 10000, None, runtime.reward_by_order, True)
    assert result["status"] == "ok" and seen
    assert seen[0][0] == {"root": True} and len(seen[0]) > 1
    assert exact.restoration_key(env, 0) == before


def proof_result(indices, values):
    """Build complete weighted certificates from a controlled root payoff table."""
    from DanKSPro.endgame import exhaustive

    exact, information, runtime = modules()
    env = small_table(("S3", "S5", "S8"))
    hand, played, counts = public_input(env)

    def payoff_solver(env, roots, max_nodes, table, reward, domain_order):
        return dict(status="ok", values={i: values[i] for i in roots}, nodes=1)

    return exhaustive.exhaustive_determinization_action(
        env=env,
        benchmark=exact,
        information_module=information,
        reward_by_order=runtime.reward_by_order,
        actor_seat=0,
        actor_hand=hand,
        played_cards=played,
        public_counts=counts,
        chosen_index=0,
        v3pro_order=[0, *indices],
        max_allocations=10000,
        max_nodes_per_allocation=250000,
        max_total_nodes=500000,
        root_candidate_indices=list(indices),
        root_action_solver=payoff_solver,
    )


def test_complete_tie_requires_positive_blueprint_and_all_five_exact_zero_margins():
    from dataclasses import replace
    from DanKSPro.endgame import tie, verifier

    _, _, runtime = modules()
    bp = proof_result((1,), {0: -1, 1: 1})
    equal = proof_result((1,), {0: -1, 1: -1})
    decision = verifier.verify_candidate_pool_blueprint_then_joint(
        baseline_index=0,
        proposed_indices=(1,),
        verify_blueprint=lambda i, rank: bp,
        verify_joint=lambda indices: equal,
    )
    assert not decision.applied
    accepted, cert = tie.apply_blueprint_tie_fallback(
        decision,
        baseline_index=0,
        proposed_indices=(1,),
        route="extended_outer",
        config=runtime.standard_config(),
    )
    assert accepted.applied and accepted.selected_index == 1
    assert set(cert["exact_margins"].values()) == {0.0}
    for malformed in (
        replace(equal, completed_allocations=0),
        replace(equal, candidate_risk_rows=()),
        replace(equal, belief_contract="other"),
    ):
        refused, proof = tie.apply_blueprint_tie_fallback(
            replace(decision, joint_result=malformed),
            baseline_index=0,
            proposed_indices=(1,),
            route="extended_outer",
            config=runtime.standard_config(),
        )
        assert not refused.applied and proof is None
    loss = proof_result((1,), {0: 1, 1: -1})
    refused, proof = tie.apply_blueprint_tie_fallback(
        replace(decision, joint_result=loss),
        baseline_index=0,
        proposed_indices=(1,),
        route="extended_outer",
        config=runtime.standard_config(),
    )
    assert not refused.applied and proof is None


def test_isolated_recovery_requires_exhausted_joint_and_copies_table():
    from dataclasses import replace
    from DanKSPro.endgame import recovery, tie, verifier

    exact, _, runtime = modules()
    bps = {i: proof_result((i,), {0: -1, i: 1}) for i in (1, 2)}
    complete_joint = proof_result((1, 2), {0: -1, 1: -1, 2: -1})
    incomplete = replace(
        complete_joint,
        status="total_node_budget_exceeded",
        selected_index=0,
        proposed_index=0,
        selection_reason="incomplete_preserves_v3pro",
        risk_metrics={},
        action_rows=(),
        candidate_risk_rows=(),
        incomplete_reason="budget",
        completed_allocations=0,
        completed_physical_weight=0,
        nodes=1,
    )
    decision = verifier.verify_candidate_pool_blueprint_then_joint(
        baseline_index=0,
        proposed_indices=(1, 2),
        verify_blueprint=lambda i, rank: bps[i],
        verify_joint=lambda indices: incomplete,
    )
    table = exact.ExactTranspositionTable(max_entries=1000000)
    table.store(("original",), 1)
    original = dict(table.entries)
    branch_tables = []

    def isolated(indices, *, max_total_nodes, table_override):
        branch_tables.append(table_override)
        assert table_override is not table and table_override.entries == original
        table_override.store(("isolated",), 2)
        return proof_result(indices, {0: -1, indices[0]: -1})

    kwargs = dict(
        decision=decision,
        baseline_index=0,
        proposed_indices=(1, 2),
        route="extended_outer",
        config=runtime.standard_config(),
        table=table,
        helper=tie,
        verifier=verifier,
        enabled=True,
        retry_budget=runtime.exact_retry_budget,
        verify_exact=isolated,
    )
    premature = recovery.recover(**kwargs, original_calls=[("initial", incomplete)])
    assert not premature["applied"] and not branch_tables
    result = recovery.recover(
        **kwargs, original_calls=[("initial", incomplete), ("retry", incomplete)]
    )
    assert result["applied"] and result["selected_index"] == 1
    assert result["certificate_kind"] == "complete_exact_tie_and_blueprint_gain"
    assert result["certificate_actual_root_indices"] == [0, 1]
    assert result["full_pool_actual_root_indices"] == [0, 1, 2]
    assert result["original_proposal_rank"] == result["local_proposal_rank"] == 1
    assert table.entries == original and len(branch_tables) == 1


def test_real_search_budget_failure_restores_and_never_exposes_partial_values():
    exact, _, runtime = modules()
    env = small_table(("S3", "S5", "S8"))
    before = exact.restoration_key(env, 0)
    result = exact.solve_root_actions_transactional(
        env,
        (0, 1, 2),
        1,
        exact.ExactTranspositionTable(),
        runtime.reward_by_order,
        True,
    )
    assert result["status"] == "node_budget_exceeded" and "values" not in result
    assert exact.restoration_key(env, 0) == before


def test_strict_risk_rejects_mean_gain_with_any_world_regression():
    from DanKSPro.endgame import exhaustive

    result = exhaustive.select_risk_constrained_action(
        values_by_allocation=[{0: 1, 1: 3}, {0: 1, 1: -1}],
        physical_weights=[9, 1],
        baseline_index=0,
        candidate_order=[0, 1],
        risk_mode="strict",
        cvar_alpha=1 / 3,
        minimum_mean_value_margin=0,
        minimum_worst_value_margin=0,
        minimum_paired_cvar_margin=0,
        minimum_paired_regret_min=0,
        minimum_win_probability_margin=0,
    )
    assert result["selected_index"] == 0 and result["proposed_index"] == 1
    assert (
        next(row for row in result["candidate_risk_rows"] if row["action_index"] == 1)[
            "paired_regret_min"
        ]
        < 0
    )


def test_exhaustive_public_api_does_not_accept_privileged_world_selection():
    import inspect
    from DanKSPro.endgame.exhaustive import exhaustive_determinization_action

    parameters = inspect.signature(exhaustive_determinization_action).parameters
    assert "calibration_true_allocation" not in parameters
    assert "allocation_mode" not in parameters
    assert "allocation_set_solver" not in parameters


@pytest.mark.parametrize(
    "size, route",
    [(8, "extended"), (9, "extended"), (10, "extended_outer"), (13, "extended_outer")],
)
def test_route_boundaries_full_legal_and_blueprint_incomplete_fail_closed(size, route):
    _, _, runtime = modules()
    cards = (
        "S3",
        "S5",
        "S8",
        "S9",
        "ST",
        "SJ",
        "SQ",
        "SK",
        "SA",
        "S2",
        "H3",
        "H5",
        "H8",
    )
    env = small_table(cards[:size])
    hand, played, counts = public_input(env)

    def incomplete(env, indices, budget, table, reward, order):
        return dict(status="node_budget_exceeded", nodes=1)

    index, trace = runtime.EndgameRefiner().refine(
        env, 0, hand, played, counts, 0, (), incomplete
    )
    assert index == 0 and trace["route"] == route
    assert set(trace["proposed_indices"]) == set(env.legal_moves.valid_range) - {0}
    assert (
        trace["reason"] == "blueprint_guard_incomplete" and trace["exact_calls"] == []
    )


def test_pair_search_uses_only_actual_pair_for_root_order(monkeypatch):
    from DanKSPro.endgame import exhaustive

    _, _, runtime = modules()
    env = small_table(("S3", "S5", "S8"))
    hand, played, counts = public_input(env)
    seen = []
    original = exhaustive.exhaustive_determinization_action

    def observe(**kwargs):
        seen.append((kwargs["v3pro_order"], kwargs["root_candidate_indices"]))
        return original(**kwargs)

    monkeypatch.setattr(exhaustive, "exhaustive_determinization_action", observe)
    runtime.EndgameRefiner().refine(
        env, 0, hand, played, counts, 0, (2, 1), runtime.FrozenPolicyContinuation()
    )
    assert seen and seen[0] == ([0, 2], [0, 2])


def test_isolated_recovery_tries_at_most_two_fresh_candidates():
    from dataclasses import replace
    from DanKSPro.endgame import recovery, tie, verifier

    exact, _, runtime = modules()
    bps = {i: proof_result((i,), {0: -1, i: 1}) for i in (1, 2)}
    joint = proof_result((1, 2), {0: -1, 1: -1, 2: -1})

    def incomplete(result):
        return replace(
            result,
            status="total_node_budget_exceeded",
            selected_index=0,
            proposed_index=0,
            selection_reason="incomplete_preserves_v3pro",
            risk_metrics={},
            action_rows=(),
            candidate_risk_rows=(),
            incomplete_reason="budget",
            completed_allocations=0,
            completed_physical_weight=0,
            nodes=1,
        )

    failed = incomplete(joint)
    decision = verifier.verify_candidate_pool_blueprint_then_joint(
        baseline_index=0,
        proposed_indices=(1, 2),
        verify_blueprint=lambda i, rank: bps[i],
        verify_joint=lambda indices: failed,
    )
    table = exact.ExactTranspositionTable(max_entries=1000000)
    table.store(("original",), 1)
    tables = []

    def verify_exact(indices, *, max_total_nodes, table_override):
        tables.append(table_override)
        if max_total_nodes == 500000:
            assert ("branch",) not in table_override.entries
            table_override.store(("branch",), 2)
        else:
            assert ("branch",) in table_override.entries
        return incomplete(proof_result(indices, {0: -1, indices[0]: -1}))

    result = recovery.recover(
        decision=decision,
        baseline_index=0,
        proposed_indices=(1, 2),
        route="extended_outer",
        config=runtime.standard_config(),
        table=table,
        helper=tie,
        verifier=verifier,
        enabled=True,
        original_calls=[("initial", failed), ("retry", failed)],
        retry_budget=runtime.exact_retry_budget,
        verify_exact=verify_exact,
    )
    assert result["status"] == "no_complete_certificate" and not result["applied"]
    assert len(result["calls"]) == 4 and result["nominal_node_budget"] == 5000000
    assert (
        tables[0] is tables[1] and tables[2] is tables[3] and tables[0] is not tables[2]
    )
    assert ("branch",) not in table.entries


def test_public_world_search_is_invariant_to_real_hidden_seat_permutation():
    from DanKSPro.endgame import exhaustive

    exact, information, runtime = modules()
    first = small_table()
    second = small_table()
    second.players[1].hand_cards, second.players[3].hand_cards = (
        second.players[3].hand_cards,
        second.players[1].hand_cards,
    )
    hand, played, counts = public_input(first)
    results = []
    for env in (first, second):
        results.append(
            exhaustive.exhaustive_determinization_action(
                env=env,
                benchmark=exact,
                information_module=information,
                reward_by_order=runtime.reward_by_order,
                actor_seat=0,
                actor_hand=hand,
                played_cards=played,
                public_counts=counts,
                chosen_index=0,
                v3pro_order=[0, 1],
                root_candidate_indices=[0, 1],
                max_allocations=10000,
                max_nodes_per_allocation=250000,
                max_total_nodes=500000,
                root_action_solver=exact.solve_root_actions_transactional,
            )
        )
    assert results[0].status == results[1].status == "complete"
    assert results[0].action_rows == results[1].action_rows
    assert results[0].candidate_risk_rows == results[1].candidate_risk_rows


def test_extended_schedule_runs_all_blueprints_before_joint_exact(monkeypatch):
    exact, _, runtime = modules()
    env = small_table(("S3", "S5", "S8", "S9", "ST", "SJ", "SQ", "SK"))
    hand, played, counts = public_input(env)
    events = []

    def solver(stage):
        def evaluate(env, roots, budget, table, reward, order):
            events.append((stage, tuple(roots)))
            return dict(
                status="ok", nodes=1, values={i: -1 if i == 0 else 1 for i in roots}
            )

        return evaluate

    monkeypatch.setattr(exact, "solve_root_actions_transactional", solver("exact"))
    selected, trace = runtime.EndgameRefiner().refine(
        env, 0, hand, played, counts, 0, (), solver("blueprint")
    )
    first_exact = next(i for i, (stage, _) in enumerate(events) if stage == "exact")
    assert all(stage == "blueprint" for stage, _ in events[:first_exact])
    assert all(stage == "exact" for stage, _ in events[first_exact:])
    assert {roots[1] for _, roots in events[:first_exact]} == set(
        trace["proposed_indices"]
    )
    assert selected in trace["proposed_indices"] and trace["applied"]
