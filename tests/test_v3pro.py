"""Public V3Pro behavior; synthetic hands only, no private checkpoints."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_v3pro_distribution_exists():
    assert (ROOT / "versions/v3pro/pyproject.toml").is_file()


def test_v3pro_packaging_excludes_generated_namespace_directories():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    metadata = tomllib.loads((ROOT / "versions/v3pro/pyproject.toml").read_text())
    assert metadata["tool"]["setuptools"]["packages"]["find"].get("namespaces") is False
    assert metadata["tool"]["setuptools"].get("include-package-data") is False


@pytest.fixture
def components():
    if importlib.util.find_spec("DanKS") is None:
        pytest.skip("install the V3 and V3Pro packages for component tests")
    from DanKSPro import safety, rules
    return safety, rules


def make_row(index, cards, rank="7", kind="Pair", remaining=None):
    from DanKS.retrieval.models import ActionCandidate, CardGroup, Partition, ScoredAction
    groups = remaining or (CardGroup("Pair", ("C9", "D9"), "9"),)
    return ScoredAction(ActionCandidate(index, kind, tuple(cards), rank), 0.,
                       Partition(tuple(groups)), 0., 1., 1.,
                       dict(my_retake_count=1., break_group_penalty=0., spend_penalty=0.))


def context(count=6, opponents=(5, 5), **kwargs):
    from DanKS.retrieval.context import build_context
    return build_context(dict(my_seat=0, curRank="2", current_kind="Lead",
                             public_counts=[count, opponents[0], 4, opponents[1]], **kwargs))


def test_asset_gate_uses_same_strength_surviving_witness(components):
    import numpy as np
    safety, _ = components
    rows = [make_row(10, ["H2", "S7"]), make_row(22, ["H7", "S7"])]
    mask = np.array([1., 1., 0.], dtype=np.float32)
    result, trace = safety.asset_safe_gate(
        ["H2", "S7", "H7", "C9", "D9", "ST"], context(), rows, mask)
    assert result.tolist() == [0., 1., 0.]
    assert mask.tolist() == [1., 1., 0.]
    assert trace["masked"] == [10]


@pytest.mark.parametrize("condition", ["weaker", "short", "finish", "missing", "must_block", "bomb"])
def test_asset_gate_does_not_invent_unconditional_bans(components, condition):
    import numpy as np
    safety, _ = components
    hand = ["H2", "S7", "H7", "C9", "D9", "ST"]
    ctx = context(opponents=(1, 5)) if condition == "short" else context()
    rows = [make_row(10, ["H2", "S7"]), make_row(22, ["H7", "S7"])]
    if condition == "weaker":
        rows[1] = make_row(22, ["H6", "S6"], rank="6")
    if condition == "finish":
        rows.append(make_row(33, hand, kind="Straight"))
    if condition == "missing":
        rows[0].details.clear()
    if condition == "must_block":
        rows[0].details["must_block"] = 1.
    if condition == "bomb":
        rows = [make_row(10, ["H2", "S7", "H7", "D7"], kind="Bomb"),
                make_row(22, ["C7", "S7", "H7", "D7"], kind="Bomb")]
    result, _ = safety.asset_safe_gate(hand, ctx, rows, np.ones(len(rows)))
    assert np.all(result == 1)


def test_pair_swap_keeps_physical_triple_and_whole_hand_structure(components):
    from DanKS.retrieval.models import CardGroup
    _, rules = components
    triple = ["S3", "H3", "C3"]
    low, high = ["S4", "H4"], ["S5", "H5"]
    selected = make_row(9, triple + high, "3", "TriplePlus", [CardGroup("Pair", tuple(low), "4")])
    alternative = make_row(27, triple + low, "3", "TriplePlus", [CardGroup("Pair", tuple(high), "5")])
    result = rules.decide(triple + low + high, context(7), [selected], 0, equivalent_rows=[alternative])
    assert result.final_action_index == 27
    assert result.trace["reason"] == "strict_equivalent_pair_swap"


def test_policy_runs_real_v3_model_and_keeps_legal_ids(components):
    torch = pytest.importorskip("torch")
    from DanKS.training.model import EfficientTeamBeliefTop10Selector
    from DanKS.training.schema import STATE_DIM, CANDIDATE_DIM
    from DanKSPro import ProPolicy
    torch.manual_seed(1)
    model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM, hidden_dim=32, candidate_hidden_dim=24)
    policy = ProPolicy(model)
    hand = ["S3", "H3", "C4", "D5", "S6"]
    actions = [dict(index=10, kind="Single", cards=["S3"], rank="3"),
               dict(index=22, kind="Pair", cards=["S3", "H3"], rank="3")]
    action, record = policy.act(hand, context(5), actions)
    assert action in {10, 22}
    assert record["training_eligible"] is False
    assert action not in record["trace"]["safe_gate"]["masked"]
    assert record["mask"].shape == (10,)


def test_policy_rejects_duplicate_action_indices(components):
    torch = pytest.importorskip("torch")
    from DanKSPro import ProPolicy
    model = torch.nn.Linear(1, 1)
    policy = ProPolicy(model)
    action = dict(index=1, kind="Single", cards=["S3"], rank="3")
    with pytest.raises(ValueError, match="unique"):
        policy.act(["S3", "H3"], context(2), [action, action])


def test_proposals_preserve_confident_expert_first(components):
    torch = pytest.importorskip("torch")
    import numpy as np
    from DanKSPro import ProPolicy

    class FixedModel(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor([values + [-100.] * (10 - len(values))]))

        def forward(self, *inputs):
            return self.values, torch.zeros(1)

    policy = ProPolicy(FixedModel([5., 0., 4.]), specialist=FixedModel([0., 10., 9.]))
    record = {k: np.zeros(1, dtype=np.float32) for k in ("state", "candidates", "history")}
    record.update(mask=np.array([1., 1., 1.] + [0.] * 7),
                  ordered_candidate_indices=[10, 22, 33], action_slot=0,
                  experience_logits=np.array([5., 0., 4.] + [-100.] * 7),
                  trace={"safe_gate": {"masked": []}})
    # Expert argmax22 clears the confidence gate; combined score favors33.
    assert policy.propose_endgame(record, total_remaining=8, baseline_index=10) == (22, 33)


def test_policy_checkpoint_expert_identity_is_checked(components, tmp_path):
    torch = pytest.importorskip("torch")
    from DanKS.training.model import EfficientTeamBeliefTop10Selector, EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE
    from DanKS.training.schema import (STATE_DIM, CANDIDATE_DIM, HISTORY_LENGTH,
                                      HISTORY_EVENT_DIM, HISTORY_PROTOCOL, TEAM_BELIEF_PROTOCOL)
    from DanKSPro import ProPolicy
    model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM, hidden_dim=32, candidate_hidden_dim=24)
    payload = dict(model_type=EFFICIENT_TEAM_BELIEF_SELECTOR_TYPE, state_dim=STATE_DIM,
                   candidate_dim=CANDIDATE_DIM, model_config=dict(hidden_dim=32, candidate_hidden_dim=24),
                   history_length=HISTORY_LENGTH, history_event_dim=HISTORY_EVENT_DIM,
                   history_protocol=HISTORY_PROTOCOL, team_belief_protocol=TEAM_BELIEF_PROTOCOL,
                   model_state_dict=model.state_dict())
    main = tmp_path / "parent.pt"
    torch.save(payload, main)
    specialist = tmp_path / "expert.pt"
    torch.save(dict(payload, endgame_exact_distillation={"input_checkpoint_sha256": "wrong"}), specialist)
    with pytest.raises(ValueError, match="parent checkpoint identity"):
        ProPolicy.from_checkpoints(main, specialist=specialist)


def test_public_adapter_rotates_counts_and_keeps_absolute_history(components):
    from DanKSPro.adapter import public_context, wire_actions
    history = [{"type": "notify", "stage": "play", "curPos": 1,
                "curAction": ["Single", "3", ["S3"]]}]
    observation = dict(seat=2, hand=["S5", "H5"], public_counts=[5, 4, 2, 3],
                       rank="2", greater_pos=1, greater_action=["Single", "3", ["S3"]],
                       action_first=False)
    actions = wire_actions([["Single", "5", ["S5"]], ["PASS", "PASS", []]])
    context_data, events = public_context(observation, history, actions)
    assert context_data["public_counts"] == [2, 3, 5, 4]
    assert context_data["my_seat"] == 2 and context_data["last_player"] == 1
    assert events[0]["pos"] == 1 and events[0]["cards"] == ["S3"]
    assert context_data["played_cards"] == ["S3"]
    assert context_data["current_kind"] == "Single"
    lead_context, _ = public_context(observation, history, actions[:1])
    assert lead_context["current_kind"] == "Lead"


def test_engine_pass_sentinel_is_not_treated_as_cards(components):
    from DanKSPro.adapter import wire_actions
    action = wire_actions([["PASS", "PASS", "PASS"]])[0]
    assert action.kind == "PASS" and action.cards == ()


def test_engine_jokers_are_translated_at_public_boundary(components):
    from DanKSPro.adapter import wire_actions, policy_cards
    actions = wire_actions([["Single", "R", ["HR"]], ["Single", "B", ["SB"]]])
    assert actions[0].cards == ("RJ",) and actions[0].rank == "RJ"
    assert actions[1].cards == ("BJ",) and actions[1].rank == "BJ"
    assert policy_cards(["HR", "SB", "H10"]) == ["RJ", "BJ", "HT"]


def test_refinement_is_explicitly_disabled_without_experts(components):
    torch = pytest.importorskip("torch")
    from DanKSPro import ProPolicy
    policy = ProPolicy(torch.nn.Linear(1, 1))
    record = dict(trace=dict(rules_action=22, safe_gate=dict(masked=[])))
    result = policy.refine_endgame(None, actor_seat=0, actor_hand=["S3"],
                                  played_cards=[], public_counts=[1, 1, 1, 1], record=record)
    assert result == 22
    assert record["trace"]["endgame"]["reason"] == "expert_unavailable"


def test_v3pro_example_is_runnable_without_weights(components):
    pytest.importorskip("torch")
    import os
    import subprocess
    import sys
    example = ROOT / "examples/v3pro_smoke.py"
    assert example.is_file()
    result = subprocess.run([sys.executable, str(example)], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
            str(path) for path in (ROOT, ROOT / "versions/v3", ROOT / "versions/v3pro"))),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DanKS V3Pro ready" in result.stdout
    assert "restored=True" in result.stdout


def test_v3pro_readmes_and_ci_explain_optional_overlay():
    for name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / name).read_text()
        assert "versions/v3pro" in readme and "DanKSPro" in readme
        assert "examples/v3pro_smoke.py" in readme
    assert "test_v3pro_endgame.py" in (ROOT / ".github/workflows/ci.yml").read_text()


def search_case():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v3pro_example_fixture", ROOT / "examples/v3pro_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from DanKSPro.adapter import public_context, wire_actions
    from DanKSPro.endgame.runtime import public_observation
    from DanKS.training.model import EfficientTeamBeliefTop10Selector
    from DanKS.training.schema import STATE_DIM, CANDIDATE_DIM
    from DanKSPro import ProPolicy
    env, played, history = module.synthetic_position()
    observation = public_observation(env)
    actions = wire_actions(env.legal_moves.action_list)
    context_data, events = public_context(observation, history, actions)
    model = EfficientTeamBeliefTop10Selector(STATE_DIM, CANDIDATE_DIM, hidden_dim=32, candidate_hidden_dim=24)
    policy = ProPolicy(model, specialist=model)
    return policy, env, played, observation, actions, context_data, events


def test_record_owns_a_copy_of_public_history(components):
    pytest.importorskip("torch")
    policy, _, _, obs, actions, ctx, events = search_case()
    _, record = policy.act(obs["hand"], ctx, actions, history=events)
    original = list(record["public_history"][0]["cards"])
    events[0]["cards"].append("S3")
    assert record["public_history"][0]["cards"] == original


def test_explicit_history_matches_context_history(components):
    pytest.importorskip("torch")
    import numpy as np
    policy, _, _, obs, actions, ctx, events = search_case()
    without_history = dict(ctx)
    without_history.pop("history")
    _, first = policy.act(obs["hand"], ctx, actions)
    _, second = policy.act(obs["hand"], without_history, actions, history=events)
    for key in ("state", "candidates", "history", "mask"):
        np.testing.assert_array_equal(first[key], second[key])


def test_direct_context_uses_explicit_history(components):
    pytest.importorskip("torch")
    import numpy as np
    from DanKS.retrieval.context import RetrievalContext
    policy, _, _, obs, actions, ctx, events = search_case()
    direct = RetrievalContext(my_seat=ctx["my_seat"], cur_rank=ctx["curRank"],
                              public_counts=tuple(ctx["public_counts"]),
                              played_cards=tuple(ctx["played_cards"]))
    _, first = policy.act(obs["hand"], ctx, actions)
    _, second = policy.act(obs["hand"], direct, actions, history=events)
    for key in ("state", "candidates", "history", "mask"):
        np.testing.assert_array_equal(first[key], second[key])


def test_prebuilt_context_rejects_inconsistent_history(components):
    pytest.importorskip("torch")
    from DanKS.retrieval.context import build_context
    policy, _, _, obs, actions, ctx, events = search_case()
    with pytest.raises(ValueError, match="prebuilt context"):
        policy.act(obs["hand"], build_context(ctx), actions, history=events[:-1])


@pytest.mark.parametrize("field", ["must_block", "opponent_short_pressure", "partner_follow_help"])
def test_nonfinite_safety_veto_preserves_mask(components, field):
    import numpy as np
    safety, _ = components
    rows = [make_row(10, ["H2", "S7"]), make_row(22, ["H7", "S7"])]
    rows[0].details[field] = float("nan")
    result, trace = safety.asset_safe_gate(
        ["H2", "S7", "H7", "C9", "D9", "ST"], context(), rows, np.ones(2))
    assert result.tolist() == [1., 1.]
    assert trace["status"] == "incomplete_preserve_mask"


@pytest.mark.parametrize("mismatch", ["seat", "counts", "level", "history"])
def test_refinement_rejects_stale_public_record(components, mismatch):
    pytest.importorskip("torch")
    policy, env, played, obs, actions, ctx, events = search_case()
    if mismatch == "seat":
        ctx["my_seat"] = 2
    elif mismatch == "counts":
        ctx["public_counts"] = [2, 2, 1, 0]
    elif mismatch == "level":
        ctx["curRank"] = "3"
    else:
        events = events[:-1]
    baseline, record = policy.act(obs["hand"], ctx, actions, history=events)
    chosen = policy.refine_endgame(env, actor_seat=0, actor_hand=obs["hand"],
                                  played_cards=played, public_counts=obs["public_counts"], record=record)
    assert chosen == baseline
    assert record["trace"]["endgame"]["reason"] == "adapter_error"
