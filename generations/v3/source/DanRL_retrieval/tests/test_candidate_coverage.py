from DanRL_retrieval.retrieval.candidate_coverage import tactical_coverage_indices
from DanRL_retrieval.retrieval.models import ActionCandidate


def _action(index: int, kind: str) -> ActionCandidate:
    cards = () if kind == "PASS" else (f"card-{index}",)
    return ActionCandidate(index=index, kind=kind, cards=cards, rank=None)


def test_response_coverage_keeps_pass_normal_and_bomb() -> None:
    actions = [_action(index, "Single") for index in range(10)]
    actions.extend((_action(10, "PASS"), _action(11, "Bomb")))

    order = tactical_coverage_indices(
        actions, list(range(len(actions))), top_k=10,
    )
    kinds = {actions[index].kind for index in order}

    assert len(order) == 10
    assert "PASS" in kinds
    assert "Single" in kinds
    assert "Bomb" in kinds
    assert order == sorted(order)


def test_response_coverage_uses_best_candidate_from_each_category() -> None:
    actions = [
        _action(0, "Single"),
        _action(1, "Pair"),
        _action(2, "Bomb"),
        _action(3, "StraightFlush"),
        _action(4, "PASS"),
    ]
    preference = [1, 3, 4, 0, 2]

    order = tactical_coverage_indices(actions, preference, top_k=3)

    assert order == [1, 3, 4]


def test_lead_without_pass_preserves_preference_prefix() -> None:
    actions = [_action(index, "Single") for index in range(5)]
    preference = [4, 2, 0, 1, 3]

    order = tactical_coverage_indices(actions, preference, top_k=3)

    assert order == [4, 2, 0]
