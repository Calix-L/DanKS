from __future__ import annotations

from typing import Any


try:
    from .native_cpp import danks_cover
except Exception:  # pragma: no cover - exercised when the extension is absent.
    danks_cover = None


def available() -> bool:
    return danks_cover is not None


def enumerate_covers(state: list[int], groups_by_first: list[list[list[int]]]) -> list[list[int]]:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    return danks_cover.enumerate_covers(state, groups_by_first)


def count_covers(state: list[int], groups_by_first: list[list[list[int]]]) -> int:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    return int(danks_cover.count_covers(state, groups_by_first))


def top_covers(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    max_results: int,
    enable_upper_bound: bool = False,
) -> list[list[int]]:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    return danks_cover.top_covers(state, groups_by_first, group_scores, tie_keys, max_results, enable_upper_bound)


def top_covers_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    max_results: int,
    enable_upper_bound: bool = False,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_batch"):
        raise RuntimeError("C++ batch top-cover kernel is not available")
    return danks_cover.top_covers_batch(
        states,
        groups_by_first,
        group_scores,
        tie_keys,
        int(max_results),
        bool(enable_upper_bound),
    )


def top_covers_selected_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_priorities: list[float],
    tie_keys: list[str],
    max_results: int,
    selected_results: int,
    enable_upper_bound: bool = False,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_selected_batch"):
        raise RuntimeError("C++ selected batch top-cover kernel is not available")
    if (
        type(groups_by_first).__name__ == "PyCapsule"
        and hasattr(danks_cover, "top_covers_selected_batch_capsule")
    ):
        return danks_cover.top_covers_selected_batch_capsule(
            states,
            groups_by_first,
            group_scores,
            group_priorities,
            tie_keys,
            int(max_results),
            int(selected_results),
            bool(enable_upper_bound),
        )
    return danks_cover.top_covers_selected_batch(
        states,
        groups_by_first,
        group_scores,
        group_priorities,
        tie_keys,
        int(max_results),
        int(selected_results),
        bool(enable_upper_bound),
    )


def top_covers_beam_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_sizes: list[int],
    tie_keys: list[str],
    beam_width: int,
    max_results: int,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_beam_batch"):
        raise RuntimeError("C++ batch beam-cover kernel is not available")
    if (
        type(groups_by_first).__name__ == "PyCapsule"
        and hasattr(danks_cover, "top_covers_beam_batch_capsule")
    ):
        return danks_cover.top_covers_beam_batch_capsule(
            states,
            groups_by_first,
            group_scores,
            group_sizes,
            tie_keys,
            int(beam_width),
            int(max_results),
        )
    return danks_cover.top_covers_beam_batch(
        states,
        groups_by_first,
        group_scores,
        group_sizes,
        tie_keys,
        int(beam_width),
        int(max_results),
    )


def top_covers_hand_count_window(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    window: int,
    max_results: int,
) -> list[list[int]]:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "top_covers_hand_count_window"):
        raise RuntimeError("C++ hand-count-window cover kernel is not available")
    return danks_cover.top_covers_hand_count_window(state, groups_by_first, group_scores, tie_keys, int(window), int(max_results))


def top_covers_effective_hand_count_window(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    group_costs: list[int],
    window: int,
    max_results: int,
) -> list[list[int]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_effective_hand_count_window"):
        raise RuntimeError("C++ effective-hand-count-window kernel is not available")
    return danks_cover.top_covers_effective_hand_count_window(
        state, groups_by_first, group_scores, tie_keys, group_costs, int(window), int(max_results)
    )


def top_covers_effective_hand_count_window_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    group_costs: list[int],
    window: int,
    max_results: int,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_effective_hand_count_window_batch"):
        raise RuntimeError("C++ batch effective-hand-count-window kernel is not available")
    return danks_cover.top_covers_effective_hand_count_window_batch(
        states, groups_by_first, group_scores, tie_keys, group_costs, int(window), int(max_results)
    )


def top_covers_effective_hand_count_window_selected(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_selection_priorities: list[float],
    tie_keys: list[str],
    group_costs: list[int],
    window: int,
    max_results: int,
    selected_results: int,
) -> list[list[int]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_effective_hand_count_window_selected"):
        raise RuntimeError("C++ selected effective-hand-count-window kernel is not available")
    return danks_cover.top_covers_effective_hand_count_window_selected(
        state,
        groups_by_first,
        group_scores,
        group_selection_priorities,
        tie_keys,
        group_costs,
        int(window),
        int(max_results),
        int(selected_results),
    )


def top_covers_effective_hand_count_window_selected_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_selection_priorities: list[float],
    tie_keys: list[str],
    group_costs: list[int],
    window: int,
    max_results: int,
    selected_results: int,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_effective_hand_count_window_selected_batch"):
        raise RuntimeError("C++ selected batch effective-hand-count-window kernel is not available")
    if (
        type(groups_by_first).__name__ == "PyCapsule"
        and hasattr(danks_cover, "top_covers_effective_hand_count_window_selected_batch_capsule")
    ):
        return danks_cover.top_covers_effective_hand_count_window_selected_batch_capsule(
            states,
            groups_by_first,
            group_scores,
            group_selection_priorities,
            tie_keys,
            group_costs,
            int(window),
            int(max_results),
            int(selected_results),
        )
    return danks_cover.top_covers_effective_hand_count_window_selected_batch(
        states,
        groups_by_first,
        group_scores,
        group_selection_priorities,
        tie_keys,
        group_costs,
        int(window),
        int(max_results),
        int(selected_results),
    )


def top_covers_hand_count_window_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
    window: int,
    max_results: int,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_hand_count_window_batch"):
        raise RuntimeError("C++ batch hand-count-window cover kernel is not available")
    return danks_cover.top_covers_hand_count_window_batch(
        states, groups_by_first, group_scores, tie_keys, int(window), int(max_results)
    )


def top_covers_hand_count_window_selected(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_priorities: list[float],
    tie_keys: list[str],
    window: int,
    max_results: int,
    selected_results: int,
) -> list[list[int]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_hand_count_window_selected"):
        raise RuntimeError("C++ selected hand-count-window kernel is not available")
    return danks_cover.top_covers_hand_count_window_selected(
        state,
        groups_by_first,
        group_scores,
        group_priorities,
        tie_keys,
        int(window),
        int(max_results),
        int(selected_results),
    )


def top_covers_hand_count_window_selected_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    group_priorities: list[float],
    tie_keys: list[str],
    window: int,
    max_results: int,
    selected_results: int,
) -> list[list[list[int]]]:
    if danks_cover is None or not hasattr(danks_cover, "top_covers_hand_count_window_selected_batch"):
        raise RuntimeError("C++ selected batch hand-count-window kernel is not available")
    if (
        type(groups_by_first).__name__ == "PyCapsule"
        and hasattr(danks_cover, "top_covers_hand_count_window_selected_batch_capsule")
    ):
        return danks_cover.top_covers_hand_count_window_selected_batch_capsule(
            states,
            groups_by_first,
            group_scores,
            group_priorities,
            tie_keys,
            int(window),
            int(max_results),
            int(selected_results),
        )
    return danks_cover.top_covers_hand_count_window_selected_batch(
        states,
        groups_by_first,
        group_scores,
        group_priorities,
        tie_keys,
        int(window),
        int(max_results),
        int(selected_results),
    )


def best_cover_by_group_scores(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_scores: list[float],
    tie_keys: list[str],
) -> list[int]:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    return list(danks_cover.best_cover_by_group_scores(state, groups_by_first, group_scores, tie_keys))


def best_selected_cover_by_score_entries(
    covers: list[list[int]],
    group_entries: list[list[float | int]],
    weights: list[float],
    pressure_values: list[float] | tuple[float, ...],
) -> tuple[int, float, tuple[float, float, float, float], float] | None:
    if danks_cover is None or not hasattr(danks_cover, "best_selected_cover_by_score_entries"):
        raise RuntimeError("C++ selected-partition score kernel is not available")
    out = danks_cover.best_selected_cover_by_score_entries(
        covers,
        group_entries,
        weights,
        pressure_values,
    )
    if out is None:
        return None
    return int(out[0]), float(out[1]), tuple(float(value) for value in out[2]), float(out[3])


def best_selected_covers_by_score_entries_batch(
    cover_batches: list[list[list[int]]],
    group_entries: list[list[float | int]],
    weights: list[float],
    pressure_values_by_batch: list[list[float] | tuple[float, ...]],
) -> list[tuple[int, float, tuple[float, float, float, float], float] | None]:
    if danks_cover is None or not hasattr(danks_cover, "best_selected_covers_by_score_entries_batch"):
        raise RuntimeError("C++ selected-partition batch score kernel is not available")
    raw = danks_cover.best_selected_covers_by_score_entries_batch(
        cover_batches,
        group_entries,
        weights,
        pressure_values_by_batch,
    )
    return [
        None
        if item is None
        else (int(item[0]), float(item[1]), tuple(float(value) for value in item[2]), float(item[3]))
        for item in raw
    ]


def best_cover_by_score_entries(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_entries: list[list[float]],
    weights: list[float],
    pressure_values: list[float],
) -> tuple[list[int], float, tuple[float, float, float, float]] | None:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "best_cover_by_score_entries"):
        raise RuntimeError("C++ score-entry kernel is not available")
    out = danks_cover.best_cover_by_score_entries(state, groups_by_first, group_entries, weights, pressure_values)
    if not out or out[0] is None:
        return None
    return list(out[0]), float(out[1]), tuple(float(value) for value in out[2])


def best_cover_by_score_entries_with_retake(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_entries: list[list[float]],
    weights: list[float],
    pressure_values: list[float],
) -> tuple[list[int], float, tuple[float, float, float, float], float] | None:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "best_cover_by_score_entries_with_retake"):
        raise RuntimeError("C++ score-entry retake kernel is not available")
    out = danks_cover.best_cover_by_score_entries_with_retake(state, groups_by_first, group_entries, weights, pressure_values)
    if not out or out[0] is None:
        return None
    return list(out[0]), float(out[1]), tuple(float(value) for value in out[2]), float(out[3])


def best_covers_by_score_entries_with_retake_batch(
    states: list[list[int]],
    groups_by_first: list[list[list[int]]],
    group_entries: list[list[float]],
    weights: list[float],
    pressure_values_by_state: list[list[float]],
) -> list[tuple[list[int], float, tuple[float, float, float, float], float] | None]:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "best_covers_by_score_entries_with_retake_batch"):
        raise RuntimeError("C++ score-entry retake batch kernel is not available")
    raw = danks_cover.best_covers_by_score_entries_with_retake_batch(
        states,
        groups_by_first,
        group_entries,
        weights,
        pressure_values_by_state,
    )
    out = []
    for item in raw:
        if not item or item[0] is None:
            out.append(None)
        else:
            out.append((list(item[0]), float(item[1]), tuple(float(value) for value in item[2]), float(item[3])))
    return out


def best_cover_by_score_entries_dp(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_entries: list[list[float]],
    weights: list[float],
    pressure_values: list[float],
    frontier_limit: int = 200000,
) -> tuple[list[int], float, tuple[float, float, float, float]] | None:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "best_cover_by_score_entries_dp"):
        raise RuntimeError("C++ score-entry DP kernel is not available")
    out = danks_cover.best_cover_by_score_entries_dp(
        state,
        groups_by_first,
        group_entries,
        weights,
        pressure_values,
        int(frontier_limit),
    )
    if not out or out[0] is None:
        return None
    return list(out[0]), float(out[1]), tuple(float(value) for value in out[2])


def best_cover_by_score_entries_dp_with_retake(
    state: list[int],
    groups_by_first: list[list[list[int]]],
    group_entries: list[list[float]],
    weights: list[float],
    pressure_values: list[float],
    frontier_limit: int = 200000,
) -> tuple[list[int], float, tuple[float, float, float, float], float] | None:
    if danks_cover is None:
        raise RuntimeError("C++ cover kernel is not available")
    if not hasattr(danks_cover, "best_cover_by_score_entries_dp_with_retake"):
        raise RuntimeError("C++ score-entry DP retake kernel is not available")
    out = danks_cover.best_cover_by_score_entries_dp_with_retake(
        state,
        groups_by_first,
        group_entries,
        weights,
        pressure_values,
        int(frontier_limit),
    )
    if not out or out[0] is None:
        return None
    return list(out[0]), float(out[1]), tuple(float(value) for value in out[2]), float(out[3])


def module() -> Any:
    return danks_cover
