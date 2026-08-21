"""
Smoke test.

  cd KSplatform/Dan_platform/python
  python3 run_smoke.py

The move engine is the bundled Python implementation.
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.environment import Environment  # noqa: E402
from engine.moves import Moves  # noqa: E402
from engine.types import Move  # noqa: E402

RANK_ORDER = {
    "2": 15, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17,
}
NUMBER_ORDER = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "B": 16, "R": 17,
}


def _pick_index(env: Environment) -> int:
    vr = env.legal_moves.valid_range
    lo, hi = vr.start, vr.stop
    if hi <= lo:
        return 0
    return random.randrange(lo, hi)


def _fresh_env() -> Environment:
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    env = Environment()
    for i in range(4):
        env.add_player(f"p{i}", i)
    random.seed()
    env.start()
    return env


def test_actions() -> None:
    print("[ok] move engine: python")

    al = Moves()
    al.parse_first_action(["S3", "H3", "C4", "D5", "S6", "H7"], 0, "3")
    assert len(al.action_list) > 0
    assert al.action_list[0][0] in ("PASS", "Single", "Pair", "Trips", "Bomb", "Straight", "StraightFlush")
    print(f"[ok] parse_first_action: {len(al.action_list)} moves")

    al2 = Moves()
    g = Move("Single", "6", ["D6"])
    al2.parse_second_action(
        ["S3", "H4", "C5", "D6", "S7", "H8"], 0, "3", g, RANK_ORDER, NUMBER_ORDER,
    )
    assert len(al2.action_list) > 0
    assert al2.action_list[0] == ["PASS", "PASS", "PASS"]
    print("[ok] parse_second_action: first is PASS")


def test_environment_loop(duration_sec: float = 2.0) -> None:
    steps = 0
    env = _fresh_env()
    deadline = time.perf_counter() + duration_sec
    while time.perf_counter() < deadline:
        fn = env.loop
        if fn == env.play:
            steps += 1
            env.play({"actIndex": _pick_index(env)})
        elif fn == env.tribute:
            env.tribute({"actIndex": _pick_index(env)})
        elif fn == env.back:
            env.back({"actIndex": _pick_index(env)})
        elif fn == env.enter_tribute_stage:
            env.enter_tribute_stage()
        elif fn == env.start_new_episode_back_2:
            env.start_new_episode_back_2()
        elif fn == env.start:
            env.start()
        else:
            env = _fresh_env()
        if env.results is not None:
            env = _fresh_env()
    print(f"[ok] random walk ~{duration_sec}s, play steps ~{steps}")


def main() -> None:
    os.chdir(ROOT)
    print("== smoke ==\n")
    test_actions()
    test_environment_loop(2.0)
    print("\nok.")


if __name__ == "__main__":
    main()
