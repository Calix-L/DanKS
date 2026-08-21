#!/usr/bin/env python3
"""Start one four-player GuanDan table with the shared DanKS engine."""

from __future__ import annotations

from guandan import Environment


def main() -> None:
    game = Environment(first_player=0)
    for seat in range(4):
        game.add_player(f"player-{seat}", seat)

    messages = game.start()
    hand_sizes = [len(player.hand_cards) for player in game.players]
    if hand_sizes != [27, 27, 27, 27] or not messages:
        raise RuntimeError("the shared engine did not create a valid four-player table")

    print("DanKS engine ready: 4 players, 27 cards each")


if __name__ == "__main__":
    main()
