from guandan.engine import Environment, Move, Moves


RANK_ORDER = {
    "2": 15, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17,
}
NUMBER_ORDER = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "B": 16, "R": 17,
}


def test_move_generation_supports_lead_and_follow() -> None:
    lead = Moves()
    lead.parse_first_action(["S3", "H3", "C4", "D5", "S6", "H7"], 0, "3")
    assert lead.action_list

    follow = Moves()
    follow.parse_second_action(
        ["S3", "H4", "C5", "D6", "S7", "H8"],
        0,
        "3",
        Move("Single", "6", ["D6"]),
        RANK_ORDER,
        NUMBER_ORDER,
    )
    assert follow.action_list[0] == ["PASS", "PASS", "PASS"]


def test_environment_starts_a_four_player_table() -> None:
    environment = Environment(first_player=0)
    for seat in range(4):
        environment.add_player(f"p{seat}", seat)
    messages = environment.start()

    assert len(environment.players) == 4
    assert all(len(player.hand_cards) == 27 for player in environment.players)
    assert messages
