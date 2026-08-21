"""Legal move list backed by the bundled Python rule implementation."""

from .python_rules import first_actions, second_actions
from .types import Move


class Moves:
    def __init__(self):
        self.valid_range = range(0, 1)
        self.action_list = []

    def __len__(self):
        return len(self.action_list)

    def __getitem__(self, item):
        return Move(*self.action_list[item])

    def parse_first_action(self, hand_cards, hearts_num, current_rank):
        if not hand_cards:
            self.action_list = []
        else:
            self.action_list = first_actions(hand_cards, hearts_num, current_rank)
        self.valid_range = range(0, len(self.action_list))

    def parse_second_action(
        self,
        hand_cards,
        hearts_num,
        current_rank,
        greater_action,
        rank_order,
        number_order,
    ):
        self.action_list = second_actions(
            hand_cards,
            hearts_num,
            current_rank,
            greater_action,
        )
        self.valid_range = range(0, len(self.action_list))
