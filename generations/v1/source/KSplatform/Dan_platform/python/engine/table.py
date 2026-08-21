"""Table runtime: play / tribute / back phases, optional undo trace."""

import copy
import logging
from copy import deepcopy
from random import randint, shuffle

from .types import (
    Card, Phase, Move, Msg, Trick, Undo, Player, Wire, Finish,
    fmt, parse, cmp_hand, cmp_rank,
)
from .moves import Moves


class Table:
    """108-card Guandan table; loop points at the active phase handler (start/play/tribute/back)."""

    # --- lifecycle & deck -------------------------------------------------

    def __init__(self, allow_step_back=False, deck_data=None, first_player=None):
        self.__cards = []
        self.__rank = 2
        self.shuffle_times = 0
        self.deck = None
        self.players = []
        self.rank_belongs = -1
        self.state = Trick()
        self.legal_moves = Moves()
        self.settlement = Finish()
        self.results = None
        self.loop = self.start

        self.allow_step_back = allow_step_back
        self.trace = []

        self.deck_data = deck_data
        self.first_player = first_player

        if self.deck_data is None:
            for i in range(2):
                digital_form = 0x0101
                for s in Card.SUITS:
                    for r in Card.RANKS[2:]:
                        digital_form += 1
                        self.__cards.append(Card(s, r, digital_form))
                    digital_form = digital_form & 0x0F01
                    digital_form += 0x0101
                self.__cards.append(Card('S', 'B', 0x0110))  # small joker
                self.__cards.append(Card('H', 'R', 0x0211))  # big joker
        else:
            assert len(self.deck_data) == 108
            self.deck = parse(self.deck_data)

        self.action_first = None
        self.belongs_to = None
        self.rank_inc = None

    @property
    def rank(self):
        return Card.RANKS[self.__rank]

    def set_rank(self, rank, belongs):
        """Set trump rank index and which partnership owns that rank."""
        self.__rank = rank
        self.rank_belongs = belongs

    def add_player(self, name, index):
        self.players.append(Player(name, index))

    def reshuffle(self):
        """Build a fresh shuffled deck and refresh ``deck_data`` string."""
        self.deck = deepcopy(self.__cards)
        shuffle(self.deck)
        self.deck_data = fmt(self.deck)

    def deal(self):
        """Deal 27 cards to each seat; insert-sort hands by rank/suit; count wild hearts."""
        count = 107
        for i in range(4):
            if len(self.players[i].hand_cards) != 0:
                self.players[i].hand_cards = []
            self.players[i].hearts_num = 0
            for j in range(27):
                if self.deck[count].rank == self.rank and self.deck[count].suit == 'H':
                    self.players[i].hearts_num += 1

                # Sort while dealing: current trump order, then suit order S,H,C,D
                index = 0
                while index != len(self.players[i].hand_cards):
                    if cmp_hand(self.deck[count], self.players[i].hand_cards[index], self.rank, False):
                        self.players[i].hand_cards.insert(index, self.deck[count])
                        break
                    index += 1
                if index == len(self.players[i].hand_cards):
                    self.players[i].hand_cards.append(self.deck[count])
                count -= 1
                self.deck.pop()

    # --- episode & phase transitions ----------------------------------------

    def start(self):
        """Start a round: reset seats, shuffle/deal, enqueue beginning + first ``act`` for lead."""

        for player in self.players:
            player.reset(play_area=None, stuck_times=0, rank=2, hearts_num=0)
        self.state.update_order('2', '2')
        if self.belongs_to is not None:
            self.set_rank(rank=2, belongs=self.belongs_to)

        if self.deck_data is None:
            self.reshuffle()
        self.deal()

        pending_messages = []
        self.settlement.clear()
        self.state.reset()
        if self.first_player is None:
            cur_pos = randint(0, 3)
        else:
            cur_pos = self.first_player
        self.first_action(cur_pos)
        self.action_first = True
        self.notify_beginning(pending_messages)
        self.act(pending_messages, cur_pos, Phase.PLAY)
        self.state.current_pos = cur_pos
        self.loop = self.play
        if self.allow_step_back:
            self.trace = []
        return pending_messages

    def start_new_episode_back_2(self):
        self.trace = []
        pending_messages = []
        self.players[self.belongs_to].rank = 2
        self.players[(self.belongs_to + 2) % 4].rank = 2
        self.players[self.belongs_to].stuck_times = 0
        self.players[(self.belongs_to + 2) % 4].stuck_times = 0
        for player in self.players:
            player.reset(play_area=None)
        self.state.update_order(old_rank=self.rank, new_rank='2')
        self.set_rank(self.players[self.belongs_to].rank, belongs=self.belongs_to)
        self.reshuffle()
        self.deal()
        self.act_tribute(pending_messages=pending_messages)
        return pending_messages

    def enter_tribute_stage(self):
        """Apply rank bump, reset hands, shuffle, then enter tribute messaging."""
        pending_messages = []
        self.players[self.belongs_to].update_rank(self.rank_inc)
        self.players[(self.belongs_to + 2) % 4].update_rank(self.rank_inc)
        self.state.update_order(old_rank=self.rank, new_rank=Card.RANKS[self.players[self.belongs_to].rank])
        self.set_rank(self.players[self.belongs_to].rank, belongs=self.belongs_to)
        for player in self.players:
            player.reset(play_area=None, hearts_num=0)
        self.reshuffle()
        self.deal()
        self.act_tribute(pending_messages=pending_messages)
        return pending_messages

    def act_tribute(self, pending_messages):
        self.loop = self.tribute
        self.notify_beginning(pending_messages)
        fourth_anti = False
        if self.players[self.settlement.fourth].red_joker_num == 2:
            fourth_anti = True
        if len(self.settlement.tri_ship) == 2:
            third_anti = False
            if self.players[self.settlement.third].red_joker_num == 2:
                third_anti = True
            if fourth_anti is True or third_anti is True:
                self.notify_anti_tribute(pending_messages, 2, [self.settlement.third, self.settlement.fourth])
                self.first_action(self.settlement.first)
                self.action_first = True
                self.act(pending_messages, self.settlement.first, Phase.PLAY)
                self.state.current_pos = self.settlement.first
                self.loop = self.play
                self.settlement.clear()
            else:
                self.legal_moves.action_list = self.players[self.settlement.fourth].max_cards(self.rank, "H" + self.rank)
                self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
                self.act(pending_messages, self.settlement.fourth, Phase.TRIBUTE)
                self.state.current_pos = self.settlement.fourth
            if self.allow_step_back:
                self.trace = []
            return pending_messages
        else:
            if fourth_anti:
                self.settlement.tri_ship = []
                self.settlement.bck_ship = []
                self.notify_anti_tribute(pending_messages, 1, [self.settlement.fourth])
                self.first_action(self.settlement.first)
                self.action_first = True
                self.act(pending_messages, self.settlement.first, Phase.PLAY)
                self.state.current_pos = self.settlement.first
                self.loop = self.play
                self.settlement.clear()
            else:
                self.legal_moves.action_list = self.players[self.settlement.fourth].max_cards(self.rank, "H" + self.rank)
                self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
                self.act(pending_messages, self.settlement.fourth, Phase.TRIBUTE)
                self.state.current_pos = self.settlement.fourth
            if self.allow_step_back:
                self.trace = []
            return pending_messages

    # --- main phase handlers ------------------------------------------------

    def play(self, content):

        action = self.legal_moves[content["actIndex"]]
        if self.allow_step_back:
            undo_frame = Undo(Phase.PLAY, self.state.current_pos, copy.deepcopy(self.state.current_action),
                                     self.state.greater_pos, copy.deepcopy(self.state.greater_action),
                                     copy.deepcopy(action), self.action_first)
        else:
            undo_frame = None
        self.state.current_action = action
        self.players[self.state.current_pos].play_area = action
        pending_messages = []
        if action.type == Move.PASS:
            self.notify_play(pending_messages)
            if self._lead_passes_after_pass_chain():
                cur_pos = (self.state.greater_pos + 2) % 4
                self.state.clear_action()
                self.first_action(cur_pos)
                self.action_first = True
            else:
                cur_pos = self._next_seat_with_cards()
                if cur_pos == self.state.greater_pos:
                    self.state.clear_action()
                    self.first_action(cur_pos)
                    self.action_first = True
                else:
                    self.second_action(cur_pos)
                    self.action_first = False
            self.act(pending_messages, cur_pos, Phase.PLAY)

            self.state.current_pos = cur_pos
            if self.allow_step_back:
                self.trace.append(undo_frame)
            return pending_messages
        else:
            player = self.players[self.state.current_pos]
            player.play_card(action.cards, "H{}".format(self.rank))
            player.play_area = action
            self.state.update_greater_action(action)
            self.notify_play(pending_messages)
            if len(player.hand_cards) == 0:
                if undo_frame is not None:
                    undo_frame.info['over_order'] = copy.deepcopy(self.settlement)
                self.settlement.add(self.state.current_pos)
                if self.settlement.episode_over():
                    for i in range(4):
                        if len(self.players[i].hand_cards) > 0:
                            self.settlement.add(i)
                    first_pos, _, inc = self.settlement.settlement()
                    self.notify_episode_over(pending_messages)
                    if self.__rank == 14:
                        if (self.rank_belongs + 2) % 4 == first_pos or self.rank_belongs == first_pos:
                            if (first_pos + 2) % 4 == self.settlement.fourth:
                                self.stuck_at_ace(first_pos, (first_pos + 2) % 4, inc, pending_messages, undo_frame)
                            else:
                                self.one_times_over(first_pos, pending_messages, undo_frame)
                        else:
                            self.belongs_to = first_pos
                            self.rank_inc = inc
                            self.loop = self.enter_tribute_stage
                    else:
                        self.belongs_to = first_pos
                        self.rank_inc = inc
                        self.loop = self.enter_tribute_stage
                else:
                    self.next_player_second_action(pending_messages)
            else:
                self.next_player_second_action(pending_messages)
            if self.allow_step_back:
                self.trace.append(undo_frame)
            return pending_messages

    def tribute(self, content):
        pending_messages = []

        tribute_pos, to = self.settlement.find_ship(self.state.current_pos, _ship="tri")
        action = self.legal_moves[content["actIndex"]]
        if self.allow_step_back:
            undo_frame = Undo(Phase.TRIBUTE, self.state.current_pos, copy.deepcopy(self.state.current_action),
                                     self.state.greater_pos, copy.deepcopy(self.state.greater_action),
                                     copy.deepcopy(action), self.action_first)
            undo_frame.info['over_order'] = copy.deepcopy(self.settlement)
        else:
            undo_frame = None
        self.settlement.tri_cards.append((tribute_pos, to, action.cards[0]))
        self.players[tribute_pos].play_card(action.cards, "H{}".format(self.rank))
        if self.settlement.index == 0 and len(self.settlement.tri_ship) == 2:
            self.settlement.index += 1
            tribute_pos, _ = self.settlement.tri_ship[-1]
            self.state.clear_action()
            self.legal_moves.action_list = self.players[tribute_pos].max_cards(self.rank, "H" + self.rank)
            self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
            self.act(pending_messages, tribute_pos, Phase.TRIBUTE)
            self.state.current_pos = tribute_pos
            if self.allow_step_back:
                self.trace.append(undo_frame)
            return pending_messages
        else:
            self._resolve_double_tribute_mapping()
            for tribute_tuple in self.settlement.tri_cards:
                tribute_pos, to, card = tribute_tuple
                self.players[to].add_card(card, self.rank)
            if len(self.settlement.tri_cards) != 2:
                self.settlement.first_play = self.settlement.fourth
            self.notify_tribute(pending_messages)
            self.settlement.index = 0
            back, to = self.settlement.bck_ship[0]
            self.state.current_pos = -1
            self.state.current_action = None
            self.legal_moves.action_list = self.players[back].less_than_ten(self.rank)
            self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
            self.act(pending_messages, back, Phase.BACK)
            self.state.current_pos = back
            self.loop = self.back
            if self.allow_step_back:
                self.trace.append(undo_frame)
            return pending_messages

    def back(self, content=None):
        back_pos, to = self.settlement.find_ship(self.state.current_pos, _ship="bck")
        action = self.legal_moves[content["actIndex"]]
        if self.allow_step_back:
            undo_frame = Undo(Phase.BACK, self.state.current_pos, copy.deepcopy(self.state.current_action),
                                     self.state.greater_pos, copy.deepcopy(self.state.greater_action),
                                     copy.deepcopy(action), self.action_first)
            undo_frame.info['over_order'] = copy.deepcopy(self.settlement)
        else:
            undo_frame = None
        self.settlement.bck_cards.append((back_pos, to, action.cards[0]))
        self.players[back_pos].play_card(action.cards, "H{}".format(self.rank))
        pending_messages = []
        if self.settlement.index == 0 and len(self.settlement.bck_ship) == 2:
            self.settlement.index += 1
            back_pos, _ = self.settlement.bck_ship[-1]
            self.legal_moves.action_list = self.players[back_pos].less_than_ten(self.rank)
            self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
            self.act(pending_messages, back_pos, Phase.BACK)
            self.state.current_pos = back_pos
        else:
            for bck_tuple in self.settlement.bck_cards:
                _, to, card = bck_tuple
                self.players[to].add_card(card, self.rank)
            self.notify_back(pending_messages)
            self.settlement.index = 0
            self.state.current_pos = -1
            self.first_action(self.settlement.first_play)
            self.action_first = True
            self.act(pending_messages, self.settlement.first_play, Phase.PLAY)
            self.state.current_pos = self.settlement.first_play
            self.loop = self.play
            self.settlement.clear()
        if self.allow_step_back:
            self.trace.append(undo_frame)
        return pending_messages

    def _resolve_double_tribute_mapping(self):
        """Map double tribute by PDF rule: bigger tribute card goes to upper."""
        if len(self.settlement.tri_cards) != 2:
            return
        tribute_a, tribute_b = self.settlement.tri_cards[0], self.settlement.tri_cards[1]
        cmp_result = cmp_rank(tribute_a[2][1], tribute_b[2][1], self.state.rank_order)
        if cmp_result == -1:
            high, low = tribute_a, tribute_b
        elif cmp_result == 1:
            high, low = tribute_b, tribute_a
        elif tribute_a[0] == self.settlement.fourth:
            high, low = tribute_a, tribute_b
        else:
            high, low = tribute_b, tribute_a

        upper = self.settlement.first
        upper_partner = (upper + 2) % 4
        high_from, _, high_card = high
        low_from, _, low_card = low
        self.settlement.tri_cards = [(high_from, upper, high_card), (low_from, upper_partner, low_card)]
        self.settlement.bck_ship = [(upper, high_from), (upper_partner, low_from)]
        self.settlement.first_play = high_from

    # --- move lists (JAR) ---------------------------------------------------

    def first_action(self, cur_pos):
        """Refresh legal moves for the opening player at ``cur_pos``."""
        self.legal_moves.parse_first_action(self.players[cur_pos].hand_cards, self.players[cur_pos].hearts_num, self.rank)

    def second_action(self, cur_pos):
        """Refresh legal moves when following a trick."""
        self.legal_moves.parse_second_action(
            self.players[cur_pos].hand_cards,
            self.players[cur_pos].hearts_num,
            self.rank,
            self.state.greater_action,
            self.state.rank_order,
            self.state.number_order
        )

    def next_player_second_action(self, pending_messages):
        """Advance to the next non-empty seat and enqueue second-player legal moves."""
        current_pos = self._next_seat_with_cards()
        self.second_action(current_pos)
        self.action_first = False
        self.act(pending_messages, current_pos, Phase.PLAY)
        self.state.current_pos = current_pos

    def generate_action_list(self, stage, is_action_first, hand_cards, hearts_num, rank, greater_action, rank_order,
                             number_order):
        if stage == Phase.TRIBUTE:
            self.legal_moves.action_list = self.players[self.state.current_pos].max_cards(self.rank, "H" + self.rank)
            self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
        elif stage == Phase.BACK:
            self.legal_moves.action_list = self.players[self.state.current_pos].less_than_ten(self.rank)
            self.legal_moves.valid_range = range(0, len(self.legal_moves.action_list))
        elif stage == Phase.PLAY:
            if is_action_first:
                self.legal_moves.parse_first_action(hand_cards, hearts_num, rank)
            else:
                self.legal_moves.parse_second_action(hand_cards, hearts_num, rank, greater_action, rank_order, number_order)
        return copy.deepcopy(self.legal_moves.action_list), copy.deepcopy(self.legal_moves.valid_range)

    # --- outbound wire helpers ----------------------------------------------

    def notify_beginning(self, msg_list):
        """Episode start: per-seat hand snapshot and ranks."""
        for i in range(len(self.players)):
            player = self.players[i]
            opponent = self.players[(i + 1) % 4]
            msg_list.append(
                Msg(
                    player.pos,
                    Wire.notify_beginning(
                        player.hand_cards, player.pos, self.__rank, player.rank, opponent.rank
                    )
                )
            )

    def notify_play(self, msg_list):
        """Broadcast current trick state after a play."""
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_play(*self.state.action_info())))

    def notify_episode_over(self, msg_list):
        """Hand over: remaining cards and finish order."""
        res = [[player.pos, player.hand2json()] for player in self.players if player.hand_cards]
        order = deepcopy(self.settlement.order)
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_episode_over(self.rank, order, res)))

    def notify_game_over(self, msg_list):
        """Match end signal (session uses fixed counters in payload)."""
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_game_over(1, 1)))

    def notify_game_result(self, msg_list):
        """Broadcast per-seat win/draw tallies."""
        respective_wins = [player.victory for player in self.players]
        respective_draws = [0, 0, 0, 0]
        self.results = [player.victory for player in self.players]
        logging.info(
            "session end: wins seat0=%s seat1=%s seat2=%s seat3=%s",
            *respective_wins,
        )
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_game_result(respective_wins, respective_draws)))

    def notify_tribute(self, msg_list):
        """Show tribute cards after tribute resolves."""
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_tribute(deepcopy(self.settlement.tri_cards))))

    def notify_back(self, msg_list):
        """Show return cards after back phase."""
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_back(deepcopy(self.settlement.bck_cards))))

    def notify_anti_tribute(self, msg_list, anti_num, anti_pos):
        """Anti-tribute: how many seats refused and which positions."""
        for player in self.players:
            msg_list.append(Msg(player.pos, Wire.notify_anti_tribute(anti_num, anti_pos)))

    def act(self, msg_list, cur_pos, stage):
        """Ask ``cur_pos`` to act in ``stage`` with full public snapshot + legal list."""
        player = self.players[cur_pos]
        opponent = self.players[(cur_pos + 1) % 4]
        cur_action = self.state.current_action.to_json() if self.state.current_action else None
        greater_action = self.state.greater_action.to_json() if self.state.greater_action else None
        act_args = (
            stage,
            [str(card) for card in player.hand_cards],
            [p.public_info() for p in self.players],
            Card.RANKS[player.rank],
            Card.RANKS[opponent.rank],
            self.rank,
            self.state.current_pos,
            cur_action,
            self.state.greater_pos,
            greater_action,
            self.legal_moves.action_list
        )
        msg_list.append(Msg(cur_pos, Wire.act(*act_args)))

    # --- scoring & episode outcomes -----------------------------------------

    def add_victory_num(self, pos, teammate_pos):
        """Credit a win to a partnership."""
        self.players[pos].victory += 1
        self.players[teammate_pos].victory += 1

    def minus_victory_num(self, pos, teammate_pos):
        self.players[pos].victory -= 1
        self.players[teammate_pos].victory -= 1

    def stuck_at_ace(self, first_pos, teammate_pos, rank_inc, pending_messages, undo_frame):
        """Ace-level stall: teammate finishes last while team is on ace; may reshuffle or tribute."""
        if undo_frame is not None:
            undo_frame.info['stuck_times'] = {'first_pos': [first_pos, self.players[first_pos].stuck_times],
                                               'teammate_pos': [first_pos, self.players[teammate_pos].stuck_times]}
        self.players[first_pos].stuck_times += 1
        self.players[teammate_pos].stuck_times += 1

        if self.players[first_pos].stuck_times > 3:
            if undo_frame is not None:
                undo_frame.info['shuffle_times'] = self.shuffle_times
            self.shuffle_times += 1
            if self.shuffle_times >= 50:
                self.notify_game_over(pending_messages)
                winner, teammate = self.shuffle_times_exceeds_threshold()
                if undo_frame is not None:
                    undo_frame.info['victory'] = [winner, teammate]
                self.add_victory_num(winner, teammate)
                self.notify_game_result(pending_messages)
            else:
                self.belongs_to = first_pos
                self.loop = self.start_new_episode_back_2
        else:
            self.belongs_to = first_pos
            self.rank_inc = rank_inc
            self.loop = self.enter_tribute_stage

    def one_times_over(self, first_pos, pending_messages, undo_frame):
        if undo_frame is not None:
            undo_frame.info['victory'] = [first_pos, (first_pos + 2) % 4]
        self.add_victory_num(first_pos, (first_pos + 2) % 4)
        self.notify_game_over(pending_messages)
        self.notify_game_result(pending_messages)

    def shuffle_times_exceeds_threshold(self):
        """After many ace downgrades, force settlement by comparing partnership ranks."""
        if self.players[0].rank > self.players[1].rank:
            return 0, 2
        elif self.players[0].rank < self.players[1].rank:
            return 1, 3
        else:
            if self.rank_belongs in (0, 2):
                return 0, 2
            return 1, 3

    # --- rule micro-checks (private) ----------------------------------------

    def _lead_passes_after_pass_chain(self):
        """After PASS, whether trick leadership jumps per Guandan pass rules."""
        c_pos, g_pos = self.state.current_pos, self.state.greater_pos
        prerequisite_a = (c_pos + 1) % 4 == g_pos
        prerequisite_b = len(self.players[self.state.greater_pos].hand_cards) == 0

        prerequisite_A = (c_pos + 2) % 4 == g_pos
        prerequisite_B = len(self.players[self.state.greater_pos].hand_cards) == 0 \
                         and len(self.players[(c_pos + 1) % 4].hand_cards) == 0

        return (prerequisite_a and prerequisite_b) or (prerequisite_A and prerequisite_B)

    def _next_seat_with_cards(self):
        """Next clockwise seat from current trick position that still has cards."""
        for i in range(1, 4):
            if len(self.players[(self.state.current_pos + i) % 4].hand_cards) > 0:
                return (self.state.current_pos + i) % 4

    # --- validation, undo, hand editing -------------------------------------

    def validate(self, pos, action):
        """True if ``pos`` is the acting seat and ``actIndex`` is in range."""
        if pos == self.state.current_pos and "actIndex" in action and \
                action["actIndex"] in self.legal_moves.valid_range:
            return True
        return False

    def loop_back(self):
        assert self.trace
        undo_frame = self.trace.pop()
        self.action_first = undo_frame.is_action_first
        self.state.current_pos = undo_frame.current_pos
        self.state.current_action = undo_frame.current_action
        self.state.greater_pos = undo_frame.greater_pos
        self.state.greater_action = undo_frame.greater_action

        if undo_frame.action_record.type != "PASS":
            self.players[self.state.current_pos].play_back(undo_frame.action_record.cards, self.rank)
        hand_cards = fmt(self.players[self.state.current_pos].hand_cards)

        if undo_frame.type == Phase.PLAY:
            self.loop = self.play
        elif undo_frame.type == Phase.TRIBUTE:
            self.loop = self.tribute
        elif undo_frame.type == Phase.BACK:
            self.loop = self.back

        if 'stuck_times' in undo_frame.info.keys():
            self.players[undo_frame.info['stuck_times']['first_pos'][0]].stuck_times = \
                undo_frame.info['stuck_times']['first_pos'][1]
            self.players[undo_frame.info['stuck_times']['teammate_pos'][0]].stuck_times = \
                undo_frame.info['stuck_times']['teammate_pos'][1]
        if 'shuffle_times' in undo_frame.info.keys():
            self.shuffle_times = undo_frame.info['shuffle_times']
        if 'victory' in undo_frame.info.keys():
            self.minus_victory_num(undo_frame.info['victory'][0], undo_frame.info['victory'][1])
        if 'over_order' in undo_frame.info.keys():
            self.settlement = undo_frame.info['over_order']

        legal_actions, _ = self.generate_action_list(undo_frame.type, undo_frame.is_action_first,
                                                     self.players[undo_frame.current_pos].hand_cards,
                                                     self.players[undo_frame.current_pos].hearts_num, self.rank,
                                                     undo_frame.greater_action, self.state.rank_order,
                                                     self.state.number_order)
        return undo_frame.type, self.state.current_pos, legal_actions, hand_cards

    def get_hand_card(self):
        return [fmt(self.players[i].hand_cards) for i in range(4)]

    def change_hand_card(self, hand_card_all):
        for i in range(4):
            hand_card = hand_card_all[i]
            hand_card = parse(hand_card)
            self.players[i].hand_cards = []
            self.players[i].hearts_num = 0
            for card in hand_card:
                if card.rank == self.rank and card.suit == 'H':
                    self.players[i].hearts_num += 1
                index = 0
                while index != len(self.players[i].hand_cards):
                    if cmp_hand(card, self.players[i].hand_cards[index], self.rank, False):
                        self.players[i].hand_cards.insert(index, card)
                        break
                    index += 1
                if index == len(self.players[i].hand_cards):
                    self.players[i].hand_cards.append(card)
        self.deck = parse(self.deck_data)
        if self.loop == self.play:
            stage = Phase.PLAY
        elif self.loop == self.tribute:
            stage = Phase.TRIBUTE
        elif self.loop == self.back:
            stage = Phase.BACK
        else:
            raise Exception('Wrong loop ' + str(self.loop))

        hand_card_all = [fmt(self.players[i].hand_cards) for i in range(4)]
        legal_actions, _ = self.generate_action_list(stage, self.action_first,
                                                     self.players[self.state.current_pos].hand_cards,
                                                     self.players[self.state.current_pos].hearts_num, self.rank,
                                                     self.state.greater_action, self.state.rank_order,
                                                     self.state.number_order)

        return hand_card_all, legal_actions
