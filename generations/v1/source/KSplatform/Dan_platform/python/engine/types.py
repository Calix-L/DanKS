"""Cards, moves, trick state, seat, wire JSON, finish order, compare helpers."""

from collections import Counter, namedtuple

RANK = ("", "", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")

DIGITAL = {
    "SA": 0x010e, "S2": 0x0102, "S3": 0x0103, "S4": 0x0104, "S5": 0x0105, "S6": 0x0106,
    "S7": 0x0107, "S8": 0x0108, "S9": 0x0109, "ST": 0x010a, "SJ": 0x010b, "SQ": 0x010c, "SK": 0x010d,
    "HA": 0x020e, "H2": 0x0202, "H3": 0x0203, "H4": 0x0204, "H5": 0x0205, "H6": 0x0206,
    "H7": 0x0207, "H8": 0x0208, "H9": 0x0209, "HT": 0x020a, "HJ": 0x020b, "HQ": 0x020c, "HK": 0x020d,
    "CA": 0x030e, "C2": 0x0302, "C3": 0x0303, "C4": 0x0304, "C5": 0x0305, "C6": 0x0306,
    "C7": 0x0307, "C8": 0x0308, "C9": 0x0309, "CT": 0x030a, "CJ": 0x030b, "CQ": 0x030c, "CK": 0x030d,
    "DA": 0x040e, "D2": 0x0402, "D3": 0x0403, "D4": 0x0404, "D5": 0x0405, "D6": 0x0406,
    "D7": 0x0407, "D8": 0x0408, "D9": 0x0409, "DT": 0x040a, "DJ": 0x040b, "DQ": 0x040c, "DK": 0x040d,
    "SB": 0x0110, "HR": 0x0211,
}


def fmt(cards):
    return [str(x) for x in cards]


def parse(labels):
    return [Card(s[0], s[1], DIGITAL[s]) for s in labels]


def cmp_hand(a, b, trump, suit_break):
    x = 15 if a.rank == trump else a.digital & 0x00FF
    y = 15 if b.rank == trump else b.digital & 0x00FF
    if x < y:
        return True
    if x > y:
        return False
    if suit_break:
        return False
    return (a.digital & 0xFF00) < (b.digital & 0xFF00)


def cmp_rank(a, b, order):
    if order[a] > order[b]:
        return -1
    if order[a] < order[b]:
        return 1
    return 0


class Phase:
    PLAY = "play"
    TRIBUTE = "tribute"
    BACK = "back"


class Card:
    __slots__ = ("suit", "rank", "digital")
    SUITS = ("S", "H", "C", "D")
    RANKS = ("", "", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")

    def __init__(self, suit, rank, digital):
        self.suit = suit
        self.rank = rank
        self.digital = digital

    def __eq__(self, other):
        if isinstance(other, str):
            return self.suit == other[0] and self.rank == other[1]
        return self.suit == other.suit and self.rank == other.rank

    def __str__(self):
        return self.suit + self.rank

    def __repr__(self):
        return str(self)


class Move:
    PASS = "PASS"

    def __init__(self, t=None, r=None, c=None):
        self._type = t
        self._rank = r
        self._cards = c

    def to_json(self):
        return [self._type, self._rank, self._cards]

    def clear(self):
        self._type = None
        self._rank = None
        self._cards = None

    @property
    def type(self):
        return self._type

    @property
    def rank(self):
        return self._rank

    @property
    def cards(self):
        return self._cards


Msg = namedtuple("Msg", ("seat", "body"))


class Trick:
    def __init__(self):
        self.deck_shuffle_times = 0
        self.current_pos = -1
        self.current_action = Move()
        self.greater_pos = -1
        self.greater_action = Move()
        self.rank_order = {
            "2": 15, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12,
            "K": 13, "A": 14, "B": 16, "R": 17,
        }
        self.number_order = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
            "Q": 12, "K": 13, "B": 16, "R": 17,
        }

    def action_info(self):
        return self.current_pos, self.current_action.to_json(), self.greater_pos, self.greater_action.to_json()

    def update_greater_action(self, action):
        self.greater_pos = self.current_pos
        self.greater_action = action

    def update_order(self, old_rank, new_rank):
        self.rank_order[old_rank] = RANK.index(old_rank)
        self.rank_order[new_rank] = 15

    def clear_action(self):
        self.current_pos = -1
        self.current_action.clear()
        self.greater_pos = -1
        self.greater_action.clear()

    def reset(self):
        self.deck_shuffle_times = 0
        self.clear_action()


class Undo:
    def __init__(self, stage_type, current_pos, current_action, greater_pos, greater_action, action_record,
                 is_action_first):
        self.type = stage_type
        self.current_pos = current_pos
        self.current_action = current_action
        self.greater_pos = greater_pos
        self.greater_action = greater_action
        self.action_record = action_record
        self.is_action_first = is_action_first
        self.info = {}


class Player:
    def __init__(self, name, index):
        self.name = name
        self.pos = index
        self.victory = 0
        self.hearts_num = 0
        self.hand_cards = []
        self.play_area = None
        self.rank = 2
        self.stuck_times = 0

    def hand2json(self):
        return [str(card) for card in self.hand_cards]

    def update_rank(self, inc):
        self.rank += inc
        self.rank = min(self.rank, 14)

    def public_info(self):
        return {"rest": len(self.hand_cards), "playArea": self.play_area.to_json() if self.play_area else None}

    def reset(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def play_card(self, cards, hearts_card):
        for card in cards:
            self.hand_cards.remove(card)
            if card == hearts_card:
                self.hearts_num -= 1

    def play_back(self, cards, current_rank):
        for card in cards:
            self.add_card(card, current_rank)
            if card == "H{}".format(current_rank):
                self.hearts_num += 1

    def add_card(self, card, current_rank):
        i = 0
        tmp = Card(card[0], card[1], digital=DIGITAL[card])
        while i != len(self.hand_cards):
            if cmp_hand(tmp, self.hand_cards[i], current_rank, False):
                self.hand_cards.insert(i, tmp)
                break
            i += 1
        if i == len(self.hand_cards):
            self.hand_cards.append(tmp)

    @property
    def red_joker_num(self):
        return sum(1 for card in self.hand_cards if card == "HR")

    def max_cards(self, rank, rank_card):
        candidates = [card for card in self.hand_cards if card != rank_card]
        if not candidates:
            return []
        highest = candidates[0]
        for card in candidates[1:]:
            if cmp_hand(highest, card, rank, False):
                highest = card
        cards = [str(c) for c in candidates if c.rank == highest.rank]
        return [["tribute", "tribute", [k]] for k in Counter(cards).keys()]

    def less_than_ten(self, current_rank):
        cards = [str(c) for c in self.hand_cards if c.digital & 0x00FF <= 10]
        return [["back", "back", [k]] for k in Counter(cards).keys()]


class Wire:
    @staticmethod
    def notify_beginning(hand_cards, index, cur_rank, pos_2_rank, oppo_rank):
        return {
            "type": "notify", "stage": "beginning",
            "handCards": [str(c) for c in hand_cards], "myPos": index,
            "curRank": cur_rank, "selfRank": pos_2_rank, "oppoRank": oppo_rank,
        }

    @staticmethod
    def notify_play(cur_pos, cur_action, greater_pos, greater_action):
        return {
            "type": "notify", "stage": "play",
            "curPos": cur_pos, "curAction": cur_action, "greaterPos": greater_pos, "greaterAction": greater_action,
        }

    @staticmethod
    def notify_episode_over(cur_rank, over_order, rest_cards):
        return {"type": "notify", "stage": "episodeOver", "curRank": cur_rank, "order": over_order, "restCards": rest_cards}

    @staticmethod
    def notify_game_over(cur_times, setting_times):
        return {"type": "notify", "stage": "gameOver", "curTimes": cur_times, "settingTimes": setting_times}

    @staticmethod
    def notify_game_result(respective_wins, respective_draws):
        return {"type": "notify", "stage": "gameResult", "victoryNum": respective_wins, "draws": respective_draws}

    @staticmethod
    def notify_tribute(tribute_cards):
        return {"type": "notify", "stage": "tribute", "result": tribute_cards}

    @staticmethod
    def notify_back(back_cards):
        return {"type": "notify", "stage": "back", "result": back_cards}

    @staticmethod
    def notify_anti_tribute(anti_num, anti_pos):
        return {"type": "notify", "stage": "anti-tribute", "antiNum": anti_num, "antiPos": anti_pos}

    @staticmethod
    def act(stage, hand_cards, public_info, self_rank, oppo_rank, cur, cur_pos, c_act, g_pos, g_act, action_list):
        return {
            "type": "act", "stage": stage, "handCards": hand_cards, "publicInfo": public_info,
            "selfRank": self_rank, "oppoRank": oppo_rank, "curRank": cur, "curPos": cur_pos,
            "curAction": c_act, "greaterPos": g_pos, "greaterAction": g_act,
            "actionList": action_list, "indexRange": len(action_list) - 1,
        }


class Finish:
    def __init__(self):
        self.order = []
        self.tri_ship = []
        self.bck_ship = []
        self.tri_cards = []
        self.bck_cards = []
        self.index = 0
        self.first_play = 0

    def add(self, pos):
        self.order.append(pos)

    def settlement(self):
        if (self.order[0] + 2) % 4 == self.order[1]:
            inc = 3
            order = [0, 1, 2, 3]
            self.tri_ship.append((self.order[-1], order[self.order[-1] - 1]))
            self.tri_ship.append((self.order[-2], order[self.order[-2] - 1]))
            self.bck_ship.append((order[self.order[-1] - 1], self.order[-1]))
            self.bck_ship.append((order[self.order[-2] - 1], self.order[-2]))
        elif (self.order[0] + 2) % 4 == self.order[2]:
            inc = 2
            self.tri_ship.append((self.order[-1], self.order[0]))
            self.bck_ship.append((self.order[0], self.order[-1]))
        else:
            inc = 1
            self.tri_ship.append((self.order[-1], self.order[0]))
            self.bck_ship.append((self.order[0], self.order[-1]))
        return self.order[0], (self.order[0] + 2) % 4, inc

    def episode_over(self):
        return (self.order[-1] + 2) % 4 in self.order

    def clear(self):
        self.first_play = -1
        self.index = 0
        self.order.clear()
        self.tri_ship.clear()
        self.bck_ship.clear()
        self.tri_cards.clear()
        self.bck_cards.clear()

    def find_ship(self, pos, _ship):
        ship = getattr(self, "{}_ship".format(_ship))
        for s in ship:
            if pos in s:
                return s

    @property
    def first(self):
        return self.order[0]

    @property
    def second(self):
        return self.order[1]

    @property
    def third(self):
        return self.order[2]

    @property
    def fourth(self):
        return self.order[3]
