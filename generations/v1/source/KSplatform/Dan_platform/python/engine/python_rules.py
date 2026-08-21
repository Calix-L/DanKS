"""Pure Python legal action generation for Guandan.

The generator intentionally works on the public action tuple shape used by the
existing Java bridge: ``[pattern, rank, cards]``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

PASS_ACTION = ["PASS", "PASS", "PASS"]
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K")
SEQUENCE_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANK_VALUE = {"A": 14, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "B": 16, "R": 17}
NUMBER_VALUE = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13}
VALUE_RANK = {v: k for k, v in NUMBER_VALUE.items()}
SUIT_ORDER = {"S": 0, "H": 1, "C": 2, "D": 3}


def first_actions(hand_cards, hearts_num=0, current_rank="2"):
    labels = _sorted_labels(hand_cards, current_rank)
    hand_counts = Counter(labels)
    wild_cards = _wild_cards(labels, current_rank)
    natural_labels = _without_wild(labels, current_rank)
    sequence_labels = _without_jokers(natural_labels)
    actions = []
    seen = set()

    def add(action):
        action = _canonicalize_action(action, current_rank)
        if action is None:
            return
        key = _action_key(action)
        if key not in seen:
            seen.add(key)
            actions.append(action)

    by_rank = _by_rank(natural_labels)
    sequence_by_rank = _by_rank(sequence_labels)
    sequence_by_suit_rank = _by_suit_rank(sequence_labels)

    for label in labels:
        add(["Single", _action_rank(label[1], current_rank), [label]])

    same_rank_groups = {}
    for rank in _candidate_group_ranks(by_rank, current_rank):
        groups_by_size = {}
        for size in range(2, len(by_rank.get(rank, [])) + len(wild_cards) + 1):
            groups = list(_same_rank_groups(by_rank.get(rank, []), wild_cards, size, rank, current_rank, allow_pure_wild=True))
            if groups:
                groups_by_size[size] = groups
        same_rank_groups[rank] = groups_by_size

    for rank, groups_by_size in same_rank_groups.items():
        for combo in groups_by_size.get(2, []):
            add(["Pair", _action_rank(rank, current_rank), combo])
        for combo in groups_by_size.get(3, []):
            add(["Trips", _action_rank(rank, current_rank), combo])
        for size, groups in groups_by_size.items():
            if size >= 4:
                for combo in groups:
                    add(["Bomb", _action_rank(rank, current_rank), combo])

    jokers = [card for card in labels if card in ("SB", "HR")]
    if labels.count("SB") >= 2:
        add(["Pair", "B", ["SB", "SB"]])
    if labels.count("HR") >= 2:
        add(["Pair", "R", ["HR", "HR"]])
    if len(jokers) >= 4:
        add(["FourKings", "R", _take_cards(jokers, 4)])

    pair_groups = {rank: groups[2] for rank, groups in same_rank_groups.items() if 2 in groups}
    if labels.count("SB") >= 2:
        pair_groups["B"] = [["SB", "SB"]]
    if labels.count("HR") >= 2:
        pair_groups["R"] = [["HR", "HR"]]
    trips_groups = {rank: groups[3] for rank, groups in same_rank_groups.items() if 3 in groups}
    sequence_pair_groups = _sequence_same_rank_groups(sequence_by_rank, wild_cards, current_rank, 2)
    sequence_trips_groups = _sequence_same_rank_groups(sequence_by_rank, wild_cards, current_rank, 3)

    for trip_rank, trip_combos in trips_groups.items():
        for pair_rank, pair_combos in pair_groups.items():
            if pair_rank == trip_rank:
                continue
            for trip in trip_combos:
                for pair in pair_combos:
                    if _cards_available(hand_counts, trip, pair):
                        selected = trip + pair
                        add(["ThreeWithTwo", _action_rank(trip_rank, current_rank), selected])

    for seq in _rank_sequences(2, current_rank):
        if all(rank in sequence_trips_groups for rank in seq):
            for left in sequence_trips_groups[seq[0]]:
                for right in sequence_trips_groups[seq[1]]:
                    if _cards_available(hand_counts, left, right):
                        selected = left + right
                        add(["TwoTrips", _action_rank(seq[-1], current_rank), selected])

    for seq in _rank_sequences(3, current_rank):
        if all(rank in sequence_pair_groups for rank in seq):
            for a in sequence_pair_groups[seq[0]]:
                for b in sequence_pair_groups[seq[1]]:
                    for c in sequence_pair_groups[seq[2]]:
                        if _cards_available(hand_counts, a, b, c):
                            selected = a + b + c
                            add(["ThreePair", _action_rank(seq[-1], current_rank), selected])

    for seq in _rank_sequences(5, current_rank):
        for straight in _sequence_cards(seq, sequence_by_rank, wild_cards):
            if not _is_plain_straight_flush_interpretation(straight, current_rank):
                add(["Straight", _action_rank(seq[-1], current_rank), straight])
        for suit in ("S", "H", "C", "D"):
            for straight_flush in _sequence_cards(seq, sequence_by_suit_rank, wild_cards, suit=suit):
                add(["StraightFlush", _action_rank(seq[-1], current_rank), straight_flush])

    return actions


def second_actions(hand_cards, hearts_num, current_rank, greater_action):
    greater = _normalize_action(greater_action)
    actions = [PASS_ACTION]
    if not greater or greater[0] == "PASS":
        return [PASS_ACTION] + first_actions(hand_cards, hearts_num, current_rank)
    g_type, g_rank, g_cards = greater
    g_rv = _rank_value(g_rank, current_rank)
    labels = _sorted_labels(hand_cards, current_rank)
    hand_counts = Counter(labels)
    wild_cards = _wild_cards(labels, current_rank)
    natural_labels = _without_wild(labels, current_rank)
    sequence_labels = _without_jokers(natural_labels)
    by_rank = _by_rank(natural_labels)
    sequence_by_rank = _by_rank(sequence_labels)
    sequence_by_suit_rank = _by_suit_rank(sequence_labels)
    seen = set()

    def add(action):
        action = _canonicalize_action(action, current_rank)
        if action is None or not can_beat(action, greater, current_rank):
            return
        key = _action_key(action)
        if key not in seen:
            seen.add(key)
            actions.append(action)

    # Always add bomb-type actions that beat greater
    # FourKings
    jokers = [card for card in labels if card in ("SB", "HR")]
    if len(jokers) >= 4:
        fk = ["FourKings", "R", _take_cards(jokers, 4)]
        if can_beat(fk, greater, current_rank):
            add(fk)
    # StraightFlush
    for seq in _rank_sequences(5, current_rank):
        for suit in ("S", "H", "C", "D"):
            for sf in _sequence_cards(seq, sequence_by_suit_rank, wild_cards, suit=suit):
                action = ["StraightFlush", _action_rank(seq[-1], current_rank), sf]
                if can_beat(action, greater, current_rank):
                    add(action)
    # Bombs
    same_rank_groups = {}
    for rank in _candidate_group_ranks(by_rank, current_rank):
        groups_by_size = {}
        for size in range(2, len(by_rank.get(rank, [])) + len(wild_cards) + 1):
            groups = list(_same_rank_groups(by_rank.get(rank, []), wild_cards, size, rank, current_rank, allow_pure_wild=True))
            if groups:
                groups_by_size[size] = groups
        same_rank_groups[rank] = groups_by_size
    for rank, groups_by_size in same_rank_groups.items():
        for size, groups in groups_by_size.items():
            if size >= 4:
                for combo in groups:
                    action = ["Bomb", _action_rank(rank, current_rank), combo]
                    if can_beat(action, greater, current_rank):
                        add(action)
    if labels.count("SB") >= 2:
        action = ["Pair", "B", ["SB", "SB"]]
        if can_beat(action, greater, current_rank):
            add(action)
    if labels.count("HR") >= 2:
        action = ["Pair", "R", ["HR", "HR"]]
        if can_beat(action, greater, current_rank):
            add(action)

    # Add same-type actions that beat greater
    if g_type == "Single":
        for label in labels:
            if _rank_value(label[1], current_rank) > g_rv:
                add(["Single", _action_rank(label[1], current_rank), [label]])
    elif g_type == "Pair":
        pair_groups = {rank: groups[2] for rank, groups in same_rank_groups.items() if 2 in groups}
        if labels.count("SB") >= 2:
            pair_groups["B"] = [["SB", "SB"]]
        if labels.count("HR") >= 2:
            pair_groups["R"] = [["HR", "HR"]]
        for rank, combos in pair_groups.items():
            if _rank_value(rank, current_rank) > g_rv:
                for combo in combos:
                    add(["Pair", _action_rank(rank, current_rank), combo])
    elif g_type == "Trips":
        trips_groups = {rank: groups[3] for rank, groups in same_rank_groups.items() if 3 in groups}
        for rank, combos in trips_groups.items():
            if _rank_value(rank, current_rank) > g_rv:
                for combo in combos:
                    add(["Trips", _action_rank(rank, current_rank), combo])
    elif g_type == "ThreeWithTwo":
        trips_groups = {rank: groups[3] for rank, groups in same_rank_groups.items() if 3 in groups}
        pair_groups = {rank: groups[2] for rank, groups in same_rank_groups.items() if 2 in groups}
        if labels.count("SB") >= 2:
            pair_groups["B"] = [["SB", "SB"]]
        if labels.count("HR") >= 2:
            pair_groups["R"] = [["HR", "HR"]]
        g_card_count = _card_count(greater)
        for trip_rank, trip_combos in trips_groups.items():
            if _rank_value(trip_rank, current_rank) > g_rv:
                for pair_rank, pair_combos in pair_groups.items():
                    if pair_rank == trip_rank:
                        continue
                    for trip in trip_combos:
                        for pair in pair_combos:
                            if _cards_available(hand_counts, trip, pair):
                                selected = trip + pair
                                action = ["ThreeWithTwo", _action_rank(trip_rank, current_rank), selected]
                                if _card_count(action) == g_card_count:
                                    add(action)
    elif g_type == "TwoTrips":
        trips_groups = _sequence_same_rank_groups(sequence_by_rank, wild_cards, current_rank, 3)
        g_seq_rv = _natural_rank_value(g_rank)
        for seq in _rank_sequences(2, current_rank):
            if _natural_rank_value(seq[-1]) > g_seq_rv and all(rank in trips_groups for rank in seq):
                for left in trips_groups[seq[0]]:
                    for right in trips_groups[seq[1]]:
                        if _cards_available(hand_counts, left, right):
                            selected = left + right
                            add(["TwoTrips", _action_rank(seq[-1], current_rank), selected])
    elif g_type == "ThreePair":
        pair_groups = _sequence_same_rank_groups(sequence_by_rank, wild_cards, current_rank, 2)
        g_seq_rv = _natural_rank_value(g_rank)
        for seq in _rank_sequences(3, current_rank):
            if _natural_rank_value(seq[-1]) > g_seq_rv and all(rank in pair_groups for rank in seq):
                for a in pair_groups[seq[0]]:
                    for b in pair_groups[seq[1]]:
                        for c in pair_groups[seq[2]]:
                            if _cards_available(hand_counts, a, b, c):
                                selected = a + b + c
                                add(["ThreePair", _action_rank(seq[-1], current_rank), selected])
    elif g_type == "Straight":
        g_seq_rv = _natural_rank_value(g_rank)
        for seq in _rank_sequences(5, current_rank):
            if _natural_rank_value(seq[-1]) > g_seq_rv:
                for straight in _sequence_cards(seq, sequence_by_rank, wild_cards):
                    if not _is_plain_straight_flush_interpretation(straight, current_rank):
                        add(["Straight", _action_rank(seq[-1], current_rank), straight])
    elif g_type == "StraightFlush":
        # Already added all StraightFlush above, nothing more needed
        pass
    elif g_type == "Bomb":
        # Already added all bombs above, nothing more needed
        pass
    elif g_type == "FourKings":
        # Nothing beats FourKings
        pass

    return actions


def can_beat(action, greater_action, current_rank="2"):
    if not greater_action or greater_action[0] == "PASS":
        return action[0] != "PASS"
    if action[0] == "PASS":
        return False
    if _is_bomb_like(action):
        if not _is_bomb_like(greater_action):
            return True
        return _bomb_key(action, current_rank) > _bomb_key(greater_action, current_rank)
    if _is_bomb_like(greater_action):
        return False
    if action[0] != greater_action[0]:
        return False
    if _card_count(action) != _card_count(greater_action):
        return False
    if action[0] in ("Straight", "ThreePair", "TwoTrips"):
        return _natural_rank_value(action[1]) > _natural_rank_value(greater_action[1])
    return _rank_value(action[1], current_rank) > _rank_value(greater_action[1], current_rank)


def _normalize_action(action):
    if hasattr(action, "to_json"):
        return action.to_json()
    if isinstance(action, list):
        return action
    return [getattr(action, "type", None), getattr(action, "rank", None), getattr(action, "cards", None)]


def _sorted_labels(cards, current_rank):
    return sorted([str(card) for card in cards], key=lambda card: (_rank_value(card[1], current_rank), SUIT_ORDER.get(card[0], 9), card))


def _by_rank(labels):
    out = defaultdict(list)
    for label in labels:
        out[label[1]].append(label)
    return dict(out)


def _by_suit_rank(labels):
    out = defaultdict(list)
    for label in labels:
        if label not in ("SB", "HR"):
            out[(label[0], label[1])].append(label)
    return dict(out)


def _wild_cards(labels, current_rank):
    return [label for label in labels if label == f"H{current_rank}"]


def _without_wild(labels, current_rank):
    wild = f"H{current_rank}"
    return [label for label in labels if label != wild]


def _without_jokers(labels):
    return [label for label in labels if label not in ("SB", "HR")]


def _candidate_group_ranks(by_rank, current_rank):
    # Jokers are handled explicitly: two small jokers or two big jokers are
    # legal pairs, and four kings is the top bomb. They are not normal triples
    # or same-rank bombs.
    ranks = {rank for rank in by_rank if rank not in ("B", "R")}
    ranks.update(rank for rank in RANK_VALUE if rank not in ("B", "R"))
    return sorted(ranks, key=lambda rank: _rank_value(rank, current_rank))


def _same_rank_groups(natural_cards, wild_cards, size, rank=None, current_rank=None, allow_pure_wild=False):
    min_wild = max(0, size - len(natural_cards))
    max_wild = min(size, len(wild_cards))
    for wild_count in range(min_wild, max_wild + 1):
        natural_count = size - wild_count
        if natural_count == 0 and not allow_pure_wild and rank != current_rank:
            continue
        for natural in _combos(natural_cards, natural_count):
            for wild in _combos(wild_cards, wild_count):
                yield natural + wild


def _sequence_same_rank_groups(by_rank, wild_cards, current_rank, size):
    out = {}
    for rank in _candidate_group_ranks(by_rank, current_rank):
        groups = list(_same_rank_groups(by_rank.get(rank, []), wild_cards, size, rank, current_rank, allow_pure_wild=True))
        if groups:
            out[rank] = groups
    return out


def _cards_available(hand_counts, *groups):
    needed = Counter()
    for group in groups:
        needed.update(group)
    return all(count <= hand_counts[card] for card, count in needed.items())


def _sequence_cards(seq, lookup, wild_cards, suit=None):
    options = [([], 0)]
    for rank in seq:
        key = (suit, rank) if suit is not None else rank
        rebuilt = []
        for selected, used_wilds in options:
            for card in lookup.get(key, []):
                rebuilt.append((selected + [card], used_wilds))
            if used_wilds < len(wild_cards):
                rebuilt.append((selected + [wild_cards[used_wilds]], used_wilds + 1))
        options = rebuilt
    return [selected for selected, _used_wilds in options]


def _is_plain_straight_flush_interpretation(cards, current_rank):
    wild = f"H{current_rank}"
    suits = []
    for card in cards:
        if card not in ("SB", "HR") and card != wild:
            suits.append(card[0])
    return bool(suits) and len(set(suits)) == 1


def _plm_sequence_value(cards, current_rank, group_size):
    counts = [0] * 13
    wild = f"H{current_rank}"
    wild_count = cards.count(wild)
    wild_left = wild_count
    for card in cards:
        if card == wild and wild_left > 0:
            wild_left -= 1
            continue
        if card in ("SB", "HR"):
            return None
        counts[NUMBER_VALUE[card[1]] - 1] += 1
    return _plm_sequence_value_from_counts(counts, wild_count, group_size)


def _plm_sequence_value_from_counts(counts, laizi, group_size):
    idx_a = NUMBER_VALUE["A"] - 1
    idx_2 = NUMBER_VALUE["2"] - 1
    idx_k = NUMBER_VALUE["K"] - 1
    min_idx = idx_2
    while min_idx <= idx_k and counts[min_idx] == 0:
        min_idx += 1
    if min_idx > idx_k:
        return None
    max_idx = idx_k
    while max_idx > min_idx and counts[max_idx] == 0:
        max_idx -= 1
    for idx in range(min_idx, max_idx + 1):
        short = group_size - counts[idx]
        if short == 0:
            continue
        if short < 0 or laizi < short:
            return None
        laizi -= short
    if counts[idx_a] > 0:
        short = group_size - counts[idx_a]
        if short < 0 or laizi < short:
            return None
        laizi -= short
        if (idx_k - max_idx) * group_size <= laizi:
            return "A"
        short = (min_idx - idx_2) * group_size
        if short > 0:
            if short > laizi:
                return None
            laizi -= short
        if laizi > 0:
            max_idx += laizi // group_size
    elif laizi > 0:
        short = (idx_k - max_idx + 1) * group_size
        if short <= laizi:
            return "A"
        max_idx += laizi // group_size
    return VALUE_RANK.get(max_idx + 1)


def _plm_triple_plus_value(cards, current_rank):
    wild = f"H{current_rank}"
    counts = Counter(card[1] if card not in ("SB", "HR") else card[1] for card in cards)
    wild_count = cards.count(wild)
    if wild_count:
        counts[current_rank] -= wild_count
        if counts[current_rank] <= 0:
            del counts[current_rank]
    bj = counts.get("B", 0)
    rj = counts.get("R", 0)
    normal = [(rank, count) for rank, count in counts.items() if rank not in ("B", "R") and count > 0]
    if bj + rj > 0:
        if bj > 0 and rj > 0:
            return None
        if bj not in (0, 2) or rj not in (0, 2):
            return None
        if len(normal) != 1:
            return None
        return normal[0][0]
    if len(normal) != 2 or any(count > 3 for _rank, count in normal):
        return None
    normal.sort(key=lambda item: NUMBER_VALUE[item[0]])
    (r1, c1), (r2, c2) = normal
    if c1 == 3:
        return r1
    if c2 == 3:
        return r2
    return r2 if _rank_value(r2, current_rank) > _rank_value(r1, current_rank) else r1


def _canonicalize_action(action, current_rank):
    action_type, _rank, cards = action
    if action_type == "PASS":
        return action
    cards = list(cards)
    if action_type == "FourKings":
        rank = "R"
    elif action_type == "Single":
        rank = cards[0][1]
    elif action_type in ("Straight", "StraightFlush"):
        rank = _plm_sequence_value(cards, current_rank, 1)
    elif action_type == "ThreePair":
        rank = _plm_sequence_value(cards, current_rank, 2)
    elif action_type == "TwoTrips":
        rank = _plm_sequence_value(cards, current_rank, 3)
    elif action_type == "ThreeWithTwo":
        rank = _plm_triple_plus_value(cards, current_rank)
    else:
        wild = f"H{current_rank}"
        natural = [card for card in cards if card != wild]
        if not natural:
            rank = current_rank
        else:
            counts = Counter(card[1] for card in natural)
            rank = max(counts, key=lambda value: counts[value])
    if rank is None:
        return None
    return [action_type, _action_rank(rank, current_rank), cards]


def _combos(cards, size):
    seen = set()
    for combo in combinations(cards, size):
        combo = list(combo)
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            yield combo


def _rank_sequences(length, current_rank):
    seq_order = list(SEQUENCE_RANKS)
    for start in range(0, len(seq_order) - length + 1):
        yield seq_order[start:start + length]


def _first_card(cards):
    return cards[0]


def _take_cards(cards, size):
    return list(cards[:size])


def _action_rank(rank, current_rank):
    return rank


def _rank_value(rank, current_rank):
    if rank == current_rank:
        return 15
    return RANK_VALUE.get(rank, 0)


def _natural_rank_value(rank):
    return RANK_VALUE.get(rank, 0)


def _is_bomb_like(action):
    return action[0] in ("Bomb", "StraightFlush", "FourKings")


def _bomb_key(action, current_rank):
    if action[0] == "FourKings":
        return (100, 0, 0)
    if action[0] == "StraightFlush":
        return (70, 0, _natural_rank_value(action[1]))
    size = _card_count(action)
    if size >= 6:
        return (80 + min(size, 8), size, _rank_value(action[1], current_rank))
    if size == 5:
        return (60, size, _rank_value(action[1], current_rank))
    return (50, size, _rank_value(action[1], current_rank))


def _card_count(action):
    cards = action[2]
    return len(cards) if isinstance(cards, list) else 0


def _action_key(action):
    cards = action[2]
    if isinstance(cards, list):
        cards = tuple(cards)
    return action[0], action[1], cards
