#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr const char* kNativeBucketsCapsuleName = "danrl.native_buckets.v1";
using NativeEncodedGroup = std::vector<unsigned short>;
using NativeBuckets = std::vector<std::vector<NativeEncodedGroup>>;
using EncodedCoverInputs = std::pair<std::vector<unsigned char>, NativeBuckets>;

const std::array<const char*, 54> kCards = {
    "C3", "D3", "H3", "S3",
    "C4", "D4", "H4", "S4",
    "C5", "D5", "H5", "S5",
    "C6", "D6", "H6", "S6",
    "C7", "D7", "H7", "S7",
    "C8", "D8", "H8", "S8",
    "C9", "D9", "H9", "S9",
    "CT", "DT", "HT", "ST",
    "CJ", "DJ", "HJ", "SJ",
    "CQ", "DQ", "HQ", "SQ",
    "CK", "DK", "HK", "SK",
    "CA", "DA", "HA", "SA",
    "C2", "D2", "H2", "S2",
    "BJ", "RJ",
};

const std::array<const char*, 54> kBeamCards = {
    "S3", "H3", "C3", "D3", "S4", "H4", "C4", "D4",
    "S5", "H5", "C5", "D5", "S6", "H6", "C6", "D6",
    "S7", "H7", "C7", "D7", "S8", "H8", "C8", "D8",
    "S9", "H9", "C9", "D9", "ST", "HT", "CT", "DT",
    "SJ", "HJ", "CJ", "DJ", "SQ", "HQ", "CQ", "DQ",
    "SK", "HK", "CK", "DK", "SA", "HA", "CA", "DA",
    "S2", "H2", "C2", "D2", "BJ", "RJ",
};

const std::unordered_map<std::string, int>& card_index() {
    static const std::unordered_map<std::string, int> index = [] {
        std::unordered_map<std::string, int> out;
        out.reserve(kCards.size());
        for (int i = 0; i < static_cast<int>(kCards.size()); ++i) {
            out.emplace(kCards[i], i);
        }
        return out;
    }();
    return index;
}

int index_for(const std::string& card) {
    const auto& index = card_index();
    const auto it = index.find(card);
    if (it == index.end()) {
        throw std::invalid_argument("invalid card label: " + card);
    }
    return it->second;
}

std::string rank_for(const std::string& card) {
    if (card == "BJ" || card == "RJ") {
        return card;
    }
    if (card.size() == 2) {
        return card.substr(1, 1);
    }
    throw std::invalid_argument("invalid card label: " + card);
}

std::string heart_level_card_for(const std::string& cur_rank) {
    if (cur_rank.size() == 1) {
        return "H" + cur_rank;
    }
    return "";
}

double rank_strength_for(
    const std::string& rank,
    const std::string& cur_rank,
    int remaining_rj
) {
    static const std::unordered_map<std::string, int> values = {
        {"2", 0}, {"3", 1}, {"4", 2}, {"5", 3}, {"6", 4},
        {"7", 5}, {"8", 6}, {"9", 7}, {"T", 8}, {"J", 9},
        {"Q", 10}, {"K", 11}, {"A", 12}, {"BJ", 13}, {"RJ", 14},
    };
    const auto found = values.find(rank);
    double base = found == values.end() ? 0.0 : static_cast<double>(found->second) / 14.0;
    if (!cur_rank.empty() && rank == cur_rank) {
        base = std::max(base, 12.5 / 14.0);
    }
    if (remaining_rj >= 0) {
        if (rank == "BJ" && remaining_rj == 0) {
            base = 0.98;
        } else if (rank == "RJ") {
            base = 1.0;
        }
    }
    return base;
}

int action_type_size_for(const std::string& kind) {
    static const std::unordered_map<std::string, int> sizes = {
        {"PASS", 0}, {"Single", 1}, {"Pair", 2}, {"Triple", 3},
        {"Straight", 5}, {"StraightPair", 6}, {"StraightTriple", 6},
        {"FullHouse", 5}, {"TriplePlus", 5}, {"Bomb", 4},
        {"StraightFlush", 5}, {"FourKings", 4},
    };
    const auto found = sizes.find(kind);
    return found == sizes.end() ? 1 : found->second;
}

bool is_bomb_kind(const std::string& kind) {
    return kind == "Bomb" || kind == "StraightFlush" || kind == "FourKings";
}

double pressure_for(
    const std::string& kind,
    const std::array<int, 4>& public_counts,
    int my_seat
) {
    const int target_size = action_type_size_for(kind);
    if (target_size <= 0) {
        return 0.0;
    }
    double best = 0.0;
    for (int seat = 0; seat < 4; ++seat) {
        const int left = public_counts[seat];
        if (seat == my_seat || (seat % 2) == (my_seat % 2) || left <= 0) {
            continue;
        }
        const int distance = (seat - my_seat + 4) % 4;
        const double position = distance == 1 ? 1.0 : distance == 2 ? 0.45 : distance == 3 ? 0.7 : 0.0;
        const double count_match = std::exp(-std::abs(left - target_size) / 1.5);
        const double threat = std::min(1.0, 10.0 / std::max(1.0, static_cast<double>(left)));
        best = std::max(best, count_match * position * threat);
    }
    return std::max(0.0, std::min(1.0, best));
}

double short_opponent_pressure_for(const std::array<int, 4>& public_counts, int my_seat) {
    double best = 0.0;
    for (int seat = 0; seat < 4; ++seat) {
        const int left = public_counts[seat];
        if (seat == my_seat || (seat % 2) == (my_seat % 2) || left <= 0) {
            continue;
        }
        if (left == 1) {
            best = std::max(best, 1.0);
        } else if (left == 2) {
            best = std::max(best, 0.75);
        } else if (left <= 4) {
            best = std::max(best, 0.35);
        }
    }
    return best;
}

std::vector<std::array<double, 6>> batch_action_static_features(
    const std::vector<py::tuple>& actions,
    const std::string& cur_rank,
    const std::string& current_rank,
    const std::string& current_kind,
    const std::array<int, 4>& public_counts,
    int my_seat,
    int last_player,
    int remaining_rj
) {
    std::vector<std::array<double, 6>> out;
    out.reserve(actions.size());
    const bool is_lead = current_kind == "Lead";
    const double short_pressure = short_opponent_pressure_for(public_counts, my_seat);
    const int teammate = (my_seat + 2) % 4;
    for (const auto& item : actions) {
        const auto kind = item[0].cast<std::string>();
        const auto cards = item[1].cast<std::vector<std::string>>();
        const auto rank = item[2].cast<std::string>();
        const int size = static_cast<int>(cards.size());
        const bool pass = kind == "PASS" || cards.empty();
        const bool bomb = is_bomb_kind(kind);
        const double rank_score = rank.empty() ? 0.0 : rank_strength_for(rank, current_rank, -1);
        const double action_strength = rank.empty() ? 0.0 : rank_strength_for(rank, cur_rank, remaining_rj);

        double current_control = 0.0;
        if (!pass && !is_lead) {
            const auto& pressure_kind = bomb ? current_kind : kind;
            double p = pressure_for(pressure_kind, public_counts, my_seat);
            if (pressure_kind == current_kind) {
                p = std::max(p, 0.35);
            }
            double strength = action_strength;
            if (rank.empty()) {
                for (const auto& card : cards) {
                    strength = std::max(strength, rank_strength_for(rank_for(card), cur_rank, remaining_rj));
                }
            }
            if (bomb) {
                strength = std::max(strength, 0.75);
            }
            current_control = p * strength * 100.0;
        }

        double lead_action = 0.0;
        if (!pass && is_lead) {
            if (kind == "Single") {
                lead_action = 8.0 + (1.0 - action_strength) * 35.0;
            } else {
                static const std::unordered_map<std::string, double> base = {
                    {"FourKings", 180.0}, {"StraightFlush", 130.0}, {"Bomb", 90.0},
                    {"StraightTriple", 40.0}, {"StraightPair", 34.0}, {"TriplePlus", 28.0},
                    {"Straight", 24.0}, {"Triple", 18.0}, {"Pair", 10.0}, {"Single", 4.0},
                };
                static const std::unordered_map<std::string, double> multiplier = {
                    {"Pair", 32.0}, {"Triple", 14.0}, {"TriplePlus", 10.0},
                    {"Straight", 12.0}, {"StraightPair", 13.0}, {"StraightTriple", 14.0},
                    {"StraightFlush", 1.2}, {"Bomb", 0.8}, {"FourKings", 0.4},
                };
                const auto base_it = base.find(kind);
                double intrinsic = base_it == base.end() ? 0.0 : base_it->second;
                if (kind == "Bomb") {
                    intrinsic += std::max(0, size - 4) * 14.0;
                }
                const auto multiplier_it = multiplier.find(kind);
                const double structure_multiplier = multiplier_it == multiplier.end() ? 0.0 : multiplier_it->second;
                lead_action = 95.0 * std::max(0, size - 1) + intrinsic * structure_multiplier;
            }
        }

        double spend = 0.0;
        if (!pass) {
            if (kind == "FourKings") spend = 2.5;
            else if (kind == "StraightFlush") spend = 1.8;
            else if (kind == "Bomb") spend = 1.0 + std::max(0, size - 4) * 0.25;
            else if (kind == "Single") spend = std::max(0.0, action_strength - 0.72);
            else if (kind == "Pair") spend = std::max(0.0, action_strength - 0.78) * 1.2;
        }
        double escape = 0.0;
        if (short_pressure > 0.0) {
            if (pass) escape = short_pressure;
            else escape = short_pressure * std::max(0.0, (is_lead ? 0.55 : 0.45) - current_control / 100.0);
        }
        double overcall = 0.0;
        if (!pass && !is_lead && last_player == teammate) {
            overcall = bomb ? 1.25 : 1.0;
        }
        out.push_back({rank_score, current_control, lead_action, spend, escape, overcall});
    }
    return out;
}

std::vector<double> batch_break_group_penalties(
    const std::vector<py::tuple>& actions,
    const std::vector<std::vector<py::tuple>>& partitions,
    const std::unordered_map<std::string, double>& base_by_kind,
    double straight_flush_to_bomb,
    double bomb_break_size_bonus
) {
    struct ActionRecord {
        std::string kind;
        std::array<unsigned char, 54> counts{};
        int size = 0;
    };
    std::vector<ActionRecord> action_records;
    action_records.reserve(actions.size());
    for (const auto& item : actions) {
        ActionRecord record;
        record.kind = item[0].cast<std::string>();
        const auto cards = item[1].cast<std::vector<std::string>>();
        record.size = static_cast<int>(cards.size());
        for (const auto& card : cards) record.counts[index_for(card)] += 1;
        action_records.push_back(std::move(record));
    }
    std::vector<double> penalties(actions.size(), 0.0);
    for (const auto& partition : partitions) {
        for (const auto& group : partition) {
            const auto group_kind = group[0].cast<std::string>();
            if (group_kind == "Single") continue;
            const auto cards = group[1].cast<std::vector<std::string>>();
            const int group_size = static_cast<int>(cards.size());
            std::array<unsigned char, 54> group_counts{};
            for (const auto& card : cards) group_counts[index_for(card)] += 1;
            for (std::size_t action_idx = 0; action_idx < action_records.size(); ++action_idx) {
                const auto& action = action_records[action_idx];
                if (action.kind == "PASS" || action.size == 0) continue;
                int overlap = 0;
                for (int card_idx = 0; card_idx < 54; ++card_idx) {
                    overlap += std::min(action.counts[card_idx], group_counts[card_idx]);
                }
                if (overlap <= 0) continue;
                const bool same_cards = overlap == group_size && overlap == action.size;
                double severity = 0.0;
                if (same_cards && action.kind == group_kind) {
                    severity = 0.0;
                } else if (group_kind == "StraightFlush" && action.kind == "Bomb") {
                    severity = straight_flush_to_bomb;
                } else if (same_cards) {
                    severity = 0.15;
                } else {
                    const auto base_it = base_by_kind.find(group_kind);
                    double base = base_it == base_by_kind.end() ? 0.0 : base_it->second;
                    if (group_kind == "Bomb") base += (group_size - 4) * bomb_break_size_bonus;
                    if (group_kind == "Straight" || group_kind == "StraightPair" ||
                        group_kind == "StraightTriple" || group_kind == "StraightFlush") {
                        base += (group_size - 5) * 1.6;
                    }
                    const double fraction = static_cast<double>(overlap) / group_size;
                    severity = base * (0.55 + 0.45 * fraction);
                }
                penalties[action_idx] = std::max(penalties[action_idx], severity);
            }
        }
    }
    return penalties;
}

const std::array<const char*, 13>& normal_ranks() {
    static const std::array<const char*, 13> ranks = {
        "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"
    };
    return ranks;
}

const std::array<const char*, 14>& straight_ranks() {
    static const std::array<const char*, 14> ranks = {
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
    };
    return ranks;
}

std::unordered_map<std::string, int> rank_position_map() {
    std::unordered_map<std::string, int> out;
    const auto& ranks = normal_ranks();
    for (int i = 0; i < static_cast<int>(ranks.size()); ++i) {
        out.emplace(ranks[i], i);
    }
    return out;
}

std::vector<std::string> sorted_hand_from_cards(const std::vector<std::string>& hand) {
    std::array<int, 54> counts{};
    for (const auto& card : hand) {
        counts[index_for(card)] += 1;
    }
    std::vector<std::string> out;
    out.reserve(hand.size());
    for (int idx = 0; idx < static_cast<int>(kCards.size()); ++idx) {
        for (int n = 0; n < counts[idx]; ++n) {
            out.emplace_back(kCards[idx]);
        }
    }
    return out;
}

bool is_plain_straight_flush_interpretation(const std::vector<std::string>& selected) {
    std::string first_suit;
    for (const auto& card : selected) {
        if (card == "BJ" || card == "RJ") {
            continue;
        }
        const auto suit = card.substr(0, 1);
        if (first_suit.empty()) {
            first_suit = suit;
        } else if (suit != first_suit) {
            return false;
        }
    }
    return !first_suit.empty();
}

std::string signature_key(
    const std::string& kind,
    const std::string& rank,
    const std::vector<std::string>& cards,
    const std::vector<std::string>& wild_as
) {
    std::string key;
    key.reserve(32 + cards.size() * 3 + wild_as.size() * 2);
    key.append(kind).push_back('|');
    key.append(rank).push_back('|');
    for (const auto& card : cards) {
        key.append(card).push_back(',');
    }
    key.push_back('|');
    for (const auto& item : wild_as) {
        key.append(item).push_back(',');
    }
    return key;
}

thread_local bool g_tuple_signature_payloads = false;

class TupleSignaturePayloadGuard {
public:
    explicit TupleSignaturePayloadGuard(bool enabled)
        : previous_(g_tuple_signature_payloads) {
        g_tuple_signature_payloads = enabled;
    }

    ~TupleSignaturePayloadGuard() {
        g_tuple_signature_payloads = previous_;
    }

    TupleSignaturePayloadGuard(const TupleSignaturePayloadGuard&) = delete;
    TupleSignaturePayloadGuard& operator=(const TupleSignaturePayloadGuard&) = delete;

private:
    bool previous_;
};

py::tuple strings_as_tuple(const std::vector<std::string>& values) {
    py::tuple result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = py::str(values[i]);
    }
    return result;
}

void add_group_signature(
    std::vector<py::tuple>& out,
    std::unordered_set<std::string>& seen,
    const std::string& kind,
    const std::string& rank,
    std::vector<std::string> cards,
    const std::vector<std::string>& wild_as
) {
    std::sort(cards.begin(), cards.end(), [](const std::string& left, const std::string& right) {
        return index_for(left) < index_for(right);
    });
    const auto key = signature_key(kind, rank, cards, wild_as);
    if (!seen.insert(key).second) {
        return;
    }
    if (g_tuple_signature_payloads) {
        out.push_back(py::make_tuple(
            kind,
            rank,
            strings_as_tuple(cards),
            strings_as_tuple(wild_as)
        ));
    } else {
        out.push_back(py::make_tuple(kind, rank, cards, wild_as));
    }
}

void choose_combinations(
    const std::vector<std::string>& values,
    int choose,
    int start,
    std::vector<std::string>& current,
    const std::function<void(const std::vector<std::string>&)>& emit
) {
    if (static_cast<int>(current.size()) == choose) {
        emit(current);
        return;
    }
    const int remaining = choose - static_cast<int>(current.size());
    for (int i = start; i <= static_cast<int>(values.size()) - remaining; ++i) {
        current.push_back(values[i]);
        choose_combinations(values, choose, i + 1, current, emit);
        current.pop_back();
    }
}

using SequenceOptions = std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>;

SequenceOptions rank_sequence_options(
    const std::vector<std::string>& wildcards,
    const std::vector<std::string>& ranks,
    const std::unordered_map<std::string, std::vector<std::string>>& cards_by_rank
) {
    SequenceOptions options;
    options.push_back({{}, {}});
    for (const auto& rank : ranks) {
        const auto found = cards_by_rank.find(rank);
        const auto empty = std::vector<std::string>{};
        const auto& candidates = found == cards_by_rank.end() ? empty : found->second;
        SequenceOptions rebuilt;
        for (const auto& option : options) {
            const auto& selected = option.first;
            const auto& wild_as = option.second;
            const auto used = wild_as.size();
            for (const auto& card : candidates) {
                auto next_selected = selected;
                next_selected.push_back(card);
                rebuilt.push_back({std::move(next_selected), wild_as});
            }
            if (used < wildcards.size()) {
                auto next_selected = selected;
                auto next_wild_as = wild_as;
                next_selected.push_back(wildcards[used]);
                next_wild_as.push_back(rank);
                rebuilt.push_back({std::move(next_selected), std::move(next_wild_as)});
            }
        }
        options = std::move(rebuilt);
    }
    return options;
}

SequenceOptions multi_rank_sequence_options(
    const std::vector<std::string>& wildcards,
    const std::vector<std::string>& ranks,
    int need_per_rank,
    const std::unordered_map<std::string, std::vector<std::string>>& cards_by_rank
) {
    SequenceOptions options;
    options.push_back({{}, {}});
    for (const auto& rank : ranks) {
        const auto found = cards_by_rank.find(rank);
        const auto empty = std::vector<std::string>{};
        const auto& rank_cards = found == cards_by_rank.end() ? empty : found->second;
        SequenceOptions rebuilt;
        for (const auto& option : options) {
            const auto& selected = option.first;
            const auto& wild_as = option.second;
            const auto used = wild_as.size();
            const int min_natural = std::max(0, need_per_rank - static_cast<int>(wildcards.size() - used));
            const int max_natural = std::min(need_per_rank, static_cast<int>(rank_cards.size()));
            for (int n_natural = min_natural; n_natural <= max_natural; ++n_natural) {
                const int n_wild = need_per_rank - n_natural;
                if (used + static_cast<std::size_t>(n_wild) > wildcards.size()) {
                    continue;
                }
                std::vector<std::string> current;
                current.reserve(n_natural);
                choose_combinations(rank_cards, n_natural, 0, current, [&](const std::vector<std::string>& naturals) {
                    auto next_selected = selected;
                    auto next_wild_as = wild_as;
                    next_selected.insert(next_selected.end(), naturals.begin(), naturals.end());
                    for (int i = 0; i < n_wild; ++i) {
                        next_selected.push_back(wildcards[used + i]);
                        next_wild_as.push_back(rank);
                    }
                    rebuilt.push_back({std::move(next_selected), std::move(next_wild_as)});
                });
            }
        }
        options = std::move(rebuilt);
    }
    return options;
}

bool native_reuse_same_rank_groups_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DANRL_NATIVE_REUSE_SAME_RANK_GROUPS");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

}  // namespace

std::vector<std::string> remove_cards_sorted(
    const std::vector<std::string>& hand,
    const std::vector<std::string>& action
) {
    std::array<int, 54> counts{};
    for (const auto& card : hand) {
        counts[index_for(card)] += 1;
    }
    for (const auto& card : action) {
        const int idx = index_for(card);
        if (counts[idx] <= 0) {
            throw std::invalid_argument("action card " + card + " not in hand");
        }
        counts[idx] -= 1;
    }

    std::vector<std::string> out;
    out.reserve(hand.size() >= action.size() ? hand.size() - action.size() : 0);
    for (int idx = 0; idx < static_cast<int>(kCards.size()); ++idx) {
        for (int n = 0; n < counts[idx]; ++n) {
            out.emplace_back(kCards[idx]);
        }
    }
    return out;
}

std::vector<std::vector<std::string>> remove_cards_sorted_batch(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& actions
) {
    std::array<int, 54> base_counts{};
    for (const auto& card : hand) {
        base_counts[index_for(card)] += 1;
    }

    std::vector<std::vector<std::string>> results;
    results.reserve(actions.size());
    for (const auto& action : actions) {
        auto counts = base_counts;
        for (const auto& card : action) {
            const int idx = index_for(card);
            if (counts[idx] <= 0) {
                throw std::invalid_argument("action card " + card + " not in hand");
            }
            counts[idx] -= 1;
        }

        std::vector<std::string> out;
        out.reserve(hand.size() >= action.size() ? hand.size() - action.size() : 0);
        for (int idx = 0; idx < static_cast<int>(kCards.size()); ++idx) {
            for (int n = 0; n < counts[idx]; ++n) {
                out.emplace_back(kCards[idx]);
            }
        }
        results.push_back(std::move(out));
    }
    return results;
}

EncodedCoverInputs encode_cover_inputs(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    std::array<int, 54> full_counts{};
    for (const auto& card : hand) {
        full_counts[index_for(card)] += 1;
    }

    std::vector<int> global_to_local(54, -1);
    std::vector<unsigned char> start;
    start.reserve(hand.size());
    for (int global_idx = 0; global_idx < static_cast<int>(kCards.size()); ++global_idx) {
        const int count = full_counts[global_idx];
        if (count <= 0) {
            continue;
        }
        global_to_local[global_idx] = static_cast<int>(start.size());
        start.push_back(static_cast<unsigned char>(count));
    }

    NativeBuckets buckets(start.size());
    for (std::size_t group_id = 0; group_id < group_cards.size(); ++group_id) {
        const auto& cards = group_cards[group_id];
        if (cards.empty()) {
            throw std::invalid_argument("group_cards must not contain empty groups");
        }
        std::vector<unsigned char> local_counts(start.size(), 0);
        for (const auto& card : cards) {
            const int global_idx = index_for(card);
            const int local_idx = global_to_local[global_idx];
            if (local_idx < 0) {
                throw std::invalid_argument("group card " + card + " not in hand");
            }
            local_counts[static_cast<std::size_t>(local_idx)] += 1;
        }
        std::vector<std::pair<int, int>> items;
        items.reserve(cards.size());
        for (std::size_t local_idx = 0; local_idx < local_counts.size(); ++local_idx) {
            if (local_counts[local_idx] != 0) {
                items.emplace_back(
                    static_cast<int>(local_idx),
                    static_cast<int>(local_counts[local_idx])
                );
            }
        }
        std::vector<unsigned short> encoded;
        encoded.reserve(1 + items.size() * 2);
        encoded.push_back(static_cast<unsigned short>(group_id));
        for (const auto& item : items) {
            encoded.push_back(static_cast<unsigned short>(item.first));
            encoded.push_back(static_cast<unsigned short>(item.second));
        }
        buckets[items.front().first].push_back(std::move(encoded));
    }

    return {std::move(start), std::move(buckets)};
}

py::capsule native_buckets_capsule(NativeBuckets buckets) {
    auto* owned = new NativeBuckets(std::move(buckets));
    return py::capsule(
        owned,
        kNativeBucketsCapsuleName,
        [](PyObject* capsule) {
            void* pointer = PyCapsule_GetPointer(capsule, kNativeBucketsCapsuleName);
            if (pointer == nullptr) {
                PyErr_Clear();
                return;
            }
            delete static_cast<NativeBuckets*>(pointer);
        }
    );
}

py::tuple build_cover_inputs(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    auto encoded = encode_cover_inputs(hand, group_cards);
    return py::make_tuple(std::move(encoded.first), std::move(encoded.second));
}

py::tuple build_cover_inputs_capsule(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    auto encoded = encode_cover_inputs(hand, group_cards);
    return py::make_tuple(
        std::move(encoded.first),
        native_buckets_capsule(std::move(encoded.second))
    );
}

EncodedCoverInputs encode_cover_inputs_beam_order(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    std::unordered_map<std::string, int> beam_index;
    beam_index.reserve(kBeamCards.size());
    for (int index = 0; index < static_cast<int>(kBeamCards.size()); ++index) {
        beam_index.emplace(kBeamCards[index], index);
    }
    std::array<int, 54> full_counts{};
    for (const auto& card : hand) {
        const auto found = beam_index.find(card);
        if (found == beam_index.end()) {
            throw std::invalid_argument("invalid card label: " + card);
        }
        full_counts[found->second] += 1;
    }
    std::vector<int> global_to_local(54, -1);
    std::vector<unsigned char> start;
    for (int global_idx = 0; global_idx < static_cast<int>(kBeamCards.size()); ++global_idx) {
        if (full_counts[global_idx] <= 0) continue;
        global_to_local[global_idx] = static_cast<int>(start.size());
        start.push_back(static_cast<unsigned char>(full_counts[global_idx]));
    }
    NativeBuckets buckets(start.size());
    for (std::size_t group_id = 0; group_id < group_cards.size(); ++group_id) {
        std::vector<unsigned char> local_counts(start.size(), 0);
        for (const auto& card : group_cards[group_id]) {
            const auto found = beam_index.find(card);
            if (found == beam_index.end()) {
                throw std::invalid_argument("invalid card label: " + card);
            }
            const int local_idx = global_to_local[found->second];
            if (local_idx < 0) {
                throw std::invalid_argument("group card " + card + " not in hand");
            }
            local_counts[static_cast<std::size_t>(local_idx)] += 1;
        }
        std::vector<unsigned short> encoded;
        encoded.push_back(static_cast<unsigned short>(group_id));
        std::size_t first = local_counts.size();
        for (std::size_t local_idx = 0; local_idx < local_counts.size(); ++local_idx) {
            if (local_counts[local_idx] == 0) continue;
            if (first == local_counts.size()) first = local_idx;
            encoded.push_back(static_cast<unsigned short>(local_idx));
            encoded.push_back(static_cast<unsigned short>(local_counts[local_idx]));
        }
        if (first == local_counts.size()) {
            throw std::invalid_argument("group_cards must not contain empty groups");
        }
        buckets[first].push_back(std::move(encoded));
    }
    return {std::move(start), std::move(buckets)};
}

py::tuple build_cover_inputs_beam_order(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    auto encoded = encode_cover_inputs_beam_order(hand, group_cards);
    return py::make_tuple(std::move(encoded.first), std::move(encoded.second));
}

py::tuple build_cover_inputs_beam_order_capsule(
    const std::vector<std::string>& hand,
    const std::vector<std::vector<std::string>>& group_cards
) {
    auto encoded = encode_cover_inputs_beam_order(hand, group_cards);
    return py::make_tuple(
        std::move(encoded.first),
        native_buckets_capsule(std::move(encoded.second))
    );
}

static std::vector<py::tuple> generate_same_rank_group_signatures_from_sorted(
    const std::vector<std::string>& sorted_hand,
    const std::string& cur_rank
) {
    const auto& ranks = normal_ranks();
    std::vector<std::vector<std::string>> natural_by_rank(ranks.size());
    auto rank_to_pos = rank_position_map();
    const auto wild_card = heart_level_card_for(cur_rank);
    std::vector<std::string> wildcards;
    int bj_count = 0;
    int rj_count = 0;
    for (const auto& card : sorted_hand) {
        if (card == "BJ") {
            bj_count += 1;
            continue;
        }
        if (card == "RJ") {
            rj_count += 1;
            continue;
        }
        const auto rank = rank_for(card);
        if (!wild_card.empty() && card == wild_card) {
            wildcards.push_back(card);
        } else {
            const auto found = rank_to_pos.find(rank);
            if (found != rank_to_pos.end()) {
                natural_by_rank[found->second].push_back(card);
            }
        }
    }

    std::vector<py::tuple> out;
    std::unordered_set<std::string> seen;
    seen.reserve(256);

    for (const auto& card : sorted_hand) {
        add_group_signature(out, seen, "Single", rank_for(card), std::vector<std::string>{card}, {});
    }

    for (int rank_idx = 0; rank_idx < static_cast<int>(ranks.size()); ++rank_idx) {
        const std::string rank = ranks[rank_idx];
        const auto& naturals = natural_by_rank[rank_idx];
        const int max_total = static_cast<int>(naturals.size() + wildcards.size());
        for (int target_size = 2; target_size <= max_total; ++target_size) {
            const std::string kind = target_size == 2 ? "Pair" : target_size == 3 ? "Triple" : "Bomb";
            const int min_natural = std::max(0, target_size - static_cast<int>(wildcards.size()));
            const int max_natural = std::min(target_size, static_cast<int>(naturals.size()));
            for (int n_natural = min_natural; n_natural <= max_natural; ++n_natural) {
                const int n_wild = target_size - n_natural;
                std::vector<std::string> selected_wilds;
                selected_wilds.reserve(n_wild);
                for (int i = 0; i < n_wild; ++i) {
                    selected_wilds.push_back(wildcards[i]);
                }
                std::vector<std::string> wild_as(n_wild, rank);
                std::vector<std::string> current;
                current.reserve(n_natural);
                choose_combinations(naturals, n_natural, 0, current, [&](const std::vector<std::string>& chosen) {
                    std::vector<std::string> cards = chosen;
                    cards.insert(cards.end(), selected_wilds.begin(), selected_wilds.end());
                    add_group_signature(out, seen, kind, rank, cards, wild_as);
                });
            }
        }
    }

    if (bj_count >= 2) {
        add_group_signature(out, seen, "Pair", "BJ", std::vector<std::string>{"BJ", "BJ"}, {});
    }
    if (rj_count >= 2) {
        add_group_signature(out, seen, "Pair", "RJ", std::vector<std::string>{"RJ", "RJ"}, {});
    }
    if (bj_count >= 2 && rj_count >= 2) {
        add_group_signature(out, seen, "FourKings", "RJ", std::vector<std::string>{"BJ", "BJ", "RJ", "RJ"}, {});
    }

    return out;
}

std::vector<py::tuple> generate_same_rank_group_signatures(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    return generate_same_rank_group_signatures_from_sorted(
        sorted_hand_from_cards(hand), cur_rank
    );
}

static std::vector<py::tuple> generate_sequence_group_signatures_from_maps(
    const std::unordered_map<std::string, std::vector<std::string>>& by_rank,
    const std::unordered_map<std::string, std::vector<std::string>>& by_suit_rank
) {
    const std::vector<std::string> no_wildcards;
    std::vector<py::tuple> out;
    std::unordered_set<std::string> seen;
    seen.reserve(256);
    const auto& ranks_all = straight_ranks();
    static const std::array<const char*, 4> suits = {"S", "H", "C", "D"};
    for (int start = 0; start <= static_cast<int>(ranks_all.size()) - 5; ++start) {
        std::vector<std::string> ranks;
        ranks.reserve(5);
        for (int offset = 0; offset < 5; ++offset) {
            ranks.emplace_back(ranks_all[start + offset]);
        }
        const auto tail_rank = ranks.back();
        for (const auto& option : rank_sequence_options(no_wildcards, ranks, by_rank)) {
            if (is_plain_straight_flush_interpretation(option.first)) {
                continue;
            }
            add_group_signature(out, seen, "Straight", tail_rank, option.first, option.second);
        }
        for (const auto* suit : suits) {
            std::unordered_map<std::string, std::vector<std::string>> suited_by_rank;
            for (const auto& rank : ranks) {
                const auto found = by_suit_rank.find(std::string(suit) + "|" + rank);
                if (found != by_suit_rank.end()) {
                    suited_by_rank.emplace(rank, found->second);
                }
            }
            for (const auto& option : rank_sequence_options(no_wildcards, ranks, suited_by_rank)) {
                add_group_signature(out, seen, "StraightFlush", tail_rank, option.first, option.second);
            }
        }
    }
    return out;
}

std::vector<py::tuple> generate_sequence_group_signatures(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    (void)cur_rank;
    const auto sorted_hand = sorted_hand_from_cards(hand);
    std::unordered_map<std::string, std::vector<std::string>> by_rank;
    std::unordered_map<std::string, std::vector<std::string>> by_suit_rank;
    for (const auto& card : sorted_hand) {
        if (card == "BJ" || card == "RJ") continue;
        const auto rank = rank_for(card);
        by_rank[rank].push_back(card);
        by_suit_rank[card.substr(0, 1) + "|" + rank].push_back(card);
    }
    return generate_sequence_group_signatures_from_maps(by_rank, by_suit_rank);
}

static std::vector<py::tuple> generate_multi_sequence_group_signatures_from_rank_map(
    const std::unordered_map<std::string, std::vector<std::string>>& by_rank
) {
    const std::vector<std::string> no_wildcards;
    std::vector<py::tuple> out;
    std::unordered_set<std::string> seen;
    seen.reserve(256);
    const auto& ranks_all = straight_ranks();

    for (int start = 0; start <= static_cast<int>(ranks_all.size()) - 3; ++start) {
        std::vector<std::string> ranks;
        ranks.reserve(3);
        for (int offset = 0; offset < 3; ++offset) {
            ranks.emplace_back(ranks_all[start + offset]);
        }
        const auto tail_rank = ranks.back();
        for (const auto& option : multi_rank_sequence_options(no_wildcards, ranks, 2, by_rank)) {
            add_group_signature(out, seen, "StraightPair", tail_rank, option.first, option.second);
        }
    }

    for (int start = 0; start <= static_cast<int>(ranks_all.size()) - 2; ++start) {
        std::vector<std::string> ranks;
        ranks.reserve(2);
        for (int offset = 0; offset < 2; ++offset) {
            ranks.emplace_back(ranks_all[start + offset]);
        }
        const auto tail_rank = ranks.back();
        for (const auto& option : multi_rank_sequence_options(no_wildcards, ranks, 3, by_rank)) {
            add_group_signature(out, seen, "StraightTriple", tail_rank, option.first, option.second);
        }
    }

    return out;
}

std::vector<py::tuple> generate_multi_sequence_group_signatures(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    (void)cur_rank;
    const auto sorted_hand = sorted_hand_from_cards(hand);
    std::unordered_map<std::string, std::vector<std::string>> by_rank;
    for (const auto& card : sorted_hand) {
        if (card == "BJ" || card == "RJ") continue;
        by_rank[rank_for(card)].push_back(card);
    }
    return generate_multi_sequence_group_signatures_from_rank_map(by_rank);
}

static std::vector<py::tuple> generate_triple_plus_group_signatures_impl(
    const std::vector<std::string>& hand,
    const std::string& cur_rank,
    const std::vector<py::tuple>* same_rank_groups
) {
    const auto sorted_hand = sorted_hand_from_cards(hand);
    std::array<int, 54> hand_counts{};
    for (const auto& card : sorted_hand) {
        hand_counts[index_for(card)] += 1;
    }

    struct SameRankGroup {
        std::string kind;
        std::string rank;
        std::vector<std::string> cards;
        std::vector<std::string> wild_as;
    };

    std::vector<SameRankGroup> triples;
    std::vector<SameRankGroup> pairs;
    std::vector<py::tuple> generated_same_rank_groups;
    if (same_rank_groups == nullptr) {
        generated_same_rank_groups = generate_same_rank_group_signatures(hand, cur_rank);
        same_rank_groups = &generated_same_rank_groups;
    }
    for (const auto& item : *same_rank_groups) {
        SameRankGroup group{
            item[0].cast<std::string>(),
            item[1].cast<std::string>(),
            item[2].cast<std::vector<std::string>>(),
            item[3].cast<std::vector<std::string>>(),
        };
        if (group.kind == "Triple") {
            triples.push_back(std::move(group));
        } else if (group.kind == "Pair") {
            pairs.push_back(std::move(group));
        }
    }

    std::vector<py::tuple> out;
    std::unordered_set<std::string> seen;
    seen.reserve(triples.size() * std::max<std::size_t>(1, pairs.size()));
    for (const auto& triple : triples) {
        std::array<int, 54> triple_counts{};
        for (const auto& card : triple.cards) {
            triple_counts[index_for(card)] += 1;
        }
        for (const auto& pair : pairs) {
            if (triple.rank == pair.rank) {
                continue;
            }
            std::array<int, 54> combined_counts = triple_counts;
            bool fits = true;
            for (const auto& card : pair.cards) {
                const int idx = index_for(card);
                combined_counts[idx] += 1;
                if (combined_counts[idx] > hand_counts[idx]) {
                    fits = false;
                    break;
                }
            }
            if (!fits) {
                continue;
            }
            auto selected = triple.cards;
            selected.insert(selected.end(), pair.cards.begin(), pair.cards.end());
            auto wild_as = triple.wild_as;
            wild_as.insert(wild_as.end(), pair.wild_as.begin(), pair.wild_as.end());
            add_group_signature(out, seen, "TriplePlus", triple.rank, selected, wild_as);
        }
    }
    return out;
}

std::vector<py::tuple> generate_triple_plus_group_signatures(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    return generate_triple_plus_group_signatures_impl(hand, cur_rank, nullptr);
}

std::vector<py::tuple> generate_all_group_signatures(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    std::vector<py::tuple> out;
    auto same_rank = generate_same_rank_group_signatures(hand, cur_rank);
    auto sequence = generate_sequence_group_signatures(hand, cur_rank);
    auto multi_sequence = generate_multi_sequence_group_signatures(hand, cur_rank);
    auto triple_plus = native_reuse_same_rank_groups_enabled()
        ? generate_triple_plus_group_signatures_impl(hand, cur_rank, &same_rank)
        : generate_triple_plus_group_signatures(hand, cur_rank);
    out.reserve(same_rank.size() + sequence.size() + multi_sequence.size() + triple_plus.size());
    out.insert(out.end(), same_rank.begin(), same_rank.end());
    out.insert(out.end(), sequence.begin(), sequence.end());
    out.insert(out.end(), multi_sequence.begin(), multi_sequence.end());
    out.insert(out.end(), triple_plus.begin(), triple_plus.end());
    return out;
}

std::vector<py::tuple> generate_all_group_signatures_tupled(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    TupleSignaturePayloadGuard guard(true);
    return generate_all_group_signatures(hand, cur_rank);
}

py::tuple build_group_records_and_cover_inputs(
    const std::vector<std::string>& hand,
    const std::string& cur_rank
) {
    const auto sorted_hand = sorted_hand_from_cards(hand);
    std::array<int, 54> full_counts{};
    for (const auto& card : sorted_hand) {
        full_counts[index_for(card)] += 1;
    }

    std::vector<int> global_to_local(54, -1);
    std::vector<unsigned char> start;
    start.reserve(sorted_hand.size());
    for (int global_idx = 0; global_idx < static_cast<int>(kCards.size()); ++global_idx) {
        const int count = full_counts[global_idx];
        if (count <= 0) {
            continue;
        }
        global_to_local[global_idx] = static_cast<int>(start.size());
        start.push_back(static_cast<unsigned char>(count));
    }

    auto signatures = generate_all_group_signatures(hand, cur_rank);
    std::vector<py::tuple> records;
    records.reserve(signatures.size());
    std::vector<std::vector<std::vector<unsigned short>>> buckets(start.size());

    for (std::size_t group_id = 0; group_id < signatures.size(); ++group_id) {
        const auto& signature = signatures[group_id];
        const auto kind = signature[0].cast<std::string>();
        const auto rank = signature[1].cast<std::string>();
        const auto cards = signature[2].cast<std::vector<std::string>>();
        const auto wild_as = signature[3].cast<std::vector<std::string>>();
        if (cards.empty()) {
            throw std::invalid_argument("group signature must not contain empty cards");
        }

        std::vector<std::pair<int, int>> items;
        items.reserve(cards.size());
        int last_idx = -1;
        for (const auto& card : cards) {
            const int global_idx = index_for(card);
            const int local_idx = global_to_local[global_idx];
            if (local_idx < 0) {
                throw std::invalid_argument("group card " + card + " not in hand");
            }
            if (local_idx == last_idx) {
                items.back().second += 1;
            } else {
                items.emplace_back(local_idx, 1);
                last_idx = local_idx;
            }
        }

        std::vector<unsigned short> encoded;
        encoded.reserve(1 + items.size() * 2);
        encoded.push_back(static_cast<unsigned short>(group_id));
        for (const auto& item : items) {
            encoded.push_back(static_cast<unsigned short>(item.first));
            encoded.push_back(static_cast<unsigned short>(item.second));
        }
        buckets[items.front().first].push_back(encoded);
        std::vector<std::vector<unsigned short>> key_items;
        key_items.reserve(items.size());
        for (const auto& item : items) {
            key_items.push_back({
                static_cast<unsigned short>(item.first),
                static_cast<unsigned short>(item.second),
            });
        }
        records.push_back(py::make_tuple(kind, rank, cards, wild_as, key_items));
    }

    return py::make_tuple(records, start, buckets);
}

PYBIND11_MODULE(danrl_actor_core, m) {
    m.doc() = "Native actor-side helpers for DanRL retrieval.";
    m.def(
        "batch_action_static_features",
        &batch_action_static_features,
        py::arg("actions"),
        py::arg("cur_rank"),
        py::arg("current_rank"),
        py::arg("current_kind"),
        py::arg("public_counts"),
        py::arg("my_seat"),
        py::arg("last_player"),
        py::arg("remaining_rj"),
        "Compute retrieval action features in one native batch."
    );
    m.def(
        "batch_break_group_penalties",
        &batch_break_group_penalties,
        py::arg("actions"),
        py::arg("partitions"),
        py::arg("base_by_kind"),
        py::arg("straight_flush_to_bomb"),
        py::arg("bomb_break_size_bonus"),
        "Compute exact partition break penalties for an action batch."
    );
    m.def(
        "remove_cards_sorted",
        &remove_cards_sorted,
        py::arg("hand"),
        py::arg("action"),
        "Remove action cards from a hand and return cards sorted by DanRL retrieval order."
    );
    m.def(
        "remove_cards_sorted_batch",
        &remove_cards_sorted_batch,
        py::arg("hand"),
        py::arg("actions"),
        "Remove each action from the same hand and return sorted after-hands."
    );
    m.def(
        "build_cover_inputs",
        &build_cover_inputs,
        py::arg("hand"),
        py::arg("group_cards"),
        "Build native exact-cover start vector and buckets for already-generated sorted groups."
    );
    m.def(
        "build_cover_inputs_capsule",
        &build_cover_inputs_capsule,
        py::arg("hand"),
        py::arg("group_cards"),
        "Build exact-cover inputs with zero-conversion capsule buckets."
    );
    m.def(
        "build_cover_inputs_beam_order",
        &build_cover_inputs_beam_order,
        py::arg("hand"),
        py::arg("group_cards"),
        "Build exact-cover inputs in the legacy beam CARD_INDEX order."
    );
    m.def(
        "build_cover_inputs_beam_order_capsule",
        &build_cover_inputs_beam_order_capsule,
        py::arg("hand"),
        py::arg("group_cards"),
        "Build beam-order exact-cover inputs with capsule buckets."
    );
    m.def(
        "generate_same_rank_group_signatures",
        &generate_same_rank_group_signatures,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate native same-rank group signatures for oracle comparison."
    );
    m.def(
        "generate_sequence_group_signatures",
        &generate_sequence_group_signatures,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate native straight and straight-flush group signatures for oracle comparison."
    );
    m.def(
        "generate_multi_sequence_group_signatures",
        &generate_multi_sequence_group_signatures,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate native straight-pair and straight-triple group signatures for oracle comparison."
    );
    m.def(
        "generate_triple_plus_group_signatures",
        &generate_triple_plus_group_signatures,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate native triple-plus group signatures for oracle comparison."
    );
    m.def(
        "generate_all_group_signatures",
        &generate_all_group_signatures,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate all native group signatures for actor-side retrieval."
    );
    m.def(
        "generate_all_group_signatures_tupled",
        &generate_all_group_signatures_tupled,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate all group signatures with tuple payloads for zero-copy Python wrapping."
    );
    m.def(
        "build_group_records_and_cover_inputs",
        &build_group_records_and_cover_inputs,
        py::arg("hand"),
        py::arg("cur_rank"),
        "Generate all group signatures plus native exact-cover inputs."
    );
}
