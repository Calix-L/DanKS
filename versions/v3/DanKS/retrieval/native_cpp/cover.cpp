#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

using State = std::vector<unsigned char>;
using EncodedGroup = std::vector<unsigned short>;  // group_id, idx,count,idx,count...
using Buckets = std::vector<std::vector<EncodedGroup>>;
using Cover = std::vector<unsigned short>;
using Covers = std::vector<Cover>;

constexpr const char* kNativeBucketsCapsuleName = "danrl.native_buckets.v1";

static const Buckets& native_buckets_from_capsule(const py::capsule& capsule) {
    void* pointer = PyCapsule_GetPointer(capsule.ptr(), kNativeBucketsCapsuleName);
    if (pointer == nullptr) {
        throw py::error_already_set();
    }
    return *static_cast<const Buckets*>(pointer);
}

struct PackedEncodedGroup {
    unsigned short group_id = 0;
    std::uint64_t require_one = 0;
    std::uint64_t require_two = 0;
    std::uint64_t require_three = 0;
    std::uint64_t subtract_value = 0;
};

using PackedBuckets = std::vector<std::vector<PackedEncodedGroup>>;

struct PackedDepthTransition {
    unsigned short group_id = 0;
    std::uint64_t next_state = 0;
    int suffix_min_depth = 0;
};

struct WindowSuffixKey {
    std::uint64_t state = 0;
    std::uint64_t prefix_score_bits = 0;
    int remaining_depth = 0;

    bool operator==(const WindowSuffixKey& other) const {
        return state == other.state &&
            prefix_score_bits == other.prefix_score_bits &&
            remaining_depth == other.remaining_depth;
    }
};

struct WindowSuffixKeyHash {
    std::size_t operator()(const WindowSuffixKey& key) const {
        std::size_t value = std::hash<std::uint64_t>{}(key.state);
        value ^= std::hash<std::uint64_t>{}(key.prefix_score_bits) +
            0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= std::hash<int>{}(key.remaining_depth) +
            0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

using PackedDepthTransitionMemo =
    std::unordered_map<std::uint64_t, std::vector<PackedDepthTransition>>;

static bool cover_tie_less(
    const Cover& left,
    const Cover& right,
    const std::vector<std::string>& tie_keys
);

struct Candidate {
    double score;
    Cover cover;
};

struct CompactPathNode {
    unsigned short group_id = 0;
    std::uint32_t tail = 0;
};

struct CompactCandidate {
    std::int64_t score_units = 0;
    std::uint32_t node = 0;
    std::uint64_t tie_prefix = 0;
};

class CompactWindowMemo {
public:
    void reserve(std::size_t expected) {
        std::size_t capacity = 16;
        while (capacity < expected * 2) capacity <<= 1;
        rehash(capacity);
    }

    const std::vector<CompactCandidate>* find(std::uint64_t key) const {
        if (indices_.empty()) return nullptr;
        std::size_t slot = hash(key) & (indices_.size() - 1);
        while (indices_[slot] != 0) {
            if (keys_[slot] == key) return &values_[indices_[slot] - 1];
            slot = (slot + 1) & (indices_.size() - 1);
        }
        return nullptr;
    }

    const std::vector<CompactCandidate>& emplace(
        std::uint64_t key,
        std::vector<CompactCandidate>&& value
    ) {
        if (indices_.empty()) rehash(16);
        if ((size_ + 1) * 10 >= indices_.size() * 7) rehash(indices_.size() * 2);
        std::size_t slot = hash(key) & (indices_.size() - 1);
        while (indices_[slot] != 0) {
            if (keys_[slot] == key) return values_[indices_[slot] - 1];
            slot = (slot + 1) & (indices_.size() - 1);
        }
        values_.push_back(std::move(value));
        keys_[slot] = key;
        indices_[slot] = static_cast<std::uint32_t>(values_.size());
        ++size_;
        return values_.back();
    }

private:
    static std::uint64_t hash(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void rehash(std::size_t capacity) {
        std::vector<std::uint64_t> new_keys(capacity, 0);
        std::vector<std::uint32_t> new_indices(capacity, 0);
        for (std::size_t old = 0; old < indices_.size(); ++old) {
            if (indices_[old] == 0) continue;
            std::size_t slot = hash(keys_[old]) & (capacity - 1);
            while (new_indices[slot] != 0) slot = (slot + 1) & (capacity - 1);
            new_keys[slot] = keys_[old];
            new_indices[slot] = indices_[old];
        }
        keys_ = std::move(new_keys);
        indices_ = std::move(new_indices);
    }

    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> indices_;
    std::deque<std::vector<CompactCandidate>> values_;
    std::size_t size_ = 0;
};

class PackedDepthMemo {
public:
    PackedDepthMemo() {
        const char* raw = std::getenv("DANRL_NATIVE_FLAT_DEPTH_MEMO");
        flat_ = raw != nullptr && std::string(raw) != "0" && std::string(raw) != "false";
    }

    void reserve(std::size_t expected) {
        if (!flat_) {
            fallback_.reserve(expected);
            return;
        }
        std::size_t capacity = 16;
        while (capacity < expected * 2) capacity <<= 1;
        rehash(capacity);
    }

    bool find(std::uint64_t key, int& value) const {
        if (!flat_) {
            const auto found = fallback_.find(key);
            if (found == fallback_.end()) return false;
            value = found->second;
            return true;
        }
        if (slot_values_.empty()) return false;
        std::size_t slot = hash(key) & (slot_values_.size() - 1);
        while (slot_values_[slot] != empty_value()) {
            if (keys_[slot] == key) {
                value = slot_values_[slot];
                return true;
            }
            slot = (slot + 1) & (slot_values_.size() - 1);
        }
        return false;
    }

    void emplace(std::uint64_t key, int value) {
        if (!flat_) {
            fallback_.emplace(key, value);
            return;
        }
        if (slot_values_.empty()) rehash(16);
        if ((size_ + 1) * 10 >= slot_values_.size() * 7) {
            rehash(slot_values_.size() * 2);
        }
        std::size_t slot = hash(key) & (slot_values_.size() - 1);
        while (slot_values_[slot] != empty_value()) {
            if (keys_[slot] == key) return;
            slot = (slot + 1) & (slot_values_.size() - 1);
        }
        keys_[slot] = key;
        slot_values_[slot] = value;
        ++size_;
    }

private:
    static constexpr int empty_value() {
        return std::numeric_limits<int>::min();
    }

    static std::uint64_t hash(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void rehash(std::size_t capacity) {
        std::vector<std::uint64_t> new_keys(capacity, 0);
        std::vector<int> new_values(capacity, empty_value());
        for (std::size_t old = 0; old < slot_values_.size(); ++old) {
            if (slot_values_[old] == empty_value()) continue;
            std::size_t slot = hash(keys_[old]) & (capacity - 1);
            while (new_values[slot] != empty_value()) {
                slot = (slot + 1) & (capacity - 1);
            }
            new_keys[slot] = keys_[old];
            new_values[slot] = slot_values_[old];
        }
        keys_ = std::move(new_keys);
        slot_values_ = std::move(new_values);
    }

    std::vector<std::uint64_t> keys_;
    std::vector<int> slot_values_;
    std::unordered_map<std::uint64_t, int> fallback_;
    std::size_t size_ = 0;
    bool flat_ = false;
};

class PackedScoreMemo {
public:
    PackedScoreMemo() {
        const char* raw = std::getenv("DANRL_NATIVE_FLAT_EFFECTIVE_SCORE_MEMO");
        flat_ = raw != nullptr && std::string(raw) != "0" && std::string(raw) != "false";
    }

    void reserve(std::size_t expected) {
        if (!flat_) {
            fallback_.reserve(expected);
            return;
        }
        std::size_t capacity = 16;
        while (capacity < expected * 2) capacity <<= 1;
        rehash(capacity);
    }

    bool find(std::uint64_t key, double& value) const {
        if (!flat_) {
            const auto found = fallback_.find(key);
            if (found == fallback_.end()) return false;
            value = found->second;
            return true;
        }
        if (occupied_.empty()) return false;
        std::size_t slot = hash(key) & (occupied_.size() - 1);
        while (occupied_[slot]) {
            if (keys_[slot] == key) {
                value = values_[slot];
                return true;
            }
            slot = (slot + 1) & (occupied_.size() - 1);
        }
        return false;
    }

    void emplace(std::uint64_t key, double value) {
        if (!flat_) {
            fallback_.emplace(key, value);
            return;
        }
        if (occupied_.empty()) rehash(16);
        if ((size_ + 1) * 10 >= occupied_.size() * 7) {
            rehash(occupied_.size() * 2);
        }
        std::size_t slot = hash(key) & (occupied_.size() - 1);
        while (occupied_[slot]) {
            if (keys_[slot] == key) return;
            slot = (slot + 1) & (occupied_.size() - 1);
        }
        keys_[slot] = key;
        values_[slot] = value;
        occupied_[slot] = 1;
        ++size_;
    }

private:
    static std::uint64_t hash(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void rehash(std::size_t capacity) {
        std::vector<std::uint64_t> new_keys(capacity, 0);
        std::vector<double> new_values(capacity, 0.0);
        std::vector<unsigned char> new_occupied(capacity, 0);
        for (std::size_t old = 0; old < occupied_.size(); ++old) {
            if (!occupied_[old]) continue;
            std::size_t slot = hash(keys_[old]) & (capacity - 1);
            while (new_occupied[slot]) slot = (slot + 1) & (capacity - 1);
            new_keys[slot] = keys_[old];
            new_values[slot] = values_[old];
            new_occupied[slot] = 1;
        }
        keys_ = std::move(new_keys);
        values_ = std::move(new_values);
        occupied_ = std::move(new_occupied);
    }

    std::vector<std::uint64_t> keys_;
    std::vector<double> values_;
    std::vector<unsigned char> occupied_;
    std::unordered_map<std::uint64_t, double> fallback_;
    std::size_t size_ = 0;
    bool flat_ = false;
};

using WindowSuffixMemo = std::unordered_map<
    WindowSuffixKey,
    std::vector<Candidate>,
    WindowSuffixKeyHash
>;

static int min_cover_depth_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    std::unordered_map<std::uint64_t, int>& memo
);

static int min_cover_depth_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    PackedDepthMemo& memo
);

struct BeamNode {
    double score = 0.0;
    State state;
    int remaining_cards = 0;
    Cover cover;
};

static Covers select_top_covers_exact(
    Covers covers,
    const std::vector<double>& group_selection_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t selected_results
) {
    std::vector<Candidate> scored;
    scored.reserve(covers.size());
    for (auto& cover : covers) {
        double score = 0.0;
        for (const auto group_id : cover) {
            score += group_selection_scores[group_id];
        }
        scored.push_back(Candidate{score, std::move(cover)});
    }
    std::sort(scored.begin(), scored.end(), [&](const Candidate& left, const Candidate& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return cover_tie_less(left.cover, right.cover, tie_keys);
    });
    Covers selected;
    selected.reserve(std::min(selected_results, scored.size()));
    for (std::size_t i = 0; i < selected_results && i < scored.size(); ++i) {
        selected.push_back(std::move(scored[i].cover));
    }
    return selected;
}

static Covers select_top_covers_python_order(
    Covers covers,
    const std::vector<double>& group_selection_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t selected_results
) {
    std::vector<Candidate> scored;
    scored.reserve(covers.size());
    for (auto& cover : covers) {
        double score = 0.0;
        for (const auto group_id : cover) {
            score += group_selection_scores[group_id];
        }
        scored.push_back(Candidate{score, std::move(cover)});
    }
    std::stable_sort(scored.begin(), scored.end(), [&](const Candidate& left, const Candidate& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return cover_tie_less(left.cover, right.cover, tie_keys);
    });
    Covers selected;
    selected.reserve(std::min(selected_results, scored.size()));
    for (std::size_t i = 0; i < selected_results && i < scored.size(); ++i) {
        selected.push_back(std::move(scored[i].cover));
    }
    return selected;
}

struct ScoreCandidate {
    double total;
    double hand_count_score;
    double card_value_score;
    double retake_score;
    double residue_score;
    Cover cover;
    double retake_count;
};

struct ScoreState {
    double card_value_score = 0.0;
    double single_debt = 0.0;
    int effective_hand_count = 0;
    int single_count = 0;
    int low_single_count = 0;
    double best_single = 0.0;
    double best_pair = 0.0;
    double best_triple = 0.0;
    double best_triple_plus = 0.0;
    double best_straight = 0.0;
    double best_straight_pair = 0.0;
    double best_straight_triple = 0.0;
    double best_bomb = 0.0;
};

struct ScoreEntry {
    double cached_value = 0.0;
    int single_count = 0;
    int low_single_count = 0;
    double group_debt = 0.0;
    int nonzero_count = 0;
    int single_control_idx = -1;
    double single_control = 0.0;
    int effective_hand_cost = 1;
    std::vector<std::pair<int, double>> controls;
};

struct SafeBoundContext {
    bool enabled = false;
    const std::vector<double>* card_value_scores = nullptr;
    const std::unordered_map<std::string, double>* shared_card_upper_memo = nullptr;
    std::unordered_map<std::string, double> card_upper_memo;
};

static std::size_t native_batch_threads() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_THREADS");
    if (raw == nullptr || raw[0] == '\0') {
        return 1;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(raw, &end, 10);
    if (end == raw || parsed == 0) {
        return 1;
    }
    const auto hw = std::thread::hardware_concurrency();
    const std::size_t cap = hw == 0 ? 64 : static_cast<std::size_t>(hw);
    return std::min<std::size_t>(static_cast<std::size_t>(parsed), cap);
}

struct SuffixScoreCandidate {
    ScoreState score_state;
    Cover cover;
};

static std::string state_key(const State& state) {
    return std::string(reinterpret_cast<const char*>(state.data()), state.size());
}

static bool can_pack_state_2bit(const State& state) {
    if (state.size() > 32) {
        return false;
    }
    for (const auto count : state) {
        if (count > 3) {
            return false;
        }
    }
    return true;
}

static std::uint64_t packed_state_key_2bit(const State& state) {
    std::uint64_t key = 0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        key |= static_cast<std::uint64_t>(state[i]) << (2 * i);
    }
    return key;
}

static std::size_t first_nonzero_index_packed(std::uint64_t state, std::size_t state_size) {
    const std::uint64_t occupied = (state | (state >> 1)) & 0x5555555555555555ULL;
    if (occupied == 0) {
        return state_size;
    }
    return static_cast<std::size_t>(__builtin_ctzll(occupied) / 2);
}

static bool subtract_group_packed(
    std::uint64_t state,
    std::size_t state_size,
    const EncodedGroup& group,
    std::uint64_t& out
) {
    out = state;
    for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
        const auto idx = static_cast<std::size_t>(group[i]);
        const auto count = static_cast<std::uint64_t>(group[i + 1]);
        if (idx >= state_size) {
            return false;
        }
        const auto shift = static_cast<unsigned int>(2 * idx);
        const auto available = (out >> shift) & 3ULL;
        if (count > available) {
            return false;
        }
        out -= count << shift;
    }
    return true;
}

static bool compile_packed_buckets(
    const Buckets& groups_by_first,
    std::size_t state_size,
    PackedBuckets& packed_buckets
) {
    if (state_size > 32) {
        return false;
    }
    packed_buckets.clear();
    packed_buckets.resize(groups_by_first.size());
    for (std::size_t bucket_idx = 0; bucket_idx < groups_by_first.size(); ++bucket_idx) {
        auto& packed_bucket = packed_buckets[bucket_idx];
        packed_bucket.reserve(groups_by_first[bucket_idx].size());
        for (const auto& group : groups_by_first[bucket_idx]) {
            if (group.size() < 3 || group.size() % 2 == 0) {
                return false;
            }
            std::array<unsigned char, 32> counts{};
            for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
                const auto idx = static_cast<std::size_t>(group[i]);
                const auto count = static_cast<unsigned int>(group[i + 1]);
                if (idx >= state_size || count == 0 || count > 3 || static_cast<unsigned int>(counts[idx]) + count > 3) {
                    return false;
                }
                counts[idx] = static_cast<unsigned char>(counts[idx] + count);
            }
            PackedEncodedGroup packed;
            packed.group_id = group[0];
            for (std::size_t idx = 0; idx < state_size; ++idx) {
                const auto count = static_cast<unsigned int>(counts[idx]);
                if (count == 0) {
                    continue;
                }
                const auto shift = static_cast<unsigned int>(2 * idx);
                const auto lane = 1ULL << shift;
                packed.require_one |= lane;
                if (count >= 2) packed.require_two |= lane;
                if (count >= 3) packed.require_three |= lane;
                packed.subtract_value |= static_cast<std::uint64_t>(count) << shift;
            }
            packed_bucket.push_back(packed);
        }
    }
    return true;
}

static bool subtract_precompiled_group(
    std::uint64_t state,
    std::uint64_t available_one,
    std::uint64_t available_two,
    std::uint64_t available_three,
    const PackedEncodedGroup& group,
    std::uint64_t& out
) {
    if ((available_one & group.require_one) != group.require_one ||
        (available_two & group.require_two) != group.require_two ||
        (available_three & group.require_three) != group.require_three) {
        return false;
    }
    out = state - group.subtract_value;
    return true;
}

static std::size_t first_nonzero_index(const State& state) {
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            return i;
        }
    }
    return state.size();
}

static bool subtract_group(const State& state, const EncodedGroup& group, State& out) {
    out = state;
    for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
        const auto idx = static_cast<std::size_t>(group[i]);
        const auto count = static_cast<unsigned char>(group[i + 1]);
        if (idx >= out.size() || count > out[idx]) {
            return false;
        }
        out[idx] = static_cast<unsigned char>(out[idx] - count);
    }
    return true;
}

static bool can_apply_group(const State& state, const EncodedGroup& group) {
    for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
        const auto idx = static_cast<std::size_t>(group[i]);
        const auto count = static_cast<unsigned char>(group[i + 1]);
        if (idx >= state.size() || count > state[idx]) {
            return false;
        }
    }
    return true;
}

static void apply_group_in_place(State& state, const EncodedGroup& group, int direction) {
    for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
        const auto idx = static_cast<std::size_t>(group[i]);
        const auto count = static_cast<unsigned char>(group[i + 1]);
        if (direction < 0) {
            state[idx] = static_cast<unsigned char>(state[idx] - count);
        } else {
            state[idx] = static_cast<unsigned char>(state[idx] + count);
        }
    }
}

static Covers suffix_covers(
    const State& state,
    const Buckets& groups_by_first,
    std::unordered_map<std::string, Covers>& memo
) {
    std::size_t first = state.size();
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == state.size()) {
        return Covers{Cover{}};
    }

    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) return found->second;

    Covers results;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            auto tails = suffix_covers(next, groups_by_first, memo);
            for (auto& tail : tails) {
                Cover cover;
                cover.reserve(tail.size() + 1);
                cover.push_back(group[0]);
                cover.insert(cover.end(), tail.begin(), tail.end());
                results.push_back(std::move(cover));
            }
        }
    }
    memo.emplace(key, results);
    return results;
}

std::uint64_t count_covers(const std::vector<unsigned char>& state, const Buckets& groups_by_first) {
    std::unordered_map<std::string, Covers> memo;
    memo.reserve(4096);
    py::gil_scoped_release release;
    return static_cast<std::uint64_t>(suffix_covers(state, groups_by_first, memo).size());
}

Covers enumerate_covers(const std::vector<unsigned char>& state, const Buckets& groups_by_first) {
    std::unordered_map<std::string, Covers> memo;
    memo.reserve(4096);
    py::gil_scoped_release release;
    return suffix_covers(state, groups_by_first, memo);
}

static void apply_score_control(ScoreState& next, int control_idx, double item_control) {
    if (control_idx == 0) {
        if (item_control > next.best_single) next.best_single = item_control;
    } else if (control_idx == 1) {
        if (item_control > next.best_pair) next.best_pair = item_control;
    } else if (control_idx == 2) {
        if (item_control > next.best_triple) next.best_triple = item_control;
    } else if (control_idx == 3) {
        if (item_control > next.best_triple_plus) next.best_triple_plus = item_control;
    } else if (control_idx == 4) {
        if (item_control > next.best_straight) next.best_straight = item_control;
    } else if (control_idx == 5) {
        if (item_control > next.best_straight_pair) next.best_straight_pair = item_control;
    } else if (control_idx == 6) {
        if (item_control > next.best_straight_triple) next.best_straight_triple = item_control;
    } else if (item_control > next.best_bomb) {
        next.best_bomb = item_control;
    }
}

static ScoreState apply_score_group(
    ScoreState next,
    unsigned short group_id,
    const std::vector<ScoreEntry>& group_entries
) {
    const auto& entry = group_entries[group_id];
    next.card_value_score += entry.cached_value;
    next.single_count += entry.single_count;
    next.low_single_count += entry.low_single_count;
    next.single_debt += entry.group_debt;
    next.effective_hand_count += entry.effective_hand_cost;
    if (entry.nonzero_count == 1) {
        apply_score_control(next, entry.single_control_idx, entry.single_control);
    } else if (entry.nonzero_count > 0) {
        for (const auto& item : entry.controls) {
            apply_score_control(next, item.first, item.second);
        }
    }
    return next;
}

static ScoreState merge_score_states(const ScoreState& left, const ScoreState& right) {
    ScoreState out;
    out.card_value_score = left.card_value_score + right.card_value_score;
    out.single_debt = left.single_debt + right.single_debt;
    out.effective_hand_count = left.effective_hand_count + right.effective_hand_count;
    out.single_count = left.single_count + right.single_count;
    out.low_single_count = left.low_single_count + right.low_single_count;
    out.best_single = std::max(left.best_single, right.best_single);
    out.best_pair = std::max(left.best_pair, right.best_pair);
    out.best_triple = std::max(left.best_triple, right.best_triple);
    out.best_triple_plus = std::max(left.best_triple_plus, right.best_triple_plus);
    out.best_straight = std::max(left.best_straight, right.best_straight);
    out.best_straight_pair = std::max(left.best_straight_pair, right.best_straight_pair);
    out.best_straight_triple = std::max(left.best_straight_triple, right.best_straight_triple);
    out.best_bomb = std::max(left.best_bomb, right.best_bomb);
    return out;
}

static bool score_state_dominates(
    const SuffixScoreCandidate& left,
    const SuffixScoreCandidate& right
) {
    constexpr double eps = 1e-12;
    const auto& a = left.score_state;
    const auto& b = right.score_state;
    bool strictly_better = false;
    if (left.cover.size() > right.cover.size()) return false;
    strictly_better = strictly_better || left.cover.size() < right.cover.size();
    if (a.effective_hand_count > b.effective_hand_count) return false;
    strictly_better = strictly_better || a.effective_hand_count < b.effective_hand_count;
    if (a.card_value_score + eps < b.card_value_score) return false;
    strictly_better = strictly_better || a.card_value_score > b.card_value_score + eps;
    if (a.single_debt > b.single_debt + eps) return false;
    strictly_better = strictly_better || a.single_debt + eps < b.single_debt;
    if (a.single_count > b.single_count) return false;
    strictly_better = strictly_better || a.single_count < b.single_count;
    if (a.low_single_count > b.low_single_count) return false;
    strictly_better = strictly_better || a.low_single_count < b.low_single_count;
    if (a.best_single + eps < b.best_single) return false;
    strictly_better = strictly_better || a.best_single > b.best_single + eps;
    if (a.best_pair + eps < b.best_pair) return false;
    strictly_better = strictly_better || a.best_pair > b.best_pair + eps;
    if (a.best_triple + eps < b.best_triple) return false;
    strictly_better = strictly_better || a.best_triple > b.best_triple + eps;
    if (a.best_triple_plus + eps < b.best_triple_plus) return false;
    strictly_better = strictly_better || a.best_triple_plus > b.best_triple_plus + eps;
    if (a.best_straight + eps < b.best_straight) return false;
    strictly_better = strictly_better || a.best_straight > b.best_straight + eps;
    if (a.best_straight_pair + eps < b.best_straight_pair) return false;
    strictly_better = strictly_better || a.best_straight_pair > b.best_straight_pair + eps;
    if (a.best_straight_triple + eps < b.best_straight_triple) return false;
    strictly_better = strictly_better || a.best_straight_triple > b.best_straight_triple + eps;
    if (a.best_bomb + eps < b.best_bomb) return false;
    strictly_better = strictly_better || a.best_bomb > b.best_bomb + eps;
    return strictly_better;
}

static void insert_pareto_candidate(
    std::vector<SuffixScoreCandidate>& frontier,
    SuffixScoreCandidate candidate
) {
    for (const auto& existing : frontier) {
        if (score_state_dominates(existing, candidate)) {
            return;
        }
    }
    auto write = frontier.begin();
    for (auto read = frontier.begin(); read != frontier.end(); ++read) {
        if (!score_state_dominates(candidate, *read)) {
            if (write != read) {
                *write = std::move(*read);
            }
            ++write;
        }
    }
    frontier.erase(write, frontier.end());
    frontier.push_back(std::move(candidate));
}

static double retake_count_from_score_state(const ScoreState& score_state) {
    double total = 0.0;
    const std::array<double, 8> controls = {
        score_state.best_single,
        score_state.best_pair,
        score_state.best_triple,
        score_state.best_triple_plus,
        score_state.best_straight,
        score_state.best_straight_pair,
        score_state.best_straight_triple,
        score_state.best_bomb,
    };
    for (const auto control : controls) {
        if (control >= 0.82) {
            total += 1.0;
        } else if (control >= 0.62) {
            total += 0.55;
        } else if (control >= 0.45) {
            total += 0.25;
        }
    }
    return total;
}

static std::size_t native_score_dp_frontier_limit() {
    const char* raw = std::getenv("DANRL_NATIVE_SCORE_DP_FRONTIER_LIMIT");
    if (raw == nullptr || raw[0] == '\0') {
        return 200000;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(raw, &end, 10);
    if (end == raw || parsed == 0) {
        return 200000;
    }
    return static_cast<std::size_t>(parsed);
}

static bool native_score_dp_enabled_for_batch() {
    const char* raw = std::getenv("DANRL_ENABLE_NATIVE_SCORE_DP");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_safe_bound_enabled() {
    const char* raw = std::getenv("DANRL_ENABLE_SAFE_BOUND");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_branch_order_enabled() {
    const char* raw = std::getenv("DANRL_ENABLE_BRANCH_ORDER");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_initial_incumbent_enabled() {
    const char* raw = std::getenv("DANRL_ENABLE_INITIAL_INCUMBENT");
    if (raw == nullptr) {
        return native_safe_bound_enabled();
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_shared_upper_memo_enabled() {
    const char* raw = std::getenv("DANRL_ENABLE_SHARED_UPPER_MEMO");
    if (raw == nullptr) {
        return native_safe_bound_enabled();
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_batch_shared_window_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_SHARED_WINDOW_MEMO");
    if (raw == nullptr) {
        return true;
    }
    const std::string value(raw);
    return !(value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off");
}

static bool native_batch_shared_top_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_SHARED_TOP_MEMO");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_batch_packed_top_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_PACKED_TOP_MEMO");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_batch_packed_window_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_PACKED_WINDOW_MEMO");
    if (raw == nullptr) {
        return true;
    }
    const std::string value(raw);
    return !(value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off");
}

static bool native_batch_window_transition_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_WINDOW_TRANSITION_MEMO");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_parallel_window_batch_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_PARALLEL_WINDOW_BATCH");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_compact_parallel_window_batch_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DANRL_NATIVE_COMPACT_PARALLEL_WINDOW_BATCH");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

static bool native_window_suffix_memo_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_WINDOW_SUFFIX_MEMO");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_depth_window_upper_bound_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_DEPTH_WINDOW_UPPER_BOUND");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_compact_window_dp_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_COMPACT_WINDOW_DP");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static std::size_t native_compact_window_min_states() {
    static const std::size_t threshold = []() {
        const char* raw = std::getenv("DANRL_NATIVE_COMPACT_WINDOW_MIN_STATES");
        if (raw == nullptr || *raw == '\0') {
            return std::size_t{16};
        }
        char* end = nullptr;
        const auto parsed = std::strtoul(raw, &end, 10);
        if (end == raw || *end != '\0') {
            return std::size_t{16};
        }
        return std::max<std::size_t>(1, static_cast<std::size_t>(parsed));
    }();
    return threshold;
}

static bool native_lazy_compact_window_dp_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DANRL_NATIVE_LAZY_COMPACT_WINDOW_DP");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

static bool native_lazy_selected_bound_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DANRL_NATIVE_LAZY_SELECTED_BOUND");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

static bool native_compact_top_dp_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_COMPACT_TOP_DP");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_lazy_compact_top_dp_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DANRL_NATIVE_LAZY_COMPACT_TOP_DP");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

static bool native_window_astar_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_WINDOW_ASTAR");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool native_batch_direct_packed_state_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_DIRECT_PACKED_STATE");
    if (raw == nullptr) {
        return true;
    }
    const std::string value(raw);
    return !(value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off");
}

static bool native_batch_precompiled_groups_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_BATCH_PRECOMPILED_GROUPS");
    if (raw == nullptr) {
        return true;
    }
    const std::string value(raw);
    return !(value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off");
}

static bool native_packed_effective_window_batch_enabled() {
    const char* raw = std::getenv("DANRL_NATIVE_PACKED_EFFECTIVE_WINDOW_BATCH");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static void finish_score_cover(
    const ScoreState& score_state,
    const Cover& chosen,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    bool& has_value,
    ScoreCandidate& best
) {
    const double hand_count_score = -static_cast<double>(score_state.effective_hand_count);
    double residue_score = (chosen.size() > 7 ? static_cast<double>(chosen.size() - 7) * 0.22 : 0.0) + score_state.single_debt;
    if (score_state.single_count >= 3) {
        residue_score += static_cast<double>(score_state.single_count - 2) * 0.35;
    }
    if (score_state.low_single_count >= 2) {
        residue_score += static_cast<double>(score_state.low_single_count - 1) * 0.45;
    }
    const double retake_total =
        pressure_values[0] * score_state.best_single +
        pressure_values[1] * score_state.best_pair +
        pressure_values[2] * score_state.best_triple +
        pressure_values[3] * score_state.best_triple_plus +
        pressure_values[4] * score_state.best_straight +
        pressure_values[5] * score_state.best_straight_pair +
        pressure_values[6] * score_state.best_straight_triple +
        pressure_values[7] * score_state.best_bomb;
    const double retake_score = retake_total * 100.0;
    const double total =
        weights[0] * hand_count_score +
        weights[1] * score_state.card_value_score +
        weights[2] * retake_score -
        weights[3] * residue_score;
    if (!has_value || total > best.total) {
        best = ScoreCandidate{
            total,
            hand_count_score,
            score_state.card_value_score,
            retake_score,
            residue_score,
            chosen,
            retake_count_from_score_state(score_state),
        };
        has_value = true;
    }
}

static std::vector<ScoreEntry> compact_score_entries(const py::sequence& group_entries);
static double max_suffix_score(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::string, double>& memo
);
static std::vector<SuffixScoreCandidate> suffix_score_frontier(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    std::unordered_map<std::string, std::vector<SuffixScoreCandidate>>& memo,
    std::size_t frontier_limit,
    bool& ok
);

static bool best_cover_by_score_entries_dp_compact(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    std::size_t frontier_limit,
    ScoreCandidate& best
) {
    std::unordered_map<std::string, std::vector<SuffixScoreCandidate>> memo;
    memo.reserve(4096);
    bool ok = true;
    const auto frontier = suffix_score_frontier(state, groups_by_first, group_entries, memo, frontier_limit, ok);
    if (!ok || frontier.empty()) {
        return false;
    }
    bool has_value = false;
    for (const auto& candidate : frontier) {
        finish_score_cover(candidate.score_state, candidate.cover, weights, pressure_values, has_value, best);
    }
    return has_value;
}

static double optimistic_retake_upper(
    const ScoreState& score_state,
    const std::vector<double>& pressure_values
) {
    const std::array<double, 8> current = {
        score_state.best_single,
        score_state.best_pair,
        score_state.best_triple,
        score_state.best_triple_plus,
        score_state.best_straight,
        score_state.best_straight_pair,
        score_state.best_straight_triple,
        score_state.best_bomb,
    };
    double total = 0.0;
    for (std::size_t i = 0; i < current.size() && i < pressure_values.size(); ++i) {
        total += pressure_values[i] * std::max(current[i], 1.0);
    }
    return total * 100.0;
}

static bool safe_bound_applicable(
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values
) {
    if (weights.size() < 4 || pressure_values.size() < 8) {
        return false;
    }
    for (const auto value : weights) {
        if (value < -1e-12) {
            return false;
        }
    }
    for (const auto value : pressure_values) {
        if (value < -1e-12) {
            return false;
        }
    }
    return true;
}

static bool can_prune_by_safe_bound(
    const State& current,
    const Buckets& groups_by_first,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    const ScoreState& score_state,
    const Cover& chosen,
    bool has_value,
    const ScoreCandidate& best,
    SafeBoundContext* bound
) {
    if (bound == nullptr || !bound->enabled || !has_value) {
        return false;
    }
    const auto first = first_nonzero_index(current);
    const double remaining_card_count = static_cast<double>(
        std::accumulate(current.begin(), current.end(), 0)
    );
    const double optimistic_hand_count_score =
        -static_cast<double>(score_state.effective_hand_count) + remaining_card_count;
    const double optimistic_card_value =
        score_state.card_value_score +
        (first == current.size()
            ? 0.0
            : [&]() {
                if (bound->shared_card_upper_memo != nullptr) {
                    const auto found = bound->shared_card_upper_memo->find(state_key(current));
                    if (found != bound->shared_card_upper_memo->end()) {
                        return found->second;
                    }
                }
                return max_suffix_score(current, groups_by_first, *bound->card_value_scores, bound->card_upper_memo);
            }());
    const double optimistic_total =
        weights[0] * optimistic_hand_count_score +
        weights[1] * optimistic_card_value +
        weights[2] * optimistic_retake_upper(score_state, pressure_values);
    return optimistic_total < best.total - 1e-12;
}

static double group_branch_priority(
    unsigned short group_id,
    const std::vector<ScoreEntry>& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values
) {
    if (group_id >= group_entries.size() || weights.size() < 4) {
        return 0.0;
    }
    const auto& entry = group_entries[group_id];
    double control_score = 0.0;
    if (entry.nonzero_count == 1) {
        if (entry.single_control_idx >= 0 && static_cast<std::size_t>(entry.single_control_idx) < pressure_values.size()) {
            control_score += pressure_values[static_cast<std::size_t>(entry.single_control_idx)] * entry.single_control;
        }
    } else if (entry.nonzero_count > 1) {
        for (const auto& item : entry.controls) {
            if (item.first >= 0 && static_cast<std::size_t>(item.first) < pressure_values.size()) {
                control_score += pressure_values[static_cast<std::size_t>(item.first)] * item.second;
            }
        }
    }
    const double hand_count_term = -weights[0] * static_cast<double>(entry.effective_hand_cost);
    const double card_term = weights[1] * entry.cached_value;
    const double retake_term = weights[2] * control_score * 100.0;
    const double residue_term = -weights[3] * entry.group_debt;
    return hand_count_term + card_term + retake_term + residue_term;
}

static bool greedy_initial_cover(
    State state,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    Cover& cover,
    ScoreState& score_state
) {
    cover.clear();
    score_state = ScoreState{};
    while (true) {
        const auto first = first_nonzero_index(state);
        if (first == state.size()) {
            return true;
        }
        if (first >= groups_by_first.size()) {
            return false;
        }
        const EncodedGroup* best_group = nullptr;
        double best_priority = 0.0;
        for (const auto& group : groups_by_first[first]) {
            if (!can_apply_group(state, group)) {
                continue;
            }
            const double priority = group_branch_priority(group[0], group_entries, weights, pressure_values);
            if (best_group == nullptr || priority > best_priority + 1e-12 || (std::abs(priority - best_priority) <= 1e-12 && group[0] < best_group->at(0))) {
                best_group = &group;
                best_priority = priority;
            }
        }
        if (best_group == nullptr) {
            return false;
        }
        const auto group_id = best_group->at(0);
        cover.push_back(group_id);
        score_state = apply_score_group(score_state, group_id, group_entries);
        apply_group_in_place(state, *best_group, -1);
    }
}

static void seed_initial_incumbent(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    bool& has_value,
    ScoreCandidate& best
) {
    if (!native_initial_incumbent_enabled()) {
        return;
    }
    Cover cover;
    ScoreState score_state;
    if (!greedy_initial_cover(state, groups_by_first, group_entries, weights, pressure_values, cover, score_state)) {
        return;
    }
    finish_score_cover(score_state, cover, weights, pressure_values, has_value, best);
}

static void dfs_best_cover_by_score_entries(
    State& current,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    const ScoreState& score_state,
    Cover& chosen,
    bool& has_value,
    ScoreCandidate& best,
    SafeBoundContext* bound = nullptr
) {
    std::size_t first = current.size();
    for (std::size_t i = 0; i < current.size(); ++i) {
        if (current[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == current.size()) {
        finish_score_cover(score_state, chosen, weights, pressure_values, has_value, best);
        return;
    }
    if (first >= groups_by_first.size()) {
        return;
    }
    if (can_prune_by_safe_bound(current, groups_by_first, weights, pressure_values, score_state, chosen, has_value, best, bound)) {
        return;
    }
    std::vector<const EncodedGroup*> ordered_groups;
    ordered_groups.reserve(groups_by_first[first].size());
    for (const auto& group : groups_by_first[first]) {
        if (can_apply_group(current, group)) {
            ordered_groups.push_back(&group);
        }
    }
    if (native_branch_order_enabled() && ordered_groups.size() > 1) {
        std::sort(
            ordered_groups.begin(),
            ordered_groups.end(),
            [&](const EncodedGroup* left, const EncodedGroup* right) {
                const double left_score = group_branch_priority(left->at(0), group_entries, weights, pressure_values);
                const double right_score = group_branch_priority(right->at(0), group_entries, weights, pressure_values);
                if (left_score > right_score + 1e-12) {
                    return true;
                }
                if (right_score > left_score + 1e-12) {
                    return false;
                }
                return left->at(0) < right->at(0);
            }
        );
    }
    for (const auto* group_ptr : ordered_groups) {
        const auto& group = *group_ptr;
        const auto group_id = group[0];
        chosen.push_back(group_id);
        apply_group_in_place(current, group, -1);
        dfs_best_cover_by_score_entries(
            current,
            groups_by_first,
            group_entries,
            weights,
            pressure_values,
            apply_score_group(score_state, group_id, group_entries),
            chosen,
            has_value,
            best,
            bound
        );
        apply_group_in_place(current, group, 1);
        chosen.pop_back();
    }
}

static std::vector<SuffixScoreCandidate> suffix_score_frontier(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<ScoreEntry>& group_entries,
    std::unordered_map<std::string, std::vector<SuffixScoreCandidate>>& memo,
    std::size_t frontier_limit,
    bool& ok
) {
    if (!ok) {
        return {};
    }
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        return std::vector<SuffixScoreCandidate>{SuffixScoreCandidate{ScoreState{}, Cover{}}};
    }
    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    std::vector<SuffixScoreCandidate> frontier;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            auto tails = suffix_score_frontier(next, groups_by_first, group_entries, memo, frontier_limit, ok);
            if (!ok) {
                return {};
            }
            const auto group_score_state = apply_score_group(ScoreState{}, group_id, group_entries);
            for (const auto& tail : tails) {
                SuffixScoreCandidate candidate;
                candidate.score_state = merge_score_states(group_score_state, tail.score_state);
                candidate.cover.reserve(tail.cover.size() + 1);
                candidate.cover.push_back(group_id);
                candidate.cover.insert(candidate.cover.end(), tail.cover.begin(), tail.cover.end());
                insert_pareto_candidate(frontier, std::move(candidate));
                if (frontier.size() > frontier_limit) {
                    ok = false;
                    return {};
                }
            }
        }
    }
    memo.emplace(key, frontier);
    return frontier;
}

static std::vector<py::object> best_cover_by_score_entries_dp_result(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    bool include_retake_count,
    std::size_t frontier_limit
) {
    const auto compact_entries = compact_score_entries(group_entries);
    std::unordered_map<std::string, std::vector<SuffixScoreCandidate>> memo;
    memo.reserve(4096);
    bool ok = true;
    std::vector<SuffixScoreCandidate> frontier;
    {
        py::gil_scoped_release release;
        frontier = suffix_score_frontier(state, groups_by_first, compact_entries, memo, frontier_limit, ok);
    }
    if (!ok || frontier.empty()) {
        return std::vector<py::object>{py::none()};
    }
    bool has_value = false;
    ScoreCandidate best;
    for (const auto& candidate : frontier) {
        finish_score_cover(candidate.score_state, candidate.cover, weights, pressure_values, has_value, best);
    }
    if (!has_value) {
        return std::vector<py::object>{py::none()};
    }
    std::vector<py::object> out{
        py::cast(best.cover),
        py::float_(best.total),
        py::make_tuple(best.hand_count_score, best.card_value_score, best.retake_score, best.residue_score)
    };
    if (include_retake_count) {
        out.push_back(py::float_(best.retake_count));
    }
    return out;
}

static std::vector<ScoreEntry> compact_score_entries(const py::sequence& group_entries) {
    std::vector<ScoreEntry> out;
    const std::size_t entry_count = static_cast<std::size_t>(py::len(group_entries));
    out.reserve(entry_count);
    for (std::size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
        py::sequence raw = py::reinterpret_borrow<py::sequence>(group_entries[entry_idx]);
        ScoreEntry entry;
        const std::size_t raw_size = static_cast<std::size_t>(py::len(raw));
        entry.cached_value = raw[0].cast<double>();
        entry.single_count = static_cast<int>(raw[1].cast<double>());
        entry.low_single_count = static_cast<int>(raw[2].cast<double>());
        entry.group_debt = raw[3].cast<double>();
        entry.nonzero_count = static_cast<int>(raw[4].cast<double>());
        entry.single_control_idx = static_cast<int>(raw[5].cast<double>());
        entry.single_control = raw[6].cast<double>();
        entry.effective_hand_cost = static_cast<int>(raw[7].cast<double>());
        if (entry.nonzero_count > 1) {
            entry.controls.reserve((raw_size - 8) / 2);
            for (std::size_t i = 8; i + 1 < raw_size; i += 2) {
                entry.controls.emplace_back(static_cast<int>(raw[i].cast<double>()), raw[i + 1].cast<double>());
            }
        }
        out.push_back(std::move(entry));
    }
    return out;
}

static py::object best_selected_cover_by_score_entries(
    const Covers& covers,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values
) {
    if (covers.empty() || weights.size() < 4 || pressure_values.size() < 8) {
        return py::none();
    }
    const auto compact_entries = compact_score_entries(group_entries);
    std::size_t best_index = 0;
    ScoreCandidate best{};
    bool has_best = false;
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < covers.size(); ++index) {
            ScoreState score_state;
            bool valid = true;
            for (const auto group_id : covers[index]) {
                if (group_id >= compact_entries.size()) {
                    valid = false;
                    break;
                }
                score_state = apply_score_group(score_state, group_id, compact_entries);
            }
            if (!valid) {
                continue;
            }
            ScoreCandidate candidate{};
            bool has_candidate = false;
            finish_score_cover(
                score_state,
                covers[index],
                weights,
                pressure_values,
                has_candidate,
                candidate
            );
            if (has_candidate && (!has_best || candidate.total > best.total)) {
                best_index = index;
                best = std::move(candidate);
                has_best = true;
            }
        }
    }
    if (!has_best) {
        return py::none();
    }
    return py::make_tuple(
        best_index,
        best.total,
        py::make_tuple(
            best.hand_count_score,
            best.card_value_score,
            best.retake_score,
            best.residue_score
        ),
        best.retake_count
    );
}

static std::vector<py::object> best_selected_covers_by_score_entries_batch(
    const std::vector<Covers>& cover_batches,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<std::vector<double>>& pressure_values_by_batch
) {
    const auto compact_entries = compact_score_entries(group_entries);
    struct SelectedResult {
        bool has_best = false;
        std::size_t best_index = 0;
        ScoreCandidate best{};
    };
    std::vector<SelectedResult> results(cover_batches.size());
    {
        py::gil_scoped_release release;
        for (std::size_t batch_idx = 0; batch_idx < cover_batches.size(); ++batch_idx) {
            if (batch_idx >= pressure_values_by_batch.size() || weights.size() < 4 ||
                pressure_values_by_batch[batch_idx].size() < 8) {
                continue;
            }
            const auto& covers = cover_batches[batch_idx];
            auto& result = results[batch_idx];
            for (std::size_t cover_idx = 0; cover_idx < covers.size(); ++cover_idx) {
                ScoreState score_state;
                bool valid = true;
                for (const auto group_id : covers[cover_idx]) {
                    if (group_id >= compact_entries.size()) {
                        valid = false;
                        break;
                    }
                    score_state = apply_score_group(score_state, group_id, compact_entries);
                }
                if (!valid) {
                    continue;
                }
                ScoreCandidate candidate{};
                bool has_candidate = false;
                finish_score_cover(
                    score_state,
                    covers[cover_idx],
                    weights,
                    pressure_values_by_batch[batch_idx],
                    has_candidate,
                    candidate
                );
                if (has_candidate && (!result.has_best || candidate.total > result.best.total)) {
                    result.best_index = cover_idx;
                    result.best = std::move(candidate);
                    result.has_best = true;
                }
            }
        }
    }
    std::vector<py::object> out;
    out.reserve(results.size());
    for (const auto& result : results) {
        if (!result.has_best) {
            out.push_back(py::none());
            continue;
        }
        out.push_back(py::make_tuple(
            result.best_index,
            result.best.total,
            py::make_tuple(
                result.best.hand_count_score,
                result.best.card_value_score,
                result.best.retake_score,
                result.best.residue_score
            ),
            result.best.retake_count
        ));
    }
    return out;
}

static std::vector<py::object> best_cover_by_score_entries_result(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    bool include_retake_count
) {
    bool has_value = false;
    ScoreCandidate best;
    Cover chosen;
    State current = state;
    const auto compact_entries = compact_score_entries(group_entries);
    std::vector<double> card_value_scores;
    if (native_safe_bound_enabled() && safe_bound_applicable(weights, pressure_values)) {
        card_value_scores.reserve(compact_entries.size());
        for (const auto& entry : compact_entries) {
            card_value_scores.push_back(entry.cached_value);
        }
    }
    SafeBoundContext bound;
    if (!card_value_scores.empty()) {
        bound.enabled = true;
        bound.card_value_scores = &card_value_scores;
        bound.card_upper_memo.reserve(4096);
    }

    {
        py::gil_scoped_release release;
        seed_initial_incumbent(
            current,
            groups_by_first,
            compact_entries,
            weights,
            pressure_values,
            has_value,
            best
        );
        dfs_best_cover_by_score_entries(
            current,
            groups_by_first,
            compact_entries,
            weights,
            pressure_values,
            ScoreState{},
            chosen,
            has_value,
            best,
            bound.enabled ? &bound : nullptr
        );
    }
    if (!has_value) {
        return std::vector<py::object>{py::none()};
    }
    std::vector<py::object> out{
        py::cast(best.cover),
        py::float_(best.total),
        py::make_tuple(best.hand_count_score, best.card_value_score, best.retake_score, best.residue_score)
    };
    if (include_retake_count) {
        out.push_back(py::float_(best.retake_count));
    }
    return out;
}

std::vector<py::object> best_cover_by_score_entries(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values
) {
    return best_cover_by_score_entries_result(state, groups_by_first, group_entries, weights, pressure_values, false);
}

std::vector<py::object> best_cover_by_score_entries_with_retake(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values
) {
    return best_cover_by_score_entries_result(state, groups_by_first, group_entries, weights, pressure_values, true);
}

std::vector<std::vector<py::object>> best_covers_by_score_entries_with_retake_batch(
    const std::vector<std::vector<unsigned char>>& states,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<std::vector<double>>& pressure_values_by_state
) {
    const auto compact_entries = compact_score_entries(group_entries);
    struct BatchResult {
        bool has_value = false;
        ScoreCandidate best;
    };
    std::vector<BatchResult> results(states.size());
    const bool use_score_dp = native_score_dp_enabled_for_batch();
    const std::size_t score_dp_frontier_limit = native_score_dp_frontier_limit();
    std::vector<double> card_value_scores;
    if (native_safe_bound_enabled()) {
        card_value_scores.reserve(compact_entries.size());
        for (const auto& entry : compact_entries) {
            card_value_scores.push_back(entry.cached_value);
        }
    }
    std::unordered_map<std::string, double> shared_card_upper_memo;
    {
        py::gil_scoped_release release;
        if (!card_value_scores.empty() && native_shared_upper_memo_enabled()) {
            shared_card_upper_memo.reserve(std::max<std::size_t>(4096, states.size() * 256));
            for (std::size_t i = 0; i < states.size() && i < pressure_values_by_state.size(); ++i) {
                if (safe_bound_applicable(weights, pressure_values_by_state[i])) {
                    State state = states[i];
                    max_suffix_score(state, groups_by_first, card_value_scores, shared_card_upper_memo);
                }
            }
        }
        const auto run_one = [&](std::size_t i) {
            if (i >= pressure_values_by_state.size()) {
                return;
            }
            State current = states[i];
            if (use_score_dp) {
                results[i].has_value = best_cover_by_score_entries_dp_compact(
                    current,
                    groups_by_first,
                    compact_entries,
                    weights,
                    pressure_values_by_state[i],
                    score_dp_frontier_limit,
                    results[i].best
                );
                if (results[i].has_value) {
                    return;
                }
            }
            Cover chosen;
            SafeBoundContext bound;
            if (!card_value_scores.empty() && safe_bound_applicable(weights, pressure_values_by_state[i])) {
                bound.enabled = true;
                bound.card_value_scores = &card_value_scores;
                if (!shared_card_upper_memo.empty()) {
                    bound.shared_card_upper_memo = &shared_card_upper_memo;
                }
                bound.card_upper_memo.reserve(4096);
            }
            seed_initial_incumbent(
                current,
                groups_by_first,
                compact_entries,
                weights,
                pressure_values_by_state[i],
                results[i].has_value,
                results[i].best
            );
            dfs_best_cover_by_score_entries(
                current,
                groups_by_first,
                compact_entries,
                weights,
                pressure_values_by_state[i],
                ScoreState{},
                chosen,
                results[i].has_value,
                results[i].best,
                bound.enabled ? &bound : nullptr
            );
        };
        const auto run_range = [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                run_one(i);
            }
        };
        const auto run_dynamic = [&]() {
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve(native_batch_threads());
            const std::size_t worker_count = std::min<std::size_t>(native_batch_threads(), states.size());
            for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                workers.emplace_back([&]() {
                    while (true) {
                        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= states.size()) {
                            break;
                        }
                        run_one(i);
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
        };
        const std::size_t requested_threads = native_batch_threads();
        const std::size_t worker_count = std::min<std::size_t>(requested_threads, states.size());
        if (worker_count <= 1) {
            run_range(0, states.size());
        } else {
            run_dynamic();
        }
    }
    std::vector<std::vector<py::object>> out;
    out.reserve(states.size());
    for (const auto& result : results) {
        if (!result.has_value) {
            out.push_back(std::vector<py::object>{py::none()});
            continue;
        }
        const auto& best = result.best;
        out.push_back(std::vector<py::object>{
            py::cast(best.cover),
            py::float_(best.total),
            py::make_tuple(best.hand_count_score, best.card_value_score, best.retake_score, best.residue_score),
            py::float_(best.retake_count),
        });
    }
    return out;
}

std::vector<py::object> best_cover_by_score_entries_dp(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    std::size_t frontier_limit
) {
    return best_cover_by_score_entries_dp_result(
        state,
        groups_by_first,
        group_entries,
        weights,
        pressure_values,
        false,
        frontier_limit
    );
}
std::vector<py::object> best_cover_by_score_entries_dp_with_retake(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const py::sequence& group_entries,
    const std::vector<double>& weights,
    const std::vector<double>& pressure_values,
    std::size_t frontier_limit
) {
    return best_cover_by_score_entries_dp_result(
        state,
        groups_by_first,
        group_entries,
        weights,
        pressure_values,
        true,
        frontier_limit
    );
}

static bool cover_tie_less(const Cover& left, const Cover& right, const std::vector<std::string>& tie_keys) {
    const auto n = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = tie_keys[left[i]];
        const auto& b = tie_keys[right[i]];
        if (a < b) {
            return true;
        }
        if (b < a) {
            return false;
        }
    }
    return left.size() < right.size();
}

static bool better_candidate(const Candidate& left, const Candidate& right, const std::vector<std::string>& tie_keys) {
    constexpr double eps = 1e-12;
    if (left.score > right.score + eps) {
        return true;
    }
    if (right.score > left.score + eps) {
        return false;
    }
    return cover_tie_less(left.cover, right.cover, tie_keys);
}

static bool better_beam_node(
    const BeamNode& left,
    const BeamNode& right,
    const std::vector<std::string>& tie_keys
) {
    if (left.score != right.score) {
        return left.score > right.score;
    }
    if (left.cover.size() != right.cover.size()) {
        return left.cover.size() < right.cover.size();
    }
    return cover_tie_less(left.cover, right.cover, tie_keys);
}

static bool better_completed_beam_node(
    const BeamNode& left,
    const BeamNode& right,
    const std::vector<std::string>& tie_keys
) {
    if (left.score != right.score) {
        return left.score > right.score;
    }
    return cover_tie_less(left.cover, right.cover, tie_keys);
}

static Covers top_covers_beam_impl(
    const State& start,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<int>& group_sizes,
    const std::vector<std::string>& tie_keys,
    std::size_t beam_width,
    std::size_t max_results
) {
    if (start.empty()) {
        return Covers{Cover{}};
    }
    const int total_cards = std::accumulate(start.begin(), start.end(), 0);
    if (total_cards <= 0 || beam_width == 0 || max_results == 0) {
        return Covers{};
    }

    std::vector<BeamNode> beam;
    beam.push_back(BeamNode{0.0, start, total_cards, Cover{}});
    std::vector<BeamNode> completed;
    for (int step = 0; step < total_cards; ++step) {
        std::vector<BeamNode> next_beam;
        for (const auto& node : beam) {
            const auto first = first_nonzero_index(node.state);
            if (first >= node.state.size() || first >= groups_by_first.size()) {
                continue;
            }
            State next_state;
            for (const auto& group : groups_by_first[first]) {
                if (!subtract_group(node.state, group, next_state)) {
                    continue;
                }
                const auto group_id = group[0];
                BeamNode next_node;
                next_node.score = node.score + group_scores[group_id];
                next_node.score -= 10.0;
                next_node.remaining_cards = node.remaining_cards - group_sizes[group_id];
                next_node.cover = node.cover;
                next_node.cover.push_back(group_id);
                if (next_node.remaining_cards == 0) {
                    completed.push_back(std::move(next_node));
                } else {
                    next_node.state = next_state;
                    next_beam.push_back(std::move(next_node));
                }
            }
        }
        if (next_beam.empty()) {
            break;
        }
        std::stable_sort(
            next_beam.begin(), next_beam.end(),
            [&](const BeamNode& left, const BeamNode& right) {
                return better_beam_node(left, right, tie_keys);
            }
        );
        if (next_beam.size() > beam_width) {
            next_beam.resize(beam_width);
        }
        beam = std::move(next_beam);
    }

    std::stable_sort(
        completed.begin(), completed.end(),
        [&](const BeamNode& left, const BeamNode& right) {
            return better_completed_beam_node(left, right, tie_keys);
        }
    );
    Covers out;
    out.reserve(std::min(max_results, completed.size()));
    for (auto& candidate : completed) {
        if (std::find(out.begin(), out.end(), candidate.cover) != out.end()) {
            continue;
        }
        out.push_back(std::move(candidate.cover));
        if (out.size() >= max_results) {
            break;
        }
    }
    return out;
}

static std::vector<Covers> top_covers_beam_batch(
    const std::vector<State>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<int>& group_sizes,
    const std::vector<std::string>& tie_keys,
    std::size_t beam_width,
    std::size_t max_results
) {
    py::gil_scoped_release release;
    std::vector<Covers> out(states.size());
    const char* raw_parallel = std::getenv("DANRL_NATIVE_PARALLEL_BEAM_BATCH");
    const bool parallel = raw_parallel != nullptr && (
        std::strcmp(raw_parallel, "1") == 0 ||
        std::strcmp(raw_parallel, "true") == 0 ||
        std::strcmp(raw_parallel, "yes") == 0 ||
        std::strcmp(raw_parallel, "on") == 0
    );
    const std::size_t worker_count = std::min<std::size_t>(
        native_batch_threads(), states.size()
    );
    if (parallel && worker_count > 1) {
        std::atomic<std::size_t> next_index{0};
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
            workers.emplace_back([&]() {
                while (true) {
                    const auto index = next_index.fetch_add(1, std::memory_order_relaxed);
                    if (index >= states.size()) {
                        break;
                    }
                    out[index] = top_covers_beam_impl(
                        states[index],
                        groups_by_first,
                        group_scores,
                        group_sizes,
                        tie_keys,
                        beam_width,
                        max_results
                    );
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        return out;
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
        out[index] = top_covers_beam_impl(
            states[index],
            groups_by_first,
            group_scores,
            group_sizes,
            tie_keys,
            beam_width,
            max_results
        );
    }
    return out;
}

static void insert_top(
    std::vector<Candidate>& top,
    Candidate candidate,
    std::size_t max_results,
    const std::vector<std::string>& tie_keys
) {
    if (max_results == 0) {
        return;
    }
    auto pos = top.begin();
    while (pos != top.end() && !better_candidate(candidate, *pos, tie_keys)) {
        ++pos;
    }
    top.insert(pos, std::move(candidate));
    if (top.size() > max_results) {
        top.pop_back();
    }
}

static double max_suffix_score(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::string, double>& memo
) {
    std::size_t first = state.size();
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == state.size()) {
        return 0.0;
    }

    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    double best = 0.0;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            const double score = group_scores[group_id] + max_suffix_score(next, groups_by_first, group_scores, memo);
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static double max_suffix_score_packed(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::uint64_t, double>& memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        return 0.0;
    }

    const auto key = packed_state_key_2bit(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    double best = 0.0;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            const double score =
                group_scores[group_id] + max_suffix_score_packed(next, groups_by_first, group_scores, memo);
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static double max_suffix_score_direct_packed(
    std::uint64_t state,
    std::size_t state_size,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::uint64_t, double>& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        return 0.0;
    }
    const auto found = memo.find(state);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    double best = 0.0;
    if (first < groups_by_first.size()) {
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group_packed(state, state_size, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            const double score = group_scores[group_id] + max_suffix_score_direct_packed(
                next, state_size, groups_by_first, group_scores, memo
            );
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static int min_cover_depth_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    PackedDepthMemo& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) return 0;
    int cached = 0;
    if (memo.find(state, cached)) return cached;
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const int suffix = min_cover_depth_precompiled_packed(
                next, state_size, groups_by_first, memo
            );
            if (suffix != INF) best = std::min(best, suffix + 1);
        }
    }
    memo.emplace(state, best);
    return best;
}

static double max_suffix_score_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::uint64_t, double>& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        return 0.0;
    }
    const auto found = memo.find(state);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    double best = 0.0;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const double score = group_scores[group.group_id] + max_suffix_score_precompiled_packed(
                next, state_size, groups_by_first, group_scores, memo
            );
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static double max_suffix_score_effective_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    PackedScoreMemo& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) return 0.0;
    double cached = 0.0;
    if (memo.find(state, cached)) return cached;

    bool has_value = false;
    double best = 0.0;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const double score = group_scores[group.group_id] +
                max_suffix_score_effective_precompiled_packed(
                    next, state_size, groups_by_first, group_scores, memo
                );
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static double max_suffix_score_depth_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::unordered_map<std::uint64_t, double>& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        return 0.0;
    }
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    if (remaining_depth <= 0 || state_size > 28) {
        return NEG_INF;
    }
    const std::uint64_t key = state | (static_cast<std::uint64_t>(remaining_depth) << 56);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    double best = NEG_INF;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const int suffix_min = min_cover_depth_precompiled_packed(
                next, state_size, groups_by_first, min_depth_memo
            );
            if (suffix_min > remaining_depth - 1) {
                continue;
            }
            const double suffix = max_suffix_score_depth_precompiled_packed(
                next,
                state_size,
                groups_by_first,
                group_scores,
                min_depth_memo,
                remaining_depth - 1,
                memo
            );
            if (!std::isfinite(suffix)) {
                continue;
            }
            const double score = group_scores[group.group_id] + suffix;
            if (!has_value || score > best) {
                best = score;
                has_value = true;
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static Candidate best_suffix_by_group_score(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::string, Candidate>& memo
) {
    std::size_t first = state.size();
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == state.size()) {
        return Candidate{0.0, Cover{}};
    }

    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    bool has_value = false;
    Candidate best;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            auto tail = best_suffix_by_group_score(next, groups_by_first, group_scores, tie_keys, memo);
            Cover cover;
            cover.reserve(tail.cover.size() + 1);
            cover.push_back(group_id);
            cover.insert(cover.end(), tail.cover.begin(), tail.cover.end());
            Candidate candidate{group_scores[group_id] + tail.score, std::move(cover)};
            if (!has_value || better_candidate(candidate, best, tie_keys)) {
                best = std::move(candidate);
                has_value = true;
            }
        }
    }
    if (!has_value) {
        best = Candidate{0.0, Cover{}};
    }
    memo.emplace(key, best);
    return best;
}

Cover best_cover_by_group_scores(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys
) {
    std::unordered_map<std::string, Candidate> memo;
    memo.reserve(4096);
    py::gil_scoped_release release;
    return best_suffix_by_group_score(state, groups_by_first, group_scores, tie_keys, memo).cover;
}

static void dfs_top(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top
) {
    std::size_t first = state.size();
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == state.size()) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }

    if (first >= groups_by_first.size()) {
        return;
    }

    State next;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group(state, group, next)) {
            continue;
        }
        const auto group_id = group[0];
        chosen.push_back(group_id);
        dfs_top(
            next,
            groups_by_first,
            group_scores,
            tie_keys,
            max_results,
            score + group_scores[group_id],
            chosen,
            top
        );
        chosen.pop_back();
    }
}

static std::vector<Candidate> suffix_top(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    bool enable_upper_bound,
    std::unordered_map<std::string, std::vector<Candidate>>& memo,
    std::unordered_map<std::string, double>& max_score_memo
) {
    std::size_t first = state.size();
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (state[i] != 0) {
            first = i;
            break;
        }
    }
    if (first == state.size()) {
        return std::vector<Candidate>{Candidate{0.0, Cover{}}};
    }

    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    std::vector<Candidate> top;
    top.reserve(max_results);
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const auto group_id = group[0];
            if (enable_upper_bound && top.size() >= max_results) {
                const double possible = group_scores[group_id] + max_suffix_score(next, groups_by_first, group_scores, max_score_memo);
                if (possible < top.back().score - 1e-12) {
                    continue;
                }
            }
            auto tails = suffix_top(
                next,
                groups_by_first,
                group_scores,
                tie_keys,
                max_results,
                enable_upper_bound,
                memo,
                max_score_memo
            );
            for (const auto& tail : tails) {
                Cover cover;
                cover.reserve(tail.cover.size() + 1);
                cover.push_back(group_id);
                cover.insert(cover.end(), tail.cover.begin(), tail.cover.end());
                insert_top(
                    top,
                    Candidate{group_scores[group_id] + tail.score, std::move(cover)},
                    max_results,
                    tie_keys
                );
            }
        }
    }
    memo.emplace(key, top);
    return top;
}

static const std::vector<Candidate>& suffix_top_direct_packed(
    std::uint64_t state,
    std::size_t state_size,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, std::vector<Candidate>>& memo
) {
    const auto found = memo.find(state);
    if (found != memo.end()) {
        return found->second;
    }

    std::vector<Candidate> top;
    if (state == 0) {
        top.push_back(Candidate{0.0, Cover{}});
    } else {
        top.reserve(max_results);
        const auto first = first_nonzero_index_packed(state, state_size);
        if (first < groups_by_first.size()) {
            for (const auto& group : groups_by_first[first]) {
                std::uint64_t next = 0;
                if (!subtract_group_packed(state, state_size, group, next)) {
                    continue;
                }
                const auto group_id = group[0];
                const auto& tails = suffix_top_direct_packed(
                    next,
                    state_size,
                    groups_by_first,
                    group_scores,
                    tie_keys,
                    max_results,
                    memo
                );
                for (const auto& tail : tails) {
                    Cover cover;
                    cover.reserve(tail.cover.size() + 1);
                    cover.push_back(group_id);
                    cover.insert(cover.end(), tail.cover.begin(), tail.cover.end());
                    insert_top(
                        top,
                        Candidate{group_scores[group_id] + tail.score, std::move(cover)},
                        max_results,
                        tie_keys
                    );
                }
            }
        }
    }
    const auto inserted = memo.emplace(state, std::move(top));
    return inserted.first->second;
}

static int min_cover_depth(
    const State& state,
    const Buckets& groups_by_first,
    std::unordered_map<std::string, int>& memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        return 0;
    }
    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const int suffix = min_cover_depth(next, groups_by_first, memo);
            if (suffix != INF) {
                best = std::min(best, suffix + 1);
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static int min_cover_depth_packed(
    const State& state,
    const Buckets& groups_by_first,
    std::unordered_map<std::uint64_t, int>& memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        return 0;
    }
    const auto key = packed_state_key_2bit(state);
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) {
                continue;
            }
            const int suffix = min_cover_depth_packed(next, groups_by_first, memo);
            if (suffix != INF) {
                best = std::min(best, suffix + 1);
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static int min_cover_depth_direct_packed(
    std::uint64_t state,
    std::size_t state_size,
    const Buckets& groups_by_first,
    std::unordered_map<std::uint64_t, int>& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        return 0;
    }
    const auto found = memo.find(state);
    if (found != memo.end()) {
        return found->second;
    }
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group_packed(state, state_size, group, next)) {
                continue;
            }
            const int suffix = min_cover_depth_direct_packed(next, state_size, groups_by_first, memo);
            if (suffix != INF) {
                best = std::min(best, suffix + 1);
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static int min_cover_depth_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    std::unordered_map<std::uint64_t, int>& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        return 0;
    }
    const auto found = memo.find(state);
    if (found != memo.end()) {
        return found->second;
    }
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const int suffix = min_cover_depth_precompiled_packed(next, state_size, groups_by_first, memo);
            if (suffix != INF) {
                best = std::min(best, suffix + 1);
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static const std::vector<PackedDepthTransition>& depth_transitions_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    PackedDepthTransitionMemo& transition_memo
) {
    const auto found = transition_memo.find(state);
    if (found != transition_memo.end()) {
        return found->second;
    }

    std::vector<PackedDepthTransition> transitions;
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first < state_size && first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        transitions.reserve(groups_by_first[first].size());
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            transitions.push_back(PackedDepthTransition{
                group.group_id,
                next,
                min_cover_depth_precompiled_packed(
                    next, state_size, groups_by_first, min_depth_memo
                ),
            });
        }
    }
    return transition_memo.emplace(state, std::move(transitions)).first->second;
}

static void collect_top_depth_window(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::string, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    std::unordered_map<std::string, double>& max_score_memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return;
    }
    if (top.size() >= max_results) {
        const double possible = score + max_suffix_score(state, groups_by_first, group_scores, max_score_memo);
        if (possible < top.back().score - 1e-12) {
            return;
        }
    }
    State next;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group(state, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth(next, groups_by_first, min_depth_memo);
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        const auto group_id = group[0];
        chosen.push_back(group_id);
        collect_top_depth_window(
            next,
            groups_by_first,
            group_scores,
            tie_keys,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            score + group_scores[group_id],
            chosen,
            top,
            max_score_memo
        );
        chosen.pop_back();
        if (top.size() >= max_results) {
            const double possible = score + max_suffix_score(state, groups_by_first, group_scores, max_score_memo);
            if (possible < top.back().score - 1e-12) {
                return;
            }
        }
    }
}

static void collect_top_depth_window_packed(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    std::unordered_map<std::uint64_t, double>& max_score_memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return;
    }
    if (top.size() >= max_results) {
        const double possible = score + max_suffix_score_packed(state, groups_by_first, group_scores, max_score_memo);
        if (possible < top.back().score - 1e-12) {
            return;
        }
    }
    State next;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group(state, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth_packed(next, groups_by_first, min_depth_memo);
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        const auto group_id = group[0];
        chosen.push_back(group_id);
        collect_top_depth_window_packed(
            next,
            groups_by_first,
            group_scores,
            tie_keys,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            score + group_scores[group_id],
            chosen,
            top,
            max_score_memo
        );
        chosen.pop_back();
        if (top.size() >= max_results) {
            const double possible =
                score + max_suffix_score_packed(state, groups_by_first, group_scores, max_score_memo);
            if (possible < top.back().score - 1e-12) {
                return;
            }
        }
    }
}

static void collect_top_depth_window_direct_packed(
    std::uint64_t state,
    std::size_t state_size,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    std::unordered_map<std::uint64_t, double>& max_score_memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return;
    }
    if (top.size() >= max_results) {
        const double possible = score + max_suffix_score_direct_packed(
            state, state_size, groups_by_first, group_scores, max_score_memo
        );
        if (possible < top.back().score - 1e-12) {
            return;
        }
    }
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group_packed(state, state_size, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth_direct_packed(next, state_size, groups_by_first, min_depth_memo);
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        const auto group_id = group[0];
        chosen.push_back(group_id);
        collect_top_depth_window_direct_packed(
            next,
            state_size,
            groups_by_first,
            group_scores,
            tie_keys,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            score + group_scores[group_id],
            chosen,
            top,
            max_score_memo
        );
        chosen.pop_back();
        if (top.size() >= max_results) {
            const double possible = score + max_suffix_score_direct_packed(
                state, state_size, groups_by_first, group_scores, max_score_memo
            );
            if (possible < top.back().score - 1e-12) {
                return;
            }
        }
    }
}

static void collect_top_depth_window_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    std::unordered_map<std::uint64_t, double>& max_score_memo,
    PackedDepthTransitionMemo* transition_memo,
    bool use_depth_upper_bound
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return;
    }
    bool suffix_upper_ready = false;
    double suffix_upper = 0.0;
    const auto pruned_by_upper_bound = [&]() {
        if (top.size() < max_results) {
            return false;
        }
        if (!suffix_upper_ready) {
            suffix_upper = use_depth_upper_bound
                ? max_suffix_score_depth_precompiled_packed(
                    state,
                    state_size,
                    groups_by_first,
                    group_scores,
                    min_depth_memo,
                    remaining_depth,
                    max_score_memo
                )
                : max_suffix_score_precompiled_packed(
                    state, state_size, groups_by_first, group_scores, max_score_memo
                );
            suffix_upper_ready = true;
        }
        // The extra margin is far above the worst roundoff of at most 27
        // additions, so the tighter bound remains conservative.
        const double possible = score + suffix_upper + (use_depth_upper_bound ? 1e-9 : 0.0);
        if (possible < top.back().score - 1e-12) {
            return true;
        }
        return false;
    };
    if (pruned_by_upper_bound()) {
        return;
    }

    if (transition_memo != nullptr) {
        const auto& transitions = depth_transitions_precompiled_packed(
            state, state_size, groups_by_first, min_depth_memo, *transition_memo
        );
        for (const auto& transition : transitions) {
            if (transition.suffix_min_depth > remaining_depth - 1) {
                continue;
            }
            chosen.push_back(transition.group_id);
            collect_top_depth_window_precompiled_packed(
                transition.next_state,
                state_size,
                groups_by_first,
                group_scores,
                tie_keys,
                min_depth_memo,
                remaining_depth - 1,
                max_results,
                score + group_scores[transition.group_id],
                chosen,
                top,
                max_score_memo,
                transition_memo,
                use_depth_upper_bound
            );
            chosen.pop_back();
            if (pruned_by_upper_bound()) {
                return;
            }
        }
        return;
    }

    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth_precompiled_packed(
            next, state_size, groups_by_first, min_depth_memo
        );
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        chosen.push_back(group.group_id);
        collect_top_depth_window_precompiled_packed(
            next,
            state_size,
            groups_by_first,
            group_scores,
            tie_keys,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            score + group_scores[group.group_id],
            chosen,
            top,
            max_score_memo,
            transition_memo,
            use_depth_upper_bound
        );
        chosen.pop_back();
        if (pruned_by_upper_bound()) {
            return;
        }
    }
}

static std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static const std::vector<Candidate>& suffix_top_depth_window_precompiled_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    double prefix_score,
    WindowSuffixMemo& memo
) {
    const WindowSuffixKey key{state, double_bits(prefix_score), remaining_depth};
    const auto found = memo.find(key);
    if (found != memo.end()) {
        return found->second;
    }

    std::vector<Candidate> top;
    top.reserve(max_results);
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        top.push_back(Candidate{prefix_score, Cover{}});
        return memo.emplace(key, std::move(top)).first->second;
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return memo.emplace(key, std::move(top)).first->second;
    }

    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth_precompiled_packed(
            next, state_size, groups_by_first, min_depth_memo
        );
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        const double next_score = prefix_score + group_scores[group.group_id];
        const auto& tails = suffix_top_depth_window_precompiled_packed(
            next,
            state_size,
            groups_by_first,
            group_scores,
            tie_keys,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            next_score,
            memo
        );
        for (const auto& tail : tails) {
            Cover cover;
            cover.reserve(tail.cover.size() + 1);
            cover.push_back(group.group_id);
            cover.insert(cover.end(), tail.cover.begin(), tail.cover.end());
            insert_top(top, Candidate{tail.score, std::move(cover)}, max_results, tie_keys);
        }
    }
    return memo.emplace(key, std::move(top)).first->second;
}

static Covers top_covers_hand_count_window_suffix_memo(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    WindowSuffixMemo& suffix_memo
) {
    const auto packed_state = packed_state_key_2bit(state);
    const int min_depth = min_cover_depth_precompiled_packed(
        packed_state, state.size(), groups_by_first, min_depth_memo
    );
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) {
        return Covers{};
    }
    const auto& top = suffix_top_depth_window_precompiled_packed(
        packed_state,
        state.size(),
        groups_by_first,
        group_scores,
        tie_keys,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        0.0,
        suffix_memo
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

static int compact_path_tie_compare(
    std::uint32_t left,
    std::uint32_t right,
    const std::vector<CompactPathNode>& arena,
    const std::vector<std::uint32_t>& tie_ranks
) {
    while (left != 0 && right != 0) {
        const auto& left_node = arena[left];
        const auto& right_node = arena[right];
        const auto left_rank = tie_ranks[left_node.group_id];
        const auto right_rank = tie_ranks[right_node.group_id];
        if (left_rank < right_rank) {
            return -1;
        }
        if (right_rank < left_rank) {
            return 1;
        }
        left = left_node.tail;
        right = right_node.tail;
    }
    if (left == 0 && right != 0) return -1;
    if (right == 0 && left != 0) return 1;
    return 0;
}

class LazyCompactWindowSolver {
public:
    LazyCompactWindowSolver(
        std::size_t state_size,
        const PackedBuckets& groups_by_first,
        const std::vector<std::int64_t>& group_score_units,
        const std::vector<std::uint32_t>& tie_ranks,
        PackedDepthMemo& min_depth_memo,
        std::vector<CompactPathNode>& arena,
        std::size_t expected_states
    ) :
        state_size_(state_size),
        groups_by_first_(groups_by_first),
        depth_groups_by_first_(groups_by_first),
        group_score_units_(group_score_units),
        tie_ranks_(tie_ranks),
        min_depth_memo_(min_depth_memo),
        arena_(arena)
    {
        for (auto& bucket : depth_groups_by_first_) {
            std::unordered_set<std::uint64_t> seen;
            seen.reserve(bucket.size());
            auto write = bucket.begin();
            for (auto read = bucket.begin(); read != bucket.end(); ++read) {
                if (!seen.insert(read->subtract_value).second) continue;
                if (write != read) *write = std::move(*read);
                ++write;
            }
            bucket.erase(write, bucket.end());
        }
        std::size_t capacity = 16;
        while (capacity < expected_states * 2) capacity <<= 1;
        keys_.assign(capacity, 0);
        indices_.assign(capacity, 0);
    }

    const std::pmr::vector<CompactCandidate>& top(
        std::uint64_t state,
        int remaining_depth,
        std::size_t count
    ) {
        const auto node_id = node_for(state, remaining_depth);
        if (count > 0) ensure(node_id, count - 1);
        return nodes_[node_id].top;
    }

    int min_depth(std::uint64_t state) {
        return min_cover_depth_precompiled_packed(
            state, state_size_, depth_groups_by_first_, min_depth_memo_
        );
    }

private:
    struct Cursor {
        unsigned short group_id = 0;
        std::uint32_t child = 0;
        std::uint32_t index = 0;
        std::uint32_t branch_order = 0;
    };

    struct Node {
        explicit Node(std::pmr::memory_resource* resource) : top(resource), heap(resource) {}
        std::uint64_t state = 0;
        int remaining_depth = 0;
        bool initialized = false;
        std::pmr::vector<CompactCandidate> top;
        std::pmr::vector<Cursor> heap;
    };

    static std::uint64_t hash(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void rehash(std::size_t capacity) {
        std::vector<std::uint64_t> new_keys(capacity, 0);
        std::vector<std::uint32_t> new_indices(capacity, 0);
        for (std::size_t old = 0; old < indices_.size(); ++old) {
            if (indices_[old] == 0) continue;
            std::size_t slot = hash(keys_[old]) & (capacity - 1);
            while (new_indices[slot] != 0) slot = (slot + 1) & (capacity - 1);
            new_keys[slot] = keys_[old];
            new_indices[slot] = indices_[old];
        }
        keys_ = std::move(new_keys);
        indices_ = std::move(new_indices);
    }

    std::uint32_t node_for(std::uint64_t state, int remaining_depth) {
        const auto key = state | (static_cast<std::uint64_t>(remaining_depth) << 56);
        if ((nodes_.size() + 1) * 10 >= indices_.size() * 7) {
            rehash(indices_.size() * 2);
        }
        std::size_t slot = hash(key) & (indices_.size() - 1);
        while (indices_[slot] != 0) {
            if (keys_[slot] == key) return indices_[slot] - 1;
            slot = (slot + 1) & (indices_.size() - 1);
        }
        nodes_.emplace_back(&node_resource_);
        auto& node = nodes_.back();
        node.state = state;
        node.remaining_depth = remaining_depth;
        node.top.reserve(16);
        node.heap.reserve(16);
        keys_[slot] = key;
        indices_[slot] = static_cast<std::uint32_t>(nodes_.size());
        return static_cast<std::uint32_t>(nodes_.size() - 1);
    }

    bool cursor_better(const Cursor& left, const Cursor& right) const {
        const auto& left_tail = nodes_[left.child].top[left.index];
        const auto& right_tail = nodes_[right.child].top[right.index];
        const auto left_score = group_score_units_[left.group_id] + left_tail.score_units;
        const auto right_score = group_score_units_[right.group_id] + right_tail.score_units;
        if (left_score != right_score) return left_score > right_score;
        const auto left_rank = tie_ranks_[left.group_id];
        const auto right_rank = tie_ranks_[right.group_id];
        if (left_rank != right_rank) return left_rank < right_rank;
        if (left_tail.tie_prefix != right_tail.tie_prefix) {
            return left_tail.tie_prefix < right_tail.tie_prefix;
        }
        const int path_order = compact_path_tie_compare(
            left_tail.node, right_tail.node, arena_, tie_ranks_
        );
        if (path_order != 0) return path_order < 0;
        if (left.branch_order != right.branch_order) {
            return left.branch_order < right.branch_order;
        }
        return left.index < right.index;
    }

    void push_cursor(Node& node, Cursor cursor) {
        node.heap.push_back(cursor);
        const auto worse = [this](const Cursor& left, const Cursor& right) {
            return cursor_better(right, left);
        };
        std::push_heap(node.heap.begin(), node.heap.end(), worse);
    }

    Cursor pop_cursor(Node& node) {
        const auto worse = [this](const Cursor& left, const Cursor& right) {
            return cursor_better(right, left);
        };
        std::pop_heap(node.heap.begin(), node.heap.end(), worse);
        auto cursor = node.heap.back();
        node.heap.pop_back();
        return cursor;
    }

    void initialize(std::uint32_t node_id) {
        auto& node = nodes_[node_id];
        if (node.initialized) return;
        node.initialized = true;
        const bool unlimited_depth = node.remaining_depth == 63;
        const auto first = first_nonzero_index_packed(node.state, state_size_);
        if (first == state_size_) {
            node.top.push_back(CompactCandidate{0, 0, 0});
            return;
        }
        if ((!unlimited_depth && node.remaining_depth <= 0) || first >= groups_by_first_.size()) {
            return;
        }
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (node.state | (node.state >> 1)) & LOW_BITS;
        const auto available_two = (node.state >> 1) & LOW_BITS;
        const auto available_three = (node.state & (node.state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        std::uint32_t branch_order = 0;
        for (const auto& group : groups_by_first_[first]) {
            if (!subtract_precompiled_group(
                    node.state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            if (!unlimited_depth) {
                const int suffix_min = min_cover_depth_precompiled_packed(
                    next, state_size_, depth_groups_by_first_, min_depth_memo_
                );
                if (suffix_min > node.remaining_depth - 1) continue;
            }
            const auto child = node_for(
                next, unlimited_depth ? 63 : node.remaining_depth - 1
            );
            ensure(child, 0);
            if (!nodes_[child].top.empty()) {
                push_cursor(node, Cursor{group.group_id, child, 0, branch_order});
            }
            ++branch_order;
        }
    }

    void ensure(std::uint32_t node_id, std::size_t index) {
        initialize(node_id);
        auto& node = nodes_[node_id];
        while (node.top.size() <= index && !node.heap.empty()) {
            auto cursor = pop_cursor(node);
            const auto& tail = nodes_[cursor.child].top[cursor.index];
            const auto path = static_cast<std::uint32_t>(arena_.size());
            arena_.push_back(CompactPathNode{cursor.group_id, tail.node});
            node.top.push_back(CompactCandidate{
                group_score_units_[cursor.group_id] + tail.score_units,
                path,
                (static_cast<std::uint64_t>(tie_ranks_[cursor.group_id] + 1) << 48) |
                    (tail.tie_prefix >> 16),
            });
            ++cursor.index;
            ensure(cursor.child, cursor.index);
            if (cursor.index < nodes_[cursor.child].top.size()) {
                push_cursor(node, cursor);
            }
        }
    }

    std::size_t state_size_;
    const PackedBuckets& groups_by_first_;
    PackedBuckets depth_groups_by_first_;
    const std::vector<std::int64_t>& group_score_units_;
    const std::vector<std::uint32_t>& tie_ranks_;
    PackedDepthMemo& min_depth_memo_;
    std::vector<CompactPathNode>& arena_;
    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> indices_;
    std::pmr::monotonic_buffer_resource node_resource_;
    std::pmr::deque<Node> nodes_{&node_resource_};
};

static Covers top_covers_hand_count_window_lazy_compact_dp(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<std::uint32_t>& tie_ranks,
    int window,
    std::size_t max_results,
    PackedDepthMemo& min_depth_memo,
    LazyCompactWindowSolver& solver,
    std::vector<CompactPathNode>& arena
) {
    const auto packed_state = packed_state_key_2bit(state);
    const int min_depth = solver.min_depth(packed_state);
    constexpr int INF = 1 << 28;
    if (min_depth >= INF || state.size() > 28) return Covers{};
    const auto& candidates = solver.top(
        packed_state, min_depth + std::max(0, window), max_results
    );
    Covers out;
    out.reserve(std::min(max_results, candidates.size()));
    for (std::size_t index = 0; index < candidates.size() && index < max_results; ++index) {
        Cover cover;
        auto path = candidates[index].node;
        while (path != 0) {
            cover.push_back(arena[path].group_id);
            path = arena[path].tail;
        }
        out.push_back(std::move(cover));
    }
    return out;
}

static Covers top_covers_hand_count_window_lazy_selected(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<double>& group_selection_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::size_t selected_results,
    PackedDepthMemo& min_depth_memo,
    LazyCompactWindowSolver& solver,
    std::vector<CompactPathNode>& arena
) {
    const auto packed_state = packed_state_key_2bit(state);
    const int min_depth = solver.min_depth(packed_state);
    constexpr int INF = 1 << 28;
    if (min_depth >= INF || state.size() > 28 || max_results == 0) return Covers{};
    const int depth_limit = min_depth + std::max(0, window);
    std::vector<double> selected_scores;
    selected_scores.reserve(selected_results);
    std::size_t generated = 0;
    while (generated < max_results) {
        const auto& candidates = solver.top(packed_state, depth_limit, generated + 1);
        if (candidates.size() <= generated) break;
        const auto& candidate = candidates[generated];
        double priority_score = 0.0;
        auto path = candidate.node;
        while (path != 0) {
            priority_score += group_selection_scores[arena[path].group_id];
            path = arena[path].tail;
        }
        if (selected_scores.size() < selected_results) {
            selected_scores.push_back(priority_score);
            std::sort(selected_scores.begin(), selected_scores.end(), std::greater<double>());
        } else if (priority_score > selected_scores.back()) {
            selected_scores.back() = priority_score;
            std::sort(selected_scores.begin(), selected_scores.end(), std::greater<double>());
        }
        ++generated;
        if (selected_scores.size() < selected_results) continue;
        const double future_priority_upper =
            static_cast<double>(candidate.score_units) / 350.0;
        // group_scores are admitted to this path only when they are rational
        // to 1/350 within 1e-9. The margin covers the maximum accumulated
        // conversion error while keeping equal-score tie candidates alive.
        if (future_priority_upper + 1e-7 < selected_scores.back()) break;
    }
    const auto& candidates = solver.top(packed_state, depth_limit, generated);
    Covers generated_covers;
    generated_covers.reserve(std::min(generated, candidates.size()));
    for (std::size_t index = 0; index < generated && index < candidates.size(); ++index) {
        Cover cover;
        auto path = candidates[index].node;
        while (path != 0) {
            cover.push_back(arena[path].group_id);
            path = arena[path].tail;
        }
        generated_covers.push_back(std::move(cover));
    }
    return select_top_covers_exact(
        std::move(generated_covers), group_selection_scores, tie_keys, selected_results
    );
}

static Covers top_covers_lazy_compact_dp(
    const State& state,
    std::size_t max_results,
    LazyCompactWindowSolver& solver,
    std::vector<CompactPathNode>& arena
) {
    const auto& candidates = solver.top(packed_state_key_2bit(state), 63, max_results);
    Covers out;
    out.reserve(std::min(max_results, candidates.size()));
    for (std::size_t index = 0; index < candidates.size() && index < max_results; ++index) {
        Cover cover;
        auto path = candidates[index].node;
        while (path != 0) {
            cover.push_back(arena[path].group_id);
            path = arena[path].tail;
        }
        out.push_back(std::move(cover));
    }
    return out;
}

static Covers top_covers_lazy_compact_selected(
    const State& state,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<double>& group_selection_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    std::size_t selected_results,
    LazyCompactWindowSolver& solver,
    std::vector<CompactPathNode>& arena
) {
    if (max_results == 0 || selected_results == 0) return Covers{};
    const auto packed_state = packed_state_key_2bit(state);
    std::vector<double> selected_scores;
    selected_scores.reserve(selected_results);
    std::size_t generated = 0;
    while (generated < max_results) {
        const auto& candidates = solver.top(packed_state, 63, generated + 1);
        if (candidates.size() <= generated) break;
        const auto& candidate = candidates[generated];
        double priority_score = 0.0;
        auto path = candidate.node;
        while (path != 0) {
            priority_score += group_selection_scores[arena[path].group_id];
            path = arena[path].tail;
        }
        if (selected_scores.size() < selected_results) {
            selected_scores.push_back(priority_score);
            std::sort(selected_scores.begin(), selected_scores.end(), std::greater<double>());
        } else if (priority_score > selected_scores.back()) {
            selected_scores.back() = priority_score;
            std::sort(selected_scores.begin(), selected_scores.end(), std::greater<double>());
        }
        ++generated;
        if (selected_scores.size() < selected_results) continue;

        // The lazy solver is ordered by the exact 1/350 score units. Future
        // candidates cannot exceed the current candidate in that ordering.
        // Keep equal-score and floating-point-near candidates so the final
        // Python-compatible ordering remains identical to the full prefix.
        const double future_priority_upper =
            static_cast<double>(candidate.score_units) / 350.0;
        if (future_priority_upper + 1e-7 < selected_scores.back()) break;
    }

    const auto& candidates = solver.top(packed_state, 63, generated);
    Covers generated_covers;
    generated_covers.reserve(std::min(generated, candidates.size()));
    for (std::size_t index = 0; index < generated && index < candidates.size(); ++index) {
        Cover cover;
        auto path = candidates[index].node;
        while (path != 0) {
            cover.push_back(arena[path].group_id);
            path = arena[path].tail;
        }
        generated_covers.push_back(std::move(cover));
    }
    return select_top_covers_python_order(
        std::move(generated_covers), group_selection_scores, tie_keys, selected_results
    );
}

static const std::vector<CompactCandidate>& compact_top_depth_window_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<std::uint32_t>& tie_ranks,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    int remaining_depth,
    std::size_t max_results,
    CompactWindowMemo& memo,
    std::vector<CompactPathNode>& arena
) {
    const std::uint64_t key = state | (static_cast<std::uint64_t>(remaining_depth) << 56);
    const auto found = memo.find(key);
    if (found != nullptr) return *found;
    std::vector<CompactCandidate> top;
    top.reserve(std::min<std::size_t>(max_results, 16));
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        top.push_back(CompactCandidate{0, 0, 0});
        return memo.emplace(key, std::move(top));
    }
    if (remaining_depth <= 0 || first >= groups_by_first.size()) {
        return memo.emplace(key, std::move(top));
    }

    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    struct MergeCursor {
        unsigned short group_id;
        const std::vector<CompactCandidate>* tails;
        std::size_t index;
        std::size_t branch_order;
    };
    const auto cursor_better = [&](const MergeCursor& left, const MergeCursor& right) {
        const auto& left_tail = (*left.tails)[left.index];
        const auto& right_tail = (*right.tails)[right.index];
        const auto left_score = group_score_units[left.group_id] + left_tail.score_units;
        const auto right_score = group_score_units[right.group_id] + right_tail.score_units;
        if (left_score != right_score) {
            return left_score > right_score;
        }
        const auto left_rank = tie_ranks[left.group_id];
        const auto right_rank = tie_ranks[right.group_id];
        if (left_rank != right_rank) {
            return left_rank < right_rank;
        }
        if (left_tail.tie_prefix != right_tail.tie_prefix) {
            return left_tail.tie_prefix < right_tail.tie_prefix;
        }
        const int path_order = compact_path_tie_compare(
            left_tail.node, right_tail.node, arena, tie_ranks
        );
        if (path_order != 0) return path_order < 0;
        if (left.branch_order != right.branch_order) {
            return left.branch_order < right.branch_order;
        }
        return left.index < right.index;
    };
    const auto cursor_worse = [&](const MergeCursor& left, const MergeCursor& right) {
        return cursor_better(right, left);
    };
    std::priority_queue<
        MergeCursor,
        std::vector<MergeCursor>,
        decltype(cursor_worse)
    > queue(cursor_worse);

    std::uint64_t next = 0;
    std::size_t branch_order = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const int suffix_min = min_cover_depth_precompiled_packed(
            next, state_size, groups_by_first, min_depth_memo
        );
        if (suffix_min > remaining_depth - 1) {
            continue;
        }
        const auto& tails = compact_top_depth_window_packed(
            next,
            state_size,
            groups_by_first,
            group_score_units,
            tie_ranks,
            min_depth_memo,
            remaining_depth - 1,
            max_results,
            memo,
            arena
        );
        if (!tails.empty()) {
            queue.push(MergeCursor{group.group_id, &tails, 0, branch_order});
        }
        ++branch_order;
    }
    while (!queue.empty() && top.size() < max_results) {
        auto cursor = queue.top();
        queue.pop();
        const auto& tail = (*cursor.tails)[cursor.index];
        const auto node = static_cast<std::uint32_t>(arena.size());
        arena.push_back(CompactPathNode{cursor.group_id, tail.node});
        top.push_back(CompactCandidate{
            group_score_units[cursor.group_id] + tail.score_units,
            node,
            (static_cast<std::uint64_t>(tie_ranks[cursor.group_id] + 1) << 48) |
                (tail.tie_prefix >> 16),
        });
        ++cursor.index;
        if (cursor.index < cursor.tails->size()) {
            queue.push(cursor);
        }
    }
    return memo.emplace(key, std::move(top));
}

static Covers top_covers_hand_count_window_compact_dp(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<std::uint32_t>& tie_ranks,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    CompactWindowMemo& memo,
    std::vector<CompactPathNode>& arena
) {
    const auto packed_state = packed_state_key_2bit(state);
    const int min_depth = min_cover_depth_precompiled_packed(
        packed_state, state.size(), groups_by_first, min_depth_memo
    );
    constexpr int INF = 1 << 28;
    if (min_depth >= INF || state.size() > 28) {
        return Covers{};
    }
    const auto& top = compact_top_depth_window_packed(
        packed_state,
        state.size(),
        groups_by_first,
        group_score_units,
        tie_ranks,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        memo,
        arena
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        Cover cover;
        auto node = candidate.node;
        while (node != 0) {
            cover.push_back(arena[node].group_id);
            node = arena[node].tail;
        }
        out.push_back(std::move(cover));
    }
    return out;
}

static const std::vector<CompactCandidate>& compact_top_all_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<std::uint32_t>& tie_ranks,
    std::size_t max_results,
    CompactWindowMemo& memo,
    std::vector<CompactPathNode>& arena
) {
    const auto found = memo.find(state);
    if (found != nullptr) return *found;
    std::vector<CompactCandidate> top;
    top.reserve(std::min<std::size_t>(max_results, 16));
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        top.push_back(CompactCandidate{0, 0, 0});
        return memo.emplace(state, std::move(top));
    }
    if (first >= groups_by_first.size()) {
        return memo.emplace(state, std::move(top));
    }
    struct MergeCursor {
        unsigned short group_id;
        const std::vector<CompactCandidate>* tails;
        std::size_t index;
        std::size_t branch_order;
    };
    const auto cursor_better = [&](const MergeCursor& left, const MergeCursor& right) {
        const auto& left_tail = (*left.tails)[left.index];
        const auto& right_tail = (*right.tails)[right.index];
        const auto left_score = group_score_units[left.group_id] + left_tail.score_units;
        const auto right_score = group_score_units[right.group_id] + right_tail.score_units;
        if (left_score != right_score) return left_score > right_score;
        const auto left_rank = tie_ranks[left.group_id];
        const auto right_rank = tie_ranks[right.group_id];
        if (left_rank != right_rank) return left_rank < right_rank;
        if (left_tail.tie_prefix != right_tail.tie_prefix) {
            return left_tail.tie_prefix < right_tail.tie_prefix;
        }
        const int path_order = compact_path_tie_compare(
            left_tail.node, right_tail.node, arena, tie_ranks
        );
        if (path_order != 0) return path_order < 0;
        if (left.branch_order != right.branch_order) return left.branch_order < right.branch_order;
        return left.index < right.index;
    };
    const auto cursor_worse = [&](const MergeCursor& left, const MergeCursor& right) {
        return cursor_better(right, left);
    };
    std::priority_queue<MergeCursor, std::vector<MergeCursor>, decltype(cursor_worse)> queue(cursor_worse);
    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::uint64_t next = 0;
    std::size_t branch_order = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const auto& tails = compact_top_all_packed(
            next, state_size, groups_by_first, group_score_units, tie_ranks,
            max_results, memo, arena
        );
        if (!tails.empty()) queue.push(MergeCursor{group.group_id, &tails, 0, branch_order});
        ++branch_order;
    }
    while (!queue.empty() && top.size() < max_results) {
        auto cursor = queue.top();
        queue.pop();
        const auto& tail = (*cursor.tails)[cursor.index];
        const auto node = static_cast<std::uint32_t>(arena.size());
        arena.push_back(CompactPathNode{cursor.group_id, tail.node});
        top.push_back(CompactCandidate{
            group_score_units[cursor.group_id] + tail.score_units,
            node,
            (static_cast<std::uint64_t>(tie_ranks[cursor.group_id] + 1) << 48) |
                (tail.tie_prefix >> 16)
        });
        if (++cursor.index < cursor.tails->size()) queue.push(cursor);
    }
    return memo.emplace(state, std::move(top));
}

static std::vector<std::uint32_t> compact_tie_ranks(const std::vector<std::string>& tie_keys) {
    std::vector<std::size_t> order(tie_keys.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return tie_keys[left] < tie_keys[right];
    });
    std::vector<std::uint32_t> ranks(tie_keys.size(), 0);
    std::uint32_t rank = 0;
    for (std::size_t pos = 0; pos < order.size(); ++pos) {
        if (pos > 0 && tie_keys[order[pos - 1]] != tie_keys[order[pos]]) ++rank;
        ranks[order[pos]] = rank;
    }
    return ranks;
}

static std::int64_t best_suffix_score_units_depth(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    int remaining_depth,
    std::unordered_map<std::uint64_t, std::int64_t>& memo
) {
    constexpr std::int64_t NEG_INF = std::numeric_limits<std::int64_t>::min() / 4;
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) return 0;
    if (remaining_depth <= 0 || first >= groups_by_first.size()) return NEG_INF;
    const std::uint64_t key = state | (static_cast<std::uint64_t>(remaining_depth) << 56);
    const auto found = memo.find(key);
    if (found != memo.end()) return found->second;
    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::int64_t best = NEG_INF;
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const auto suffix = best_suffix_score_units_depth(
            next, state_size, groups_by_first, group_score_units,
            remaining_depth - 1, memo
        );
        if (suffix != NEG_INF) best = std::max(best, group_score_units[group.group_id] + suffix);
    }
    memo.emplace(key, best);
    return best;
}

static Covers top_covers_hand_count_window_astar(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<std::int64_t>& group_score_units,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    std::unordered_map<std::uint64_t, std::int64_t>& suffix_score_memo
) {
    struct SearchNode {
        std::uint64_t state = 0;
        std::int64_t score = 0;
        std::int64_t upper = 0;
        std::uint32_t path = 0;
        int remaining_depth = 0;
        std::uint64_t serial = 0;
    };
    struct Worse {
        bool operator()(const SearchNode& left, const SearchNode& right) const {
            if (left.upper != right.upper) return left.upper < right.upper;
            return left.serial > right.serial;
        }
    };
    constexpr std::int64_t NEG_INF = std::numeric_limits<std::int64_t>::min() / 4;
    if (state.empty() || max_results == 0) return state.empty() ? Covers{Cover{}} : Covers{};
    const auto packed_state = packed_state_key_2bit(state);
    const int min_depth = min_cover_depth_precompiled_packed(
        packed_state, state.size(), groups_by_first, min_depth_memo
    );
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) return Covers{};
    const int depth_limit = min_depth + std::max(0, window);
    const auto root_upper = best_suffix_score_units_depth(
        packed_state, state.size(), groups_by_first, group_score_units,
        depth_limit, suffix_score_memo
    );
    if (root_upper == NEG_INF) return Covers{};
    std::vector<CompactPathNode> arena;
    arena.reserve(max_results * 32);
    arena.push_back(CompactPathNode{});
    std::priority_queue<SearchNode, std::vector<SearchNode>, Worse> queue;
    std::uint64_t serial = 0;
    queue.push(SearchNode{packed_state, 0, root_upper, 0, depth_limit, serial++});
    struct Completed {
        std::int64_t score;
        Cover cover;
    };
    std::vector<Completed> completed;
    completed.reserve(max_results * 2);
    std::int64_t threshold = NEG_INF;
    while (!queue.empty()) {
        if (completed.size() >= max_results && queue.top().upper < threshold) break;
        const auto node = queue.top();
        queue.pop();
        const auto first = first_nonzero_index_packed(node.state, state.size());
        if (first == state.size()) {
            Cover cover;
            auto path = node.path;
            while (path != 0) {
                cover.push_back(arena[path].group_id);
                path = arena[path].tail;
            }
            std::reverse(cover.begin(), cover.end());
            completed.push_back(Completed{node.score, std::move(cover)});
            if (completed.size() == max_results) threshold = node.score;
            continue;
        }
        if (node.remaining_depth <= 0 || first >= groups_by_first.size()) continue;
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (node.state | (node.state >> 1)) & LOW_BITS;
        const auto available_two = (node.state >> 1) & LOW_BITS;
        const auto available_three = (node.state & (node.state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    node.state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            const int suffix_min = min_cover_depth_precompiled_packed(
                next, state.size(), groups_by_first, min_depth_memo
            );
            if (suffix_min > node.remaining_depth - 1) continue;
            const auto suffix_upper = best_suffix_score_units_depth(
                next, state.size(), groups_by_first, group_score_units,
                node.remaining_depth - 1, suffix_score_memo
            );
            if (suffix_upper == NEG_INF) continue;
            const auto next_score = node.score + group_score_units[group.group_id];
            const auto path = static_cast<std::uint32_t>(arena.size());
            arena.push_back(CompactPathNode{group.group_id, node.path});
            queue.push(SearchNode{
                next,
                next_score,
                next_score + suffix_upper,
                path,
                node.remaining_depth - 1,
                serial++,
            });
        }
    }
    std::stable_sort(completed.begin(), completed.end(), [&](const Completed& left, const Completed& right) {
        if (left.score != right.score) return left.score > right.score;
        return cover_tie_less(left.cover, right.cover, tie_keys);
    });
    Covers out;
    out.reserve(std::min(max_results, completed.size()));
    for (auto& candidate : completed) {
        out.push_back(std::move(candidate.cover));
        if (out.size() >= max_results) break;
    }
    return out;
}

static Covers top_covers_impl(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    bool enable_upper_bound
) {
    std::unordered_map<std::string, std::vector<Candidate>> memo;
    std::unordered_map<std::string, double> max_score_memo;
    memo.reserve(4096);
    max_score_memo.reserve(4096);
    auto top = suffix_top(state, groups_by_first, group_scores, tie_keys, max_results, enable_upper_bound, memo, max_score_memo);

    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

Covers top_covers(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    bool enable_upper_bound
) {
    py::gil_scoped_release release;
    return top_covers_impl(
        state, groups_by_first, group_scores, tie_keys, max_results, enable_upper_bound
    );
}

std::vector<Covers> top_covers_batch(
    const std::vector<std::vector<unsigned char>>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    bool enable_upper_bound
) {
    std::vector<Covers> results;
    results.reserve(states.size());
    py::gil_scoped_release release;
    if (native_batch_shared_top_memo_enabled()) {
        const bool use_packed = native_batch_packed_top_memo_enabled()
            && !enable_upper_bound
            && !states.empty()
            && std::all_of(
            states.begin(),
            states.end(),
            [&states](const State& state) {
                return state.size() == states.front().size() && can_pack_state_2bit(state);
            }
        );
        if (use_packed) {
            if (native_compact_top_dp_enabled() && states.front().size() <= 28) {
                std::vector<std::int64_t> group_score_units;
                group_score_units.reserve(group_scores.size());
                bool rational_scores = true;
                for (const double score : group_scores) {
                    const auto units = static_cast<std::int64_t>(std::llround(score * 350.0));
                    if (std::abs(score - static_cast<double>(units) / 350.0) > 1e-9) {
                        rational_scores = false;
                        break;
                    }
                    group_score_units.push_back(units);
                }
                PackedBuckets packed_buckets;
                if (rational_scores && compile_packed_buckets(
                        groups_by_first, states.front().size(), packed_buckets)) {
                    const auto tie_ranks = compact_tie_ranks(tie_keys);
                    CompactWindowMemo compact_memo;
                    const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
                    compact_memo.reserve(memo_capacity);
                    std::vector<CompactPathNode> arena;
                    arena.reserve(memo_capacity * 4);
                    arena.push_back(CompactPathNode{});
                    PackedDepthMemo unused_min_depth_memo;
                    unused_min_depth_memo.reserve(memo_capacity);
                    std::unique_ptr<LazyCompactWindowSolver> lazy_solver;
                    if (native_lazy_compact_top_dp_enabled()) {
                        lazy_solver = std::make_unique<LazyCompactWindowSolver>(
                            states.front().size(),
                            packed_buckets,
                            group_score_units,
                            tie_ranks,
                            unused_min_depth_memo,
                            arena,
                            memo_capacity
                        );
                    }
                    for (const auto& state : states) {
                        if (lazy_solver) {
                            results.push_back(top_covers_lazy_compact_dp(
                                state, max_results, *lazy_solver, arena
                            ));
                            continue;
                        }
                        const auto& top = compact_top_all_packed(
                            packed_state_key_2bit(state),
                            state.size(),
                            packed_buckets,
                            group_score_units,
                            tie_ranks,
                            max_results,
                            compact_memo,
                            arena
                        );
                        Covers covers;
                        covers.reserve(top.size());
                        for (const auto& candidate : top) {
                            Cover cover;
                            auto node = candidate.node;
                            while (node != 0) {
                                cover.push_back(arena[node].group_id);
                                node = arena[node].tail;
                            }
                            covers.push_back(std::move(cover));
                        }
                        results.push_back(std::move(covers));
                    }
                    return results;
                }
            }
            std::unordered_map<std::uint64_t, std::vector<Candidate>> memo;
            const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
            memo.reserve(memo_capacity);
            const auto state_size = states.front().size();
            for (const auto& state : states) {
                const auto& top = suffix_top_direct_packed(
                    packed_state_key_2bit(state),
                    state_size,
                    groups_by_first,
                    group_scores,
                    tie_keys,
                    max_results,
                    memo
                );
                Covers covers;
                covers.reserve(top.size());
                for (const auto& candidate : top) {
                    covers.push_back(candidate.cover);
                }
                results.push_back(std::move(covers));
            }
            return results;
        }
        std::unordered_map<std::string, std::vector<Candidate>> memo;
        std::unordered_map<std::string, double> max_score_memo;
        const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
        memo.reserve(memo_capacity);
        max_score_memo.reserve(memo_capacity);
        for (const auto& state : states) {
            auto top = suffix_top(
                state,
                groups_by_first,
                group_scores,
                tie_keys,
                max_results,
                enable_upper_bound,
                memo,
                max_score_memo
            );
            Covers covers;
            covers.reserve(top.size());
            for (const auto& candidate : top) {
                covers.push_back(candidate.cover);
            }
            results.push_back(std::move(covers));
        }
        return results;
    }
    for (const auto& state : states) {
        results.push_back(top_covers_impl(
            state, groups_by_first, group_scores, tie_keys, max_results, enable_upper_bound
        ));
    }
    return results;
}

std::vector<Covers> top_covers_selected_batch(
    const std::vector<std::vector<unsigned char>>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_priorities,
    const std::vector<std::string>& tie_keys,
    std::size_t max_results,
    std::size_t selected_results,
    bool enable_upper_bound
) {
    const bool use_lazy_exact_selection =
        native_lazy_selected_bound_enabled() &&
        native_batch_shared_top_memo_enabled() &&
        native_batch_packed_top_memo_enabled() &&
        native_compact_top_dp_enabled() &&
        native_lazy_compact_top_dp_enabled() &&
        !enable_upper_bound &&
        selected_results > 0 &&
        !states.empty() &&
        states.front().size() <= 28 &&
        std::all_of(states.begin(), states.end(), [&states](const State& state) {
            return state.size() == states.front().size() && can_pack_state_2bit(state);
        });
    if (use_lazy_exact_selection) {
        std::vector<std::int64_t> group_score_units;
        group_score_units.reserve(group_scores.size());
        bool compatible_scores = group_scores.size() == group_priorities.size();
        for (std::size_t index = 0; compatible_scores && index < group_scores.size(); ++index) {
            const double score = group_scores[index];
            const auto units = static_cast<std::int64_t>(std::llround(score * 350.0));
            compatible_scores =
                std::abs(score - static_cast<double>(units) / 350.0) <= 1e-9 &&
                std::abs(score - (group_priorities[index] - 10.0)) <= 1e-9;
            group_score_units.push_back(units);
        }
        PackedBuckets packed_buckets;
        compatible_scores = compatible_scores && compile_packed_buckets(
            groups_by_first, states.front().size(), packed_buckets
        );
        if (compatible_scores) {
            const auto tie_ranks = compact_tie_ranks(tie_keys);
            const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
            PackedDepthMemo min_depth_memo;
            min_depth_memo.reserve(memo_capacity);
            std::vector<CompactPathNode> arena;
            arena.reserve(memo_capacity * 4);
            arena.push_back(CompactPathNode{});
            LazyCompactWindowSolver solver(
                states.front().size(),
                packed_buckets,
                group_score_units,
                tie_ranks,
                min_depth_memo,
                arena,
                memo_capacity
            );
            std::vector<Covers> results;
            results.reserve(states.size());
            py::gil_scoped_release release;
            for (const auto& state : states) {
                results.push_back(top_covers_lazy_compact_selected(
                    state,
                    group_score_units,
                    group_scores,
                    tie_keys,
                    max_results,
                    selected_results,
                    solver,
                    arena
                ));
            }
            return results;
        }
    }
    auto results = top_covers_batch(
        states,
        groups_by_first,
        group_scores,
        tie_keys,
        max_results,
        enable_upper_bound
    );
    py::gil_scoped_release release;
    for (auto& covers : results) {
        covers = select_top_covers_python_order(
            std::move(covers), group_scores, tie_keys, selected_results
        );
    }
    return results;
}

static Covers top_covers_hand_count_window_impl_with_memo(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::string, int>& min_depth_memo,
    std::unordered_map<std::string, double>& max_score_memo
) {
    const int min_depth = min_cover_depth(state, groups_by_first, min_depth_memo);
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) {
        return Covers{};
    }
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(static_cast<std::size_t>(std::max(0, min_depth + std::max(0, window))));
    collect_top_depth_window(
        state,
        groups_by_first,
        group_scores,
        tie_keys,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

static Covers top_covers_hand_count_window_impl_with_packed_memo(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    std::unordered_map<std::uint64_t, double>& max_score_memo
) {
    const int min_depth = min_cover_depth_packed(state, groups_by_first, min_depth_memo);
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) {
        return Covers{};
    }
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(static_cast<std::size_t>(std::max(0, min_depth + std::max(0, window))));
    collect_top_depth_window_packed(
        state,
        groups_by_first,
        group_scores,
        tie_keys,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

static Covers top_covers_hand_count_window_impl_with_direct_packed_memo(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    std::unordered_map<std::uint64_t, double>& max_score_memo
) {
    const auto packed_state = packed_state_key_2bit(state);
    const auto state_size = state.size();
    const int min_depth = min_cover_depth_direct_packed(
        packed_state, state_size, groups_by_first, min_depth_memo
    );
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) {
        return Covers{};
    }
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(static_cast<std::size_t>(std::max(0, min_depth + std::max(0, window))));
    collect_top_depth_window_direct_packed(
        packed_state,
        state_size,
        groups_by_first,
        group_scores,
        tie_keys,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

static Covers top_covers_hand_count_window_impl_with_precompiled_packed_memo(
    const std::vector<unsigned char>& state,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::unordered_map<std::uint64_t, int>& min_depth_memo,
    std::unordered_map<std::uint64_t, double>& max_score_memo,
    PackedDepthTransitionMemo* transition_memo
) {
    const auto packed_state = packed_state_key_2bit(state);
    const auto state_size = state.size();
    const int min_depth = min_cover_depth_precompiled_packed(
        packed_state, state_size, groups_by_first, min_depth_memo
    );
    constexpr int INF = 1 << 28;
    if (min_depth >= INF) {
        return Covers{};
    }
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(static_cast<std::size_t>(std::max(0, min_depth + std::max(0, window))));
    collect_top_depth_window_precompiled_packed(
        packed_state,
        state_size,
        groups_by_first,
        group_scores,
        tie_keys,
        min_depth_memo,
        min_depth + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo,
        transition_memo,
        native_depth_window_upper_bound_enabled()
    );
    Covers out;
    out.reserve(top.size());
    for (const auto& candidate : top) {
        out.push_back(candidate.cover);
    }
    return out;
}

static Covers top_covers_hand_count_window_impl(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results
) {
    std::unordered_map<std::string, int> min_depth_memo;
    std::unordered_map<std::string, double> max_score_memo;
    min_depth_memo.reserve(4096);
    max_score_memo.reserve(4096);
    return top_covers_hand_count_window_impl_with_memo(
        state,
        groups_by_first,
        group_scores,
        tie_keys,
        window,
        max_results,
        min_depth_memo,
        max_score_memo
    );
}

Covers top_covers_hand_count_window(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results
) {
    py::gil_scoped_release release;
    return top_covers_hand_count_window_impl(state, groups_by_first, group_scores, tie_keys, window, max_results);
}

std::vector<Covers> top_covers_hand_count_window_batch(
    const std::vector<std::vector<unsigned char>>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results
) {
    std::vector<Covers> results(states.size());
    if (!native_batch_shared_window_memo_enabled()) {
        py::gil_scoped_release release;
        for (std::size_t i = 0; i < states.size(); ++i) {
            results[i] = top_covers_hand_count_window_impl(
                states[i], groups_by_first, group_scores, tie_keys, window, max_results
            );
        }
        return results;
    }
    const auto packed_state_size = states.empty() ? 0 : states.front().size();
    const bool use_packed_memo =
        !states.empty() && native_batch_packed_window_memo_enabled() &&
        std::all_of(states.begin(), states.end(), [&](const State& state) {
            return state.size() == packed_state_size && can_pack_state_2bit(state);
        });
    if (use_packed_memo) {
        std::unordered_map<std::uint64_t, int> min_depth_memo;
        std::unordered_map<std::uint64_t, double> max_score_memo;
        PackedDepthTransitionMemo transition_memo;
        const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
        min_depth_memo.reserve(memo_capacity);
        max_score_memo.reserve(memo_capacity);
        const bool reuse_transitions = native_batch_window_transition_memo_enabled();
        if (reuse_transitions) {
            transition_memo.reserve(memo_capacity);
        }
        const bool direct_packed = native_batch_direct_packed_state_enabled();
        PackedBuckets packed_buckets;
        const bool precompiled_groups =
            direct_packed && native_batch_precompiled_groups_enabled() &&
            compile_packed_buckets(groups_by_first, packed_state_size, packed_buckets);
        py::gil_scoped_release release;
        if (precompiled_groups) {
            for (std::size_t i = 0; i < states.size(); ++i) {
                results[i] = top_covers_hand_count_window_impl_with_precompiled_packed_memo(
                    states[i],
                    packed_buckets,
                    group_scores,
                    tie_keys,
                    window,
                    max_results,
                    min_depth_memo,
                    max_score_memo,
                    reuse_transitions ? &transition_memo : nullptr
                );
            }
            return results;
        }
        for (std::size_t i = 0; i < states.size(); ++i) {
            results[i] = direct_packed
                ? top_covers_hand_count_window_impl_with_direct_packed_memo(
                    states[i],
                    groups_by_first,
                    group_scores,
                    tie_keys,
                    window,
                    max_results,
                    min_depth_memo,
                    max_score_memo
                )
                : top_covers_hand_count_window_impl_with_packed_memo(
                    states[i],
                    groups_by_first,
                    group_scores,
                    tie_keys,
                    window,
                    max_results,
                    min_depth_memo,
                    max_score_memo
                );
        }
        return results;
    }
    std::unordered_map<std::string, int> min_depth_memo;
    std::unordered_map<std::string, double> max_score_memo;
    const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
    min_depth_memo.reserve(memo_capacity);
    max_score_memo.reserve(memo_capacity);
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < states.size(); ++i) {
        results[i] = top_covers_hand_count_window_impl_with_memo(
            states[i],
            groups_by_first,
            group_scores,
            tie_keys,
            window,
            max_results,
            min_depth_memo,
            max_score_memo
        );
    }
    return results;
}

Covers top_covers_hand_count_window_selected(
    const std::vector<unsigned char>& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_priorities,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::size_t selected_results
) {
    py::gil_scoped_release release;
    return select_top_covers_exact(
        top_covers_hand_count_window_impl(
            state, groups_by_first, group_scores, tie_keys, window, max_results
        ),
        group_scores,
        tie_keys,
        selected_results
    );
}

std::vector<Covers> top_covers_hand_count_window_selected_batch(
    const std::vector<std::vector<unsigned char>>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_priorities,
    const std::vector<std::string>& tie_keys,
    int window,
    std::size_t max_results,
    std::size_t selected_results
) {
    std::vector<Covers> results;
    results.reserve(states.size());
    if (!native_batch_shared_window_memo_enabled()) {
        py::gil_scoped_release release;
        for (const auto& state : states) {
            results.push_back(select_top_covers_exact(
                top_covers_hand_count_window_impl(
                    state, groups_by_first, group_scores, tie_keys, window, max_results
                ),
                group_scores,
                tie_keys,
                selected_results
            ));
        }
        return results;
    }
    const auto packed_state_size = states.empty() ? 0 : states.front().size();
    const bool use_packed_memo =
        !states.empty() && native_batch_packed_window_memo_enabled() &&
        std::all_of(states.begin(), states.end(), [&](const State& state) {
            return state.size() == packed_state_size && can_pack_state_2bit(state);
        });
    if (use_packed_memo) {
        std::unordered_map<std::uint64_t, int> min_depth_memo;
        std::unordered_map<std::uint64_t, double> max_score_memo;
        PackedDepthTransitionMemo transition_memo;
        const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
        min_depth_memo.reserve(memo_capacity);
        max_score_memo.reserve(memo_capacity);
        const bool reuse_transitions = native_batch_window_transition_memo_enabled();
        if (reuse_transitions) {
            transition_memo.reserve(memo_capacity);
        }
        const bool direct_packed = native_batch_direct_packed_state_enabled();
        PackedBuckets packed_buckets;
        const bool precompiled_groups =
            direct_packed && native_batch_precompiled_groups_enabled() &&
            compile_packed_buckets(groups_by_first, packed_state_size, packed_buckets);
        py::gil_scoped_release release;
        if (precompiled_groups) {
            if (
                native_compact_window_dp_enabled() &&
                packed_state_size <= 28 &&
                states.size() >= native_compact_window_min_states()
            ) {
                std::vector<std::int64_t> group_score_units;
                group_score_units.reserve(group_scores.size());
                bool rational_scores = true;
                for (const double score : group_scores) {
                    const auto units = static_cast<std::int64_t>(std::llround(score * 350.0));
                    if (std::abs(score - static_cast<double>(units) / 350.0) > 1e-9) {
                        rational_scores = false;
                        break;
                    }
                    group_score_units.push_back(units);
                }
                if (rational_scores) {
                    const auto tie_ranks = compact_tie_ranks(tie_keys);
                    if (native_window_astar_enabled()) {
                        std::unordered_map<std::uint64_t, std::int64_t> suffix_score_memo;
                        suffix_score_memo.reserve(memo_capacity);
                        for (const auto& state : states) {
                            results.push_back(select_top_covers_exact(
                                top_covers_hand_count_window_astar(
                                    state,
                                    packed_buckets,
                                    group_score_units,
                                    tie_keys,
                                    window,
                                    max_results,
                                    min_depth_memo,
                                    suffix_score_memo
                                ),
                                group_scores,
                                tie_keys,
                                selected_results
                            ));
                        }
                        return results;
                    }
                    const std::size_t compact_worker_count = std::min<std::size_t>(
                        native_batch_threads(), states.size()
                    );
                    if (
                        native_compact_parallel_window_batch_enabled() &&
                        native_parallel_window_batch_enabled() &&
                        compact_worker_count > 1
                    ) {
                        std::vector<Covers> parallel_results(states.size());
                        std::atomic<std::size_t> next_index{0};
                        std::vector<std::thread> workers;
                        workers.reserve(compact_worker_count);
                        for (std::size_t worker_id = 0; worker_id < compact_worker_count; ++worker_id) {
                            workers.emplace_back([&]() {
                                PackedDepthMemo local_lazy_min_depth_memo;
                                local_lazy_min_depth_memo.reserve(memo_capacity);
                                std::unordered_map<std::uint64_t, int> local_eager_min_depth_memo;
                                if (!native_lazy_compact_window_dp_enabled()) {
                                    local_eager_min_depth_memo.reserve(memo_capacity);
                                }
                                CompactWindowMemo local_compact_memo;
                                local_compact_memo.reserve(memo_capacity);
                                std::vector<CompactPathNode> local_arena;
                                local_arena.reserve(memo_capacity * 4);
                                local_arena.push_back(CompactPathNode{});
                                std::unique_ptr<LazyCompactWindowSolver> local_lazy_solver;
                                if (native_lazy_compact_window_dp_enabled()) {
                                    local_lazy_solver = std::make_unique<LazyCompactWindowSolver>(
                                        packed_state_size,
                                        packed_buckets,
                                        group_score_units,
                                        tie_ranks,
                                        local_lazy_min_depth_memo,
                                        local_arena,
                                        memo_capacity
                                    );
                                }
                                while (true) {
                                    const auto index = next_index.fetch_add(1, std::memory_order_relaxed);
                                    if (index >= states.size()) break;
                                    if (local_lazy_solver && native_lazy_selected_bound_enabled()) {
                                        parallel_results[index] = top_covers_hand_count_window_lazy_selected(
                                            states[index],
                                            packed_buckets,
                                            group_score_units,
                                            group_scores,
                                            tie_keys,
                                            window,
                                            max_results,
                                            selected_results,
                                            local_lazy_min_depth_memo,
                                            *local_lazy_solver,
                                            local_arena
                                        );
                                    } else {
                                        parallel_results[index] = select_top_covers_exact(
                                            local_lazy_solver
                                                ? top_covers_hand_count_window_lazy_compact_dp(
                                                states[index],
                                                packed_buckets,
                                                group_score_units,
                                                tie_ranks,
                                                window,
                                                max_results,
                                                local_lazy_min_depth_memo,
                                                *local_lazy_solver,
                                                local_arena
                                            )
                                            : top_covers_hand_count_window_compact_dp(
                                                states[index],
                                                packed_buckets,
                                                group_score_units,
                                                tie_ranks,
                                                window,
                                                max_results,
                                                local_eager_min_depth_memo,
                                                local_compact_memo,
                                                local_arena
                                                ),
                                            group_scores,
                                            tie_keys,
                                            selected_results
                                        );
                                    }
                                }
                            });
                        }
                        for (auto& worker : workers) worker.join();
                        return parallel_results;
                    }
                    CompactWindowMemo compact_memo;
                    compact_memo.reserve(memo_capacity);
                    std::vector<CompactPathNode> arena;
                    arena.reserve(memo_capacity * 4);
                    arena.push_back(CompactPathNode{});
                    PackedDepthMemo lazy_min_depth_memo;
                    std::unique_ptr<LazyCompactWindowSolver> lazy_solver;
                    if (native_lazy_compact_window_dp_enabled()) {
                        lazy_min_depth_memo.reserve(memo_capacity);
                        lazy_solver = std::make_unique<LazyCompactWindowSolver>(
                            packed_state_size,
                            packed_buckets,
                            group_score_units,
                            tie_ranks,
                            lazy_min_depth_memo,
                            arena,
                            memo_capacity
                        );
                    }
                    for (const auto& state : states) {
                        if (lazy_solver && native_lazy_selected_bound_enabled()) {
                            results.push_back(top_covers_hand_count_window_lazy_selected(
                                state,
                                packed_buckets,
                                group_score_units,
                                group_scores,
                                tie_keys,
                                window,
                                max_results,
                                selected_results,
                                lazy_min_depth_memo,
                                *lazy_solver,
                                arena
                            ));
                            continue;
                        }
                        results.push_back(select_top_covers_exact(
                            lazy_solver
                                ? top_covers_hand_count_window_lazy_compact_dp(
                                    state,
                                    packed_buckets,
                                    group_score_units,
                                    tie_ranks,
                                    window,
                                    max_results,
                                    lazy_min_depth_memo,
                                    *lazy_solver,
                                    arena
                                )
                                : top_covers_hand_count_window_compact_dp(
                                    state,
                                    packed_buckets,
                                    group_score_units,
                                    tie_ranks,
                                    window,
                                    max_results,
                                    min_depth_memo,
                                    compact_memo,
                                    arena
                                ),
                            group_scores,
                            tie_keys,
                            selected_results
                        ));
                    }
                    return results;
                }
            }
            if (native_window_suffix_memo_enabled()) {
                WindowSuffixMemo suffix_memo;
                suffix_memo.reserve(memo_capacity);
                for (const auto& state : states) {
                    results.push_back(select_top_covers_exact(
                        top_covers_hand_count_window_suffix_memo(
                            state,
                            packed_buckets,
                            group_scores,
                            tie_keys,
                            window,
                            max_results,
                            min_depth_memo,
                            suffix_memo
                        ),
                        group_scores,
                        tie_keys,
                        selected_results
                    ));
                }
                return results;
            }
            const std::size_t worker_count = std::min<std::size_t>(
                native_batch_threads(), states.size()
            );
            if (native_parallel_window_batch_enabled() && worker_count > 1) {
                // Populate every reachable min-depth state before sharing the
                // memo across workers. Recursive collection can then only hit
                // existing entries, so concurrent access is read-only.
                for (const auto& state : states) {
                    min_cover_depth_precompiled_packed(
                        packed_state_key_2bit(state),
                        state.size(),
                        packed_buckets,
                        min_depth_memo
                    );
                }
                std::vector<Covers> parallel_results(states.size());
                std::atomic<std::size_t> next_index{0};
                std::vector<std::thread> workers;
                workers.reserve(worker_count);
                for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                    workers.emplace_back([&]() {
                        std::unordered_map<std::uint64_t, double> local_max_score_memo;
                        local_max_score_memo.reserve(memo_capacity);
                        while (true) {
                            const std::size_t index = next_index.fetch_add(
                                1, std::memory_order_relaxed
                            );
                            if (index >= states.size()) {
                                break;
                            }
                            parallel_results[index] = select_top_covers_exact(
                                top_covers_hand_count_window_impl_with_precompiled_packed_memo(
                                    states[index],
                                    packed_buckets,
                                    group_scores,
                                    tie_keys,
                                    window,
                                    max_results,
                                    min_depth_memo,
                                    local_max_score_memo,
                                    nullptr
                                ),
                                group_scores,
                                tie_keys,
                                selected_results
                            );
                        }
                    });
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                return parallel_results;
            }
            for (const auto& state : states) {
                results.push_back(select_top_covers_exact(
                    top_covers_hand_count_window_impl_with_precompiled_packed_memo(
                        state,
                        packed_buckets,
                        group_scores,
                        tie_keys,
                        window,
                        max_results,
                        min_depth_memo,
                        max_score_memo,
                        reuse_transitions ? &transition_memo : nullptr
                    ),
                    group_scores,
                    tie_keys,
                    selected_results
                ));
            }
            return results;
        }
        for (const auto& state : states) {
            results.push_back(select_top_covers_exact(
                direct_packed
                    ? top_covers_hand_count_window_impl_with_direct_packed_memo(
                        state,
                        groups_by_first,
                        group_scores,
                        tie_keys,
                        window,
                        max_results,
                        min_depth_memo,
                        max_score_memo
                    )
                    : top_covers_hand_count_window_impl_with_packed_memo(
                        state,
                        groups_by_first,
                        group_scores,
                        tie_keys,
                        window,
                        max_results,
                        min_depth_memo,
                        max_score_memo
                    ),
                group_scores,
                tie_keys,
                selected_results
            ));
        }
        return results;
    }
    std::unordered_map<std::string, int> min_depth_memo;
    std::unordered_map<std::string, double> max_score_memo;
    const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
    min_depth_memo.reserve(memo_capacity);
    max_score_memo.reserve(memo_capacity);
    py::gil_scoped_release release;
    for (const auto& state : states) {
        results.push_back(select_top_covers_exact(
            top_covers_hand_count_window_impl_with_memo(
                state,
                groups_by_first,
                group_scores,
                tie_keys,
                window,
                max_results,
                min_depth_memo,
                max_score_memo
            ),
            group_scores,
            tie_keys,
            selected_results
        ));
    }
    return results;
}

static int min_effective_hand_count(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<int>& group_costs,
    std::unordered_map<std::string, int>& memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) return 0;
    const auto key = state_key(state);
    const auto found = memo.find(key);
    if (found != memo.end()) return found->second;
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        State next;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_group(state, group, next)) continue;
            const auto group_id = group[0];
            if (group_id >= group_costs.size()) continue;
            const int suffix = min_effective_hand_count(
                next, groups_by_first, group_costs, memo
            );
            if (suffix < INF) {
                best = std::min(best, group_costs[group_id] + suffix);
            }
        }
    }
    memo.emplace(key, best);
    return best;
}

static void collect_optimal_effective_group_ids(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<int>& group_costs,
    std::unordered_map<std::string, int>& min_cost_memo,
    std::unordered_set<std::string>& visited,
    std::unordered_set<std::size_t>& selected
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size() || first >= groups_by_first.size()) return;
    const auto key = state_key(state);
    if (!visited.emplace(key).second) return;
    constexpr int INF = 1 << 28;
    const int optimum = min_effective_hand_count(
        state, groups_by_first, group_costs, min_cost_memo
    );
    if (optimum >= INF) return;
    State next;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group(state, group, next)) continue;
        const auto group_id = group[0];
        if (group_id >= group_costs.size()) continue;
        const int suffix = min_effective_hand_count(
            next, groups_by_first, group_costs, min_cost_memo
        );
        if (suffix >= INF || group_costs[group_id] + suffix != optimum) continue;
        selected.emplace(group_id);
        collect_optimal_effective_group_ids(
            next,
            groups_by_first,
            group_costs,
            min_cost_memo,
            visited,
            selected
        );
    }
}

static int min_effective_hand_count_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<int>& group_costs,
    PackedDepthMemo& memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) return 0;
    int memoized = 0;
    if (memo.find(state, memoized)) return memoized;
    constexpr int INF = 1 << 28;
    int best = INF;
    if (first < groups_by_first.size()) {
        constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
        const auto available_one = (state | (state >> 1)) & LOW_BITS;
        const auto available_two = (state >> 1) & LOW_BITS;
        const auto available_three = (state & (state >> 1)) & LOW_BITS;
        std::uint64_t next = 0;
        for (const auto& group : groups_by_first[first]) {
            if (!subtract_precompiled_group(
                    state, available_one, available_two, available_three, group, next)) {
                continue;
            }
            if (group.group_id >= group_costs.size()) continue;
            const int suffix = min_effective_hand_count_packed(
                next, state_size, groups_by_first, group_costs, memo
            );
            if (suffix < INF) {
                best = std::min(best, group_costs[group.group_id] + suffix);
            }
        }
    }
    memo.emplace(state, best);
    return best;
}

static void collect_optimal_effective_group_ids_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<int>& group_costs,
    PackedDepthMemo& min_cost_memo,
    std::unordered_set<std::uint64_t>& visited,
    std::unordered_set<std::size_t>& selected
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size || first >= groups_by_first.size()) return;
    if (!visited.emplace(state).second) return;
    constexpr int INF = 1 << 28;
    const int optimum = min_effective_hand_count_packed(
        state, state_size, groups_by_first, group_costs, min_cost_memo
    );
    if (optimum >= INF) return;
    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        if (group.group_id >= group_costs.size()) continue;
        const int suffix = min_effective_hand_count_packed(
            next, state_size, groups_by_first, group_costs, min_cost_memo
        );
        if (suffix >= INF || group_costs[group.group_id] + suffix != optimum) continue;
        selected.emplace(group.group_id);
        collect_optimal_effective_group_ids_packed(
            next,
            state_size,
            groups_by_first,
            group_costs,
            min_cost_memo,
            visited,
            selected
        );
    }
}

static void validate_optimal_effective_inputs(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<int>& group_costs
) {
    for (std::size_t bucket_idx = 0; bucket_idx < groups_by_first.size(); ++bucket_idx) {
        if (bucket_idx >= state.size() && !groups_by_first[bucket_idx].empty()) {
            throw py::value_error("non-empty group bucket index exceeds state size");
        }
        for (const auto& group : groups_by_first[bucket_idx]) {
            if (group.size() < 3 || group.size() % 2 == 0) {
                throw py::value_error(
                    "encoded group must contain group_id followed by one or more index/count pairs"
                );
            }
            if (static_cast<std::size_t>(group[0]) >= group_costs.size()) {
                throw py::value_error("encoded group_id exceeds group_costs size");
            }
            bool removes_bucket_anchor = false;
            for (std::size_t i = 1; i + 1 < group.size(); i += 2) {
                const auto idx = static_cast<std::size_t>(group[i]);
                const auto count = static_cast<unsigned int>(group[i + 1]);
                if (idx >= state.size()) {
                    throw py::value_error("encoded group card index exceeds state size");
                }
                if (count == 0) {
                    throw py::value_error("encoded group count must be positive");
                }
                if (idx == bucket_idx) {
                    removes_bucket_anchor = true;
                }
            }
            if (!removes_bucket_anchor) {
                throw py::value_error("encoded group must remove a card from its bucket index");
            }
        }
    }
}

std::vector<std::size_t> optimal_effective_group_ids(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<int>& group_costs
) {
    validate_optimal_effective_inputs(state, groups_by_first, group_costs);
    std::unordered_set<std::size_t> selected;
    selected.reserve(group_costs.size());
    {
        py::gil_scoped_release release;
        PackedBuckets packed_buckets;
        if (
            can_pack_state_2bit(state) &&
            compile_packed_buckets(groups_by_first, state.size(), packed_buckets)
        ) {
            PackedDepthMemo min_cost_memo;
            std::unordered_set<std::uint64_t> visited;
            min_cost_memo.reserve(4096);
            visited.reserve(4096);
            collect_optimal_effective_group_ids_packed(
                packed_state_key_2bit(state),
                state.size(),
                packed_buckets,
                group_costs,
                min_cost_memo,
                visited,
                selected
            );
        } else {
            std::unordered_map<std::string, int> min_cost_memo;
            std::unordered_set<std::string> visited;
            min_cost_memo.reserve(4096);
            visited.reserve(4096);
            collect_optimal_effective_group_ids(
                state,
                groups_by_first,
                group_costs,
                min_cost_memo,
                visited,
                selected
            );
        }
    }
    std::vector<std::size_t> out(selected.begin(), selected.end());
    std::sort(out.begin(), out.end());
    return out;
}

static void collect_top_effective_hand_count_window(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    std::unordered_map<std::string, int>& min_cost_memo,
    int remaining_cost_budget,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    std::unordered_map<std::string, double>& max_score_memo
) {
    const auto first = first_nonzero_index(state);
    if (first == state.size()) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (first >= groups_by_first.size()) return;
    if (top.size() >= max_results) {
        const double possible = score + max_suffix_score(
            state, groups_by_first, group_scores, max_score_memo
        );
        if (possible < top.back().score - 1e-12) return;
    }
    State next;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_group(state, group, next)) continue;
        const auto group_id = group[0];
        if (group_id >= group_costs.size() || group_id >= group_scores.size()) continue;
        const int suffix_min = min_effective_hand_count(
            next, groups_by_first, group_costs, min_cost_memo
        );
        if (group_costs[group_id] + suffix_min > remaining_cost_budget) continue;
        chosen.push_back(group_id);
        collect_top_effective_hand_count_window(
            next,
            groups_by_first,
            group_scores,
            tie_keys,
            group_costs,
            min_cost_memo,
            remaining_cost_budget - group_costs[group_id],
            max_results,
            score + group_scores[group_id],
            chosen,
            top,
            max_score_memo
        );
        chosen.pop_back();
    }
}

static void collect_top_effective_hand_count_window_packed(
    std::uint64_t state,
    std::size_t state_size,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    PackedDepthMemo& min_cost_memo,
    int remaining_cost_budget,
    std::size_t max_results,
    double score,
    Cover& chosen,
    std::vector<Candidate>& top,
    PackedScoreMemo& max_score_memo
) {
    const auto first = first_nonzero_index_packed(state, state_size);
    if (first == state_size) {
        insert_top(top, Candidate{score, chosen}, max_results, tie_keys);
        return;
    }
    if (first >= groups_by_first.size()) return;
    if (top.size() >= max_results) {
        const double possible = score + max_suffix_score_effective_precompiled_packed(
            state, state_size, groups_by_first, group_scores, max_score_memo
        );
        if (possible < top.back().score - 1e-12) return;
    }
    constexpr std::uint64_t LOW_BITS = 0x5555555555555555ULL;
    const auto available_one = (state | (state >> 1)) & LOW_BITS;
    const auto available_two = (state >> 1) & LOW_BITS;
    const auto available_three = (state & (state >> 1)) & LOW_BITS;
    std::uint64_t next = 0;
    for (const auto& group : groups_by_first[first]) {
        if (!subtract_precompiled_group(
                state, available_one, available_two, available_three, group, next)) {
            continue;
        }
        const auto group_id = group.group_id;
        if (group_id >= group_costs.size() || group_id >= group_scores.size()) continue;
        const int suffix_min = min_effective_hand_count_packed(
            next, state_size, groups_by_first, group_costs, min_cost_memo
        );
        if (group_costs[group_id] + suffix_min > remaining_cost_budget) continue;
        chosen.push_back(group_id);
        collect_top_effective_hand_count_window_packed(
            next,
            state_size,
            groups_by_first,
            group_scores,
            tie_keys,
            group_costs,
            min_cost_memo,
            remaining_cost_budget - group_costs[group_id],
            max_results,
            score + group_scores[group_id],
            chosen,
            top,
            max_score_memo
        );
        chosen.pop_back();
    }
}

static Covers top_covers_effective_hand_count_window_impl_with_memo(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results,
    std::unordered_map<std::string, int>& min_cost_memo,
    std::unordered_map<std::string, double>& max_score_memo
) {
    constexpr int INF = 1 << 28;
    const int minimum = min_effective_hand_count(
        state, groups_by_first, group_costs, min_cost_memo
    );
    if (minimum >= INF) return Covers{};
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(std::accumulate(state.begin(), state.end(), std::size_t{0}));
    collect_top_effective_hand_count_window(
        state,
        groups_by_first,
        group_scores,
        tie_keys,
        group_costs,
        min_cost_memo,
        minimum + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo
    );
    Covers out;
    out.reserve(top.size());
    for (auto& candidate : top) out.push_back(std::move(candidate.cover));
    return out;
}

static Covers top_covers_effective_hand_count_window_impl_with_packed_memo(
    const State& state,
    const PackedBuckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results,
    PackedDepthMemo& min_cost_memo,
    PackedScoreMemo& max_score_memo
) {
    constexpr int INF = 1 << 28;
    const auto packed_state = packed_state_key_2bit(state);
    const auto state_size = state.size();
    const int minimum = min_effective_hand_count_packed(
        packed_state, state_size, groups_by_first, group_costs, min_cost_memo
    );
    if (minimum >= INF) return Covers{};
    std::vector<Candidate> top;
    top.reserve(max_results);
    Cover chosen;
    chosen.reserve(std::accumulate(state.begin(), state.end(), std::size_t{0}));
    collect_top_effective_hand_count_window_packed(
        packed_state,
        state_size,
        groups_by_first,
        group_scores,
        tie_keys,
        group_costs,
        min_cost_memo,
        minimum + std::max(0, window),
        max_results,
        0.0,
        chosen,
        top,
        max_score_memo
    );
    Covers out;
    out.reserve(top.size());
    for (auto& candidate : top) out.push_back(std::move(candidate.cover));
    return out;
}

Covers top_covers_effective_hand_count_window(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results
) {
    std::unordered_map<std::string, int> min_cost_memo;
    std::unordered_map<std::string, double> max_score_memo;
    min_cost_memo.reserve(4096);
    max_score_memo.reserve(4096);
    py::gil_scoped_release release;
    return top_covers_effective_hand_count_window_impl_with_memo(
        state, groups_by_first, group_scores, tie_keys, group_costs, window,
        max_results, min_cost_memo, max_score_memo
    );
}

std::vector<Covers> top_covers_effective_hand_count_window_batch(
    const std::vector<State>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results
) {
    std::vector<Covers> results;
    results.reserve(states.size());
    std::unordered_map<std::string, int> min_cost_memo;
    std::unordered_map<std::string, double> max_score_memo;
    const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
    min_cost_memo.reserve(memo_capacity);
    max_score_memo.reserve(memo_capacity);
    py::gil_scoped_release release;
    for (const auto& state : states) {
        results.push_back(top_covers_effective_hand_count_window_impl_with_memo(
            state, groups_by_first, group_scores, tie_keys, group_costs, window,
            max_results, min_cost_memo, max_score_memo
        ));
    }
    return results;
}

Covers top_covers_effective_hand_count_window_selected(
    const State& state,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_selection_priorities,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results,
    std::size_t selected_results
) {
    std::unordered_map<std::string, int> min_cost_memo;
    std::unordered_map<std::string, double> max_score_memo;
    min_cost_memo.reserve(4096);
    max_score_memo.reserve(4096);
    py::gil_scoped_release release;
    return select_top_covers_exact(
        top_covers_effective_hand_count_window_impl_with_memo(
            state, groups_by_first, group_scores, tie_keys, group_costs, window,
            max_results, min_cost_memo, max_score_memo
        ),
        group_scores,
        tie_keys,
        selected_results
    );
}

std::vector<Covers> top_covers_effective_hand_count_window_selected_batch_packed(
    const std::vector<State>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_selection_priorities,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results,
    std::size_t selected_results
) {
    const auto state_size = states.empty() ? 0 : states.front().size();
    PackedBuckets packed_buckets;
    const bool packable =
        !states.empty() &&
        std::all_of(states.begin(), states.end(), [&](const State& state) {
            return state.size() == state_size && can_pack_state_2bit(state);
        }) &&
        compile_packed_buckets(groups_by_first, state_size, packed_buckets);
    if (!packable) {
        auto results = top_covers_effective_hand_count_window_batch(
            states, groups_by_first, group_scores, tie_keys, group_costs, window, max_results
        );
        for (auto& covers : results) {
            covers = select_top_covers_exact(
                std::move(covers), group_scores, tie_keys, selected_results
            );
        }
        return results;
    }

    std::vector<Covers> results;
    results.reserve(states.size());
    PackedDepthMemo min_cost_memo;
    PackedScoreMemo max_score_memo;
    const auto memo_capacity = std::max<std::size_t>(4096, states.size() * 256);
    min_cost_memo.reserve(memo_capacity);
    max_score_memo.reserve(memo_capacity);
    py::gil_scoped_release release;
    for (const auto& state : states) {
        results.push_back(select_top_covers_exact(
            top_covers_effective_hand_count_window_impl_with_packed_memo(
                state,
                packed_buckets,
                group_scores,
                tie_keys,
                group_costs,
                window,
                max_results,
                min_cost_memo,
                max_score_memo
            ),
            group_scores,
            tie_keys,
            selected_results
        ));
    }
    return results;
}

std::vector<Covers> top_covers_effective_hand_count_window_selected_batch(
    const std::vector<State>& states,
    const Buckets& groups_by_first,
    const std::vector<double>& group_scores,
    const std::vector<double>& group_selection_priorities,
    const std::vector<std::string>& tie_keys,
    const std::vector<int>& group_costs,
    int window,
    std::size_t max_results,
    std::size_t selected_results
) {
    if (native_packed_effective_window_batch_enabled()) {
        return top_covers_effective_hand_count_window_selected_batch_packed(
            states,
            groups_by_first,
            group_scores,
            group_selection_priorities,
            tie_keys,
            group_costs,
            window,
            max_results,
            selected_results
        );
    }
    auto results = top_covers_effective_hand_count_window_batch(
        states, groups_by_first, group_scores, tie_keys, group_costs, window, max_results
    );
    for (auto& covers : results) {
        covers = select_top_covers_exact(
            std::move(covers), group_scores, tie_keys, selected_results
        );
    }
    return results;
}

PYBIND11_MODULE(danrl_cover, m) {
    m.attr("packed_effective_window_flat_score_memo_supported") = py::bool_(true);
    m.attr("packed_effective_window_flat_depth_memo_supported") = py::bool_(true);
    m.def(
        "optimal_effective_group_ids",
        &optimal_effective_group_ids,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_costs")
    );
    m.def("count_covers", &count_covers, py::arg("state"), py::arg("groups_by_first"));
    m.def("enumerate_covers", &enumerate_covers, py::arg("state"), py::arg("groups_by_first"));
    m.def(
        "top_covers",
        &top_covers,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("max_results"),
        py::arg("enable_upper_bound") = false
    );
    m.def(
        "top_covers_batch",
        &top_covers_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("max_results"),
        py::arg("enable_upper_bound") = false
    );
    m.def(
        "top_covers_selected_batch",
        &top_covers_selected_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_priorities"),
        py::arg("tie_keys"),
        py::arg("max_results"),
        py::arg("selected_results"),
        py::arg("enable_upper_bound") = false
    );
    m.def(
        "top_covers_selected_batch_capsule",
        [](const std::vector<State>& states,
           const py::capsule& groups_by_first,
           const std::vector<double>& group_scores,
           const std::vector<double>& group_priorities,
           const std::vector<std::string>& tie_keys,
           std::size_t max_results,
           std::size_t selected_results,
           bool enable_upper_bound) {
            return top_covers_selected_batch(
                states,
                native_buckets_from_capsule(groups_by_first),
                group_scores,
                group_priorities,
                tie_keys,
                max_results,
                selected_results,
                enable_upper_bound
            );
        },
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_priorities"),
        py::arg("tie_keys"),
        py::arg("max_results"),
        py::arg("selected_results"),
        py::arg("enable_upper_bound") = false
    );
    m.def(
        "top_covers_beam_batch",
        &top_covers_beam_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_sizes"),
        py::arg("tie_keys"),
        py::arg("beam_width"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_beam_batch_capsule",
        [](const std::vector<State>& states,
           const py::capsule& groups_by_first,
           const std::vector<double>& group_scores,
           const std::vector<int>& group_sizes,
           const std::vector<std::string>& tie_keys,
           std::size_t beam_width,
           std::size_t max_results) {
            return top_covers_beam_batch(
                states,
                native_buckets_from_capsule(groups_by_first),
                group_scores,
                group_sizes,
                tie_keys,
                beam_width,
                max_results
            );
        },
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_sizes"),
        py::arg("tie_keys"),
        py::arg("beam_width"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_effective_hand_count_window",
        &top_covers_effective_hand_count_window,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_effective_hand_count_window_batch",
        &top_covers_effective_hand_count_window_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_effective_hand_count_window_selected",
        &top_covers_effective_hand_count_window_selected,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_selection_priorities"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_effective_hand_count_window_selected_batch_packed",
        &top_covers_effective_hand_count_window_selected_batch_packed,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_selection_priorities"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_effective_hand_count_window_selected_batch",
        &top_covers_effective_hand_count_window_selected_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_selection_priorities"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_effective_hand_count_window_selected_batch_capsule",
        [](const std::vector<State>& states,
           const py::capsule& groups_by_first,
           const std::vector<double>& group_scores,
           const std::vector<double>& group_selection_priorities,
           const std::vector<std::string>& tie_keys,
           const std::vector<int>& group_costs,
           int window,
           std::size_t max_results,
           std::size_t selected_results) {
            return top_covers_effective_hand_count_window_selected_batch(
                states,
                native_buckets_from_capsule(groups_by_first),
                group_scores,
                group_selection_priorities,
                tie_keys,
                group_costs,
                window,
                max_results,
                selected_results
            );
        },
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_selection_priorities"),
        py::arg("tie_keys"),
        py::arg("group_costs"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_hand_count_window",
        &top_covers_hand_count_window,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("window"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_hand_count_window_batch",
        &top_covers_hand_count_window_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys"),
        py::arg("window"),
        py::arg("max_results")
    );
    m.def(
        "top_covers_hand_count_window_selected",
        &top_covers_hand_count_window_selected,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_priorities"),
        py::arg("tie_keys"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_hand_count_window_selected_batch",
        &top_covers_hand_count_window_selected_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_priorities"),
        py::arg("tie_keys"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "top_covers_hand_count_window_selected_batch_capsule",
        [](const std::vector<State>& states,
           const py::capsule& groups_by_first,
           const std::vector<double>& group_scores,
           const std::vector<double>& group_priorities,
           const std::vector<std::string>& tie_keys,
           int window,
           std::size_t max_results,
           std::size_t selected_results) {
            return top_covers_hand_count_window_selected_batch(
                states,
                native_buckets_from_capsule(groups_by_first),
                group_scores,
                group_priorities,
                tie_keys,
                window,
                max_results,
                selected_results
            );
        },
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("group_priorities"),
        py::arg("tie_keys"),
        py::arg("window"),
        py::arg("max_results"),
        py::arg("selected_results")
    );
    m.def(
        "best_selected_cover_by_score_entries",
        &best_selected_cover_by_score_entries,
        py::arg("covers"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values")
    );
    m.def(
        "best_selected_covers_by_score_entries_batch",
        &best_selected_covers_by_score_entries_batch,
        py::arg("cover_batches"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values_by_batch")
    );
    m.def(
        "best_cover_by_group_scores",
        &best_cover_by_group_scores,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_scores"),
        py::arg("tie_keys")
    );
    m.def(
        "best_cover_by_score_entries",
        &best_cover_by_score_entries,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values")
    );
    m.def(
        "best_cover_by_score_entries_with_retake",
        &best_cover_by_score_entries_with_retake,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values")
    );
    m.def(
        "best_covers_by_score_entries_with_retake_batch",
        &best_covers_by_score_entries_with_retake_batch,
        py::arg("states"),
        py::arg("groups_by_first"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values_by_state")
    );
    m.def(
        "best_cover_by_score_entries_dp",
        &best_cover_by_score_entries_dp,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values"),
        py::arg("frontier_limit") = 200000
    );
    m.def(
        "best_cover_by_score_entries_dp_with_retake",
        &best_cover_by_score_entries_dp_with_retake,
        py::arg("state"),
        py::arg("groups_by_first"),
        py::arg("group_entries"),
        py::arg("weights"),
        py::arg("pressure_values"),
        py::arg("frontier_limit") = 200000
    );
}
