// a finite subnetwork D and its truncated rate matrix R
#pragma once

#include "linear/sparse/sparse.hpp"
#include "stats/selection.hpp"
#include "subsweep/chain.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace subsweep {

template <typename State>
class subnetwork;

template <typename State, detail::oracle<State> Rates>
void update_subnetwork(subnetwork<State> &D, array<State> next, const Rates &rates);

// D and R; a retained state keeps its row and its entries of R
template <typename State>
class subnetwork {
  public:
    [[nodiscard]] idx size() const {
        require_compact_rows();
        return states_.size();
    }
    [[nodiscard]] const State &state(idx i) const { return states_[i]; }
    [[nodiscard]] view<const State> states() const {
        require_compact_rows();
        return states_;
    }

    [[nodiscard]] std::optional<idx> find(const State &x) const {
        auto found = row_of_.find(x);
        return found == row_of_.end() ? std::nullopt : std::optional<idx>(found->second);
    }

    // R, with R_ij = -rate(i -> j) off the diagonal and R_ii the total rate out of i
    [[nodiscard]] const num::spmat &matrix() const;

    // The same R written directly into dense storage.
    [[nodiscard]] num::mat<real> dense_matrix() const;

    // R_ii and (R 1)_i, read directly from the retained rows.
    [[nodiscard]] view<const real> total_rates() const {
        require_compact_rows();
        return diagonal_;
    }
    [[nodiscard]] num::vec<real> exit_rates() const {
        require_compact_rows();
        num::vec<real> result(size(), 0.0);
        for (idx i = 0; i < size(); ++i) {
            result[i] = diagonal_[i];
            for (auto [j, value] : row_[i]) {
                result[i] += value;
            }
            result[i] = std::max(real{0.0}, result[i]);
        }
        return result;
    }

    // the level of each row, when the rate oracle has levels
    [[nodiscard]] view<const idx> levels() const { return level_; }

    // the rows whose state changed in the last update
    [[nodiscard]] const array<idx> &changed() const {
        require_compact_rows();
        return changed_;
    }
    // whether row i holds a state that entered D in the last update
    [[nodiscard]] bool entered(idx i) const { return entered_[i]; }

  private:
    template <typename S, detail::oracle<S> Rates>
    friend void update_subnetwork(subnetwork<S> &D, array<S> next, const Rates &rates);

    // The steps of an update: empty rows, assign the new states, compact, and fill their rates.
    template <detail::oracle<State> Rates>
    void insert(State x, const Rates &rates);
    template <detail::oracle<State> Rates>
    void populate(idx i, const Rates &rates);
    void remove(idx i);
    void compact_rows();
    void begin_update() {
        entered_.assign(states_.size(), 0);
        changed_row_.assign(states_.size(), 0);
    }
    void finish_update() {
        changed_ = num::filter(states_.size(), [&](idx i) { return changed_row_[i]; });
        changed_row_.clear();
    }

    void require_compact_rows() const {
        if (!empty_rows_.empty()) {
            throw std::logic_error("during an update a rate oracle may only call find");
        }
    }
    // R_ij += value for i != j
    void accumulate(idx i, idx j, real value);
    // Rename row `from` to `to` in every entry that refers to it.
    void move_row(idx from, idx to);

    array<State> states_;
    unordered_map<State, idx> row_of_;
    // (j, R_ij) of each row, the rows with an entry in each column, and R_ii
    array<array<std::pair<idx, real>>> row_;
    array<array<idx>> column_;
    array<real> diagonal_;
    array<idx> level_;
    array<unsigned char> entered_;
    array<unsigned char> changed_row_;
    array<idx> changed_;
    // The empty rows, lowest first.
    num::set<idx> empty_rows_;
    mutable std::optional<num::spmat> R_;
};

template <typename State>
template <detail::oracle<State> Rates>
void subnetwork<State>::insert(State x, const Rates &rates) {
    if (find(x)) {
        throw std::invalid_argument("the state is already in the subnetwork");
    }
    // The lowest empty row, or a new one.
    idx i = states_.size();
    if (!empty_rows_.empty()) {
        i = *empty_rows_.begin();
        empty_rows_.erase(empty_rows_.begin());
    } else {
        states_.emplace_back();
        row_.emplace_back();
        column_.emplace_back();
        diagonal_.push_back(0.0);
        entered_.push_back(0);
        changed_row_.push_back(0);
        if constexpr (leveled_oracle<Rates, State>) {
            level_.push_back(0);
        }
    }
    row_of_[x] = i;
    states_[i] = std::move(x);
    entered_[i] = 1;
    changed_row_[i] = 1;
    if constexpr (leveled_oracle<Rates, State>) {
        level_[i] = static_cast<idx>(rates.level(states_[i]));
    }
    R_.reset();
}

template <typename State>
template <detail::oracle<State> Rates>
void subnetwork<State>::populate(idx i, const Rates &rates) {
    auto entries = [&] {
        if constexpr (rate_oracle<Rates, State>) {
            return rates.add_to_subnetwork(states_[i], *this);
        } else {
            return std::tuple{rates.total_rate(states_[i]), rates.outgoing(states_[i], *this),
                              rates.incoming(states_[i], *this)};
        }
    }();
    auto &[total, outgoing, incoming] = entries;
    if (!(total >= 0.0)) {
        throw std::invalid_argument("total rates must be nonnegative");
    }
    diagonal_[i] = total;

    // a transition between x and D[j] must change the level by at most one
    auto check_level = [&](idx j) {
        if constexpr (leveled_oracle<Rates, State>) {
            idx a = level_[i], b = level_[j];
            if ((a > b ? a - b : b - a) > 1) {
                throw std::invalid_argument("a transition changes the level by more than one");
            }
        }
    };
    auto check_rate = [&](idx j, real rate) {
        if (j >= states_.size() || empty_rows_.contains(j)) {
            throw std::invalid_argument("the rate oracle returned a row outside the subnetwork");
        }
        if (!(rate >= 0.0)) {
            throw std::invalid_argument("transition rates must be nonnegative");
        }
        if (j == i) {
            throw std::invalid_argument("a transition must change the state");
        }
        check_level(j);
    };

    // row i: the transitions x -> D[j]
    real inside = 0.0;
    for (auto [j, rate] : outgoing) {
        check_rate(j, rate);
        inside += rate;
        // For two entering states, the lower row supplies both directions once.
        bool add = !entered(j) || i < j;
        if (rate > 0.0 && add) {
            accumulate(i, j, -rate);
        }
    }
    if (inside > total + 1e-12 * std::max(real{1.0}, total)) {
        throw std::invalid_argument("the rates into the subnetwork exceed the total rate");
    }
    // column i: the transitions D[j] -> x
    for (auto [j, rate] : incoming) {
        check_rate(j, rate);
        bool add = !entered(j) || i < j;
        if (rate > 0.0 && add) {
            accumulate(j, i, -rate);
        }
    }
}

template <typename State>
void subnetwork<State>::remove(idx i) {
    R_.reset();
    for (auto [j, value] : row_[i]) {
        std::erase(column_[j], i);
    }
    for (idx r : column_[i]) {
        std::erase_if(row_[r], [&](const auto &entry) { return entry.first == i; });
    }
    row_[i].clear();
    column_[i].clear();
    row_of_.erase(states_[i]);
    empty_rows_.insert(i);
}

template <typename State>
void subnetwork<State>::compact_rows() {
    // States past the new size fill empty rows below it, both in ascending order.
    idx n = states_.size() - empty_rows_.size();
    array<idx> from = num::filter(states_.size(),
                                  [&](idx row) { return row >= n && !empty_rows_.contains(row); });
    array<idx> to = num::filter(n, [&](idx row) { return empty_rows_.contains(row); });
    for (idx k = 0; k < from.size(); ++k) {
        idx a = from[k], b = to[k];
        move_row(a, b);
        states_[b] = std::move(states_[a]);
        row_of_[states_[b]] = b;
        row_[b] = std::move(row_[a]);
        column_[b] = std::move(column_[a]);
        diagonal_[b] = diagonal_[a];
        entered_[b] = entered_[a];
        changed_row_[b] = 1;
        if (!level_.empty()) {
            level_[b] = level_[a];
        }
    }
    states_.resize(n);
    row_.resize(n);
    column_.resize(n);
    diagonal_.resize(n);
    entered_.resize(n);
    changed_row_.resize(n);
    if (!level_.empty()) {
        level_.resize(n);
    }
    empty_rows_.clear();
    R_.reset();
}

template <typename State>
const num::spmat &subnetwork<State>::matrix() const {
    require_compact_rows();
    if (!R_) {
        array<idx> rows, columns;
        array<real> values;
        for (idx i = 0; i < states_.size(); ++i) {
            rows.push_back(i);
            columns.push_back(i);
            values.push_back(diagonal_[i]);
            for (auto [j, value] : row_[i]) {
                rows.push_back(i);
                columns.push_back(j);
                values.push_back(value);
            }
        }
        R_ = num::spmat::from_triplets(states_.size(), states_.size(), rows, columns, values);
    }
    return *R_;
}

template <typename State>
num::mat<real> subnetwork<State>::dense_matrix() const {
    require_compact_rows();
    num::mat<real> R(states_.size(), states_.size(), 0.0);
    for (idx i = 0; i < states_.size(); ++i) {
        R(i, i) = diagonal_[i];
        for (auto [j, value] : row_[i]) {
            R(i, j) = value;
        }
    }
    return R;
}

template <typename State>
void subnetwork<State>::accumulate(idx i, idx j, real value) {
    for (auto &[column, existing] : row_[i]) {
        if (column == j) {
            existing += value;
            return;
        }
    }
    row_[i].push_back({j, value});
    column_[j].push_back(i);
}

template <typename State>
void subnetwork<State>::move_row(idx from, idx to) {
    for (auto [j, value] : row_[from]) {
        std::replace(column_[j].begin(), column_[j].end(), from, to);
    }
    for (idx r : column_[from]) {
        for (auto &entry : row_[r]) {
            if (entry.first == from) {
                entry.first = to;
            }
        }
    }
}

// Make D exactly `next`, keeping each retained state's row when possible.
template <typename State, detail::oracle<State> Rates>
void update_subnetwork(subnetwork<State> &D, array<State> next, const Rates &rates) {
    D.begin_update();
    array<unsigned char> retained(D.states_.size(), 0);
    array<unsigned char> already_present(next.size(), 0);
    for (idx k = 0; k < next.size(); ++k) {
        if (const auto i = D.find(next[k])) {
            retained[*i] = 1;
            already_present[k] = 1;
        }
    }
    for (idx i = 0; i < retained.size(); ++i) {
        if (!retained[i]) {
            D.remove(i);
        }
    }
    for (idx k = 0; k < next.size(); ++k) {
        if (!already_present[k] && !D.find(next[k])) {
            D.insert(std::move(next[k]), rates);
        }
    }
    D.compact_rows();
    D.finish_update();
    if (D.size() == 0) {
        throw std::invalid_argument("a subnetwork must contain at least one state");
    }
    // The oracle sees the complete, compact D while filling each entering row.
    for (idx i = 0; i < D.size(); ++i) {
        if (D.entered(i)) {
            D.populate(i, rates);
        }
    }
}

// D and R for `states`, with every row new
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] subnetwork<State> make_subnetwork(const Rates &rates, array<State> states) {
    subnetwork<State> D;
    update_subnetwork(D, std::move(states), rates);
    return D;
}

} // namespace subsweep
