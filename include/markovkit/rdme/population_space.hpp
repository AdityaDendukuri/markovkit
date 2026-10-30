// RDME state spaces of fixed species totals, numbered arithmetically instead of by hashing
#pragma once

#include "markovkit/chain.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace markovkit {

namespace detail {

inline idx binomial(idx n, idx k) {
    k = std::min(k, n - k);
    idx result = 1;
    for (idx i = 1; i <= k; ++i) {
        if (result > std::numeric_limits<idx>::max() / (n - k + i)) {
            throw std::overflow_error("population state space is too large");
        }
        result = result * (n - k + i) / i;
    }
    return result;
}

// the number of ways to put `total` molecules in `voxels` voxels
inline idx compositions(idx voxels, int total) {
    return binomial(static_cast<idx>(total) + voxels - 1, voxels - 1);
}

// append every state with these totals, in rank order: species 0 is the most significant
// digit, and within a species the count in voxel 0 is the most significant
inline void enumerate(idx voxels, const array<int> &totals, idx species, idx voxel, int remaining,
                      state &x, array<state> &out) {
    const idx S = totals.size();
    if (species == S) {
        out.push_back(x);
        return;
    }
    const idx at = (voxel * S) + species;
    if (voxel + 1 == voxels) {
        x[at] = remaining;
        enumerate(voxels, totals, species + 1, 0, species + 1 < S ? totals[species + 1] : 0, x,
                  out);
        return;
    }
    for (int count = 0; count <= remaining; ++count) {
        x[at] = count;
        enumerate(voxels, totals, species, voxel + 1, remaining - count, x, out);
    }
}

inline void enumerate(idx voxels, const array<int> &totals, array<state> &out) {
    if (voxels == 0 || totals.empty()) {
        throw std::invalid_argument("population state space needs a voxel and a species");
    }
    idx size = 1;
    for (int total : totals) {
        if (total < 0) {
            throw std::invalid_argument("species totals must be nonnegative");
        }
        const idx factor = compositions(voxels, total);
        if (size > std::numeric_limits<idx>::max() / factor) {
            throw std::overflow_error("population state space is too large");
        }
        size *= factor;
    }
    out.reserve(out.size() + size);
    state x(voxels * totals.size(), 0);
    enumerate(voxels, totals, 0, 0, totals[0], x, out);
}

// the position of x among the states with these totals, or nothing if its totals differ
inline std::optional<idx> rank(idx voxels, const array<int> &totals, const state &x) {
    const idx S = totals.size();
    if (x.size() != voxels * S) {
        return std::nullopt;
    }
    idx result = 0;
    for (idx s = 0; s < S; ++s) {
        int remaining = totals[s];
        idx local = 0;
        for (idx v = 0; v + 1 < voxels; ++v) {
            const int count = x[(v * S) + s];
            if (count < 0 || count > remaining) {
                return std::nullopt;
            }
            // skip the states with fewer molecules in voxel v
            const idx rest = voxels - v - 1;
            local += binomial(remaining + rest, rest) - binomial(remaining - count + rest, rest);
            remaining -= count;
        }
        if (x[((voxels - 1) * S) + s] != remaining) {
            return std::nullopt;
        }
        result = (result * compositions(voxels, totals[s])) + local;
    }
    return result;
}

} // namespace detail

// every state with totals[s] molecules of species s; diffusion never leaves it
class population_sector {
  public:
    population_sector(idx n_voxels, array<int> totals)
        : voxels_(n_voxels), totals_(std::move(totals)) {
        detail::enumerate(voxels_, totals_, states_);
    }

    [[nodiscard]] idx size() const { return states_.size(); }
    [[nodiscard]] const state &operator[](idx i) const { return states_[i]; }
    [[nodiscard]] std::optional<idx> find(const state &x) const {
        return detail::rank(voxels_, totals_, x);
    }

    [[nodiscard]] idx n_voxels() const { return voxels_; }
    [[nodiscard]] idx n_species() const { return totals_.size(); }
    [[nodiscard]] const array<int> &totals() const { return totals_; }
    [[nodiscard]] const array<state> &states() const { return states_; }

  private:
    idx voxels_;
    array<int> totals_;
    array<state> states_;
};

// every sector with 0 <= totals[s] <= max_totals[s], stored one after another; a reaction
// that changes a total by one couples neighboring blocks of the generator
class population_box {
  public:
    population_box(idx n_voxels, array<int> max_totals)
        : voxels_(n_voxels), max_totals_(std::move(max_totals)) {
        begin_.push_back(0);
        array<int> totals(max_totals_.size(), 0);
        add_sectors(0, totals);
    }

    [[nodiscard]] idx size() const { return states_.size(); }
    [[nodiscard]] const state &operator[](idx i) const { return states_[i]; }
    [[nodiscard]] std::optional<idx> find(const state &x) const {
        array<int> totals(n_species(), 0);
        for (idx i = 0; i < x.size(); ++i) {
            totals[i % n_species()] += x[i];
        }
        const auto k = find_sector(totals);
        const auto local = k ? detail::rank(voxels_, totals, x) : std::nullopt;
        return local ? std::optional<idx>(begin_[*k] + *local) : std::nullopt;
    }

    [[nodiscard]] idx n_voxels() const { return voxels_; }
    [[nodiscard]] idx n_species() const { return max_totals_.size(); }
    [[nodiscard]] idx n_sectors() const { return totals_.size(); }

    // sector k holds the states sector_begin(k), ..., sector_end(k) - 1
    [[nodiscard]] const array<int> &sector_totals(idx k) const { return totals_[k]; }
    [[nodiscard]] idx sector_begin(idx k) const { return begin_[k]; }
    [[nodiscard]] idx sector_end(idx k) const { return begin_[k + 1]; }

    [[nodiscard]] idx sector_of(idx i) const {
        return static_cast<idx>(std::upper_bound(begin_.begin(), begin_.end(), i) -
                                begin_.begin() - 1);
    }

    [[nodiscard]] std::optional<idx> find_sector(const array<int> &totals) const {
        idx k = 0;
        for (idx s = 0; s < n_species(); ++s) {
            if (totals[s] < 0 || totals[s] > max_totals_[s]) {
                return std::nullopt;
            }
            k = (k * (max_totals_[s] + 1)) + totals[s];
        }
        return k;
    }

  private:
    void add_sectors(idx species, array<int> &totals) {
        if (species == n_species()) {
            totals_.push_back(totals);
            detail::enumerate(voxels_, totals, states_);
            begin_.push_back(states_.size());
            return;
        }
        for (int total = 0; total <= max_totals_[species]; ++total) {
            totals[species] = total;
            add_sectors(species + 1, totals);
        }
    }

    idx voxels_;
    array<int> max_totals_;
    array<array<int>> totals_;
    array<idx> begin_;
    array<state> states_;
};

} // namespace markovkit
