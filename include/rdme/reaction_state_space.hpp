/// @file rdme/reaction_state_space.hpp
/// @brief Contiguous union of conserved-total sectors for reactive RDMEs.
#pragma once

#include "core/types.hpp"
#include "rdme/configuration.hpp"
#include "rdme/diffusion_state_space.hpp"
#include "rdme/state_layout.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rdme {

/// All per-species total sectors in the box 0 <= k_s <= max_totals[s].
/// States belonging to one sector are contiguous, exposing the block structure.
class reaction_state_space {
  public:
    reaction_state_space(std::size_t voxel_count, num::array<int> max_totals)
        : layout_(voxel_count, max_totals.size()), max_totals_(std::move(max_totals)) {
        if (std::any_of(max_totals_.begin(), max_totals_.end(),
                        [](int total) { return total < 0; })) {
            throw std::invalid_argument("RDME maximum sector totals must be nonnegative");
        }
        sector_offsets_.push_back(0);
        num::array<int> totals(max_totals_.size(), 0);
        enumerate_sector_totals(0, totals);
    }

    [[nodiscard]] const state_layout &layout() const noexcept { return layout_; }
    [[nodiscard]] std::size_t size() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t n_sectors() const noexcept { return sector_totals_.size(); }
    [[nodiscard]] const num::array<int> &max_totals() const noexcept { return max_totals_; }
    [[nodiscard]] const num::array<num::array<int>> &sector_totals() const noexcept {
        return sector_totals_;
    }
    [[nodiscard]] const num::array<num::idx> &sector_offsets() const noexcept {
        return sector_offsets_;
    }
    [[nodiscard]] const num::array<configuration> &states() const noexcept { return states_; }
    [[nodiscard]] const configuration &operator[](num::idx state) const {
        return states_.at(state);
    }
    [[nodiscard]] num::idx sector(num::idx state) const {
        if (state >= size())
            throw std::out_of_range("RDME state index is outside the state space");
        return static_cast<num::idx>(
            std::upper_bound(sector_offsets_.begin(), sector_offsets_.end(), state) -
            sector_offsets_.begin() - 1);
    }

    [[nodiscard]] num::idx find(const configuration &state) const {
        if (state.size() != layout_.size())
            return npos;
        num::array<int> totals(layout_.species_count(), 0);
        for (std::size_t voxel = 0; voxel < layout_.voxel_count(); ++voxel) {
            for (std::size_t species = 0; species < layout_.species_count(); ++species) {
                const int count = state[layout_.index(voxel, species)];
                if (count < 0 || totals[species] > max_totals_[species] - count)
                    return npos;
                totals[species] += count;
            }
        }

        num::idx sector_index = 0;
        for (std::size_t species = 0; species < layout_.species_count(); ++species) {
            sector_index = sector_index * static_cast<num::idx>(max_totals_[species] + 1) +
                           static_cast<num::idx>(totals[species]);
        }
        const num::idx local = diffusion_state_space::rank_configuration(layout_, totals, state);
        return local == diffusion_state_space::npos ? npos : sector_offsets_[sector_index] + local;
    }

    static constexpr num::idx npos = std::numeric_limits<num::idx>::max();

  private:
    void enumerate_sector_totals(std::size_t species, num::array<int> &totals) {
        if (species == max_totals_.size()) {
            append_sector(totals);
            return;
        }
        for (int total = 0; total <= max_totals_[species]; ++total) {
            totals[species] = total;
            enumerate_sector_totals(species + 1, totals);
        }
    }

    void append_sector(const num::array<int> &totals) {
        sector_totals_.push_back(totals);
        diffusion_state_space sector_space(layout_.voxel_count(), totals);
        states_.reserve(states_.size() + sector_space.size());
        for (const configuration &state : sector_space.states()) {
            states_.push_back(state);
        }
        sector_offsets_.push_back(states_.size());
    }

    state_layout layout_;
    num::array<int> max_totals_;
    num::array<num::array<int>> sector_totals_;
    num::array<num::idx> sector_offsets_;
    num::array<configuration> states_;
};

} // namespace rdme
