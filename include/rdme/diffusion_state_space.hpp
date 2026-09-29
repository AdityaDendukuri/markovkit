/// @file rdme/diffusion_state_space.hpp
/// @brief Enumeration of fixed-population RDME diffusion sectors.
#pragma once

#include "core/types.hpp"
#include "rdme/configuration.hpp"
#include "rdme/state_layout.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rdme {

/// Every voxel configuration with prescribed total count for each species.
///
/// Diffusion never leaves this state space. With V voxels and totals k_s, its
/// size is the product of binomial(k_s + V - 1, V - 1).
class diffusion_state_space {
  public:
    diffusion_state_space(std::size_t voxel_count, num::array<int> species_totals)
        : layout_(voxel_count, species_totals.size()), species_totals_(std::move(species_totals)) {
        if (std::any_of(species_totals_.begin(), species_totals_.end(),
                        [](int total) { return total < 0; })) {
            throw std::invalid_argument("RDME sector totals must be nonnegative");
        }

        const std::size_t expected = expected_size(voxel_count, species_totals_);
        states_.reserve(expected);

        configuration state(layout_.size(), 0);
        enumerate_species(0, state);
        if (states_.size() != expected) {
            throw std::logic_error("RDME sector enumeration produced an inconsistent size");
        }
    }

    [[nodiscard]] const state_layout &layout() const noexcept { return layout_; }
    [[nodiscard]] std::size_t size() const noexcept { return states_.size(); }
    [[nodiscard]] bool empty() const noexcept { return states_.empty(); }
    [[nodiscard]] const num::array<int> &species_totals() const noexcept { return species_totals_; }
    [[nodiscard]] const num::array<configuration> &states() const noexcept { return states_; }
    [[nodiscard]] const configuration &operator[](std::size_t index) const {
        return states_.at(index);
    }

    [[nodiscard]] num::idx find(const configuration &state) const {
        return rank_configuration(layout_, species_totals_, state);
    }

    /// Rank a configuration in the same mixed-radix order used by enumeration.
    /// No hash table or allocation is required.
    [[nodiscard]] num::idx rank(const configuration &state) const { return find(state); }

    /// Recover a configuration from its rank without consulting the stored state array.
    [[nodiscard]] configuration unrank(num::idx index) const {
        return unrank_configuration(layout_, species_totals_, index);
    }

    [[nodiscard]] static num::idx rank_configuration(const state_layout &layout,
                                                     num::view<const int> species_totals,
                                                     const configuration &state) {
        if (species_totals.size() != layout.species_count() || state.size() != layout.size()) {
            return npos;
        }

        num::idx rank = 0;
        for (std::size_t species = 0; species < layout.species_count(); ++species) {
            int remaining = species_totals[species];
            if (remaining < 0)
                return npos;
            num::idx composition_rank = 0;
            for (std::size_t voxel = 0; voxel + 1 < layout.voxel_count(); ++voxel) {
                const int count = state[layout.index(voxel, species)];
                if (count < 0 || count > remaining)
                    return npos;
                const std::size_t remaining_voxels = layout.voxel_count() - voxel - 1;
                composition_rank +=
                    binomial(static_cast<std::size_t>(remaining) + remaining_voxels,
                             remaining_voxels) -
                    binomial(static_cast<std::size_t>(remaining - count) + remaining_voxels,
                             remaining_voxels);
                remaining -= count;
            }
            if (state[layout.index(layout.voxel_count() - 1, species)] != remaining)
                return npos;
            const num::idx radix = composition_count(layout.voxel_count(), species_totals[species]);
            rank = rank * radix + composition_rank;
        }
        return rank;
    }

    [[nodiscard]] static configuration unrank_configuration(const state_layout &layout,
                                                            num::view<const int> species_totals,
                                                            num::idx index) {
        const num::idx size = expected_size(layout.voxel_count(), species_totals);
        if (species_totals.size() != layout.species_count() || index >= size) {
            throw std::out_of_range("RDME configuration rank is outside its sector");
        }

        num::array<num::idx> composition_ranks(layout.species_count(), 0);
        for (std::size_t species = layout.species_count(); species-- > 0;) {
            const num::idx radix = composition_count(layout.voxel_count(), species_totals[species]);
            composition_ranks[species] = index % radix;
            index /= radix;
        }

        configuration state(layout.size(), 0);
        for (std::size_t species = 0; species < layout.species_count(); ++species) {
            int remaining = species_totals[species];
            num::idx rank = composition_ranks[species];
            for (std::size_t voxel = 0; voxel + 1 < layout.voxel_count(); ++voxel) {
                const std::size_t remaining_voxels = layout.voxel_count() - voxel - 1;
                int count = 0;
                for (; count <= remaining; ++count) {
                    const num::idx completions =
                        composition_count(remaining_voxels, remaining - count);
                    if (rank < completions)
                        break;
                    rank -= completions;
                }
                state[layout.index(voxel, species)] = count;
                remaining -= count;
            }
            state[layout.index(layout.voxel_count() - 1, species)] = remaining;
        }
        return state;
    }

    [[nodiscard]] static std::size_t expected_size(std::size_t voxel_count,
                                                   num::view<const int> species_totals) {
        if (voxel_count == 0 || species_totals.empty()) {
            throw std::invalid_argument("RDME sectors need at least one voxel and species");
        }
        std::size_t result = 1;
        for (int total : species_totals) {
            if (total < 0) {
                throw std::invalid_argument("RDME sector totals must be nonnegative");
            }
            const std::size_t factor =
                binomial(static_cast<std::size_t>(total) + voxel_count - 1, voxel_count - 1);
            if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor) {
                throw std::overflow_error("RDME sector size overflows size_t");
            }
            result *= factor;
        }
        return result;
    }

    static constexpr num::idx npos = std::numeric_limits<num::idx>::max();

  private:
    [[nodiscard]] static std::size_t composition_count(std::size_t voxel_count, int total) {
        return binomial(static_cast<std::size_t>(total) + voxel_count - 1, voxel_count - 1);
    }

    [[nodiscard]] static std::size_t binomial(std::size_t n, std::size_t k) {
        k = std::min(k, n - k);
        std::size_t result = 1;
        for (std::size_t i = 1; i <= k; ++i) {
            const std::size_t numerator = n - k + i;
            if (result > std::numeric_limits<std::size_t>::max() / numerator) {
                throw std::overflow_error("RDME binomial coefficient overflows size_t");
            }
            result = (result * numerator) / i;
        }
        return result;
    }

    void enumerate_species(std::size_t species, configuration &state) {
        if (species == layout_.species_count()) {
            states_.push_back(state);
            return;
        }
        enumerate_compositions(species, 0, species_totals_[species], state);
    }

    void enumerate_compositions(std::size_t species, std::size_t voxel, int remaining,
                                configuration &state) {
        if (voxel + 1 == layout_.voxel_count()) {
            state[layout_.index(voxel, species)] = remaining;
            enumerate_species(species + 1, state);
            return;
        }
        for (int count = 0; count <= remaining; ++count) {
            state[layout_.index(voxel, species)] = count;
            enumerate_compositions(species, voxel + 1, remaining - count, state);
        }
    }

    state_layout layout_;
    num::array<int> species_totals_;
    num::array<configuration> states_;
};

} // namespace rdme
