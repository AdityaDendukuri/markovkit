/// @file rdme/state_layout.hpp
/// @brief Mapping between voxel/species coordinates and a flat RDME state.
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rdme {

/// Voxel-major layout: `(voxel, species)` maps to `voxel * species_count + species`.
class state_layout {
  public:
    state_layout(std::size_t voxel_count, std::size_t species_count)
        : voxel_count_(voxel_count), species_count_(species_count) {
        if (voxel_count == 0 || species_count == 0) {
            throw std::invalid_argument("RDME state layout dimensions must be positive");
        }
        if (voxel_count > std::numeric_limits<std::size_t>::max() / species_count) {
            throw std::overflow_error("RDME state layout size overflows size_t");
        }
    }

    [[nodiscard]] std::size_t voxel_count() const noexcept { return voxel_count_; }
    [[nodiscard]] std::size_t species_count() const noexcept { return species_count_; }
    [[nodiscard]] std::size_t size() const noexcept { return voxel_count_ * species_count_; }

    [[nodiscard]] std::size_t index(std::size_t voxel, std::size_t species) const {
        if (voxel >= voxel_count_ || species >= species_count_) {
            throw std::out_of_range("RDME voxel/species coordinate is outside the state layout");
        }
        return (voxel * species_count_) + species;
    }

    template <class State>
    [[nodiscard]] int count(const State &state, std::size_t voxel, std::size_t species) const {
        validate(state);
        return state[index(voxel, species)];
    }

    template <class State>
    void validate(const State &state) const {
        if (state.size() != size()) {
            throw std::invalid_argument("RDME state size does not match its layout");
        }
    }

  private:
    std::size_t voxel_count_;
    std::size_t species_count_;
};

} // namespace rdme
