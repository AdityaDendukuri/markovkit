/// @file rdme/configuration.hpp
/// @brief Dynamic molecule-count state for scalable RDME calculations.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <span>
#include <stdexcept>

namespace rdme {

/// Dynamic voxel-major molecule counts, independent of the small fixed-capacity CME state.
class Configuration {
  public:
    Configuration() = default;

    explicit Configuration(std::size_t size, int value = 0) : counts_(size, value) {
        validate_nonnegative();
    }

    Configuration(std::initializer_list<int> counts) : counts_(counts) { validate_nonnegative(); }

    explicit Configuration(std::span<const int> counts) : counts_(counts.begin(), counts.end()) {
        validate_nonnegative();
    }

    [[nodiscard]] std::size_t size() const noexcept { return counts_.size(); }
    [[nodiscard]] bool empty() const noexcept { return counts_.empty(); }
    [[nodiscard]] const int *data() const noexcept { return counts_.data(); }
    [[nodiscard]] int *data() noexcept { return counts_.data(); }
    [[nodiscard]] const int *begin() const noexcept { return counts_.data(); }
    [[nodiscard]] const int *end() const noexcept { return counts_.data() + counts_.size(); }
    [[nodiscard]] int *begin() noexcept { return counts_.data(); }
    [[nodiscard]] int *end() noexcept { return counts_.data() + counts_.size(); }
    [[nodiscard]] int operator[](std::size_t index) const noexcept { return counts_[index]; }
    [[nodiscard]] int &operator[](std::size_t index) noexcept { return counts_[index]; }

    [[nodiscard]] bool operator==(const Configuration &) const noexcept = default;

  private:
    void validate_nonnegative() const {
        if (std::any_of(counts_.begin(), counts_.end(), [](int count) { return count < 0; })) {
            throw std::invalid_argument("RDME molecule counts must be nonnegative");
        }
    }

    num::array<int> counts_;
};

struct ConfigurationHash {
    [[nodiscard]] std::size_t operator()(const Configuration &state) const noexcept {
        std::size_t seed = state.size();
        for (int count : state) {
            const auto value = static_cast<std::size_t>(count);
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

} // namespace rdme
