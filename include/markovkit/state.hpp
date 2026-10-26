#pragma once

#include "core/types.hpp"

namespace markovkit {

/// Molecule counts, or signed stoichiometric changes, with one entry per coordinate.
using state = num::array<int>;

/// Hash molecule counts for the CME state-to-index lookup.
struct state_hash {
    std::size_t operator()(const state &x) const noexcept {
        std::size_t seed = x.size();
        for (int count : x)
            seed ^= static_cast<std::size_t>(count) + 0x9e3779b97f4a7c15ULL + (seed << 6U) +
                    (seed >> 2U);
        return seed;
    }
};

} // namespace markovkit
