#pragma once

#include "core/types.hpp"
#include "markovkit/state.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace cme {

using propensity =
    std::function<double(const markovkit::state &, const num::array<double> &, double)>;

/// Stoichiometric changes and their state-dependent reaction rates.
struct reaction_system {
    // changes[r] and propensities[r] describe the same reaction.
    num::array<markovkit::state> changes;
    num::array<propensity> propensities;

    [[nodiscard]] std::size_t size() const noexcept { return changes.size(); }

    void validate() const {
        if (changes.size() != propensities.size()) {
            throw std::invalid_argument(
                "reaction changes and propensities must have the same size");
        }
        for (const auto &change : changes) {
            if (change.size() != changes.front().size())
                throw std::invalid_argument("reaction changes must have the same species count");
        }
    }

    [[nodiscard]] double evaluate_propensities(const markovkit::state &state,
                                               const num::array<double> &rates, double time,
                                               num::view<double> output) const {
        if (output.size() != propensities.size()) {
            throw std::invalid_argument("propensity output size must match the reaction count");
        }
        double total = 0.0;
        for (std::size_t reaction = 0; reaction < propensities.size(); ++reaction) {
            output[reaction] = std::max(0.0, propensities[reaction](state, rates, time));
            total += output[reaction];
        }
        return total;
    }

    [[nodiscard]] double total_propensity(const markovkit::state &state,
                                          const num::array<double> &rates, double time) const {
        double total = 0.0;
        for (const auto &propensity : propensities) {
            total += std::max(0.0, propensity(state, rates, time));
        }
        return total;
    }
};

} // namespace cme
