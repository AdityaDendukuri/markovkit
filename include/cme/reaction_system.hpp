#pragma once

#include "markovkit/state.hpp"
#include <algorithm>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace cme {

using Propensity =
    std::function<double(const markovkit::State &, const std::vector<double> &, double)>;

/// Stoichiometric changes and their state-dependent reaction rates.
struct ReactionSystem {
    // changes[r] and propensities[r] describe the same reaction.
    std::vector<markovkit::State> changes;
    std::vector<Propensity> propensities;

    [[nodiscard]] std::size_t size() const noexcept { return changes.size(); }

    void validate() const {
        if (changes.size() != propensities.size()) {
            throw std::invalid_argument(
                "reaction changes and propensities must have the same size");
        }
    }

    [[nodiscard]] double evaluate_propensities(const markovkit::State &state,
                                               const std::vector<double> &rates, double time,
                                               std::span<double> output) const {
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

    [[nodiscard]] double total_propensity(const markovkit::State &state,
                                          const std::vector<double> &rates, double time) const {
        double total = 0.0;
        for (const auto &propensity : propensities) {
            total += std::max(0.0, propensity(state, rates, time));
        }
        return total;
    }
};

} // namespace cme
