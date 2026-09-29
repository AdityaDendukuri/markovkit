#pragma once

#include "core/types.hpp"
#include "markovkit/state.hpp"
#include <stdexcept>

namespace markovkit {

/// Compute probability-weighted means for every species.
inline num::vec species_means(num::view<const state> states,
                              num::view<const double> probabilities) {
    if (states.size() != probabilities.size()) {
        throw std::invalid_argument("states and probabilities must have the same size");
    }
    if (states.empty()) {
        return {};
    }
    num::vec means(states.front().size(), 0.0);
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (states[index].size() != means.size()) {
            throw std::invalid_argument("states must have the same species count");
        }
        for (std::size_t species = 0; species < means.size(); ++species) {
            means[species] += probabilities[index] * states[index][species];
        }
    }
    return means;
}

/// Aggregate a joint distribution by one species count.
inline num::map<int, double> marginal_distribution(num::view<const state> states,
                                                            num::view<const double> probabilities,
                                                            std::size_t species) {
    if (states.size() != probabilities.size()) {
        throw std::invalid_argument("states and probabilities must have the same size");
    }
    num::map<int, double> marginal;
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (species >= states[index].size()) {
            throw std::out_of_range("species index is out of range");
        }
        marginal[states[index][species]] += probabilities[index];
    }
    return marginal;
}

} // namespace markovkit
