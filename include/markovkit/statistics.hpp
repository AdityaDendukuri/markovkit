#pragma once

#include "markovkit/state.hpp"
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

namespace markovkit {

/// Compute probability-weighted means for every species.
inline std::vector<double> species_means(std::span<const State> states,
                                         std::span<const double> probabilities) {
  if (states.size() != probabilities.size()) {
    throw std::invalid_argument("states and probabilities must have the same size");
  }
  if (states.empty()) {
    return {};
  }
  std::vector<double> means(states.front().size(), 0.0);
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
inline std::map<int, double> marginal_distribution(std::span<const State> states,
                                                   std::span<const double> probabilities,
                                                   std::size_t species) {
  if (states.size() != probabilities.size()) {
    throw std::invalid_argument("states and probabilities must have the same size");
  }
  std::map<int, double> marginal;
  for (std::size_t index = 0; index < states.size(); ++index) {
    if (species >= states[index].size()) {
      throw std::out_of_range("species index is out of range");
    }
    marginal[states[index][species]] += probabilities[index];
  }
  return marginal;
}

} // namespace markovkit
