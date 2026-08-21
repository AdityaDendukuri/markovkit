#pragma once

#include "markovkit/state.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace cme {

using Propensity =
    std::function<double(const markovkit::State&, const std::vector<double>&, double)>;

/// Stoichiometric changes and their state-dependent reaction rates.
struct ReactionSystem {
  // changes[r] and propensities[r] describe the same reaction.
  std::vector<markovkit::State> changes;
  std::vector<Propensity> propensities;

  [[nodiscard]] std::size_t size() const noexcept {
    return changes.size();
  }

  void validate() const {
    if (changes.size() != propensities.size()) {
      throw std::invalid_argument("reaction changes and propensities must have the same size");
    }
  }
};

} // namespace cme
