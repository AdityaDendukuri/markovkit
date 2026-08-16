/// @file reaction_system.hpp
/// @brief ReactionSystem: stoichiometry + propensity functions for the CME.
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include <functional>
#include <vector>

namespace fsp {

/// @brief Propensity function signature: a_k(x, rates, t) -> double
using PropensityFn = std::function<double(const MultiIndex &,
                                          const std::vector<double> &, double)>;

/// @brief Discrete stochastic reaction network.
///
/// Each reaction k is described by:
///   - stoich_vecs[k]: net change in molecule counts (delta x when reaction
///   fires)
///   - propensities[k]: a_k(x, rates, t) -- the rate at which reaction k fires
///     given current state x, parameter vector rates, and time t.
struct ReactionSystem {
  std::vector<MultiIndex> stoich_vecs;    ///< net stoichiometry per reaction
  std::vector<PropensityFn> propensities; ///< propensity functions

  std::size_t n_reactions() const { return stoich_vecs.size(); }
};

} // namespace fsp
