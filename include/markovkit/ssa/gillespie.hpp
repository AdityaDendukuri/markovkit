/// @file ssa/gillespie.hpp
/// @brief Gillespie SSA (Direct Method & Next Reaction Method) for CME
/// stochastic trajectory simulation.
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include <numerics.hpp>
#include <random>
#include <vector>

namespace markovkit::ssa {

/// @brief Single trajectory result from Gillespie SSA.
struct SSATrajectory {
  std::vector<num::real> t;       ///< event jump times
  std::vector<fsp::MultiIndex> x; ///< multi-index state history
  std::vector<num::idx> rx_idx;   ///< index of reaction fired per step
};

/// @brief Simulate a single trajectory using Gillespie Direct Method SSA.
SSATrajectory gillespie_direct_simulate(const fsp::ReactionSystem &model,
                                        const std::vector<num::real> &rates,
                                        const fsp::MultiIndex &u0, num::real t0,
                                        num::real tf, int rng_seed = 42);

} // namespace markovkit::ssa
