/// @file ssa/tau_leaping.hpp
/// @brief Explicit Poisson Tau-Leaping for accelerated stochastic simulation.
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include "markovkit/ssa/gillespie.hpp"
#include <numerics.hpp>
#include <random>
#include <vector>

namespace markovkit::ssa {

struct TauLeapingParams {
  num::real tau_fixed = 0.0; ///< fixed step tau (0 = adaptive tau control)
  num::real epsilon = 0.03;  ///< relative propensity change tolerance for adaptive step
  num::real min_tau = 1e-6;
};

/// @brief Simulate a single trajectory using Explicit Poisson Tau-Leaping.
SSATrajectory tau_leaping_simulate(const fsp::ReactionSystem& model,
                                   const std::vector<num::real>& rates,
                                   const fsp::MultiIndex& u0,
                                   num::real t0,
                                   num::real tf,
                                   TauLeapingParams params = {},
                                   int rng_seed = 42);

} // namespace markovkit::ssa
