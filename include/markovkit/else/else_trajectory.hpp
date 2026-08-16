/// @file else/else_trajectory.hpp
/// @brief Expected-Visits Low-Probability State Extrapolation (ELSE) trajectory solver.
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include <numerics.hpp>
#include <random>
#include <vector>

namespace markovkit::else_solver {

struct ELSETrajectory {
  std::vector<double> t;
  std::vector<fsp::MultiIndex> x;
  int number_of_else_steps = 0;
};

/// @brief Simulate a single trajectory using Expected-Visits ELSE.
ELSETrajectory else_trajectory_simulate(const fsp::ReactionSystem& model,
                                        const std::vector<double>& rates,
                                        const fsp::MultiIndex& u0,
                                        double t0,
                                        double tf,
                                        int radius = 3,
                                        int rng_seed = 42);

} // namespace markovkit::else_solver
