/// @file else/ensemble.hpp
/// @brief Ensemble ELSE: multi-trajectory stochastic sampling sharing
/// subnetwork solves.
#pragma once

#include "markovkit/else/subnetwork.hpp"
#include <vector>

namespace markovkit::else_solver {

struct EnsembleTrajectory {
  int current_state;
  double time;
  bool active;
};

struct EnsembleResult {
  int n_trajectories;
  double final_time;
  std::vector<int> final_states;
  std::vector<double> exit_times;
};

/// @brief Run Ensemble ELSE over N parallel trajectories sharing a subnetwork
/// generator solve.
EnsembleResult ensemble_else_simulate(std::size_t n_trajectories,
                                      const Subnetwork& subnetwork,
                                      double t_max,
                                      int rng_seed = 42);

} // namespace markovkit::else_solver
