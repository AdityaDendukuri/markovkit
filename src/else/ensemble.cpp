/// @file src/else/ensemble.cpp
/// @brief Compiled Ensemble ELSE solver with OpenMP multi-threading.
#include "markovkit/else/ensemble.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace markovkit::else_solver {

EnsembleResult ensemble_else_simulate(std::size_t n_trajectories,
                                      const Subnetwork &subnetwork,
                                      double t_max, int rng_seed) {
  num::idx n_states = subnetwork.R.rows();

  EnsembleResult res;
  res.n_trajectories = static_cast<int>(n_trajectories);
  res.final_time = t_max;
  res.final_states.resize(n_trajectories, 0);
  res.exit_times.resize(n_trajectories, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
  for (std::size_t k = 0; k < n_trajectories; ++k) {
    std::mt19937 rng(rng_seed + static_cast<int>(k));
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int state = 0;
    double t = 0.0;

    while (t < t_max) {
      double tau = subnetwork.Z(state, state);
      if (tau <= 0.0)
        break;

      double u = dist(rng);
      double dt = -tau * std::log(u);
      if (t + dt > t_max) {
        t = t_max;
        break;
      }

      t += dt;

      double r_ii = -subnetwork.R(state, state);
      double r = dist(rng) * r_ii;
      double cumulative = 0.0;
      int next_state = state;

      for (num::idx j = 0; j < n_states; ++j) {
        if (j == (num::idx)state)
          continue;
        cumulative += subnetwork.R(state, j);
        if (r <= cumulative) {
          next_state = static_cast<int>(j);
          break;
        }
      }

      state = next_state;
    }

    res.final_states[k] = state;
    res.exit_times[k] = t;
  }

  return res;
}

} // namespace markovkit::else_solver
