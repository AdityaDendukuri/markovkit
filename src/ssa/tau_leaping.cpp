/// @file src/ssa/tau_leaping.cpp
/// @brief Compiled Explicit Poisson Tau-Leaping simulation.
#include "markovkit/ssa/tau_leaping.hpp"

namespace markovkit::ssa {

SSATrajectory tau_leaping_simulate(const fsp::ReactionSystem &model,
                                   const std::vector<num::real> &rates,
                                   const fsp::MultiIndex &u0, num::real t0,
                                   num::real tf, TauLeapingParams params,
                                   int rng_seed) {
  SSATrajectory traj;
  traj.t.push_back(t0);
  traj.x.push_back(u0);

  std::mt19937 rng(rng_seed);

  fsp::MultiIndex current_x = u0;
  num::real current_t = t0;
  num::idx n_rx = model.propensities.size();

  std::vector<num::real> a(n_rx, 0.0);

  while (current_t < tf) {
    num::real a0 = 0.0;
    for (num::idx r = 0; r < n_rx; ++r) {
      a[r] = model.propensities[r](current_x, rates, current_t);
      if (a[r] < 0.0)
        a[r] = 0.0;
      a0 += a[r];
    }

    if (a0 <= 0.0)
      break;

    num::real tau = params.tau_fixed;
    if (tau <= 0.0) {
      tau = std::max(params.min_tau, params.epsilon / a0);
    }

    if (current_t + tau > tf)
      tau = tf - current_t;

    current_t += tau;

    for (num::idx r = 0; r < n_rx; ++r) {
      if (a[r] <= 0.0)
        continue;
      std::poisson_distribution<int> pois(a[r] * tau);
      int k_r = pois(rng);

      for (std::size_t c = 0; c < current_x.size(); ++c) {
        current_x[c] += k_r * model.stoich_vecs[r][c];
        if (current_x[c] < 0)
          current_x[c] = 0; // Boundary clamp
      }
    }

    traj.t.push_back(current_t);
    traj.x.push_back(current_x);
  }

  return traj;
}

} // namespace markovkit::ssa
