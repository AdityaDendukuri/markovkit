/// @file src/ssa/gillespie.cpp
/// @brief Compiled Gillespie Direct Method SSA simulation.
#include "markovkit/ssa/gillespie.hpp"

namespace markovkit::ssa {

SSATrajectory gillespie_direct_simulate(const fsp::ReactionSystem &model,
                                        const std::vector<num::real> &rates,
                                        const fsp::MultiIndex &u0, num::real t0,
                                        num::real tf, int rng_seed) {
  SSATrajectory traj;
  traj.t.push_back(t0);
  traj.x.push_back(u0);

  std::mt19937 rng(rng_seed);
  std::uniform_real_distribution<num::real> dist(0.0, 1.0);

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

    if (a0 <= 0.0) {
      current_t = tf;
      traj.t.push_back(current_t);
      traj.x.push_back(current_x);
      break;
    }

    num::real u1 = dist(rng);
    num::real dt = -std::log(u1) / a0;

    if (current_t + dt > tf) {
      current_t = tf;
      traj.t.push_back(current_t);
      traj.x.push_back(current_x);
      break;
    }

    current_t += dt;

    num::real u2 = dist(rng) * a0;
    num::real cumulative = 0.0;
    num::idx fired_rx = 0;

    for (num::idx r = 0; r < n_rx; ++r) {
      cumulative += a[r];
      if (u2 <= cumulative) {
        fired_rx = r;
        break;
      }
    }

    for (std::size_t c = 0; c < current_x.size(); ++c) {
      current_x[c] += model.stoich_vecs[fired_rx][c];
    }

    traj.t.push_back(current_t);
    traj.x.push_back(current_x);
    traj.rx_idx.push_back(fired_rx);
  }

  return traj;
}

} // namespace markovkit::ssa
