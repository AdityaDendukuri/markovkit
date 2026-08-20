#include "ssa/tau_leaping.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace ssa {

Trajectory tau_leaping(const fsp::ReactionSystem& model,
                       const std::vector<double>& rates,
                       const MultiIndex& initial,
                       double t0,
                       double tf,
                       LeapOptions options,
                       int seed) {
  std::mt19937 rng(seed);
  Trajectory path;
  path.t.push_back(t0);
  path.states.push_back(initial);

  MultiIndex current = initial;
  double t = t0;
  std::vector<double> prop(model.n_reactions(), 0.0);

  while (t < tf) {
    double total = 0.0;
    for (std::size_t r = 0; r < model.n_reactions(); ++r) {
      prop[r] = std::max(0.0, model.propensities[r](current, rates, t));
      total += prop[r];
    }
    if (!(total > 0.0)) break;

    double dt = options.fixed_step > 0.0 ? options.fixed_step
                                         : std::max(options.min_step, options.epsilon / total);
    dt = std::min(dt, tf - t);
    t += dt;

    for (std::size_t r = 0; r < model.n_reactions(); ++r) {
      if (prop[r] <= 0.0) continue;
      std::poisson_distribution<int> poisson(prop[r] * dt);
      int count = poisson(rng);
      if (count <= 0) continue;

      for (std::size_t c = 0; c < current.size(); ++c) {
        current[c] = std::max(0, current[c] + count * model.stoich_vecs[r][c]);
      }
    }

    path.t.push_back(t);
    path.states.push_back(current);
  }
  return path;
}

} // namespace ssa
