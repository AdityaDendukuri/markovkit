#include "ssa/gillespie.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace ssa {

Trajectory gillespie(const fsp::ReactionSystem& model,
                     const std::vector<double>& rates,
                     const MultiIndex& initial,
                     double t0,
                     double tf,
                     int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

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

    double dt = -std::log(std::max(uniform(rng), 1e-300)) / total;
    if (t + dt > tf) break;
    t += dt;

    double target = uniform(rng) * total;
    double csum = 0.0;
    std::size_t fired = model.n_reactions() - 1;
    for (std::size_t r = 0; r < model.n_reactions(); ++r) {
      csum += prop[r];
      if (target <= csum) {
        fired = r;
        break;
      }
    }

    current = current + model.stoich_vecs[fired];
    path.t.push_back(t);
    path.states.push_back(current);
    path.reactions.push_back(fired);
  }

  path.t.push_back(tf);
  path.states.push_back(current);
  return path;
}

} // namespace ssa
